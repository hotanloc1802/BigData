import os
import sys
import time
import json
import re
import fasttext
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, lit, current_timestamp, pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, LongType, FloatType

# Pyvi import
from pyvi import ViTokenizer

# Hugging Face Transformers imports
from transformers import AutoTokenizer, AutoModel

# TorchCRF import (ensure you have it installed: pip install TorchCRF)
# Corrected import: 'CRF' does not take 'batch_first' in its __init__
from TorchCRF import CRF 

# --- A. Thiet lap moi truong Python cho Spark ---
# KIEM TRA LAI DUONG DAN NAY!
# Dam bao duong dan nay tro den Python interpreter trong moi truong Conda cua ban.
CONDA_ENV_PYTHON_PATH = r"D:\DevTools\Anacoda\envs\spark311_env\python.exe" 

os.environ['PYSPARK_PYTHON'] = CONDA_ENV_PYTHON_PATH
os.environ['PYSPARK_DRIVER_PYTHON'] = CONDA_ENV_PYTHON_PATH

print(f"Da dat PYSPARK_PYTHON va PYSPARK_DRIVER_PYTHON thanh: {CONDA_ENV_PYTHON_PATH}")
print("--- Thong tin Python cua Driver Script ---")
print(f"Duong dan Python thuc thi (Driver): {sys.executable}")
print(f"Phien ban Python (Driver): {sys.version}")
print("--- Ket thong tin ---")

# --- B0. Khoi tao SparkSession va Cau hinh ---
KAFKA_CONNECTOR_VERSION = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"
COMMONS_POOL2_VERSION = "org.apache.commons:commons-pool2:2.12.0"

spark = SparkSession.builder \
    .appName("ABSA_Kafka_Prediction_Stream_Consolidated") \
    .config("spark.jars.packages", f"{KAFKA_CONNECTOR_VERSION},{COMMONS_POOL2_VERSION}") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.python.worker.memory", "4g") \
    .config("spark.sql.adaptive.enabled", "false") \
    .config("spark.driver.maxResultSize", "0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("SparkSession da duoc khoi tao va cau hinh.")

# --- Dinh nghia cac Hang so va Duong dan (tu model.py, embedding_utils.py, prediction_pipeline.py) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) 

FASTTEXT_MODEL_PATH = os.path.join(project_root, "models", "fasttext", "cc.vi.300.bin") 
XLMR_MODEL_NAME = os.path.join(project_root, "models", "huggingface", "hub", "models--xlm-roberta-base", "snapshots", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089")
MODEL_SAVE_PATH = os.path.join(project_root, "models", "absa_bilstm_crf_model.pth")

FASTTEXT_EMBEDDING_DIM = 300
CHAR_EMBEDDING_DIM = 50
CHAR_LSTM_HIDDEN_DIM = 50
CHAR_FEATURE_OUTPUT_DIM = CHAR_LSTM_HIDDEN_DIM * 2 
XLMR_EMBEDDING_DIM = 768 

FUSED_EMBEDDING_DIM = FASTTEXT_EMBEDDING_DIM + CHAR_FEATURE_OUTPUT_DIM + XLMR_EMBEDDING_DIM

LSTM_HIDDEN_DIM = 256
NUM_LSTM_LAYERS = 2
DROPOUT_RATE = 0.5

# --- Char_to_idx (tu embedding_utils.py) ---
char_to_idx = {char: i for i, char in enumerate(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789áàảạãăằẳẵặâầẩẫậéèẻẹẽêềểễệíìỉịĩóòỏọõôồổỗộơờởỡợúùủụũưừửữựýỳỷỵỹđĐ,.:;?!'\"()[]{}<>/-_&@#$%^&*=+~` "
)}
char_to_idx["<PAD>"] = 0
char_to_idx["<unk>"] = len(char_to_idx)
CHAR_VOCAB_SIZE = len(char_to_idx)


# --- Embedder Classes (tu embedding_utils.py) ---
class FastTextEmbedder:
    def __init__(self, model_path=FASTTEXT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FastText model not found at: {model_path}")
        self.model = fasttext.load_model(model_path)
    def get_embedding(self, word):
        return torch.from_numpy(self.model.get_word_vector(word)).float()

class CharBiLSTMEmbedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=CHAR_EMBEDDING_DIM, hidden_dim=CHAR_LSTM_HIDDEN_DIM):
        super(CharBiLSTMEmbedder, self).__init__()
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=char_to_idx["<PAD>"])
        self.char_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.output_dim = hidden_dim * 2
    def forward(self, char_indices_list_per_word):
        word_char_representations = []
        for char_indices_for_word in char_indices_list_per_word:
            if char_indices_for_word.numel() == 0: 
                word_char_representations.append(torch.zeros(self.output_dim))
                continue
            char_embs = self.char_embedding(char_indices_for_word.unsqueeze(0)) 
            _, (h_n, _) = self.char_lstm(char_embs)
            char_word_representation = torch.cat((h_n[0], h_n[1]), dim=1).squeeze(0)
            word_char_representations.append(char_word_representation)
        if not word_char_representations:
            return torch.empty(0, self.output_dim) 
        return torch.stack(word_char_representations)

class XLMREmbedder:
    def __init__(self, model_name=XLMR_MODEL_NAME):
        if not os.path.exists(model_name) or \
           not (os.path.exists(os.path.join(model_name, 'tokenizer.json')) or \
                os.path.exists(os.path.join(model_name, 'config.json'))):
            raise FileNotFoundError(f"XLM-RoBERTa model not found at: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    def get_token_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        offset_mapping = inputs.pop('offset_mapping') 
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state.squeeze(0) 
        return token_embeddings, inputs['input_ids'].squeeze(0), offset_mapping.squeeze(0)

# --- Preprocessing Helpers (tu embedding_utils.py) ---
def clean_text(text): 
    text = re.sub(r'[\r\n\t]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def process_text_to_word_tokens(text):
    cleaned_text = clean_text(text)
    tokens_str = ViTokenizer.tokenize(cleaned_text)
    tokens = tokens_str.replace("_", " ").split()
    
    token_info_list = []
    current_char_idx = 0
    for i, token in enumerate(tokens):
        while current_char_idx < len(cleaned_text) and cleaned_text[current_char_idx].isspace():
            current_char_idx += 1
            
        token_start = cleaned_text.find(token, current_char_idx)
        if token_start != -1:
            token_end = token_start + len(token)
            token_info_list.append({
                'token_text': token,
                'start_char': token_start,
                'end_char': token_end,
                'token_idx': i
            })
            current_char_idx = token_end
        else:
            token_info_list.append({'token_text': token, 'start_char': -1, 'end_char': -1, 'token_idx': i})
            current_char_idx += len(token) 
    return tokens, cleaned_text, token_info_list

def get_char_indices_for_words(cleaned_text, token_info_list, char_to_idx_map):
    char_indices_list_per_word = []
    for token_info in token_info_list:
        char_span = cleaned_text[token_info['start_char']:token_info['end_char']] if token_info['start_char'] != -1 else token_info['token_text']
        
        word_char_indices = [char_to_idx_map.get(c, char_to_idx_map['<unk>']) for c in char_span]
        char_indices_list_per_word.append(torch.tensor(word_char_indices, dtype=torch.long))
    return char_indices_list_per_word

def align_xlmr_embeddings_to_words(
    xlmr_token_embeddings, xlmr_input_ids, xlmr_offset_mapping,
    pyvi_tokens, xlmr_tokenizer_instance, original_text_for_alignment
):
    word_embeddings = []
    pyvi_token_spans = []
    temp_current_char_idx = 0
    for token in pyvi_tokens:
        token_start = original_text_for_alignment.find(token, temp_current_char_idx)
        if token_start != -1:
            token_end = token_start + len(token)
            pyvi_token_spans.append((token_start, token_end))
            temp_current_char_idx = token_end
        else:
            pyvi_token_spans.append((-1, -1)) 
            temp_current_char_idx += len(token) 

    xlmr_word_ids = [] 
    for i, (subword_start_char, subword_end_char) in enumerate(xlmr_offset_mapping):
        if xlmr_input_ids[i] in xlmr_tokenizer_instance.all_special_ids:
            xlmr_word_ids.append(None)
            continue
        
        found_word_id = None
        for pyvi_idx, (pyvi_start_char, pyvi_end_char) in enumerate(pyvi_token_spans):
            if pyvi_start_char == -1: continue
            
            if max(subword_start_char, pyvi_start_char) < min(subword_end_char, pyvi_end_char):
                found_word_id = pyvi_idx
                break
        xlmr_word_ids.append(found_word_id)

    grouped_embeddings = [[] for _ in range(len(pyvi_tokens))]
    for i, word_id in enumerate(xlmr_word_ids):
        if word_id is not None and word_id < len(grouped_embeddings):
            grouped_embeddings[word_id].append(xlmr_token_embeddings[i])

    for group in grouped_embeddings:
        if group:
            word_embeddings.append(torch.stack(group).mean(dim=0))
        else:
            word_embeddings.append(torch.zeros(XLMR_EMBEDDING_DIM))

    if len(word_embeddings) > len(pyvi_tokens):
        word_embeddings = word_embeddings[:len(pyvi_tokens)]
    elif len(word_embeddings) < len(pyvi_tokens):
        word_embeddings.extend([torch.zeros(XLMR_EMBEDDING_DIM)] * (len(pyvi_tokens) - len(word_embeddings)))

    if not word_embeddings:
        return torch.empty(0, XLMR_EMBEDDING_DIM)

    return torch.stack(word_embeddings)


def get_fused_embeddings_and_iob2_labels(sample, fasttext_embedder, char_bilstm_embedder, xlmr_embedder):
    text = sample["text"]
    raw_labels = sample.get("labels", []) 

    tokens, cleaned_text, token_info_list = process_text_to_word_tokens(text)
    if not tokens:
        return None, None, None

    iob2_labels = ['O'] * len(tokens)
    for start_char_label, end_char_label, label_tag in raw_labels:
        affected_token_indices = []
        for token_info in token_info_list:
            token_start = token_info['start_char']
            token_end = token_info['end_char']
            token_idx = token_info['token_idx']
            if max(token_start, start_char_label) < min(token_end, end_char_label):
                affected_token_indices.append(token_idx)
        
        if affected_token_indices:
            affected_token_indices.sort()
            iob2_labels[affected_token_indices[0]] = f"B-{label_tag}"
            for i in range(1, len(affected_token_indices)):
                iob2_labels[affected_token_indices[i]] = f"I-{label_tag}"

    if not tokens: # Double check after initial check
        return None, None, None

    syllable_embeddings = torch.stack([fasttext_embedder.get_embedding(token) for token in tokens])
    
    char_indices_list_per_word = get_char_indices_for_words(cleaned_text, token_info_list, char_to_idx) 
    character_embeddings = char_bilstm_embedder(char_indices_list_per_word)
    
    xlmr_token_embs, xlmr_input_ids, xlmr_offset_mapping = xlmr_embedder.get_token_embeddings(text)
    contextual_embeddings = align_xlmr_embeddings_to_words(
        xlmr_token_embs,
        xlmr_input_ids,
        xlmr_offset_mapping,
        tokens,
        xlmr_embedder.tokenizer,
        cleaned_text
    )

    min_len = min(syllable_embeddings.shape[0], character_embeddings.shape[0], contextual_embeddings.shape[0])
    
    if not (syllable_embeddings.shape[0] == character_embeddings.shape[0] == contextual_embeddings.shape[0]):
        syllable_embeddings = syllable_embeddings[:min_len]
        character_embeddings = character_embeddings[:min_len]
        contextual_embeddings = contextual_embeddings[:min_len]
        tokens = tokens[:min_len] 
        iob2_labels = iob2_labels[:min_len] 
        
    if min_len == 0:
        return None, None, None

    fused_embeddings = torch.cat(
        (syllable_embeddings, character_embeddings, contextual_embeddings), dim=1
    )

    return fused_embeddings, iob2_labels, tokens 


# --- BiLSTM_CRF_Model (tu model.py) ---
class BiLSTM_CRF_Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_tags, num_layers=1, dropout=0.0):
        super(BiLSTM_CRF_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

        self.hidden2tag = nn.Linear(hidden_dim, num_tags)

        # CORRECTED LINE: Removed batch_first=True
        self.crf = CRF(num_tags)

    def forward(self, embeddings, tags=None, mask=None):
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.hidden2tag(lstm_out)

        if tags is not None:
            # The 'mask' argument should be passed to the CRF's loss calculation
            # For TorchCRF, the mask is typically handled by the forward/decode method.
            # Make sure your 'tags' also align with the batch_first behavior if needed.
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            # The 'mask' argument should be passed to the CRF's decode method
            return self.crf.decode(emissions, mask=mask)
# --- Global Model Cache ---
_model_cache = {}

# --- Model Loading Function for UDF ---
_udf_fasttext_embedder = None
_udf_char_bilstm_embedder = None
_udf_xlmr_embedder = None
_udf_absa_model = None
_udf_idx_to_tag_map = None
_udf_tag_to_idx_map = None
_udf_model_device = None
_udf_char_to_idx_loaded = None 

def load_absa_model_for_udf(model_path=MODEL_SAVE_PATH):
    """
    Tải mô hình ABSA đã huấn luyện và các embedder cho Pandas UDF.
    Sử dụng cache toàn cục _model_cache.
    """
    global _udf_fasttext_embedder, _udf_char_bilstm_embedder, _udf_xlmr_embedder, \
           _udf_absa_model, _udf_idx_to_tag_map, _udf_tag_to_idx_map, \
           _udf_model_device, _udf_char_to_idx_loaded

    if 'absa_model' in _model_cache and _model_cache['absa_model'] != "LOAD_ERROR":
        _udf_fasttext_embedder = _model_cache['fasttext_embedder']
        _udf_char_bilstm_embedder = _model_cache['char_bilstm_embedder']
        _udf_xlmr_embedder = _model_cache['xlmr_embedder']
        _udf_absa_model = _model_cache['absa_model']
        _udf_idx_to_tag_map = _model_cache['idx_to_tag_map']
        _udf_tag_to_idx_map = _model_cache['tag_to_idx_map']
        _udf_model_device = _model_cache['device']
        _udf_char_to_idx_loaded = _model_cache['char_to_idx_loaded']
        return True

    print(f"Worker process: Loading ABSA model from {model_path}...")
    
    _udf_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(model_path, map_location=_udf_model_device)

        _udf_fasttext_embedder = FastTextEmbedder(FASTTEXT_MODEL_PATH)

        _udf_char_to_idx_loaded = checkpoint.get('char_to_idx') 
        if _udf_char_to_idx_loaded is None:
            raise ValueError("'char_to_idx' not found in checkpoint. Cannot initialize CharBiLSTMEmbedder.")

        _udf_char_bilstm_embedder = CharBiLSTMEmbedder(
            vocab_size=len(_udf_char_to_idx_loaded), 
            embedding_dim=CHAR_EMBEDDING_DIM, 
            hidden_dim=CHAR_LSTM_HIDDEN_DIM 
        ).to(_udf_model_device)
        
        if 'char_bilstm_state_dict' in checkpoint:
            _udf_char_bilstm_embedder.load_state_dict(checkpoint['char_bilstm_state_dict'])
            print("Worker process: CharBiLSTMEmbedder state_dict loaded.")
        else:
            print("Worker process: WARNING: 'char_bilstm_state_dict' not found in checkpoint. CharBiLSTMEmbedder initialized with random weights. This will impact performance if it was trained.")


        _udf_xlmr_embedder = XLMREmbedder(XLMR_MODEL_NAME)
        _udf_xlmr_embedder.model.to(_udf_model_device) 

        _udf_absa_model = BiLSTM_CRF_Model(
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_tags=checkpoint['num_tags'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout']
        ).to(_udf_model_device)
        
        _udf_absa_model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        _udf_absa_model.eval() 

        _udf_idx_to_tag_map = checkpoint['idx_to_tag']
        _udf_tag_to_idx_map = checkpoint['current_tag_to_idx'] 

        _model_cache['fasttext_embedder'] = _udf_fasttext_embedder
        _model_cache['char_bilstm_embedder'] = _udf_char_bilstm_embedder
        _model_cache['xlmr_embedder'] = _udf_xlmr_embedder
        _model_cache['absa_model'] = _udf_absa_model
        _model_cache['idx_to_tag_map'] = _udf_idx_to_tag_map
        _model_cache['tag_to_idx_map'] = _udf_tag_to_idx_map
        _model_cache['device'] = _udf_model_device
        _model_cache['char_to_idx_loaded'] = _udf_char_to_idx_loaded 

        print("Worker process: ABSA model and embedders loaded successfully.")
        return True

    except FileNotFoundError:
        print(f"Worker process: ERROR: Model file not found at {model_path}. Please ensure the model is trained and saved.")
        _model_cache['absa_model'] = "LOAD_ERROR" 
        return False
    except Exception as e:
        print(f"Worker process: ERROR loading model or embedders: {e}")
        import traceback
        traceback.print_exc()
        _model_cache['absa_model'] = "LOAD_ERROR"
        return False

# --- Schemas (UPDATED to match flat Kafka message) ---
tiktok_comment_schema_flat = StructType([
    StructField("comment_text", StringType(), True),
    StructField("video_id", StringType(), True),
    StructField("comment_id", StringType(), True),
    StructField("created_time", LongType(), True) 
])

prediction_udf_schema = ArrayType(StructType([
    StructField("span_text", StringType(), True),
    StructField("tag_type", StringType(), True),
    StructField("start_token_idx", LongType(), True),
    StructField("end_token_idx", LongType(), True)
]))


# --- UDF Definition (predict_absa_spans_udf) ---
@pandas_udf(prediction_udf_schema, PandasUDFType.SCALAR) 
def predict_absa_spans_udf(comment_texts: pd.Series, video_ids: pd.Series, comment_ids: pd.Series) -> pd.Series: 
    load_absa_model_for_udf() 
    
    absa_model = _udf_absa_model
    idx_to_tag_map = _udf_idx_to_tag_map
    fasttext_embedder = _udf_fasttext_embedder
    char_bilstm_embedder = _udf_char_bilstm_embedder
    xlmr_embedder = _udf_xlmr_embedder
    model_device = _udf_model_device
    char_to_idx_from_checkpoint = _udf_char_to_idx_loaded 

    if absa_model == "LOAD_ERROR" or absa_model is None:
        print("Worker process: Model not available, returning empty spans for batch.")
        # --- FIX: Removed dtype argument for pd.Series ---
        return pd.Series([[] for _ in range(len(comment_texts))]) 
    
    print("Worker process: UDF received a batch. Starting processing.")

    if comment_texts.empty:
        print("Worker process: Empty comment batch received.")
        return pd.Series([]) # Empty Series, no need for list comprehension

    current_batch_df = pd.DataFrame({
        'comment_text': comment_texts,
        'video_id': video_ids,
        'comment_id': comment_ids
    })

    print(f"Worker process: Inside batch. Type of batch DataFrame: {type(current_batch_df)}")
    print(f"Worker process: Is batch empty? {current_batch_df.empty}")
    print(f"Worker process: Batch columns: {current_batch_df.columns.tolist()}")
    print(f"Worker process: First row of batch: \n{current_batch_df.head(1).to_string()}")

    results_list = []
    
    print(f"Worker process: Processing batch with {len(current_batch_df)} comments.")
    for idx, row in current_batch_df.iterrows(): 
        comment_text = row['comment_text']
        comment_id = row['comment_id']
        
        print(f"Worker process: Processing comment_id: {comment_id}, text: '{comment_text[:50]}...'")

        if not pd.notnull(comment_text) or not isinstance(comment_text, str) or not comment_text.strip():
            print(f"Worker process: Skipping invalid comment_text for comment_id: {comment_id}")
            results_list.append([])
            continue
        
        try:
            sample_data_for_embeddings = {"text": comment_text, "labels": []} 

            print(f"Worker process: Calling get_fused_embeddings_and_iob2_labels for comment_id: {comment_id}")
            fused_embeddings, _, word_tokens = get_fused_embeddings_and_iob2_labels(
                sample_data_for_embeddings, fasttext_embedder, char_bilstm_embedder, xlmr_embedder
            )
            print(f"Worker process: Fused embeddings generated for comment_id: {comment_id}. Tokens count: {len(word_tokens) if word_tokens else 0}")

            if fused_embeddings is None or len(word_tokens) == 0:
                print(f"Worker process: Fused embeddings are None or no tokens for comment_id: {comment_id}. Skipping prediction.")
                results_list.append([])
                continue
            
            print(f"Worker process: Moving embeddings to device {model_device} for comment_id: {comment_id}")
            if fused_embeddings.dim() == 2:
                fused_embeddings_batch = fused_embeddings.unsqueeze(0).to(model_device)
            else: 
                fused_embeddings_batch = fused_embeddings.to(model_device)
            
            mask_batch = torch.ones(fused_embeddings_batch.shape[0], fused_embeddings_batch.shape[1], dtype=torch.bool, device=model_device)
            
            print(f"Worker process: Performing model inference for comment_id: {comment_id}")
            absa_model.eval() 
            with torch.no_grad():
                predicted_tag_ids_list_of_lists = absa_model(fused_embeddings_batch, mask=mask_batch)
            
            if not predicted_tag_ids_list_of_lists: 
                predicted_iob_labels = []
                print(f"Worker process: WARNING: Model returned an empty list of tag IDs for comment_id: {comment_id}")
            else:
                predicted_tag_ids_single_seq = predicted_tag_ids_list_of_lists[0] 
                
                # The CRF.decode method from TorchCRF (without batch_first in init) typically returns 
                # a list of lists of ints for a batch, or a list of ints for a single sequence.
                # So, predicted_tag_ids_single_seq should already be a list of ints.
                # No need for .item() or .ndim checks here IF TorchCRF.CRF.decode behaves as expected.
                # However, for robustness if it *still* returns a tensor for some reason:
                if isinstance(predicted_tag_ids_single_seq, torch.Tensor):
                    # Convert to CPU and then to a Python list
                    predicted_tag_ids_single_seq = predicted_tag_ids_single_seq.cpu().tolist()
                
                predicted_iob_labels = [idx_to_tag_map[tag_id] for tag_id in predicted_tag_ids_single_seq]
            
            print(f"Worker process: Model inference complete for comment_id: {comment_id}. Predicted labels count: {len(predicted_iob_labels)}")

            print(f"Worker process: Extracting spans for comment_id: {comment_id}")
            extracted_spans = _extract_spans_from_iob2(word_tokens, predicted_iob_labels) 
            results_list.append(extracted_spans)
            print(f"Worker process: Spans extracted for comment_id: {comment_id}. Number of spans: {len(extracted_spans)}")

        except Exception as e:
            print(f"Worker process: ERROR processing comment '{comment_id}': {e}")
            import traceback
            traceback.print_exc() 
            results_list.append([]) 
            continue 

    # --- FIX: Removed dtype argument for pd.Series ---
    return pd.Series(results_list) 

# --- B1. Doc stream tu Kafka ---
KAFKA_TOPIC_INPUT = "tiktok_comment" 
KAFKA_TOPIC_OUTPUT_PREDICTIONS = "absa_predictions" 
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

print(f"Dang thiet lap doc stream tu Kafka topic: {KAFKA_TOPIC_INPUT} tren server: {KAFKA_BOOTSTRAP_SERVERS}")
kafkaDF = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", KAFKA_TOPIC_INPUT) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()
print("Thiet lap doc stream tu Kafka thanh cong.")

# --- B2. Parse JSON va chuan bi du lieu (UPDATED for flat JSON) ---
print("Dang thiet lap parsing JSON va trich xuat comment tu cau truc phang.")
parsedDF = kafkaDF.selectExpr("CAST(value AS STRING) as json_str") \
    .withColumn("data", from_json(col("json_str"), tiktok_comment_schema_flat)) \
    .filter(col("data").isNotNull()) 

df_for_udf = parsedDF.select(
    col("data.comment_text").alias("comment_text"),
    col("data.video_id").alias("video_id"),
    col("data.comment_id").alias("comment_id"),
    col("data.created_time").alias("created_time_ms")
) \
.filter(col("comment_text").isNotNull() & (col("comment_text") != "") & (col("comment_text").cast(StringType()) != "NaN"))

print("Da thiet lap parsing JSON va trich xuat comment tu cau truc phang thanh cong.")


# --- B3. Ap dung Pandas UDF de du doan ABSA ---
df_with_predictions = df_for_udf.withColumn(
    "extracted_spans", predict_absa_spans_udf(col("comment_text"), col("video_id"), col("comment_id")) 
)

df_final_output = df_with_predictions.withColumn("processing_timestamp", current_timestamp())

print("Da ap dung UDF va tao DataFrame voi cot 'extracted_spans' va 'processing_timestamp'.")

# --- B4. Khoi chay Streaming Query de ghi ra Console (hoac Kafka topic khac) ---
output_mode = "append" 
checkpoint_dir_consumer = os.path.join("spark_checkpoints", "absa_prediction_console")

print(f"Dang khoi chay streaming query (ABSA Prediction - Console Output) voi output mode: {output_mode} va checkpoint: {checkpoint_dir_consumer}")

query = df_final_output.select(
        "video_id", 
        "comment_id", 
        "comment_text", 
        "extracted_spans", 
        "processing_timestamp"
    ) \
    .writeStream \
    .format("console") \
    .outputMode(output_mode) \
    .option("truncate", "false") \
    .option("numRows", 20) \
    .option("checkpointLocation", checkpoint_dir_consumer) \
    .trigger(processingTime="10 seconds") \
    .start()

print("Streaming query (ABSA Prediction - Console Output) da khoi chay! Cho du lieu tu Kafka...")
print("Ket qua se duoc in truc tiep ra console moi 10 giay.")

try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("Da nhan tin hieu dung (Ctrl+C). Dang dung query...")
    if query:
        try:
            query.stop()
        except Exception as e_stop:
            print(f"Loi khi dung query: {e_stop}")
    print("Query da dung.")
except Exception as e_stream:
    print(f"Loi trong qua trinh streaming: {e_stream}")
    if query:
        try:
            query.stop()
        except Exception as e_stop:
            print(f"Loi khi dung query do loi streaming: {e_stop}")
    print("Query da dung do loi.")
finally:
    if 'spark' in locals() and spark.getActiveSession():
        try:
            spark.stop()
        except Exception as e_spark_stop:
            print(f"Loi khi dong SparkSession: {e_spark_stop}")
    print("SparkSession da dong (neu co).")