import os
import json
import re
import fasttext
import torch
import torch.nn as nn
import numpy as np
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from TorchCRF import CRF # ĐÃ SỬA: Import đúng thư viện CRF (chữ t thường)
import asyncio # Thêm import asyncio
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
# --- Cấu hình chung ---
FASTTEXT_MODEL_PATH = "models/fasttext/cc.vi.300.bin" 
FASTTEXT_EMBEDDING_DIM = 300

CHAR_EMBEDDING_DIM = 50
CHAR_LSTM_HIDDEN_DIM = 50
CHAR_FEATURE_OUTPUT_DIM = CHAR_LSTM_HIDDEN_DIM * 2 

XLMR_MODEL_NAME = "models/huggingface/hub/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
XLMR_EMBEDDING_DIM = 768 

FUSED_EMBEDDING_DIM = FASTTEXT_EMBEDDING_DIM + CHAR_FEATURE_OUTPUT_DIM + XLMR_EMBEDDING_DIM

LSTM_HIDDEN_DIM = 256
NUM_LSTM_LAYERS = 2
DROPOUT_RATE = 0.5

# --- FastText Embedder ---
class FastTextEmbedder:
    def __init__(self, model_path=FASTTEXT_MODEL_PATH):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Could not find FastText model at: {model_path}")
            self.model = fasttext.load_model(model_path)
            self.embedding_dim = self.model.get_word_vector("test").shape[0] 
            print(f"✅ FastText model loaded from: {model_path} (dim: {self.embedding_dim})")
        except Exception as e:
            raise RuntimeError(f"Error loading FastText model from '{model_path}': {e}. "
                               "Please check the path and FastText library installation.")

    def get_embedding(self, word):
        return torch.from_numpy(self.model.get_word_vector(word)).float()

# --- Char BiLSTM Embedder ---
char_to_idx = {char: i for i, char in enumerate(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789áàảạãăằẳẵặâầẩẫậéèẻẹẽêềểễệíìỉịĩóòỏọõôồổỗộơờởỡợúùủụũưừửữựýỳỷỵỹđĐ,.:;?!'\"()[]{}<>/-_&@#$%^&*=+~` "
)}
char_to_idx["<PAD>"] = 0
char_to_idx["<unk>"] = len(char_to_idx)
CHAR_VOCAB_SIZE = len(char_to_idx)

class CharBiLSTMEmbedder(nn.Module):
    def __init__(self, vocab_size=CHAR_VOCAB_SIZE, embedding_dim=CHAR_EMBEDDING_DIM, hidden_dim=CHAR_LSTM_HIDDEN_DIM):
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

# --- XLM-RoBERta Embedder ---
class XLMREmbedder:
    def __init__(self, model_name=XLMR_MODEL_NAME):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.embedding_dim = self.model.config.hidden_size
            print(f"✅ XLM-RoBERTa model loaded from: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            raise RuntimeError(f"Error loading XLM-RoBERTa model from '{model_name}': {e}. "
                               "Please check model name or internet connection.")

    def get_token_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        offset_mapping = inputs.pop('offset_mapping') 
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state.squeeze(0) 
            
        return token_embeddings, inputs['input_ids'].squeeze(0), offset_mapping.squeeze(0)

# --- Preprocessing Helpers ---
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
            print(f"Warning: Could not align token '{token}' at index {i} in cleaned text.")
            token_info_list.append({'token_text': token, 'start_char': -1, 'end_char': -1, 'token_idx': i})
            current_char_idx += len(token)
    return tokens, cleaned_text, token_info_list

def get_char_indices_for_words(cleaned_text, token_info_list, char_to_idx):
    char_indices_list_per_word = []
    for token_info in token_info_list:
        char_span = cleaned_text[token_info['start_char']:token_info['end_char']] if token_info['start_char'] != -1 else token_info['token_text']
        word_char_indices = [char_to_idx.get(c, char_to_idx['<unk>']) for c in char_span]
        char_indices_list_per_word.append(torch.tensor(word_char_indices, dtype=torch.long))
    return char_indices_list_per_word

def align_xlmr_embeddings_to_words(
    xlmr_token_embeddings, xlmr_input_ids, xlmr_offset_mapping,
    pyvi_tokens, xlmr_tokenizer_instance, cleaned_text 
):
    word_embeddings = []

    pyvi_token_spans = []
    temp_current_char_idx = 0
    for token in pyvi_tokens:
        token_start = cleaned_text.find(token, temp_current_char_idx)
        if token_start != -1:
            token_end = token_start + len(token)
            pyvi_token_spans.append((token_start, token_end))
            temp_current_char_idx = token_end
        else:
            pyvi_token_spans.append((-1, -1)) 

    xlmr_word_ids = []
    for i, (subword_start_char, subword_end_char) in enumerate(xlmr_offset_mapping):
        if xlmr_input_ids[i] in xlmr_tokenizer_instance.all_special_ids:
            xlmr_word_ids.append(None)
            continue

        found_word_id = None
        for pyvi_idx, (pyvi_start_char, pyvi_end_char) in enumerate(pyvi_token_spans):
            if pyvi_start_char == -1:
                continue
            if max(subword_start_char, pyvi_start_char) < min(subword_end_char, pyvi_end_char):
                found_word_id = pyvi_idx
                break
        xlmr_word_ids.append(found_word_id)

    grouped_embeddings = [[] for _ in range(len(pyvi_tokens))]
    for i, word_id in enumerate(xlmr_word_ids):
        if word_id is not None and word_id < len(pyvi_tokens):
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

    return torch.stack(word_embeddings)

# --- Main function to generate Fused Embeddings and IOB2 Labels ---
def get_fused_embeddings_and_iob2_labels(sample, fasttext_embedder, char_bilstm_embedder, xlmr_embedder):
    text = sample["text"]
    raw_labels = sample.get("labels", [])

    tokens, cleaned_text, token_info_list = process_text_to_word_tokens(text)
    if not tokens:
        print(f"Warning: No tokens generated from text: '{text}'")
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

    if not (syllable_embeddings.shape[0] == character_embeddings.shape[0] == contextual_embeddings.shape[0]):
        print(f"⚠️ Embedding dimensions mismatch: Syllable={syllable_embeddings.shape[0]}, "
              f"Character={character_embeddings.shape[0]}, Contextual={contextual_embeddings.shape[0]}. "
              f"Adjusting to min_len to continue.")
        min_len = min(syllable_embeddings.shape[0], character_embeddings.shape[0], contextual_embeddings.shape[0])
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


# --- BiLSTM-CRF Model Definition ---
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

        self.crf = CRF(num_tags, batch_first=True) 

    def forward(self, embeddings, tags=None, mask=None):
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.hidden2tag(lstm_out)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=mask) 

# --- Custom Dataset for training ---
class ABSADataset(Dataset):
    def __init__(self, data, tag_to_idx_map):
        self.data = data
        self.tag_to_idx_map = tag_to_idx_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        embeddings = torch.tensor(item['embeddings'], dtype=torch.float)
        tags = torch.tensor([self.tag_to_idx_map.get(tag, self.tag_to_idx_map['O']) for tag in item['iob2_labels']], dtype=torch.long)
        
        return embeddings, tags, len(item['word_tokens'])

# --- Collate function for DataLoader (to handle variable sequence lengths) ---
tag_to_idx = {'<PAD>': 0, 'O': 1} 

def collate_fn(batch, tag_to_idx_map=None): # Add tag_to_idx_map parameter
    if tag_to_idx_map is None:
        global tag_to_idx
        tag_to_idx_map = tag_to_idx 

    embeddings_list, tags_list, lengths = zip(*batch)
    
    padded_embeddings = pad_sequence(embeddings_list, batch_first=True, padding_value=0.0)
    padded_tags = pad_sequence(tags_list, batch_first=True, padding_value=tag_to_idx_map['<PAD>']) 

    mask = (padded_tags != tag_to_idx_map['<PAD>']) 

    return padded_embeddings, padded_tags, torch.tensor(lengths, dtype=torch.long), mask 

# --- Helper functions for Span-level Evaluation ---
# Based on the paper: "A predicted span is correct only if it exactly matches the gold standard span."
def extract_spans(iob2_labels):
    """
    Extracts (start_idx, end_idx, tag_type) spans from an IOB2 tag sequence.
    Example: ['O', 'B-ASPECT', 'I-ASPECT', 'O'] -> [(1, 2, 'ASPECT')]
    """
    spans = []
    current_span = None

    for i, tag in enumerate(iob2_labels):
        if tag.startswith('B-'):
            if current_span: # End previous span if exists
                spans.append(current_span)
            tag_type = tag[2:]
            current_span = [i, i, tag_type] # [start, end, type]
        elif tag.startswith('I-'):
            if current_span and current_span[2] == tag[2:]: # Continue current span
                current_span[1] = i
            else: # Invalid I-tag without B-tag or type mismatch
                if current_span: # End previous valid span
                    spans.append(current_span)
                current_span = None # Reset, as this I-tag is ill-formed
        else: # 'O' tag
            if current_span: # End current span
                spans.append(current_span)
            current_span = None
    
    if current_span: # Add last span if it exists
        spans.append(current_span)
    
    # Convert to immutable tuples
    return [tuple(s) for s in spans]

def calculate_span_metrics(true_iob2_labels_list, pred_iob2_labels_list, all_span_tag_types):
    """
    Calculates span-level Precision, Recall, F1-score (micro and macro).
    Args:
        true_iob2_labels_list (list of list of str): List of true IOB2 tag sequences.
        pred_iob2_labels_list (list of list of str): List of predicted IOB2 tag sequences.
        all_span_tag_types (list of str): All possible span tag types (e.g., 'ASPECT', 'PRICE').
                                            These are the parts after 'B-' or 'I-'.
    Returns:
        dict: Dictionary containing micro and macro P, R, F1.
    """
    true_positives = {tag_type: 0 for tag_type in all_span_tag_types}
    false_positives = {tag_type: 0 for tag_type in all_span_tag_types}
    false_negatives = {tag_type: 0 for tag_type in all_span_tag_types}

    total_tp_micro = 0
    total_fp_micro = 0
    total_fn_micro = 0

    for true_tags, pred_tags in zip(true_iob2_labels_list, pred_iob2_labels_list):
        true_spans = extract_spans(true_tags)
        pred_spans = extract_spans(pred_tags)

        # Create sets for efficient checking
        true_spans_set = set(true_spans)
        pred_spans_set = set(pred_spans)

        # Count TP, FP, FN for this sample (micro and per-tag)
        for span in pred_spans_set:
            if span in true_spans_set:
                total_tp_micro += 1
                if span[2] in true_positives: true_positives[span[2]] += 1
            else:
                total_fp_micro += 1
                if span[2] in false_positives: false_positives[span[2]] += 1
        
        for span in true_spans_set:
            if span not in pred_spans_set:
                total_fn_micro += 1
                if span[2] in false_negatives: false_negatives[span[2]] += 1

    # Calculate Macro P, R, F1
    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []
    per_tag_metrics = {}

    for tag_type in all_span_tag_types:
        tp = true_positives[tag_type]
        fp = false_positives[tag_type]
        fn = false_negatives[tag_type]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_tag_metrics[tag_type] = {"precision": precision, "recall": recall, "f1": f1}

        if (tp + fp + fn) > 0: # Only include in macro average if there were any instances for this tag type
            macro_precision_list.append(precision)
            macro_recall_list.append(recall)
            macro_f1_list.append(f1)
    
    macro_precision = sum(macro_precision_list) / len(macro_precision_list) if macro_precision_list else 0
    macro_recall = sum(macro_recall_list) / len(macro_recall_list) if macro_recall_list else 0
    macro_f1 = sum(macro_f1_list) / len(macro_f1_list) if macro_f1_list else 0

    # Calculate Micro P, R, F1
    micro_precision = total_tp_micro / (total_tp_micro + total_fp_micro) if (total_tp_micro + total_fp_micro) > 0 else 0
    micro_recall = total_tp_micro / (total_tp_micro + total_fn_micro) if (total_tp_micro + total_fn_micro) > 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_tag_metrics": per_tag_metrics
    }

# --- Helper functions for Token-level Evaluation ---
# --- Helper functions for Token-level Evaluation ---
def calculate_token_metrics(true_labels_list_of_lists, predicted_labels_list_of_lists, all_eval_tags):
    """
    Calculates token-level Precision, Recall, F1-score (micro and macro).
    Args:
        true_labels_list_of_lists (list of list of str): List of true IOB2 tag sequences.
        predicted_labels_list_of_lists (list of list of str): List of predicted IOB2 tag sequences.
        all_eval_tags (list of str): All relevant IOB2 tags to evaluate (excluding 'O' and '<PAD>').
    Returns:
        dict: Dictionary containing micro and macro P, R, F1.
    """
    # Flatten the lists of lists
    # SỬA LỖI Ở ĐÂY: đổi true_labels_of_lists thành true_labels_list_of_lists
    flat_true_labels = [tag for sublist in true_labels_list_of_lists for tag in sublist]
    flat_predicted_labels = [tag for sublist in predicted_labels_list_of_lists for tag in sublist]

    # Create a unified set of all tags present in true and predicted labels for accurate mapping
    unique_tags_in_data = sorted(list(set(flat_true_labels + flat_predicted_labels)))
    temp_tag_to_id = {tag: i for i, tag in enumerate(unique_tags_in_data)}
    
    true_ids = [temp_tag_to_id[tag] for tag in flat_true_labels]
    predicted_ids = [temp_tag_to_id[tag] for tag in flat_predicted_labels]

    # Filter out 'O' and '<PAD>' tags from true and predicted IDs for evaluation
    # This is a common practice for sequence tagging where 'O' is the majority class
    tags_to_ignore_for_metrics = ['O', '<PAD>']
    ids_to_ignore = [temp_tag_to_id[tag] for tag in tags_to_ignore_for_metrics if tag in temp_tag_to_id]
    
    filtered_true_ids = []
    filtered_predicted_ids = []
    
    for t, p in zip(true_ids, predicted_ids):
        if t not in ids_to_ignore: 
             filtered_true_ids.append(t)
             filtered_predicted_ids.append(p)
             
    # Get the numerical IDs of tags to be evaluated
    # This is important if 'all_eval_tags' is a subset of all possible tags
    target_ids_for_metrics = [temp_tag_to_id[tag] for tag in all_eval_tags if tag in temp_tag_to_id]
    
    if not filtered_true_ids: # Handle case where there are no relevant tags to evaluate
        return {
            "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0,
            "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
            "per_tag_metrics": {tag: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for tag in all_eval_tags}
        }

    # Calculate metrics
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        filtered_true_ids, filtered_predicted_ids, average='micro', labels=target_ids_for_metrics, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        filtered_true_ids, filtered_predicted_ids, average='macro', labels=target_ids_for_metrics, zero_division=0
    )
    
    # Calculate per-tag metrics for all_eval_tags
    p_per_tag, r_per_tag, f1_per_tag, _ = precision_recall_fscore_support(
        filtered_true_ids, filtered_predicted_ids, labels=target_ids_for_metrics, zero_division=0
    )
    
    per_tag_metrics = {}
    for i, tag_id in enumerate(target_ids_for_metrics):
        tag_name = unique_tags_in_data[tag_id] 
        per_tag_metrics[tag_name] = {
            "precision": p_per_tag[i],
            "recall": r_per_tag[i],
            "f1": f1_per_tag[i]
        }


    return {
        "micro_precision": p_micro,
        "micro_recall": r_micro,
        "micro_f1": f1_micro,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "per_tag_metrics": per_tag_metrics
    }

# --- New function to run ABSA model training and evaluation ---
async def run_absa_model_training_and_evaluation_pipeline(fused_data_filepath: str, model_save_path: str = "models/absa_bilstm_crf_model.pth"):
    """
    Chạy pipeline huấn luyện và đánh giá mô hình ABSA (BiLSTM-CRF) từ dữ liệu đã gộp.
    Thêm tùy chọn lưu mô hình.
    """
    # Define control flags for this specific run
    RUN_TRAINING = True 
    RUN_TESTING = True

    print(f"Current model pipeline mode: {'Training + Testing' if RUN_TRAINING and RUN_TESTING else ('Training Only' if RUN_TRAINING else 'Preprocessing Only')}")

    print("\n--- GIAI ĐOẠN 1: Chuẩn bị Dữ liệu & Xây dựng Mô hình ABSA (Dựa trên phương pháp của bài báo) ---")

    # 1.1 Tải và Trích xuất Dữ liệu Bình luận từ JSON đã gộp
    print("\n--- 1.1 Tải và Trích xuất Dữ liệu Bình luận từ JSON đã gộp ---")
    all_comments_for_training = []
    try:
        with open(fused_data_filepath, 'r', encoding='utf-8') as f:
            fused_data = json.load(f)
        
        for video_entry in fused_data:
            if "comments" in video_entry and isinstance(video_entry["comments"], list):
                for comment in video_entry["comments"]:
                    if 'embeddings' in comment and isinstance(comment['embeddings'], list):
                        comment['embeddings'] = np.array(comment['embeddings']) # Convert list to numpy array
                all_comments_for_training.extend(video_entry["comments"])
        
        print(f"✅ Đã tải và trích xuất {len(all_comments_for_training)} bình luận từ {len(fused_data)} video để huấn luyện.")
        if not all_comments_for_training:
            print("❌ Không có bình luận nào được trích xuất để huấn luyện. Dừng huấn luyện mô hình.")
            return

    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy tệp dữ liệu gộp tại '{fused_data_filepath}'. Vui lòng đảm bảo đã chạy bước gộp dữ liệu.")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi: Không thể đọc tệp JSON gộp '{fused_data_filepath}'. Vui lòng kiểm tra định dạng tệp: {e}")
        return
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu gộp: {e}")
        return

    # 1.2 Tiền xử lý Dữ liệu Gán nhãn cho Mô hình (Đã có Embeddings)
    print("\n--- 1.2 Tiền xử lý Dữ liệu Gán nhãn cho Mô hình (Đã có Embeddings) ---")
    all_iob2_tags = sorted(list(set(tag for item in all_comments_for_training for tag in item['iob2_labels'])))
    
    if 'O' not in all_iob2_tags:
        all_iob2_tags.append('O')
    if '<PAD>' not in all_iob2_tags:
        all_iob2_tags.append('<PAD>')
    all_iob2_tags = sorted(list(set(all_iob2_tags)))

    current_tag_to_idx = {tag: i for i, tag in enumerate(all_iob2_tags)}
    idx_to_tag = {i: tag for tag, i in current_tag_to_idx.items()}
    NUM_TAGS = len(current_tag_to_idx)
    print(f"\nDefined {NUM_TAGS} unique tags: {current_tag_to_idx}")

    all_span_tag_types = sorted(list(set(tag[2:] for tag in all_iob2_tags if tag.startswith(('B-', 'I-')))))
    print(f"Span Tag Types for Evaluation: {all_span_tag_types}")

    # For token-level evaluation, consider all non-padding tags.
    all_eval_tags_for_token_metrics = [tag for tag in all_iob2_tags if tag != '<PAD>']


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1.3 Xây dựng & Huấn luyện Mô hình ABSA (Span Detection Model)
    if RUN_TRAINING:
        print("\n--- 1.3 Xây dựng & Huấn luyện Mô hình ABSA (Span Detection Model) ---")
        if len(all_comments_for_training) > 1:
            train_data, test_data = train_test_split(all_comments_for_training, test_size=0.2, random_state=42)
            print(f"Data split: {len(train_data)} samples for training, {len(test_data)} samples for testing.")
        else:
            print("Not enough samples to split data. Using all data for training (no separate test set for this run).")
            train_data = all_comments_for_training
            test_data = []

        train_dataset = ABSADataset(train_data, current_tag_to_idx)
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda b: collate_fn(b, current_tag_to_idx)) 
        
        model = BiLSTM_CRF_Model(
            embedding_dim=FUSED_EMBEDDING_DIM,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_tags=NUM_TAGS,
            num_layers=NUM_LSTM_LAYERS,
            dropout=DROPOUT_RATE
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print(f"\n--- Starting model training for {len(train_data)} samples ---")
        num_epochs = 5
        for epoch in range(num_epochs):
            total_loss = 0
            model.train() 
            for batch_idx, (embeddings_batch, tags_batch, lengths, mask_batch) in enumerate(train_dataloader): 
                optimizer.zero_grad() 

                embeddings_batch = embeddings_batch.to(device)
                tags_batch = tags_batch.to(device)
                mask_batch = mask_batch.to(device)

                loss = model(embeddings_batch, tags=tags_batch, mask=mask_batch)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_dataloader):.4f}")

        print("\n--- Model Training Finished ---")

        # --- LƯU MÔ HÌNH VÀ CÁC THÔNG TIN LIÊN QUAN ---
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'current_tag_to_idx': current_tag_to_idx,
            'idx_to_tag': idx_to_tag,
            'embedding_dim': FUSED_EMBEDDING_DIM,
            'hidden_dim': LSTM_HIDDEN_DIM,
            'num_tags': NUM_TAGS,
            'num_layers': NUM_LSTM_LAYERS,
            'dropout': DROPOUT_RATE,
            'char_to_idx': char_to_idx # Cần lưu char_to_idx cho CharBiLSTMEmbedder
        }, model_save_path)
        print(f"✅ Model và cấu hình đã được lưu tại: {model_save_path}")


        # 1.4 Đánh giá (Evaluation) Mô hình (Cấp độ Span)
        if RUN_TESTING and len(test_data) > 0:
            print("\n--- 1.4 Đánh giá (Evaluation) Mô hình (Cấp độ Span) ---")
            model.eval() 
            
            test_dataset = ABSADataset(test_data, current_tag_to_idx)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda b: collate_fn(b, current_tag_to_idx)) 

            all_true_labels_span = [] 
            all_predicted_labels_span = [] 

            all_true_labels_token = [] 
            all_predicted_labels_token = [] 

            with torch.no_grad(): 
                for i, (embeddings_batch, tags_batch, lengths, mask_batch) in enumerate(test_dataloader): 
                    original_text = test_data[i]['original_text']
                    word_tokens = test_data[i]['word_tokens']
                    true_labels_str = [idx_to_tag[id.item()] for id in tags_batch.squeeze(0)] 

                    embeddings_batch = embeddings_batch.to(device)
                    mask_batch = mask_batch.to(device)

                    predicted_tag_ids_list = model(embeddings_batch, mask=mask_batch) 
                    
                    predicted_tag_ids = predicted_tag_ids_list[0] 

                    predicted_labels_str = [idx_to_tag[id_val] for id_val in predicted_tag_ids]

                    actual_length = lengths.item()
                    predicted_labels_str = predicted_labels_str[:actual_length]
                    true_labels_str = true_labels_str[:actual_length] 
                    word_tokens_display = word_tokens[:actual_length]

                    # Tính toán độ chính xác token trên mỗi mẫu
                    num_correct_tokens = sum(1 for t, p in zip(true_labels_str, predicted_labels_str) if t == p)
                    accuracy_per_token_sample = num_correct_tokens / actual_length if actual_length > 0 else 0


                    print(f"\n--- Test Sample {i+1} ---")
                    print(f"Original Text: {original_text}")
                    print(f"Word Tokens: {word_tokens_display}")
                    print(f"True Labels (IOB2): {true_labels_str}")
                    print(f"Predicted Labels (IOB2): {predicted_labels_str}")
                    print(f"Token-level Accuracy for this sample: {accuracy_per_token_sample:.2f}") # In ra độ chính xác token

                    # Accumulate for overall span-level metrics
                    all_true_labels_span.append(true_labels_str)
                    all_predicted_labels_span.append(predicted_labels_str)

                    # Accumulate for overall token-level metrics (flattened)
                    all_true_labels_token.extend(true_labels_str)
                    all_predicted_labels_token.extend(predicted_labels_str)
            
            # Calculate overall span-level metrics
            print("\n--- Overall Span-level Evaluation Metrics ---")
            span_metrics_results = calculate_span_metrics(
                all_true_labels_span, 
                all_predicted_labels_span, 
                all_span_tag_types # Use all_span_tag_types for macro average categories
            )
            
            print(f"Micro-Precision: {span_metrics_results['micro_precision']:.4f}")
            print(f"Micro-Recall: {span_metrics_results['micro_recall']:.4f}")
            print(f"Micro-F1: {span_metrics_results['micro_f1']:.4f}")
            print(f"Macro-Precision: {span_metrics_results['macro_precision']:.4f}")
            print(f"Macro-Recall: {span_metrics_results['macro_recall']:.4f}")
            print(f"Macro-F1: {span_metrics_results['macro_f1']:.4f}")

            print("\n--- Per-Tag F1-score (Span-level) ---")
            for tag_type, metrics in span_metrics_results['per_tag_metrics'].items():
                print(f"   {tag_type}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

            # Calculate overall token-level metrics
            print("\n--- Overall Token-level Evaluation Metrics ---")
            token_metrics_results = calculate_token_metrics(
                [all_true_labels_token], # Pass as list of list for consistency with how calculate_token_metrics expects it
                [all_predicted_labels_token],
                all_eval_tags_for_token_metrics
            )
            
            print(f"Micro-Precision: {token_metrics_results['micro_precision']:.4f}")
            print(f"Micro-Recall: {token_metrics_results['micro_recall']:.4f}")
            print(f"Micro-F1: {token_metrics_results['micro_f1']:.4f}")
            print(f"Macro-Precision: {token_metrics_results['macro_precision']:.4f}")
            print(f"Macro-Recall: {token_metrics_results['macro_recall']:.4f}")
            print(f"Macro-F1: {token_metrics_results['macro_f1']:.4f}")

            print("\n--- Per-Tag F1-score (Token-level) ---")
            for tag_type, metrics in token_metrics_results['per_tag_metrics'].items():
                print(f"   {tag_type}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

            print("\n--- Model Testing/Inference Finished ---")
            print("Remember: This is a basic demonstration. For robust evaluation:")
            print("1. The evaluation here is at span-level, matching exact spans.")
            print("2. For more advanced evaluation (e.g., partial matches), custom logic is needed.")

        elif RUN_TESTING and len(test_data) == 0:
            print("Model testing skipped: No test data available (due to small dataset size).")
        else:
            print("Model testing skipped (RUN_TESTING is False).")

    else:
        print("\n--- Model Training Skipped (RUN_TRAINING is False) ---")

    print("\n--- (Conceptual) GIAI ĐOẠN 2 & 3: Phân tích Dữ liệu Bình luận & Script Video & Xây dựng Hệ thống Đề xuất KOL ---")
    print("Các bước này sẽ được xây dựng dựa trên kết quả của Giai đoạn 1 (mô hình ABSA) và dữ liệu gộp.")
    print("Bạn có thể sử dụng mô hình đã huấn luyện để dự đoán các khía cạnh/sentiment trên dữ liệu chưa gán nhãn, sau đó kết hợp với dữ liệu script và các chỉ số khác để xây dựng hệ thống đề xuất KOL.")

    print("\n--- End of Model Pipeline Execution ---")


# --- Khối chính khi file được chạy trực tiếp ---
if __name__ == "__main__":
    print("Running src/model.py directly for demonstration purposes.")
    
    fused_data_filepath = "../data/combined/fused_video_comment_features.json"
    model_save_path = "../models/absa_bilstm_crf_model.pth" # Đường dẫn lưu mô hình

    try:
        asyncio.run(run_absa_model_training_and_evaluation_pipeline(fused_data_filepath, model_save_path))
        print("Huấn luyện mô hình ABSA hoàn tất khi chạy trực tiếp.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp dữ liệu gộp tại '{fused_data_filepath}'. Vui lòng đảm bảo đã chạy bước gộp dữ liệu ('python main.py --process full' hoặc 'python main.py --process fuse').")
    except Exception as e:
        print(f"Lỗi khi chạy pipeline huấn luyện mô hình ABSA trực tiếp: {e}")
        import traceback
        traceback.print_exc()