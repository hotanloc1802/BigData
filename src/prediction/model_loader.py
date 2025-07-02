import os
import torch
import fasttext
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from TorchCRF import CRF # Import CRF
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import các hàm và lớp từ các module gốc để tái sử dụng
# Lưu ý: Các hằng số (như embedding_dim) sẽ được lấy từ file model.py khi tải mô hình ABSA
from src.model import BiLSTM_CRF_Model # Tải kiến trúc mô hình ABSA
from src.embedding_utils import char_to_idx, CHAR_VOCAB_SIZE, CHAR_EMBEDDING_DIM, CHAR_LSTM_HIDDEN_DIM, CHAR_FEATURE_OUTPUT_DIM, FASTTEXT_EMBEDDING_DIM, XLMR_EMBEDDING_DIM , process_text_to_word_tokens, get_char_indices_for_words# Để tái tạo Embedder

# Biến cache toàn cục cho mỗi worker process
_model_cache = {}

# --- Đường dẫn tới các mô hình đã huấn luyện/tải về ---
# Đảm bảo các đường dẫn này khớp với cấu trúc thư mục thực tế của bạn
# (Giống như trong model.py và video_processor.py)
ABSA_MODEL_PATH = r"D:\Storage_document\Effort_Ki2_Nam3\BigData\Do_an\tiktok_comment_cleaner\models\absa_bilstm_crf_model.pth" # Đường dẫn tới file model ABSA đã huấn luyện
FASTTEXT_MODEL_PATH = r"D:\Storage_document\Effort_Ki2_Nam3\BigData\Do_an\tiktok_comment_cleaner\models\fasttext\cc.vi.300.bin"
XLMR_MODEL_NAME = r"D:\Storage_document\Effort_Ki2_Nam3\BigData\Do_an\tiktok_comment_cleaner\models\huggingface\hub\models--xlm-roberta-base\snapshots\e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"

# --- Các lớp Embedder được tái tạo để dùng riêng trong UDF nếu cần ---
# Đây là sự trùng lặp code để tránh phụ thuộc trực tiếp vào embedding_utils trong UDF,
# đảm bảo các dependency của UDF được quản lý gọn gàng hơn trên worker.
class FastTextEmbedderForUDF:
    def __init__(self, model_path=FASTTEXT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"UDF: FastText model not found at: {model_path}")
        self.model = fasttext.load_model(model_path)
    def get_embedding(self, word):
        return torch.from_numpy(self.model.get_word_vector(word)).float()

class CharBiLSTMEmbedderForUDF(nn.Module):
    def __init__(self, vocab_size=CHAR_VOCAB_SIZE, embedding_dim=CHAR_EMBEDDING_DIM, hidden_dim=CHAR_LSTM_HIDDEN_DIM):
        super(CharBiLSTMEmbedderForUDF, self).__init__()
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

class XLMREmbedderForUDF:
    def __init__(self, model_name=XLMR_MODEL_NAME):
        if not os.path.exists(model_name) or \
           not (os.path.exists(os.path.join(model_name, 'tokenizer.json')) or \
                os.path.exists(os.path.join(model_name, 'config.json'))):
            raise FileNotFoundError(f"UDF: XLM-RoBERTa model not found at: {model_name}")
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

# Hàm trợ giúp để căn chỉnh embeddings XLM-R cấp subword với word tokens
def align_xlmr_embeddings_to_words_for_udf(
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
            if pyvi_start_char == -1: continue
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

# Hàm để tạo fused embeddings cho một câu/comment
def get_fused_embeddings_for_udf(text, fasttext_embedder, char_bilstm_embedder, xlmr_embedder):
    # Sử dụng hàm process_text_to_word_tokens từ text_utils
    tokens, cleaned_text, token_info_list = process_text_to_word_tokens(text) 
    if not tokens:
        return None, None # Trả về None nếu không có token

    syllable_embeddings = torch.stack([fasttext_embedder.get_embedding(token) for token in tokens])
    # Sử dụng hàm get_char_indices_for_words từ text_utils
    char_indices_list_per_word = get_char_indices_for_words(cleaned_text, token_info_list) 
    character_embeddings = char_bilstm_embedder(char_indices_list_per_word)
    xlmr_token_embs, xlmr_input_ids, xlmr_offset_mapping = xlmr_embedder.get_token_embeddings(text)
    contextual_embeddings = align_xlmr_embeddings_to_words_for_udf(
        xlmr_token_embs, xlmr_input_ids, xlmr_offset_mapping,
        tokens, xlmr_embedder.tokenizer, cleaned_text
    )

    if not (syllable_embeddings.shape[0] == character_embeddings.shape[0] == contextual_embeddings.shape[0]):
        min_len = min(syllable_embeddings.shape[0], character_embeddings.shape[0], contextual_embeddings.shape[0])
        syllable_embeddings = syllable_embeddings[:min_len]
        character_embeddings = character_embeddings[:min_len]
        contextual_embeddings = contextual_embeddings[:min_len]
        tokens = tokens[:min_len]

    fused_embeddings = torch.cat(
        (syllable_embeddings, character_embeddings, contextual_embeddings), dim=1
    )
    return fused_embeddings, tokens # Trả về tensor embeddings và tokens đã xử lý

def load_all_absa_prediction_models(force_cpu: bool = False): # Thêm tham số force_cpu
    """
    Tải tất cả các mô hình cần thiết cho pipeline dự đoán ABSA trên Spark worker.
    Sẽ được cache trên mỗi worker.
    Args:
        force_cpu (bool): Nếu True, buộc tất cả các model PyTorch chạy trên CPU.
    """
    if 'absa_model' not in _model_cache:
        print("Worker process: Loading ABSA prediction models...")
        try:
            device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
            
            # Tải ABSA model
            absa_model, tag_to_idx_map = _load_absa_model_internal(ABSA_MODEL_PATH, device)
            
            # Tải các Embedder
            fasttext_embedder = FastTextEmbedderForUDF(FASTTEXT_MODEL_PATH)
            char_bilstm_embedder = CharBiLSTMEmbedderForUDF()
            xlmr_embedder = XLMREmbedderForUDF(XLMR_MODEL_NAME)
            
            _model_cache['absa_model'] = absa_model
            _model_cache['tag_to_idx_map'] = tag_to_idx_map
            _model_cache['idx_to_tag_map'] = {v: k for k, v in tag_to_idx_map.items()} # Cần cho decoding
            _model_cache['fasttext_embedder'] = fasttext_embedder
            _model_cache['char_bilstm_embedder'] = char_bilstm_embedder
            _model_cache['xlmr_embedder'] = xlmr_embedder
            _model_cache['device'] = device

            print(f"Worker process: ABSA prediction models loaded successfully on {device}.")
        except Exception as e:
            print(f"Worker process: ERROR loading ABSA prediction models: {e}")
            _model_cache['absa_model'] = "LOAD_ERROR" # Đánh dấu lỗi
            raise RuntimeError(f"Failed to load ABSA prediction models: {e}")

# Hàm nội bộ để tải BiLSTM_CRF_Model từ checkpoint
def _load_absa_model_internal(path, device):
    """Tải BiLSTM_CRF_Model và tag_to_idx_map từ checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ABSA model checkpoint not found at: {path}")
    
    checkpoint = torch.load(path, map_location=device)

    model = BiLSTM_CRF_Model(
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_tags=checkpoint['num_tags'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 👉 Sửa dòng này:
    tag_to_idx_map = checkpoint.get('tag_to_idx_map') or checkpoint.get('current_tag_to_idx')
    if tag_to_idx_map is None:
        raise KeyError("Checkpoint không chứa 'tag_to_idx_map' hoặc 'current_tag_to_idx'")

    return model, tag_to_idx_map