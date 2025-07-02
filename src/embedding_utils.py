import os
import json
import re
import fasttext
import torch
import torch.nn as nn
import numpy as np
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModel


FASTTEXT_MODEL_PATH = "models/fasttext/cc.vi.300.bin" 
FASTTEXT_EMBEDDING_DIM = 300

CHAR_EMBEDDING_DIM = 50
CHAR_LSTM_HIDDEN_DIM = 50
CHAR_FEATURE_OUTPUT_DIM = CHAR_LSTM_HIDDEN_DIM * 2

# Đã sửa lại đường dẫn của mô hình XLM-RoBERTa để trỏ đến thư mục cục bộ
XLMR_MODEL_NAME = "models\huggingface\hub\models--xlm-roberta-base\snapshots\e73636d4f797dec63c3081bb6ed5c7b0bb3f2089" # Đường dẫn cục bộ dự kiến
XLMR_EMBEDDING_DIM = 768

# Ánh xạ ký tự sang chỉ số
char_to_idx = {char: i for i, char in enumerate(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789áàảạãăằẳẵặâầẩẫậéèẻẹẽêềểễệíìỉịĩóòỏọõôồổỗộơờởỡợúùủụũưừửữựýỳỷỵỹđĐ,.:;?!'\"()[]{}<>/-_&@#$%^&*=+~` "
)}
char_to_idx["<PAD>"] = 0
char_to_idx["<unk>"] = len(char_to_idx)
CHAR_VOCAB_SIZE = len(char_to_idx)

class FastTextEmbedder:
    """Lớp để tải và cung cấp embeddings từ mô hình FastText."""
    def __init__(self, model_path=FASTTEXT_MODEL_PATH):
        try:
            if not os.path.exists(model_path):
                print(f"⚠️ Không tìm thấy mô hình FastText tại: {model_path}. "
                      "Vui lòng tải xuống mô hình FastText tiếng Việt (ví dụ: 'cc.vi.300.bin') và đặt vào thư mục này.")
                raise FileNotFoundError(f"Không tìm thấy mô hình FastText tại: {model_path}")
                
            self.model = fasttext.load_model(model_path)
            self.embedding_dim = self.model.get_word_vector("test").shape[0]
            print(f"✅ Đã tải mô hình FastText từ: {model_path} (dim: {self.embedding_dim})")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải mô hình FastText từ '{model_path}': {e}. "
                               "Kiểm tra lại đường dẫn và cài đặt thư viện FastText.")

    def get_embedding(self, word):
        """Lấy vector embedding cho một từ."""
        return torch.from_numpy(self.model.get_word_vector(word)).float()

class CharBiLSTMEmbedder(nn.Module):
    """Lớp để tạo embeddings cấp ký tự sử dụng BiLSTM."""
    def __init__(self, vocab_size=CHAR_VOCAB_SIZE, embedding_dim=CHAR_EMBEDDING_DIM, hidden_dim=CHAR_LSTM_HIDDEN_DIM):
        super(CharBiLSTMEmbedder, self).__init__()
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=char_to_idx["<PAD>"])
        self.char_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.output_dim = hidden_dim * 2

    def forward(self, char_indices_list_per_word):
        """
        Input: list of tensors, each tensor contains char indices for a word.
        Output: Tensor of shape (num_words, output_dim)
        """
        word_char_representations = []
        for char_indices_for_word in char_indices_list_per_word:
            if char_indices_for_word.numel() == 0:
                # Xử lý trường hợp từ rỗng hoặc không có ký tự
                word_char_representations.append(torch.zeros(self.output_dim))
                continue

            char_embs = self.char_embedding(char_indices_for_word.unsqueeze(0))
            _, (h_n, _) = self.char_lstm(char_embs)
            # Nối hidden states từ cả hai chiều (forward và backward)
            char_word_representation = torch.cat((h_n[0], h_n[1]), dim=1).squeeze(0)
            word_char_representations.append(char_word_representation)

        if not word_char_representations:
            # Trả về tensor rỗng nếu không có từ nào được xử lý
            return torch.empty(0, self.output_dim)
        return torch.stack(word_char_representations)

class XLMREmbedder:
    """Lớp để tải và cung cấp contextual embeddings từ mô hình XLM-RoBERTa."""
    def __init__(self, model_name=XLMR_MODEL_NAME): # Sử dụng đường dẫn cục bộ
        try:
            # Kiểm tra xem đường dẫn cục bộ có tồn tại và chứa các file cần thiết không
            if not os.path.exists(model_name) or \
               not (os.path.exists(os.path.join(model_name, 'tokenizer.json')) or \
                    os.path.exists(os.path.join(model_name, 'config.json'))):
                print(f"⚠️ Không tìm thấy mô hình XLM-RoBERTa tại: {model_name}. "
                      "Vui lòng tải xuống mô hình 'xlm-roberta-base' từ Hugging Face "
                      "và giải nén vào thư mục này (e.g., models/huggingface/xlm-roberta-base/config.json, etc.).")
                # Nếu không tìm thấy cục bộ, thử tải từ Hugging Face Hub (nếu cần) hoặc báo lỗi
                # Hiện tại, nếu không tìm thấy cục bộ sẽ báo lỗi để người dùng tải thủ công.
                raise FileNotFoundError(f"Không tìm thấy mô hình XLM-RoBERTa tại: {model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval() # Đặt mô hình ở chế độ đánh giá
            self.embedding_dim = self.model.config.hidden_size
            print(f"✅ Đã tải mô hình XLM-RoBERTa từ: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải mô hình XLM-RoBERTa từ '{model_name}': {e}. "
                               "Kiểm tra lại tên mô hình hoặc kết nối internet.")

    def get_token_embeddings(self, text):
        """
        Mã hóa văn bản và lấy embeddings cấp subword.
        Trả về: token_embeddings, input_ids, offset_mapping
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        offset_mapping = inputs.pop('offset_mapping')

        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state.squeeze(0)

        return token_embeddings, inputs['input_ids'].squeeze(0), offset_mapping.squeeze(0)

def clean_text_for_tokenization(text):
    """Làm sạch văn bản bằng cách loại bỏ ký tự đặc biệt và chuẩn hóa khoảng trắng."""
    # Giữ lại các ký tự tiếng Việt, số, dấu câu và khoảng trắng
    text = re.sub(r'[\r\n\t]', ' ', text) # Loại bỏ xuống dòng, tab
    text = re.sub(r'\s+', ' ', text).strip() # Chuẩn hóa khoảng trắng
    return text

def process_text_to_word_tokens(text):
    """
    Tách văn bản thành các word/syllable tokens bằng pyvi.
    Đồng thời cố gắng căn chỉnh vị trí ký tự của các token.
    Trả về: tokens (list[str]), cleaned_text (str), token_info_list (list[dict])
    """
    cleaned_text = clean_text_for_tokenization(text)
    
    # PyviTokenizer đôi khi thêm dấu gạch dưới cho từ ghép, chúng ta muốn tách ra
    tokens_str = ViTokenizer.tokenize(cleaned_text)
    tokens = tokens_str.replace("_", " ").split()

    token_info_list = []
    current_char_idx = 0
    for i, token in enumerate(tokens):
        # Bỏ qua khoảng trắng trước khi tìm token
        while current_char_idx < len(cleaned_text) and cleaned_text[current_char_idx].isspace():
            current_char_idx += 1

        # Tìm vị trí của token trong cleaned_text
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
            # Nếu không tìm thấy, có thể do tokenization không hoàn hảo hoặc ký tự đặc biệt
            # Cố gắng tìm kiếm không phân biệt chữ hoa/thường hoặc bỏ qua các ký tự không khớp
            # Để đơn giản, chỉ ghi cảnh báo và ước tính vị trí
            print(f"Warning: Không thể căn chỉnh token '{token}' (idx: {i}) trong văn bản gốc. Ước tính vị trí.")
            token_info_list.append({
                'token_text': token,
                'start_char': current_char_idx, # Ước tính
                'end_char': current_char_idx + len(token), # Ước tính
                'token_idx': i
            })
            current_char_idx += len(token) # Di chuyển con trỏ để tiếp tục tìm kiếm

    return tokens, cleaned_text, token_info_list

def get_char_indices_for_words(cleaned_text, token_info_list, char_to_idx):
    """
    Tạo danh sách các tensor chỉ số ký tự cho mỗi từ/âm tiết.
    """
    char_indices_list_per_word = []
    for token_info in token_info_list:
        # Lấy đoạn văn bản tương ứng với token từ cleaned_text
        # Sử dụng token_text nếu start_char là -1 (không căn chỉnh được)
        char_span = cleaned_text[token_info['start_char']:token_info['end_char']] \
                    if token_info['start_char'] != -1 else token_info['token_text']
        
        # Chuyển đổi từng ký tự thành chỉ số, sử dụng <unk> nếu không tìm thấy
        word_char_indices = [char_to_idx.get(c, char_to_idx['<unk>']) for c in char_span]
        char_indices_list_per_word.append(torch.tensor(word_char_indices, dtype=torch.long))
    return char_indices_list_per_word

def align_xlmr_embeddings_to_words(
    xlmr_token_embeddings, xlmr_input_ids, xlmr_offset_mapping,
    pyvi_tokens, xlmr_tokenizer_instance, original_text_for_alignment
):
    """
    Ánh xạ contextual embeddings từ XLM-RoBERTa (subword level) sang word tokens (pyvi level).
    Sử dụng mean pooling cho các subword thuộc cùng một word.
    """
    word_embeddings = []

    # Tạo spans cho pyvi tokens trong original_text_for_alignment
    pyvi_token_spans = []
    temp_current_char_idx = 0
    for token in pyvi_tokens:
        token_start = original_text_for_alignment.find(token, temp_current_char_idx)
        if token_start != -1:
            token_end = token_start + len(token)
            pyvi_token_spans.append((token_start, token_end))
            temp_current_char_idx = token_end
        else:
            # Nếu không tìm thấy, có thể do sự khác biệt nhỏ sau khi làm sạch
            # Cố gắng tìm kiếm không phân biệt chữ hoa/thường hoặc bỏ qua các ký tự không khớp
            # Để đơn giản, chỉ ghi cảnh báo và gán span rỗng
            print(f"Warning: Không thể căn chỉnh Pyvi token '{token}' trong văn bản gốc cho XLM-R alignment.")
            pyvi_token_spans.append((-1, -1)) # Đánh dấu là không tìm thấy

    xlmr_word_ids = [] # Lưu trữ chỉ số của word token (pyvi) mà mỗi subword (xlmr) thuộc về
    for i, (subword_start_char, subword_end_char) in enumerate(xlmr_offset_mapping):
        # Bỏ qua các token đặc biệt của XLM-R (như <s>, </s>, <pad>)
        if xlmr_input_ids[i] in xlmr_tokenizer_instance.all_special_ids:
            xlmr_word_ids.append(None)
            continue

        found_word_id = None
        for pyvi_idx, (pyvi_start_char, pyvi_end_char) in enumerate(pyvi_token_spans):
            if pyvi_start_char == -1: # Bỏ qua các pyvi token không tìm thấy span
                continue
            
            # Kiểm tra sự chồng lấn giữa span của subword và span của pyvi token
            if max(subword_start_char, pyvi_start_char) < min(subword_end_char, pyvi_end_char):
                found_word_id = pyvi_idx
                break
        xlmr_word_ids.append(found_word_id)

    # Nhóm các embeddings của subword lại theo word token
    grouped_embeddings = [[] for _ in range(len(pyvi_tokens))]
    for i, word_id in enumerate(xlmr_word_ids):
        if word_id is not None and word_id < len(pyvi_tokens):
            grouped_embeddings[word_id].append(xlmr_token_embeddings[i])

    # Tính toán embedding trung bình cho mỗi word token
    for group in grouped_embeddings:
        if group:
            word_embeddings.append(torch.stack(group).mean(dim=0))
        else:
            # Nếu không có subword nào thuộc về word token này, gán vector 0
            word_embeddings.append(torch.zeros(XLMR_EMBEDDING_DIM))

    # Đảm bảo số lượng embeddings khớp với số lượng pyvi tokens
    if len(word_embeddings) > len(pyvi_tokens):
        word_embeddings = word_embeddings[:len(pyvi_tokens)]
    elif len(word_embeddings) < len(pyvi_tokens):
        word_embeddings.extend([torch.zeros(XLMR_EMBEDDING_DIM)] * (len(pyvi_tokens) - len(word_embeddings)))

    return torch.stack(word_embeddings)

def get_fused_embeddings_and_iob2_labels(sample, fasttext_embedder, char_bilstm_embedder, xlmr_embedder):
    """
    Xử lý một mẫu dữ liệu (văn bản và nhãn span) để tạo ra fused embeddings và IOB2 labels.
    
    Args:
        sample (dict): Một dictionary chứa 'text' (văn bản bình luận) và 'labels' (danh sách các span nhãn).
                       Ví dụ: {"text": "...", "labels": [[start_char, end_char, label_tag], ...]}
        fasttext_embedder (FastTextEmbedder): Instance của FastTextEmbedder.
        char_bilstm_embedder (CharBiLSTMEmbedder): Instance của CharBiLSTMEmbedder.
        xlmr_embedder (XLMREmbedder): Instance của XLMREmbedder.
        
    Returns:
        tuple: (fused_embeddings, iob2_labels, tokens) hoặc (None, None, None) nếu có lỗi.
    """
    text = sample["text"]
    raw_labels = sample.get("labels", [])

    # Bước 1: Tokenization và căn chỉnh
    tokens, cleaned_text, token_info_list = process_text_to_word_tokens(text)
    if not tokens:
        print(f"Warning: Không có token nào được tạo từ văn bản: '{text}'")
        return None, None, None

    # Bước 2: Chuyển đổi nhãn span sang IOB2
    iob2_labels = ['O'] * len(tokens)
    for start_char_label, end_char_label, label_tag in raw_labels:
        affected_token_indices = []
        for token_info in token_info_list:
            token_start = token_info['start_char']
            token_end = token_info['end_char']
            token_idx = token_info['token_idx']
            
            # Kiểm tra sự chồng lấn giữa span của nhãn và span của token
            if max(token_start, start_char_label) < min(token_end, end_char_label):
                affected_token_indices.append(token_idx)

        if affected_token_indices:
            affected_token_indices.sort()
            iob2_labels[affected_token_indices[0]] = f"B-{label_tag}"
            for i in range(1, len(affected_token_indices)):
                iob2_labels[affected_token_indices[i]] = f"I-{label_tag}"

    # Bước 3: Tạo Syllable Embeddings (FastText)
    syllable_embeddings = torch.stack([fasttext_embedder.get_embedding(token) for token in tokens])

    # Bước 4: Tạo Character Embeddings (CharBiLSTM)
    char_indices_list_per_word = get_char_indices_for_words(cleaned_text, token_info_list, char_to_idx)
    character_embeddings = char_bilstm_embedder(char_indices_list_per_word)

    # Bước 5: Tạo Contextual Embeddings (XLM-RoBERTa)
    xlmr_token_embs, xlmr_input_ids, xlmr_offset_mapping = xlmr_embedder.get_token_embeddings(text)
    contextual_embeddings = align_xlmr_embeddings_to_words(
        xlmr_token_embs,
        xlmr_input_ids,
        xlmr_offset_mapping,
        tokens,
        xlmr_embedder.tokenizer,
        text # Sử dụng văn bản gốc để căn chỉnh XLM-R
    )

    # Bước 6: Đảm bảo kích thước embeddings khớp nhau và nối chúng lại
    if not (syllable_embeddings.shape[0] == character_embeddings.shape[0] == contextual_embeddings.shape[0]):
        print(f"⚠️ Kích thước embeddings không khớp: Syllable={syllable_embeddings.shape[0]}, "
              f"Character={character_embeddings.shape[0]}, Contextual={contextual_embeddings.shape[0]}. "
              f"Đang điều chỉnh về min_len để tiếp tục.")
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