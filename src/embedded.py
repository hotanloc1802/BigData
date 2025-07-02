import os
import json
import re
import fasttext
import torch
import torch.nn as nn
import numpy as np
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModel

# Configuration parameters
FASTTEXT_MODEL_PATH = "../models/fasttext/cc.vi.300.bin"
FASTTEXT_EMBEDDING_DIM = 300

CHAR_EMBEDDING_DIM = 50
CHAR_LSTM_HIDDEN_DIM = 50

CHAR_FEATURE_OUTPUT_DIM = CHAR_LSTM_HIDDEN_DIM * 2

XLMR_MODEL_NAME = "xlm-roberta-base"
XLMR_EMBEDDING_DIM = 768

class FastTextEmbedder:
    def __init__(self, model_path=FASTTEXT_MODEL_PATH):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy mô hình FastText tại: {model_path}")
            self.model = fasttext.load_model(model_path)
            self.embedding_dim = self.model.get_word_vector("test").shape[0]
            print(f"✅ Đã tải mô hình FastText từ: {model_path} (dim: {self.embedding_dim})")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải mô hình FastText từ '{model_path}': {e}. "
                               "Kiểm tra lại đường dẫn và cài đặt thư viện FastText.")

    def get_embedding(self, word):
        return torch.from_numpy(self.model.get_word_vector(word)).float()

# Character to index mapping
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

class XLMREmbedder:
    def __init__(self, model_name=XLMR_MODEL_NAME):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.embedding_dim = self.model.config.hidden_size
            print(f"✅ Đã tải mô hình XLM-RoBERTa từ: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải mô hình XLM-RoBERTa từ '{model_name}': {e}. "
                               "Kiểm tra lại tên mô hình hoặc kết nối internet.")

    def get_token_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        offset_mapping = inputs.pop('offset_mapping')

        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state.squeeze(0)

        return token_embeddings, inputs['input_ids'].squeeze(0), offset_mapping.squeeze(0)

def clean_text(text):
    """Làm sạch văn bản bằng cách loại bỏ ký tự đặc biệt và chuẩn hóa khoảng trắng."""
    text = re.sub(r'[\r\n\t]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def process_text_to_word_tokens(text):
    """
    Tách văn bản thành các word/syllable tokens bằng pyvi.
    Trả về: tokens (list[str]), cleaned_text (str), token_info_list (list[dict])
    """
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
    """
    Tạo danh sách các tensor chỉ số ký tự cho mỗi từ/âm tiết.
    """
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
    """
    Ánh xạ contextual embeddings từ XLM-RoBERTa (subword level) sang word tokens (pyvi level).
    Sử dụng mean pooling cho các subword thuộc cùng một word.
    """
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

def get_fused_embeddings_and_iob2_labels(sample, fasttext_embedder, char_bilstm_embedder, xlmr_embedder):
    """
    Xử lý một mẫu dữ liệu để tạo ra fused embeddings và IOB2 labels.
    """
    text = sample["text"]
    raw_labels = sample.get("labels", [])

    tokens, cleaned_text, token_info_list = process_text_to_word_tokens(text)
    if not tokens:
        print(f"Warning: Không có token nào được tạo từ văn bản: '{text}'")
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


if __name__ == "__main__":
    try:
        fasttext_embedder = FastTextEmbedder()
        char_bilstm_embedder = CharBiLSTMEmbedder()
        xlmr_embedder = XLMREmbedder()
    except Exception as e:
        print(f"Không thể khởi tạo một hoặc nhiều Embedder: {e}")
        print("Vui lòng kiểm tra lại đường dẫn mô hình, kết nối internet và cài đặt thư viện.")
        exit()

    input_file = '../data/labeled/video_comments_KOL1_1_spans.json'
    output_file = '../data/enriched/video_comments_KOL1_1_embeeded.jsonl'

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    input_data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        print(f"✅ Đã tải dữ liệu từ: {input_file} ({len(input_data)} mẫu)")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy tệp input tại '{input_file}'. Vui lòng kiểm tra đường dẫn.")
        exit()
    except json.JSONDecodeError:
        print(f"❌ Lỗi: Không thể đọc tệp JSON '{input_file}'. Vui lòng kiểm tra định dạng tệp.")
        exit()

    processed_samples = []
    print("--- Bắt đầu xử lý và tạo Embeddings ---")
    for i, sample in enumerate(input_data):
        print(f"\n--- Xử lý mẫu {i+1}/{len(input_data)}: {sample['text'][:50]}... ---")
        fused_embs, iob2_lbls, tokens = get_fused_embeddings_and_iob2_labels(
            sample, fasttext_embedder, char_bilstm_embedder, xlmr_embedder
        )

        if fused_embs is None:
            print(f"Bỏ qua mẫu {i+1} do lỗi trong quá trình tạo embeddings/nhãn hoặc không có token.")
            continue

        print(f"Original Text: {sample['text']}")
        print(f"Word Tokens ({len(tokens)}): {tokens}")
        print(f"IOB2 Labels ({len(iob2_lbls)}): {iob2_lbls}")
        print(f"Shape của Fused Embeddings: {fused_embs.shape}")

        processed_samples.append({
            "original_text": sample['text'],
            "word_tokens": tokens,
            "iob2_labels": iob2_lbls,
            "embeddings": fused_embs.tolist()
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_samples:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n✅ Kết quả đã được lưu vào: {output_file}")
    print("\n--- Hoàn tất quá trình tạo Fused Embeddings và IOB2 Labels ---")
    print("Dữ liệu đã được tiền xử lý và lưu, sẵn sàng cho giai đoạn huấn luyện mô hình.")
