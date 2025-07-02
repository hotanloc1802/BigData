import json
from pyvi import ViTokenizer 
from transformers import AutoTokenizer 
import re # Để làm sạch văn bản

# Load XLM-RoBERTa tokenizer
try:
    xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
except Exception as e:
    print(f"Lỗi khi tải AutoTokenizer: {e}. Vui lòng kiểm tra kết nối internet hoặc tên mô hình.")
    # Fallback to a dummy tokenizer or handle gracefully if not for actual model training
    class DummyTokenizer:
        def __call__(self, text, return_tensors, add_special_tokens):
            return {'input_ids': [[0] + [ord(c) for c in text] + [1]], 'attention_mask': [[1]*(len(text)+2)]}
        def convert_ids_to_tokens(self, ids):
            return [chr(i) if i not in [0,1] else ('<s>' if i==0 else '</s>') for i in ids]
    xlmr_tokenizer = DummyTokenizer()


def clean_text(text):
    """
    Hàm làm sạch văn bản: loại bỏ ký tự đặc biệt, nhiều khoảng trắng thừa.
    Bạn có thể thêm các quy tắc làm sạch khác tùy theo nhu cầu.
    """
    text = re.sub(r'[\r\n\t]', ' ', text) # Loại bỏ xuống dòng, tab
    text = re.sub(r'\s+', ' ', text).strip() # Thay thế nhiều khoảng trắng bằng một khoảng trắng
    return text

def process_sentence_for_absa(data_sample):
    original_text = data_sample["text"]
    labels_spans = data_sample["labels"]

    # Bước làm sạch văn bản
    text = clean_text(original_text)

    # --- 1. Tách Token ở cấp độ từ/âm tiết (Syllable/Word Tokenization) ---
    tokens_str = ViTokenizer.tokenize(text)
    tokens = tokens_str.replace("_", " ").split()  # Chuyển chuỗi gạch dưới thành list các từ

    # Khởi tạo nhãn IOB2 cho từng token là 'O' (Outside)
    token_labels = ['O'] * len(tokens)

    # Ánh xạ từ index ký tự của văn bản gốc sang index của token
    token_info_list = []
    current_char_idx = 0
    for i, token in enumerate(tokens):
        while current_char_idx < len(text) and text[current_char_idx].isspace():
            current_char_idx += 1
        
        token_start = text.find(token, current_char_idx)
        
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
            print(f"Warning: Token '{token}' at token list index {i} not found in text from char index {current_char_idx}. Skipping for labeling alignment.")
            current_char_idx += len(token)

    # --- 2. Gán nhãn IOB2 cho từng token dựa trên spans đã gán nhãn ---
    for start_char_label, end_char_label, label_tag in labels_spans:
        affected_token_indices = []
        for token_info in token_info_list:
            token_start = token_info['start_char']
            token_end = token_info['end_char']
            token_idx = token_info['token_idx']

            if max(token_start, start_char_label) < min(token_end, end_char_label):
                affected_token_indices.append(token_idx)
        
        if affected_token_indices:
            affected_token_indices.sort()
            token_labels[affected_token_indices[0]] = f"B-{label_tag}"
            for i in range(1, len(affected_token_indices)):
                token_labels[affected_token_indices[i]] = f"I-{label_tag}"

    # --- 3. Tách Ký tự (Character Tokenization) ---
    char_tokens = list(text) 

    # --- 4. Tách Subword với XLM-RoBERTa Tokenizer ---
    xlmr_tokenized_output = xlmr_tokenizer(text, add_special_tokens=True)
    xlmr_tokens = xlmr_tokenizer.convert_ids_to_tokens(xlmr_tokenized_output['input_ids'])

    return {
        "text": text,                
        "tokens": tokens,            
        "token_labels": token_labels, 
        "char_tokens": char_tokens,  
        "xlmr_tokens": xlmr_tokens   
    }

# --- Đọc Dữ Liệu từ File JSON ---
input_file = '../data/labeled/video_comments_KOL1_1_spans.json'  
output_file = '../data/enriched/video_comments_KOL1_1_tokens.json'  

with open(input_file, 'r', encoding='utf-8') as f:
    input_data = json.load(f)

# Xử lý từng mẫu dữ liệu
processed_samples = []
for sample in input_data:
    processed_sample = process_sentence_for_absa(sample)
    processed_samples.append(processed_sample)

# Lưu kết quả vào tệp JSON
output_data = {
    "KOL1_1": processed_samples
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Kết quả đã được lưu vào: {output_file}")
