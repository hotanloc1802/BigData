import json
from pyvi import ViTokenizer # Đã thay thế underthesea để tách từ/âm tiết tiếng Việt
from transformers import AutoTokenizer # Để tách subword với XLM-RoBERTa
import re # Để làm sạch văn bản

# Load XLM-RoBERTa tokenizer. Bạn có thể thay đổi sang một tokenizer cụ thể cho tiếng Việt
# nếu có sẵn và phù hợp hơn với mô hình của bạn (ví dụ: 'vietnamese-nlp/viRoberta-base')
# 'xlm-roberta-base' là một lựa chọn tốt cho mục đích chung và đa ngôn ngữ.
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
    """
    Xử lý một mẫu dữ liệu đã gán nhãn thành định dạng token-level
    phù hợp cho mô hình Span Detection (BiLSTM-CRF).

    Args:
        data_sample (dict): Một dictionary chứa "text" và "labels"
                            theo format guideline của bạn.

    Returns:
        dict: Chứa:
            - "text": Văn bản gốc đã được làm sạch.
            - "tokens": Danh sách các token (từ/âm tiết) đã được tách.
            - "token_labels": Danh sách các nhãn IOB2 cho từng token,
                              kết hợp với ENTITY#ATTRIBUTE#SENTIMENT.
            - "char_tokens": Danh sách các ký tự (cho character embedding).
            - "xlmr_tokens": Danh sách các subword tokens từ XLM-RoBERTa (cho contextual embedding).
    """
    original_text = data_sample["text"]
    labels_spans = data_sample["labels"]

    # Bước làm sạch văn bản
    text = clean_text(original_text)

    # --- 1. Tách Token ở cấp độ từ/âm tiết (Syllable/Word Tokenization) ---
    # Sử dụng pyvi.ViTokenizer để tách từ/âm tiết
    # ViTokenizer.tokenize trả về một chuỗi với các từ được nối bằng dấu gạch dưới '_'
    # Ví dụ: "Mê_chị_chou quá , em_tưởng em follow_chị rồi nhưng_hoá_ra chưa"
    # Sau đó cần split chuỗi này thành list các từ riêng biệt
    tokens_str = ViTokenizer.tokenize(text)
    tokens = tokens_str.replace("_", " ").split() # Chuyển chuỗi gạch dưới thành list các từ

    # Khởi tạo nhãn IOB2 cho từng token là 'O' (Outside)
    token_labels = ['O'] * len(tokens)

    # Ánh xạ từ index ký tự của văn bản gốc sang index của token
    # Tạo một danh sách các dictionary, mỗi dict chứa thông tin của một token:
    # text, start_char, end_char, token_idx
    token_info_list = []
    current_char_idx = 0
    for i, token in enumerate(tokens):
        # Bỏ qua khoảng trắng đầu
        while current_char_idx < len(text) and text[current_char_idx].isspace():
            current_char_idx += 1
        
        # Tìm vị trí chính xác của token trong văn bản gốc từ vị trí hiện tại
        token_start = text.find(token, current_char_idx)
        
        if token_start != -1: # Nếu tìm thấy token
            token_end = token_start + len(token)
            token_info_list.append({
                'token_text': token,
                'start_char': token_start,
                'end_char': token_end,
                'token_idx': i
            })
            current_char_idx = token_end # Cập nhật vị trí tìm kiếm tiếp theo
        else:
            # Xử lý lỗi nếu token không tìm thấy hoặc underthesea xử lý quá khác
            # Cho giải pháp mạnh mẽ, bạn có thể cần tái căn chỉnh hoặc ghi log các trường hợp này
            print(f"Warning: Token '{token}' at token list index {i} not found in text from char index {current_char_idx}. Skipping for labeling alignment.")
            current_char_idx += len(token) # Advance anyway to prevent infinite loop


    # --- 2. Gán nhãn IOB2 cho từng token dựa trên spans đã gán nhãn ---
    for start_char_label, end_char_label, label_tag in labels_spans:
        # Giả định: labels_spans dựa trên original_text và vẫn hợp lệ sau khi clean_text
        # Trong dự án thực tế, bạn có thể cần điều chỉnh lại các chỉ mục span sau khi làm sạch.

        affected_token_indices = []
        for token_info in token_info_list:
            token_start = token_info['start_char']
            token_end = token_info['end_char']
            token_idx = token_info['token_idx']

            # Kiểm tra sự trùng lặp giữa span được gán nhãn và token
            # Span (start_char_label, end_char_label) bao trùm hoặc trùng lặp với token (token_start, token_end)
            if max(token_start, start_char_label) < min(token_end, end_char_label):
                affected_token_indices.append(token_idx)
        
        if affected_token_indices:
            # Sắp xếp để đảm bảo thứ tự B-I-I-I chính xác
            affected_token_indices.sort()
            
            # Gán nhãn B- (Beginning) cho token đầu tiên trong span
            token_labels[affected_token_indices[0]] = f"B-{label_tag}"
            
            # Gán nhãn I- (Inside) cho các token tiếp theo trong span
            for i in range(1, len(affected_token_indices)):
                token_labels[affected_token_indices[i]] = f"I-{label_tag}"
        else:
            print(f"Warning: No tokens found for labeled span [{start_char_label}, {end_char_label}] with label '{label_tag}' in cleaned text: '{text}'. Original: '{original_text}'")

    # --- 3. Tách Ký tự (Character Tokenization) ---
    char_tokens = list(text) # Mỗi ký tự là một token riêng

    # --- 4. Tách Subword với XLM-RoBERTa Tokenizer ---
    # XLM-RoBERTa tokenizer sẽ xử lý văn bản gốc thành các subword token
    # `add_special_tokens=True` sẽ thêm các token như <s> và </s>
    xlmr_tokenized_output = xlmr_tokenizer(text, add_special_tokens=True)
    xlmr_tokens = xlmr_tokenizer.convert_ids_to_tokens(xlmr_tokenized_output['input_ids'])

    return {
        "text": text,                # Cleaned original text
        "tokens": tokens,            # Word/Syllable tokens from pyvi
        "token_labels": token_labels, # IOB2 labels for Word/Syllable tokens
        "char_tokens": char_tokens,  # Character tokens
        "xlmr_tokens": xlmr_tokens   # XLM-RoBERTa subword tokens
    }

# --- Chạy ví dụ ---
sample_data_1 = {
    "text": "Mê chị chou quá, em tưởng em follow chị rồi nhưng hoá ra chưa",
    "labels": [
      [0, 11, "KOL#null#POSITIVE"] # Span "Mê chị chou"
    ]
}

sample_data_2 = {
  "text": "Son ớt cay vậy đánh lâu có bị sưng mỏ không chị",
  "labels": [
    [0, 11, "PRODUCT#null#NEUTRAL"],
    [29, 36, "PRODUCT#null#NEGATIVE"]
  ]
}

sample_data_3 = {
  "text": "Chuyên văn có khác bà ta nói câu nào tôi cười khằng khặc câu đó",
  "labels": [
    [16, 62, "KOL#CONTENT_QUALITY#POSITIVE"] # Span: "bà ta nói câu nào tôi cười khằng khặc câu đó"
  ]
}

print("--- Processing Sample Data 1 ---")
processed_data_1 = process_sentence_for_absa(sample_data_1)
print(f"Original Text: {processed_data_1['text']}\n")
print(f"Syllable/Word Tokens: {processed_data_1['tokens']}")
print(f"IOB2 Labels for Syllable/Word Tokens: {processed_data_1['token_labels']}\n")
print(f"Character Tokens: {processed_data_1['char_tokens']}\n")
print(f"XLM-RoBERTa Subword Tokens: {processed_data_1['xlmr_tokens']}\n")

print("--- Processing Sample Data 2 ---")
processed_data_2 = process_sentence_for_absa(sample_data_2)
print(f"Original Text: {processed_data_2['text']}\n")
print(f"Syllable/Word Tokens: {processed_data_2['tokens']}")
print(f"IOB2 Labels for Syllable/Word Tokens: {processed_data_2['token_labels']}\n")
print(f"Character Tokens: {processed_data_2['char_tokens']}\n")
print(f"XLM-RoBERTa Subword Tokens: {processed_data_2['xlmr_tokens']}\n")

print("--- Processing Sample Data 3 ---")
processed_data_3 = process_sentence_for_absa(sample_data_3)
print(f"Original Text: {processed_data_3['text']}\n")
print(f"Syllable/Word Tokens: {processed_data_3['tokens']}")
print(f"IOB2 Labels for Syllable/Word Tokens: {processed_data_3['token_labels']}\n")
print(f"Character Tokens: {processed_data_3['char_tokens']}\n")
print(f"XLM-RoBERTa Subword Tokens: {processed_data_3['xlmr_tokens']}\n")
