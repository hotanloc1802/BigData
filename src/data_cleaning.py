import re
import regex as re_unicode

# --- Hàm tiền xử lý cơ bản ---
def to_lowercase(text):
    """Chuyển đổi văn bản thành chữ thường."""
    return text.lower()

def remove_urls(text):
    """Loại bỏ các URL khỏi văn bản."""
    url_pattern = re.compile(
        r'https?://[^\s/$.?#].[^\s]*|www\.[^\s/$.?#].[^\s]*|'
        r'[a-zA-Z0-9.\-]+(?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?'
    )
    return url_pattern.sub(' ', text)

def remove_special_characters(text):
    """
    Loại bỏ các ký tự đặc biệt, giữ lại chữ cái, số, khoảng trắng và dấu câu cơ bản.
    Chuẩn hóa dấu câu: loại bỏ lặp lại, khoảng trắng thừa trước dấu.
    """
    cleaned_text = re_unicode.sub(r'[^\p{L}\p{N}\s.,!?]', ' ', text, flags=re_unicode.UNICODE)
    cleaned_text = re.sub(r'([.,!?-]){2,}', r'\1', cleaned_text) # Loại bỏ dấu câu lặp lại (ví dụ: !!! -> !)
    cleaned_text = re.sub(r'\s([.,!?-])', r'\1', cleaned_text) # Loại bỏ khoảng trắng trước dấu câu
    cleaned_text = re.sub(r'([.,!?-])(?=\S)', r'\1 ', cleaned_text) # Thêm khoảng trắng sau dấu câu nếu không có
    return cleaned_text

def normalize_whitespace(text):
    """Chuẩn hóa khoảng trắng thừa."""
    return re.sub(r'\s+', ' ', text).strip()

def normalize_repetitions(text):
    """Chuẩn hóa các ký tự lặp lại (ví dụ: helooo -> helo)."""
    return re.sub(r'(.)\1{2,}', r'\1', text)

# --- Từ điển từ lóng ---
slang_dict = {
    'iu': 'yêu', 'k': 'không', 'ko': 'không', 'hok': 'không', 'hong': 'không',
    'hg': 'không', 'hongg': 'không', 'khum': 'không', 'hem': 'không', 'hăm': 'không',
    'đượt': 'được', 'duoc': 'được', 'đc': 'được', 'đx': 'được', 'dc': 'được',
    'bt': 'biết', 'bít': 'biết', 'bic': 'biết', 'bik': 'biết', 'bthg': 'bình thường',
    'bth': 'bình thường', 'ad': 'admin', 'sp': 'sản phẩm', 'tl': 'trả lời',
    'trl': 'trả lời', 'mik': 'mình', 'mìn': 'mình', 'mn': 'mọi người', 'mng': 'mọi người',
    'nx': 'nữa', 'nũa': 'nữa', 'vs': 'với', 'dới': 'với', 'ntn': 'như thế nào',
    'cx': 'cũng', 'cug': 'cũng', 'cg': 'cũng', 'z': 'vậy', 'j': 'gì',
    'lun': 'luôn', 'nun': 'luôn', 'okela': 'ổn', 'hịn': 'xịn', 'ib': 'inbox',
    'cmt': 'bình luận', 'ngiu': 'người yêu', 'cb': 'chuẩn bị', 'chớt': 'chết',
    'chếc': 'chết', 'sml': 'sấp mặt luôn', 'vkl': 'vãi cả lúa', 'vcl': 'vãi cả lúa',
    'vl': 'vãi lúa', 'qá': 'quá', 'quáaa': 'quá', 'quó': 'quá', 'bả': 'bà ta',
    'bà': 'bà', 'chj': 'chị', 'c': 'chị', 'cj': 'chị', 'ch': 'chưa',
    'chế': 'chị', 'thg': 'thằng', 'con': 'con', 'đ': 'không', 'đou': 'đâu',
    'gì zậy': 'gì vậy', 'jz': 'gì vậy', 'btay': 'bó tay', 'nhma': 'nhưng mà',
    'nm': 'nhưng mà', 'đúng r': 'đúng rồi', 'r': 'rồi', 'rùi': 'rồi',
    'gòi': 'rồi', 'gùi': 'rồi', 'thik': 'thích', 'e': 'em', 'elm': 'em',
    'chỵ': 'chị', 'chuỵ': 'chị', 'nt': 'như thế', 'm.n': 'mọi người', 'v': 'vậy',
    'roài': 'rồi', 'xốp': 'shop', 'quánh': 'đánh', 'rì': 'gì', 'che': 'che',
    'ln': 'luôn', 'makep': 'make up', 'quện quện': 'vón cục', 'cục cục': 'vón cục',
    'nhấc lên': 'bị bong', 'thấy gớm': 'xấu', 'in4': 'thông tin', 't': 'tôi',
    'rv': 'review', 'ord': 'order', 'auth': 'authentic', 'phake': 'fake',
    'rep': 'fake', 'chấn động': 'gây ấn tượng mạnh', 'keo': 'đẹp', 'dịu keo': 'quá dễ thương',
    'hàn quắc': 'Hàn Quốc', 'âu mỹ': 'Âu Mỹ', 'pass': 'bán lại', 'thuiii': 'thôi',
    'thui mò': 'thôi mà', 'bn': 'bao nhiêu', 'bnhiu': 'bao nhiêu', 'dk': 'đúng không',
    'ng': 'người', 'mih': 'mình', 'cnay': 'cái này', 'kh': 'không', 'ê hề': 'nhiều',
    'zị': 'vậy', 'kím': 'kiếm', 'nhvay': 'như vậy', 'lm': 'làm', 'na ná': 'giống',
    'vay chèn': 'vậy trời', 'mom': 'chị', 'oiiii': 'ơi', 'camon': 'cảm ơn', 'b': 'bạn',
    'típ': 'tiếp', 'tr': 'trời', 'tậu': 'mua', 'Rcm': 'recommend', 'nhiu': 'nhiêu',
    'hp': 'hạnh phúc', 'đth': 'điện thoại', 'dth': 'dễ thương', 'vid': 'video',
    'vd': 'ví dụ', 'ui': 'ơi', 'tut': 'tutorial', 'hai': 'chị', 'h': 'giờ',
    'ms': 'mới'
}

def normalize_slang(text, slang_dict):
    """
    Chuẩn hóa các từ lóng trong văn bản dựa trên từ điển slang_dict.
    """
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

def capitalize_sentences(text):
    """Viết hoa chữ cái đầu câu sau dấu chấm, hỏi, than."""
    return re_unicode.sub(r'(^|\.\s*|\?\s*|!\s*)(\p{L})', lambda m: m.group(1) + m.group(2).upper(), text, flags=re_unicode.UNICODE)

def clean_comment_text(text):
    """
    Hàm chính để làm sạch một đoạn văn bản bình luận.
    Bao gồm các bước: chữ thường, loại bỏ URL, chuẩn hóa lặp lại,
    chuẩn hóa từ lóng, loại bỏ ký tự đặc biệt, chuẩn hóa khoảng trắng,
    và viết hoa chữ cái đầu câu.
    """
    # Bỏ qua các bình luận quá ngắn (ví dụ: chỉ có 1 hoặc 2 từ)
    if len(text.split()) < 3:
        return '' 
    
    text = to_lowercase(text)
    text = remove_urls(text)
    text = normalize_repetitions(text) 
    text = normalize_slang(text, slang_dict) # Sử dụng slang_dict đã định nghĩa
    text = remove_special_characters(text)
    text = normalize_whitespace(text)
    text = capitalize_sentences(text) 
    return text

