import subprocess
import os
import whisper
import time
import json
from datetime import datetime
from src.data_cleaning import clean_comment_text
import torch 
import torch.nn as nn 
import fasttext 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel 
from pyvi import ViTokenizer 

# --- Bắt đầu phần code trùng lặp để tạo Fused Embedding cho video script ---
# VÌ YÊU CẦU "K THAY ĐỔI EMBEDDING_UTILIS" VÀ "K SỬ DỤNG EMBEDDING_UTILIS TRONG VIDEO_PROCESSOR"
# Đây là các hằng số và lớp/hàm được sao chép/tái tạo từ src/embedding_utils.py để sử dụng riêng cho video_processor
# để có được fused embedding 1168 chiều.

# Tham số cấu hình cho các Embedder trong video_processor
FASTTEXT_MODEL_PATH = os.path.join("models", "fasttext", "cc.vi.300.bin") 
FASTTEXT_EMBEDDING_DIM = 300

CHAR_EMBEDDING_DIM = 50
CHAR_LSTM_HIDDEN_DIM = 50
CHAR_FEATURE_OUTPUT_DIM = CHAR_LSTM_HIDDEN_DIM * 2

# Đã sửa lại đường dẫn của mô hình XLM-RoBERTa để trỏ đến thư mục cục bộ (Đã sử dụng os.path.join)
XLMR_MODEL_NAME = os.path.join("models", "huggingface", "hub", "models--xlm-roberta-base", "snapshots", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089")
XLMR_EMBEDDING_DIM = 768

# Cấu hình mô hình tóm tắt (Đã sửa lại tên mô hình và sử dụng os.path.join)
SUMMARIZATION_MODEL_NAME = os.path.join("models", "huggingface", "hub", "models--VietAI--vit5-large-vietnews-summarization", "snapshots", "8926b4edb541fd4f28260e5469d8a00121785fcf")
summarization_tokenizer = None
summarization_model = None


# Ánh xạ ký tự sang chỉ số (Cần thiết cho CharBiLSTMEmbedder)
# Đã sửa lỗi ký tự không in được U+00A0 và dấu nháy đơn
video_char_to_idx = {char: i for i, char in enumerate(
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789áàảạãăằẳẵặâầẩẫậéèẻẹẽêềểễệíìỉịĩóòỏọõôồổỗộơờởỡợúùủụũưừửữựýỳỷỵỹđĐ,.:;?!\'\"()[]{}<>/-_&@#$%^&*=+~`'
)}
video_char_to_idx["<PAD>"] = 0
video_char_to_idx["<unk>"] = len(video_char_to_idx)
VIDEO_CHAR_VOCAB_SIZE = len(video_char_to_idx)

class VideoFastTextEmbedder:
    def __init__(self, model_path=FASTTEXT_MODEL_PATH):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy mô hình FastText cho video tại: {model_path}")
            self.model = fasttext.load_model(model_path)
            self.embedding_dim = self.model.get_word_vector("test").shape[0]
            print(f"✅ Đã tải mô hình FastText cho video từ: {model_path} (dim: {self.embedding_dim})")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải mô hình FastText cho video từ '{model_path}': {e}.")
    def get_embedding(self, word):
        return torch.from_numpy(self.model.get_word_vector(word)).float()

class VideoCharBiLSTMEmbedder(nn.Module):
    def __init__(self, vocab_size=VIDEO_CHAR_VOCAB_SIZE, embedding_dim=CHAR_EMBEDDING_DIM, hidden_dim=CHAR_LSTM_HIDDEN_DIM):
        super(VideoCharBiLSTMEmbedder, self).__init__()
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=video_char_to_idx["<PAD>"])
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

class VideoXLMREmbedder: # Tái sử dụng nhưng với tên khác để tránh nhầm lẫn
    def __init__(self, model_name=XLMR_MODEL_NAME):
        try:
            if not os.path.exists(model_name) or \
               not (os.path.exists(os.path.join(model_name, 'tokenizer.json')) or \
                    os.path.exists(os.path.join(model_name, 'config.json'))):
                raise FileNotFoundError(f"Không tìm thấy mô hình XLM-RoBERta cho video tại: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.embedding_dim = self.model.config.hidden_size
            print(f"✅ Đã tải mô hình XLM-RoBERTa cho video từ: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải mô hình XLM-RoBERTa cho video từ '{model_name}': {e}.")

    def get_token_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        offset_mapping = inputs.pop('offset_mapping')
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state.squeeze(0)
        return token_embeddings, inputs['input_ids'].squeeze(0), offset_mapping.squeeze(0)

# Các instance global cho embedder trong video_processor
video_fasttext_embedder_instance = None
video_char_bilstm_embedder_instance = None
video_xlm_roberta_embedder_instance = None # Instance riêng cho video_processor

def load_video_script_embedders():
    """Tải và khởi tạo tất cả các embedder cần thiết cho video script."""
    global video_fasttext_embedder_instance, video_char_bilstm_embedder_instance, video_xlm_roberta_embedder_instance
    if video_fasttext_embedder_instance is None:
        try:
            video_fasttext_embedder_instance = VideoFastTextEmbedder()
            video_char_bilstm_embedder_instance = VideoCharBiLSTMEmbedder()
            video_xlm_roberta_embedder_instance = VideoXLMREmbedder()
            print("✅ Đã khởi tạo tất cả Embedder cho video script.")
        except RuntimeError as e:
            print(f"❌ Lỗi khi khởi tạo Embedder cho video script: {e}")
            raise # Báo lỗi để dừng nếu không thể tải mô hình

def process_text_to_word_tokens_for_video(text): # Sao chép từ embedding_utils/text_utils
    """Tách văn bản thành các word/syllable tokens bằng pyvi."""
    cleaned_text = text # Giả định đã clean bởi clean_text_content
    tokens_str = ViTokenizer.tokenize(cleaned_text)
    tokens = tokens_str.replace("_", " ").split()
    # (Để đơn giản, bỏ qua logic căn chỉnh phức tạp nếu nó không cần cho fused sentence embedding)
    return tokens, cleaned_text, None # Trả về None cho token_info_list nếu không dùng

def get_char_indices_for_words_for_video(cleaned_text, tokens, char_to_idx): # Sao chép từ embedding_utils/text_utils
    """Tạo danh sách các tensor chỉ số ký tự cho mỗi từ/âm tiết."""
    char_indices_list_per_word = []
    for token in tokens:
        word_char_indices = [char_to_idx.get(c, char_to_idx['<unk>']) for c in token] # Dùng token trực tiếp
        char_indices_list_per_word.append(torch.tensor(word_char_indices, dtype=torch.long))
    return char_indices_list_per_word

def get_fused_sentence_embedding_for_script(script_text: str) -> list:
    """
    Tạo fused sentence embedding (1168 chiều) cho script.
    """
    if video_fasttext_embedder_instance is None: # Đảm bảo đã tải các embedder
        load_video_script_embedders()

    # Bước 1: Tokenize văn bản
    tokens, _, _ = process_text_to_word_tokens_for_video(script_text)
    if not tokens:
        return torch.zeros(FASTTEXT_EMBEDDING_DIM + CHAR_FEATURE_OUTPUT_DIM + XLMR_EMBEDDING_DIM).tolist() # Sử dụng hằng số chung

    # Bước 2: Tạo FastText Embeddings và Mean Pooling
    fasttext_word_embs = torch.stack([video_fasttext_embedder_instance.get_embedding(token) for token in tokens])
    fasttext_sentence_emb = torch.mean(fasttext_word_embs, dim=0)

    # Bước 3: Tạo Character Embeddings và Mean Pooling
    char_indices_list_per_word = get_char_indices_for_words_for_video(script_text, tokens, video_char_to_idx)
    character_word_embs = video_char_bilstm_embedder_instance(char_indices_list_per_word)
    character_sentence_emb = torch.mean(character_word_embs, dim=0)

    # Bước 4: Tạo XLM-RoBERTa Sentence Embedding (dùng CLS token)
    # Tái sử dụng get_sentence_embedding_from_xlmr đã có
    xlmr_sentence_emb = torch.tensor(get_sentence_embedding_from_xlmr_internal(script_text))


    # Bước 5: Nối các embedding
    fused_embedding = torch.cat(
        (fasttext_sentence_emb, character_sentence_emb, xlmr_sentence_emb), dim=0
    )
    return fused_embedding.tolist()

def get_sentence_embedding_from_xlmr_internal(text: str) -> list:
    """
    Tạo sentence embedding từ văn bản bằng mô hình XLM-RoBERTa được tải cục bộ (dùng trong video_processor).
    """
    if video_xlm_roberta_embedder_instance is None:
        load_video_script_embedders() # Tải tất cả embedder nếu chưa có
    
    inputs = video_xlm_roberta_embedder_instance.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = video_xlm_roberta_embedder_instance.model(**inputs)
        sentence_embedding = outputs.last_hidden_state[0, 0, :] 
    return sentence_embedding.tolist()

# --- Các hàm khác (download_video, extract_audio, transcribe_audio, summarize_transcript) giữ nguyên ---

# 🔧 Đường dẫn tương đối đến ffmpeg (cần đảm bảo ffmpeg có trong PATH hoặc cung cấp đường dẫn đầy đủ)
FFMPEG_PATH = "ffmpeg" 


summarization_tokenizer = None
summarization_model = None


def load_summarization_model():
    """Tải tokenizer và mô hình tóm tắt nếu chúng chưa được tải."""
    global summarization_tokenizer, summarization_model
    if summarization_tokenizer is None:
        print(f"✨ Đang tải mô hình tóm tắt: {SUMMARIZATION_MODEL_NAME}...")
        try:
            tokenizer_loaded = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
            model_loaded = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)
            
            if not callable(tokenizer_loaded): 
                raise TypeError("Đối tượng tokenizer được tải không phải là một hàm (callable).")

            summarization_tokenizer = tokenizer_loaded
            summarization_model = model_loaded
            summarization_model.eval() 
            print(f"✅ Đã tải mô hình tóm tắt từ: {SUMMARIZATION_MODEL_NAME}")
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình tóm tắt '{SUMMARIZATION_MODEL_NAME}': {e}")
            summarization_tokenizer = None 
            summarization_model = None 
            raise RuntimeError(f"Không thể tải mô hình tóm tắt.") 

def get_video_info(url):
    """Lấy thông tin JSON của video từ URL bằng yt-dlp."""
    try:
        cmd = ["yt-dlp", "--print-json", url]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        video_info = json.loads(result.stdout)
        return video_info
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Lỗi khi lấy thông tin video bằng yt-dlp: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi giải mã JSON từ yt-dlp: {e}")
        return None

def download_video(url, output_file_path):
    """Tải video từ URL bằng yt-dlp."""
    try:
        print(f"📥 Đang tải video: {url}...")
        cmd = ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4", "-o", output_file_path, url] 
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Đã tải video: {output_file_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi yt-dlp:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Lỗi: yt-dlp không tìm thấy. Vui lòng đảm bảo yt-dlp đã được cài đặt và có trong PATH.")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi tải video: {e}")
        return False

def extract_audio(input_video_path, output_audio_path):
    """Tách âm thanh từ video và chuyển đổi sang định dạng WAV (16kHz, mono) bằng FFmpeg."""
    try:
        print(f"🎧 Đang tách âm thanh từ {input_video_path} (chuẩn .wav)...")
        cmd = [
            FFMPEG_PATH,
            "-i", input_video_path,
            "-vn", 
            "-acodec", "pcm_s16le", 
            "-ar", "16000",  
            "-ac", "1",      
            "-y", 
            output_audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Âm thanh đã lưu: {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg lỗi:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        print("Vui lòng đảm bảo FFmpeg đã được cài đặt và có trong PATH hệ thống.")
        return None
    except FileNotFoundError:
        print("❌ Lỗi: FFmpeg không tìm thấy. Vui lòng đảm bảo FFmpeg đã được cài đặt và có trong PATH.")
        return None
    except Exception as e:
        print(f"❌ Lỗi khi tách âm thanh: {e}")
        return None

def transcribe_audio(audio_file_path, whisper_model_name="medium", download_root="models/whisper"):
    """Thực hiện chuyển đổi giọng nói thành văn bản bằng mô hình Whisper."""
    try:
        print(f"🧠 Đang nhận dạng với Whisper ({whisper_model_name}): {audio_file_path}")
        if not os.path.exists(audio_file_path):
            print("❌ File âm thanh không tồn tại.")
            return ""
        
        if not os.path.exists(download_root):
            os.makedirs(download_root)
            print(f"✅ Đã tạo thư mục Whisper model: {download_root}")

        model = whisper.load_model(whisper_model_name, download_root=download_root)
        result = model.transcribe(audio_file_path, language="vi")
        print("📝 Transcript:", result["text"][:200], "...") 
        return result["text"]
    except Exception as e:
        print(f"❌ Lỗi khi nhận dạng với Whisper: {e}")
        import traceback
        traceback.print_exc()
        return ""

async def summarize_transcript(transcript_text: str) -> str:
    """
    Tóm tắt nội dung của transcript sử dụng mô hình VietAI/vit5-large-vietnews-summarization.
    """
    if not transcript_text:
        return "Không có nội dung để tóm tắt."

    try:
        if summarization_model is None or summarization_tokenizer is None:
            load_summarization_model()

        print(f"\n✨ Đang tóm tắt transcript bằng mô hình: {SUMMARIZATION_MODEL_NAME}...")
        
        input_ids = summarization_tokenizer(transcript_text, return_tensors="pt", max_length=512, truncation=True).input_ids
        
        with torch.no_grad(): 
            generated_ids = summarization_model.generate(
                input_ids, 
                max_length=150, 
                min_length=30, 
                num_beams=5, 
                early_stopping=True
            )
        summary = summarization_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"✅ Đã tóm tắt: {summary}")
        return summary
    except Exception as e:
        print(f"❌ Lỗi khi tóm tắt bằng mô hình: {e}")
        import traceback
        traceback.print_exc()
        return "Không thể tóm tắt do lỗi."

async def process_video_pipeline(video_url: str, output_raw_dir: str, output_summary_dir: str): 
    """
    Pipeline xử lý video: tải xuống, trích xuất âm thanh, chuyển đổi giọng nói thành văn bản, làm sạch, tóm tắt,
    và tạo dữ liệu có cấu trúc để gán nhãn thuộc tính với fused embedding 1168 chiều.
    """
    video_id = video_url.split("/")[-1].split("?")[0] 
    
    os.makedirs(output_raw_dir, exist_ok=True)
    os.makedirs(output_summary_dir, exist_ok=True)

    video_filename = os.path.join(output_raw_dir, f"video_{video_id}.mp4")
    audio_filename = os.path.join(output_raw_dir, f"audio_{video_id}.wav")
    raw_transcript_filename = os.path.join(output_raw_dir, f"transcript_raw_{video_id}.txt") 
    cleaned_transcript_filename = os.path.join(output_raw_dir, f"transcript_cleaned_{video_id}.txt") 
    transcript_summary_filepath = os.path.join(output_summary_dir, f"transcript_summary_{video_id}.json")

    # Lấy thông tin video (tiêu đề) trước khi tải
    video_info = get_video_info(video_url)
    video_title = video_info.get("title", "Unknown Title") if video_info else "Unknown Title"
    print(f"🎥 Tiêu đề video: {video_title}")

    # Bước 1: Tải video
    if not os.path.exists(video_filename):
        if not download_video(video_url, video_filename):
            print("❌ Dừng pipeline xử lý video do lỗi tải video.")
            return
    else:
        print(f"✅ Video đã tồn tại: {video_filename}. Bỏ qua tải xuống.")

    # Bước 2: Tách âm thanh
    if not os.path.exists(audio_filename):
        audio_path = extract_audio(video_filename, audio_filename)
        if not audio_path:
            print("❌ Dừng pipeline xử lý video do lỗi tách âm thanh.")
            return
    else:
        print(f"✅ File âm thanh đã tồn tại: {audio_filename}. Bỏ qua tách âm thanh.")
        audio_path = audio_filename

    # Bước 3: Nhận dạng giọng nói (transcript thô)
    raw_transcript_text = ""
    if not os.path.exists(raw_transcript_filename):
        raw_transcript_text = transcribe_audio(audio_path)
        if raw_transcript_text:
            with open(raw_transcript_filename, "w", encoding="utf-8") as f:
                f.write(raw_transcript_text)
            print(f"📄 Transcript thô đã lưu: {raw_transcript_filename}")
        else:
            print("❌ Không thể tạo transcript thô. Dừng pipeline xử lý video.")
            return
    else:
        print(f"✅ File transcript thô đã tồn tại: {raw_transcript_filename}. Bỏ qua nhận dạng giọng nói.")
        with open(raw_transcript_filename, "r", encoding="utf-8") as f:
            raw_transcript_text = f.read()

    # Bước 4: Làm sạch transcript
    cleaned_transcript_text = ""
    if raw_transcript_text and not os.path.exists(cleaned_transcript_filename):
        print("🧹 Đang làm sạch transcript...")
        cleaned_transcript_text = clean_comment_text(raw_transcript_text) 
        if cleaned_transcript_text:
            with open(cleaned_transcript_filename, "w", encoding="utf-8") as f:
                f.write(cleaned_transcript_text)
            print(f"✅ Transcript đã làm sạch đã lưu: {cleaned_transcript_filename}")
        else:
            print("⚠️ Transcript trống sau khi làm sạch. Bỏ qua các bước tiếp theo.")
            return 
    elif os.path.exists(cleaned_transcript_filename):
        print(f"✅ File transcript đã làm sạch đã tồn tại: {cleaned_transcript_filename}. Bỏ qua làm sạch.")
        with open(cleaned_transcript_filename, "r", encoding="utf-8") as f:
            cleaned_transcript_text = f.read()
    else: 
        print("⚠️ Không có transcript thô để làm sạch.")
        return

    # Bước 5: Tóm tắt transcript (sử dụng mô hình VietAI/vit5-large-vietnews-summarization)
    # Đã di chuyển lên trước để tạo embedding dựa trên tóm tắt
    summary_text = ""
    if cleaned_transcript_text and not os.path.exists(transcript_summary_filepath):
        summary_text = await summarize_transcript(cleaned_transcript_text)
        if summary_text:
            summary_output = {
                "video_id": video_id,
                "video_url": video_url,
                "cleaned_transcript_filepath": cleaned_transcript_filename, 
                "summary": summary_text,
                "timestamp": datetime.now().isoformat()
            }
            with open(transcript_summary_filepath, "w", encoding="utf-8") as f:
                json.dump(summary_output, f, indent=2, ensure_ascii=False)
            print(f"✅ Tóm tắt abstractive đã lưu: {transcript_summary_filepath}")
        else:
            print("❌ Không thể tạo tóm tắt abstractive.")
    elif os.path.exists(transcript_summary_filepath):
        print(f"✅ File tóm tắt abstractive đã tồn tại: {transcript_summary_filepath}. Bỏ qua tóm tắt.")
        with open(transcript_summary_filepath, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
            summary_text = summary_data.get("summary", "")
    else:
        print("⚠️ Không có transcript đã làm sạch để tạo tóm tắt abstractive.")
        return # Dừng pipeline nếu không có tóm tắt để tạo embedding

    # Bước 6: Tạo script_context_vector (1168 chiều) dựa trên TÓM TẮT và lưu dữ liệu có cấu trúc để gán nhãn
    script_context_vector = None
    if summary_text: # Sử dụng summary_text thay vì cleaned_transcript_text
        print("� Đang tạo script context vector (1168 chiều) từ TÓM TẮT...")
        try:
            load_video_script_embedders() 
            script_context_vector = get_fused_sentence_embedding_for_script(summary_text) # Thay đổi ở đây
            print(f"✅ Đã tạo script context vector (dim: {len(script_context_vector)}) từ tóm tắt.")
        except Exception as e:
            print(f"❌ Lỗi khi tạo script context vector từ tóm tắt: {e}")
            script_context_vector = None
    else:
        print("⚠️ Không thể tạo script context vector (tóm tắt rỗng).")

    # Lưu dữ liệu có cấu trúc cho việc gán nhãn
    structured_output_for_labeling_dir = os.path.join("data", "script", "preprocessed") 
    os.makedirs(structured_output_for_labeling_dir, exist_ok=True)
    structured_output_filepath = os.path.join(structured_output_for_labeling_dir, f"video_script_features_for_labeling_{video_id}.json")

    structured_output = {
        "video_id": video_id,
        "video_url": video_url,
        "video_title": video_title,
        "script_text": cleaned_transcript_text, # Vẫn giữ script đầy đủ để tham chiếu
        "summary_text": summary_text, # Thêm tóm tắt vào output
        "script_context_vector_from_summary": script_context_vector, # Đổi tên để rõ ràng hơn
        "script_attributes_mentioned": [] # Để trống cho việc gán nhãn thủ công
    }
    with open(structured_output_filepath, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=2, ensure_ascii=False)
    print(f"✅ Dữ liệu script video đã lưu ở định dạng sẵn sàng gán nhãn: {structured_output_filepath}")


    # 🧹 Xoá file tạm (video và audio)
    for file in [video_filename, audio_filename]:
        if os.path.exists(file):
            os.remove(file)
            print(f"🧹 Đã xoá file tạm: {file}")

    print(f"\n--- Pipeline xử lý video cho {video_id} hoàn tất ---")