import subprocess
import os
import whisper
import time

# 🧾 Thông tin video TikTok
video_url = "https://www.tiktok.com/@yu.theeditor/video/7256733622533721350"
video_id = video_url.split("/")[-1]
video_filename = f"video_{video_id}.mp4"
audio_filename = f"audio_{video_id}.wav"
transcript_filename = f"transcript_{video_id}.txt"

# 🔧 Đường dẫn tuyệt đối đến ffmpeg
FFMPEG_PATH = r"D:\Program Files\ffmpeg-2025-06-16-git-e6fb8f373e-essentials_build\bin\ffmpeg.exe"

# 📥 Tải video bằng yt-dlp
def download_video(url, output_file):
    try:
        print("📥 Đang tải video bằng yt-dlp...")
        cmd = ["python", "-m", "yt_dlp", "-f", "mp4", "-o", output_file, url]
        subprocess.run(cmd, check=True)
        print(f"✅ Đã tải video: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi tải video: {e}")
        return False

# 🎧 Tách âm thanh và chuyển sang .wav chuẩn cho Whisper
def extract_audio(input_video, output_audio):
    try:
        print("🎧 Đang tách âm thanh (chuẩn .wav)...")
        cmd = [
            FFMPEG_PATH,
            "-i", input_video,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # 16kHz
            "-ac", "1",      # Mono
            "-y", output_audio
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Âm thanh đã lưu: {output_audio}")
        return output_audio
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg lỗi:\n{e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Lỗi khi tách âm thanh: {e}")
        return None

# 🧠 Nhận dạng âm thanh bằng Whisper large-v3
def transcribe_audio(audio_file):
    try:
        audio_path = os.path.abspath(audio_file)
        print(f"🧠 Đang nhận dạng với Whisper (medium): {audio_path}")
        if not os.path.exists(audio_path):
            print("❌ File âm thanh không tồn tại.")
            return ""
        model = whisper.load_model("medium")
        result = model.transcribe(audio_path, language="vi")
        print("📝 Transcript:", result["text"])
        return result["text"]
    except Exception as e:
        print(f"❌ Lỗi khi nhận dạng: {e}")
        import traceback
        traceback.print_exc()
        return ""

# 🔁 Luồng xử lý chính
def process_tiktok_video():
    if download_video(video_url, video_filename):
        time.sleep(1)
        audio_path = extract_audio(video_filename, audio_filename)
        if audio_path:
            transcript = transcribe_audio(audio_path)
            with open(transcript_filename, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"📄 Transcript đã lưu: {transcript_filename}")
        # 🧹 Xoá file tạm
        for file in [video_filename, audio_filename]:
            if os.path.exists(file):
                os.remove(file)
                print(f"🧹 Đã xoá: {file}")

# ▶️ Bắt đầu chạy
if __name__ == "__main__":
    process_tiktok_video()
