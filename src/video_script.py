import subprocess
import os
import whisper

FFMPEG_PATH = r"D:\Program Files\ffmpeg-2025-06-16-git-e6fb8f373e-essentials_build\bin\ffmpeg.exe"

def extract_audio(input_video_path, output_audio_path):
    try:
        if not os.path.exists(input_video_path):
            print(f"❌ File video không tồn tại: {input_video_path}")
            return None

        ffmpeg_cmd_base = [FFMPEG_PATH] if os.path.exists(FFMPEG_PATH) else ["ffmpeg"]

        try:
            subprocess.run(ffmpeg_cmd_base + ["-version"], check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Không tìm thấy ffmpeg.")
            return None

        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

        cmd = [
            *ffmpeg_cmd_base,
            "-i", input_video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y", output_audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Đã tạo âm thanh: {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg lỗi:\n{e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")
        return None

def transcribe_audio(audio_file_path):
    try:
        if not os.path.exists(audio_file_path):
            print("❌ File âm thanh không tồn tại.")
            return ""

        model = whisper.load_model("medium")
        result = model.transcribe(audio_file_path, language="vi")
        return result["text"]
    except Exception as e:
        print(f"❌ Lỗi khi nhận dạng âm thanh: {e}")
        return ""

def process_video_from_local_path(local_video_path):
    if not os.path.exists(local_video_path):
        print(f"❌ File video không tồn tại: {local_video_path}")
        return

    base_name = os.path.splitext(os.path.basename(local_video_path))[0]
    script_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../data/script"))
    audio_path = os.path.join(script_dir, f"{base_name}.wav")
    transcript_path = os.path.join(script_dir, f"{base_name}_transcript.txt")

    audio_file = extract_audio(local_video_path, audio_path)
    if audio_file:
        transcript = transcribe_audio(audio_file)
        if transcript:
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"📄 Đã lưu transcript: {transcript_path}")
        else:
            print("❗ Transcript trống.")

    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            print(f"🧹 Đã xoá file tạm: {audio_path}")
        except OSError as e:
            print(f"❌ Lỗi khi xoá {audio_path}: {e}")

if __name__ == "__main__":
    local_video_path_input = input("Nhập đường dẫn video: ")
    if os.path.exists(local_video_path_input):
        process_video_from_local_path(local_video_path_input)
    else:
        print(f"❌ File không tồn tại: {local_video_path_input}")
