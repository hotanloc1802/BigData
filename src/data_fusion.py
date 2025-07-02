import os
import json

def fuse_data_pipelines(comments_filepath: str, video_scripts_dir: str, output_filepath: str):
    """
    Gộp dữ liệu bình luận đã nhúng và dữ liệu script video đã nhúng thành một file JSON duy nhất.

    Args:
        comments_filepath (str): Đường dẫn đến file .jsonl chứa các bình luận đã nhúng.
        video_scripts_dir (str): Đường dẫn đến thư mục chứa các file JSON của script video.
        output_filepath (str): Đường dẫn của file JSON đầu ra chứa dữ liệu gộp.
    """
    print(f"\n===== Bắt đầu bước Gộp dữ liệu (Data Fusion) =====")

    # 1. Đọc dữ liệu bình luận và nhóm theo video_id
    comments_by_video = {}
    try:
        with open(comments_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                comment_data = json.loads(line)
                video_id = comment_data.get("video_id") # Lấy video_id từ comment
                if video_id:
                    if video_id not in comments_by_video:
                        comments_by_video[video_id] = []
                    comments_by_video[video_id].append(comment_data)
                else:
                    print(f"⚠️ Bình luận không có video_id: {comment_data.get('original_text', comment_data.get('text'))[:50]}...")
        print(f"✅ Đã đọc {sum(len(v) for v in comments_by_video.values())} bình luận từ {len(comments_by_video)} video.")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file bình luận đã nhúng: {comments_filepath}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi giải mã JSON trong file bình luận {comments_filepath}: {e}")
        return
    except Exception as e:
        print(f"❌ Lỗi khi đọc dữ liệu bình luận: {e}")
        return

    # 2. Đọc dữ liệu script video
    combined_data = []
    video_script_files = [f for f in os.listdir(video_scripts_dir) if f.endswith(".json")]
    print(f"🔎 Tìm thấy {len(video_script_files)} file script video trong {video_scripts_dir}.")

    if not video_script_files:
        print("⚠️ Không có file script video nào để gộp.")
        return

    for filename in video_script_files:
        filepath = os.path.join(video_scripts_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                script_data = json.load(f)
                script_video_id = script_data.get("video_id")

                if script_video_id:
                    # Gán danh sách bình luận vào trường "comments" của script
                    script_data["comments"] = comments_by_video.get(script_video_id, [])
                    if not script_data["comments"]:
                        print(f"⚠️ Không tìm thấy bình luận nào cho video_id: {script_video_id}")
                    combined_data.append(script_data)
                else:
                    print(f"⚠️ File script {filename} không có video_id.")
        except json.JSONDecodeError as e:
            print(f"❌ Lỗi giải mã JSON trong file script {filename}: {e}")
        except Exception as e:
            print(f"❌ Lỗi khi đọc file script {filename}: {e}")

    # 3. Lưu dữ liệu gộp
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Đã gộp thành công {len(combined_data)} video và bình luận liên quan vào: {output_filepath}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu file gộp dữ liệu {output_filepath}: {e}")

    print(f"===== Hoàn tất bước Gộp dữ liệu =====")