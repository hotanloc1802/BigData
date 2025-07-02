import json
import os
import logging

# Import hàm làm sạch và từ điển từ data_cleaning.py
from src.data_cleaning import clean_comment_text

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_raw_comments(input_path, output_dir):
    """
    Đọc dữ liệu bình luận thô từ file JSON, làm sạch từng bình luận
    và lưu kết quả vào một file JSON mới.
    
    Args:
        input_path (str): Đường dẫn đến file JSON chứa dữ liệu bình luận thô.
        output_dir (str): Đường dẫn thư mục để lưu file JSON đã làm sạch.
    """
    if not os.path.exists(input_path):
        logging.error(f"❌ Không tìm thấy file đầu vào: {input_path}")
        return

    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_json_path = os.path.join(output_dir, f"{base_filename}_cleaned.json")
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"❌ Lỗi đọc file JSON {input_path}: {e}")
        return
    except Exception as e:
        logging.error(f"❌ Lỗi không xác định khi đọc file {input_path}: {e}")
        return

    # Chuẩn hóa định dạng đầu vào: luôn là một list các video
    if isinstance(raw_data, dict) and 'video_url' in raw_data:
        raw_data = [raw_data]

    logging.info(f"📥 Đã đọc {len(raw_data)} video từ {input_path}")
    logging.info(f"📄 File đầu ra sẽ là: {output_json_path}")

    processed_entries = []

    for video in raw_data:
        if not isinstance(video, dict):
            logging.warning(f"⚠️ Bỏ qua phần tử không hợp lệ trong dữ liệu gốc: {video}")
            continue

        for comment in video.get("comments", []):
            raw_text = comment.get("text", "")
            cleaned_text = clean_comment_text(raw_text)

            if cleaned_text.strip(): # Chỉ thêm vào nếu văn bản đã làm sạch không rỗng
                processed_entries.append({
                    "text": cleaned_text,
                    "labels": [] # Giữ nguyên trường labels để tương thích với định dạng sau này
                })

            for reply in comment.get("replies", []):
                reply_text = reply.get("text", "")
                cleaned_reply = clean_comment_text(reply_text)

                if cleaned_reply.strip(): # Chỉ thêm vào nếu văn bản đã làm sạch không rỗng
                    processed_entries.append({
                        "text": cleaned_reply,
                        "labels": []
                    })
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(processed_entries, f, ensure_ascii=False, indent=2)
        logging.info(f"✅ Đã lưu file JSON đã làm sạch: {output_json_path}")
    except Exception as e:
        logging.error(f"❌ Lỗi khi lưu file JSON {output_json_path}: {e}")

