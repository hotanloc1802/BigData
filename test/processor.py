import json
import os
from src.vncorenlp_wrapper import VnCoreNLPWrapper
from src.cleaner import clean_comment_text # Assuming cleaner.py exists and clean_comment_text is defined
import sys
import os

def process_file(input_path, output_annotated_path, jar_path, annotators):
    # === B1: Đọc dữ liệu gốc ===
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Hỗ trợ cả dạng 1 video lẻ
    if isinstance(raw_data, dict) and 'video_url' in raw_data:
        raw_data = [raw_data]

    print(f"📥 Đã đọc {len(raw_data)} video")

    # === B2: Khởi tạo NLP duy nhất 1 lần ===
    nlp = VnCoreNLPWrapper(
        jar_path=jar_path,
        annotators=annotators,
        models_dir="vncorenlp/models" # Ensure this path is correct relative to where you run the script
    )

    all_texts_to_annotate = []
    # Store references to where each text came from to map back results
    # Each item in this list will be a dict: {"comment_obj": ..., "type": "comment"|"reply"}
    text_references = [] 

    # === B2.1: Thu thập tất cả văn bản cần phân tích ===
    print("Collecting all comments and replies for batch processing...")
    for video in raw_data:
        if not isinstance(video, dict):
            print(f"⚠️ Bỏ qua phần tử không hợp lệ: {video}")
            continue

        for comment in video.get("comments", []):
            raw_text = comment.get("text", "")
            cleaned_text = clean_comment_text(raw_text)
            
            all_texts_to_annotate.append(cleaned_text)
            text_references.append({
                "source_type": "comment",
                "original_obj": comment # Keep a reference to the original comment object
            })

            for reply in comment.get("replies", []):
                reply_text = reply.get("text", "")
                cleaned_reply = clean_comment_text(reply_text)
                
                all_texts_to_annotate.append(cleaned_reply)
                text_references.append({
                    "source_type": "reply",
                    "original_obj": reply # Keep a reference to the original reply object
                })

    # === B2.2: Gọi NLP một lần duy nhất cho tất cả văn bản ===
    print(f"Calling VnCoreNLP to annotate {len(all_texts_to_annotate)} texts in batch...")
    # The annotate_batch method returns a list of lists of dictionaries
    # where each inner list corresponds to one of the input texts.
    all_annotations = nlp.annotate_batch(all_texts_to_annotate)
    print("Batch annotation complete.")

    # === B2.3: Gắn kết quả phân tích trở lại dữ liệu gốc ===
    annotated_data = []
    
    if len(all_annotations) != len(all_texts_to_annotate):
        print(f"❌ Cảnh báo: Số lượng kết quả phân tích ({len(all_annotations)}) không khớp với số lượng văn bản đầu vào ({len(all_texts_to_annotate)}). Có thể có lỗi trong quá trình phân tích.")
    
    # Iterate through the original references and attach their corresponding annotations
    for i, ref in enumerate(text_references):
        current_annotations = all_annotations[i] if i < len(all_annotations) else []
        current_cleaned_text = all_texts_to_annotate[i] # The cleaned text itself

        # Construct the annotated entry
        entry = {
            "text": current_cleaned_text,
            "username": ref["original_obj"].get("author", {}).get("username", ""),
            "likes_count": ref["original_obj"].get("likes_count", 0),
            "annotations": current_annotations # This is the key change!
        }
        annotated_data.append(entry)

    # === B3: Ghi ra file kết quả ===
    os.makedirs(os.path.dirname(output_annotated_path), exist_ok=True)

    with open(output_annotated_path, "w", encoding="utf-8") as f:
        json.dump(annotated_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu file phân tích: {output_annotated_path}")