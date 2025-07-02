import asyncio
import os
import json
import argparse
from src.data_collection import fetch_tiktok_comments
from src.data_cleaning import clean_comment_text
from src.embedding_utils import FastTextEmbedder, CharBiLSTMEmbedder, XLMREmbedder, get_fused_embeddings_and_iob2_labels
from src.video_processor import process_video_pipeline 
from src.data_fusion import fuse_data_pipelines # Import hàm gộp dữ liệu

# --- Khởi tạo Embedder (khởi tạo một lần để tái sử dụng) ---
# Các instance này chỉ dùng cho pipeline bình luận (comments pipeline)
fasttext_embedder_instance = None
char_bilstm_embedder_instance = None
xlmr_embedder_instance = None 

def initialize_embedders():
    """Khởi tạo các embedder nếu chúng chưa được khởi tạo. Các embedder này chỉ dùng cho pipeline bình luận."""
    global fasttext_embedder_instance, char_bilstm_embedder_instance, xlmr_embedder_instance
    if fasttext_embedder_instance is None: # Chỉ khởi tạo nếu chưa có
        try:
            fasttext_embedder_instance = FastTextEmbedder()
            char_bilstm_embedder_instance = CharBiLSTMEmbedder()
            xlmr_embedder_instance = XLMREmbedder() 
            print("✅ Đã khởi tạo tất cả Embedder cho pipeline bình luận.")
        except RuntimeError as e:
            print(f"❌ Không thể khởi tạo một hoặc nhiều Embedder: {e}")
            print("Vui lòng kiểm tra lại đường dẫn mô hình, kết nối internet và cài đặt thư viện.")
            raise 

async def run_comments_pipeline(args, ms_token, video_url_for_comments):
    """Chạy toàn bộ pipeline xử lý bình luận."""
    # Đã giữ nguyên đường dẫn bạn cung cấp
    raw_comments_dir = "data/comment/raw"
    raw_comments_filename = "video_comments_raw.json"
    raw_comments_filepath = os.path.join(raw_comments_dir, raw_comments_filename)

    # Đã giữ nguyên đường dẫn bạn cung cấp
    cleaned_comments_dir = "data/comment/cleaned"
    cleaned_comments_filename = "comments_formatted_for_labeling.json" 
    cleaned_comments_filepath = os.path.join(cleaned_comments_dir, cleaned_comments_filename)

    # Đã giữ nguyên đường dẫn bạn cung cấp
    preprocessed_comments_dir = "data/comment/preprocessed"
    preprocessed_comments_filename = "comments_with_embeddings_and_labels.jsonl"
    preprocessed_comments_filepath = os.path.join(preprocessed_comments_dir, preprocessed_comments_filename)

    print("\n===== Bắt đầu Pipeline Xử lý Bình luận =====")
    
    # Bước 1: Thu thập dữ liệu thô
    if args.comment_step in ["full", "collect"]:
        print(f"\n--- Bắt đầu quá trình thu thập bình luận cho video: {video_url_for_comments} ---")
        await fetch_tiktok_comments(ms_token, video_url_for_comments, raw_comments_filepath, count=50)
        print("--- Quá trình thu thập bình luận thô hoàn tất ---")
        if args.comment_step == "collect":
            print("\n--- Chỉ chạy bước thu thập bình luận. Hoàn tất pipeline bình luận. ---")
            return 

    # Bước 2: Làm sạch dữ liệu
    if args.comment_step in ["full", "clean"]:
        print("\n--- Bắt đầu quá trình làm sạch dữ liệu bình luận ---")
        if not os.path.exists(cleaned_comments_dir):
            os.makedirs(cleaned_comments_dir)
            print(f"✅ Đã tạo thư mục: {cleaned_comments_dir}")

        formatted_cleaned_comments_list = []

        if os.path.exists(raw_comments_filepath):
            try:
                with open(raw_comments_filepath, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                
                total_processed_count = 0
                for comment_entry in raw_data.get("comments", []):
                    original_text = comment_entry.get("text", "")
                    cleaned_text = clean_comment_text(original_text)
                    
                    total_processed_count += 1

                    if cleaned_text: 
                        formatted_comment = {
                            "text": cleaned_text,
                            "labels": [] 
                        }
                        # Đảm bảo video_id và video_url được bảo toàn sau bước làm sạch nếu chúng có trong raw_data
                        if "video_id" in comment_entry: formatted_comment["video_id"] = comment_entry["video_id"]
                        if "video_url" in comment_entry: formatted_comment["video_url"] = comment_entry["video_url"]

                        formatted_cleaned_comments_list.append(formatted_comment)
                    else:
                        print(f"  ⚠️ Bỏ qua bình luận (ID gốc: {comment_entry.get('id')}) vì quá ngắn sau khi làm sạch.")
                
                with open(cleaned_comments_filepath, "w", encoding="utf-8") as f:
                    json.dump(formatted_cleaned_comments_list, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Hoàn tất làm sạch và định dạng! Đã lưu {len(formatted_cleaned_comments_list)} bình luận vào {cleaned_comments_filepath} theo định dạng sẵn sàng gán nhãn.")
                print(f"💡 Bước tiếp theo: Vui lòng gán nhãn thủ công cho dữ liệu trong '{cleaned_comments_filepath}' và lưu nó vào thư mục 'data/labeled/' (ví dụ: 'data/labeled/your_manually_labeled_data.json').")

            except json.JSONDecodeError as e:
                print(f"❌ Lỗi đọc file JSON thô {raw_comments_filepath}: {e}")
            except Exception as e:
                print(f"❌ Lỗi trong quá trình làm sạch và định dạng dữ liệu: {e}")
        else:
            print(f"❌ Không tìm thấy file dữ liệu thô: {raw_comments_filepath}. Vui lòng đảm bảo bước thu thập dữ liệu thành công trước khi chạy bước làm sạch.")
        
        if args.comment_step == "clean":
            print("\n--- Chỉ chạy bước làm sạch bình luận. Hoàn tất pipeline bình luận. ---")
            return 

    # Bước 3: Tạo Embeddings và IOB2 Labels (Yêu cầu dữ liệu đã gán nhãn)
    if args.comment_step in ["full", "embed"]:
        print("\n--- Bắt đầu quá trình tạo Embeddings và IOB2 Labels cho bình luận ---")
        
        # Khởi tạo các embedder nếu chưa có
        if fasttext_embedder_instance is None: 
            try:
                initialize_embedders()
            except RuntimeError:
                print("Bỏ qua bước tạo embeddings do không thể khởi tạo Embedder.")
                return 

        # Đã giữ nguyên đường dẫn bạn cung cấp (giả định file này đã được gán nhãn thủ công)
        labeled_data_filepath = r"data\comment\labeled\video_comments_KOL1_1_spans.json" 
        
        if not os.path.exists(preprocessed_comments_dir):
            os.makedirs(preprocessed_comments_dir)
            print(f"✅ Đã tạo thư mục: {preprocessed_comments_dir}")

        labeled_input_data = []
        if os.path.exists(labeled_data_filepath):
            try:
                with open(labeled_data_filepath, 'r', encoding="utf-8") as f:
                    labeled_input_data = json.load(f)
                print(f"✅ Đã tải dữ liệu đã gán nhãn từ: {labeled_data_filepath} ({len(labeled_input_data)} mẫu)")
                
                processed_samples = []
                for i, sample in enumerate(labeled_input_data):
                    print(f"\n--- Xử lý mẫu {i+1}/{len(labeled_input_data)}: {sample['text'][:50]}... ---")
                    
                    # Truyền video_id và video_url từ sample đã gán nhãn nếu có
                    temp_video_id = sample.get("video_id")
                    temp_video_url = sample.get("video_url")

                    fused_embs, iob2_lbls, tokens = get_fused_embeddings_and_iob2_labels(
                        sample, fasttext_embedder_instance, char_bilstm_embedder_instance, xlmr_embedder_instance
                    )

                    if fused_embs is None:
                        print(f"Bỏ qua mẫu {i+1} do lỗi trong quá trình tạo embeddings/nhãn hoặc không có token.")
                        continue

                    processed_comment = {
                        "video_id": "7519112491503226119",
                        "original_text": sample['text'],
                        "word_tokens": tokens,
                        "iob2_labels": iob2_lbls,
                        "embeddings": fused_embs.tolist()
                    }
                    if temp_video_id: processed_comment["video_id"] = temp_video_id
                    if temp_video_url: processed_comment["video_url"] = temp_video_url
                    
                    processed_samples.append(processed_comment)
                
                with open(preprocessed_comments_filepath, 'w', encoding="utf-8") as f:
                    for entry in processed_samples:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

                print(f"\n✅ Kết quả Embeddings và IOB2 Labels đã được lưu vào: {preprocessed_comments_filepath}")
                print("\n--- Hoàn tất quá trình tạo Fused Embeddings và IOB2 Labels cho bình luận ---")

            except json.JSONDecodeError as e:
                print(f"❌ Lỗi đọc file JSON đã gán nhãn {labeled_data_filepath}: {e}")
            except Exception as e:
                print(f"❌ Lỗi trong quá trình tạo Embeddings và IOB2 Labels: {e}")
        else:
            print(f"❌ Không tìm thấy file dữ liệu đã gán nhãn: {labeled_data_filepath}.")
            print("Vui lòng gán nhãn thủ công cho dữ liệu đã làm sạch và lưu vào đường dẫn này để tiếp tục bước tạo embeddings.")
    
    print("\n===== Hoàn tất Pipeline Xử lý Bình luận =====")


async def run_videos_pipeline(video_url_for_videos): 
    """Chạy toàn bộ pipeline xử lý video."""
    # Đã giữ nguyên đường dẫn bạn cung cấp
    video_raw_dir = "data/script/raw" 
    video_summary_dir = "data/script/summarized" 

    print("\n===== Bắt đầu Pipeline Xử lý Video =====")
    await process_video_pipeline(video_url_for_videos, video_raw_dir, video_summary_dir)
    print("\n===== Hoàn tất Pipeline Xử lý Video =====")


async def main():
    parser = argparse.ArgumentParser(description="Pipeline xử lý bình luận và video TikTok.")
    parser.add_argument(
        "--process", 
        type=str, 
        default="full", 
        choices=["full", "comments", "videos", "fuse"], # Thêm lựa chọn 'fuse'
        help="Chọn quy trình xử lý: 'full' (cả bình luận và video), 'comments' (chỉ bình luận), 'videos' (chỉ video), 'fuse' (chỉ gộp dữ liệu)."
    )
    parser.add_argument(
        "--comment_step", 
        type=str, 
        default="full", 
        choices=["full", "collect", "clean", "embed"],
        help="Chọn bước xử lý bình luận: 'full' (mặc định), 'collect' (chỉ thu thập), 'clean' (chỉ làm sạch), 'embed' (chỉ tạo embeddings)."
    )
    args = parser.parse_args()

    # --- Cấu hình chung ---
    ms_token = "mgsvpemIp8ZglZO-QXrDRhSwIVqO3kI7wf5k9EHE80rP-KpO8-FwK2u9d4yommFAM5FJbxZLdNIGzftBRBq8mzbCUuHSfnph24FbgxhZwtkOCs3XtUTLEuq5u6qF5O9TSmWEyxVXnrEquZbRFRCmALDz"
    video_url_for_comments = "https://www.tiktok.com/@halinhofficial/video/7519112491503226119" 
    video_url_for_videos = "https://www.tiktok.com/@chouchinchan/video/7519112491503226119"

    # --- Định nghĩa các đường dẫn cho bước Gộp dữ liệu ---
    # Đã sửa lỗi: Giờ là đường dẫn đầy đủ đến file JSONL
    comments_embedded_filepath = os.path.join("data", "comment", "preprocessed", "comments_with_embeddings_and_labels.jsonl")
    
    # Đường dẫn đến thư mục chứa các file JSON của script video đã được xử lý và sẵn sàng gán nhãn
    # (output từ video_processor.py)
    # Đã giữ nguyên đường dẫn bạn cung cấp
    video_scripts_for_fusion_dir = os.path.join("data", "script", "preprocessed") # <-- Dữ liệu script nhúng sẽ ở đây

    # Đường dẫn file đầu ra sau khi gộp
    # Đã sửa lỗi: Giờ là đường dẫn đầy đủ đến file JSON
    output_fused_filepath = os.path.join("data", "combined", "fused_video_comment_features.json")
    
    # Đảm bảo thư mục 'combined' tồn tại
    os.makedirs(os.path.dirname(output_fused_filepath), exist_ok=True)


    # --- Chạy các Pipeline ---
    if args.process == "full":
        print("\n--- Bắt đầu chạy song song Pipeline Xử lý Bình luận và Video ---")
        try:
            # Khởi tạo embedder cho pipeline bình luận (chỉ khi cần)
            # Pipeline video tự quản lý việc tải model của nó
            await asyncio.gather(
                run_comments_pipeline(args, ms_token, video_url_for_comments),
                run_videos_pipeline(video_url_for_videos) 
            )
        except RuntimeError: 
            print("Một trong các pipeline đã gặp lỗi khởi tạo. Dừng toàn bộ quá trình.")
            return
        
        # Sau khi cả hai pipeline hoàn thành, chạy bước gộp dữ liệu
        print("\n--- Bắt đầu bước gộp dữ liệu sau khi các pipeline chính hoàn thành ---")
        fuse_data_pipelines(comments_embedded_filepath, video_scripts_for_fusion_dir, output_fused_filepath)


    elif args.process == "comments":
        print("\n--- Bắt đầu chạy Pipeline Xử lý Bình luận ---")
        try:
            await run_comments_pipeline(args, ms_token, video_url_for_comments)
        except RuntimeError:
            print("Pipeline bình luận đã gặp lỗi khởi tạo. Dừng quá trình.")
            return

    elif args.process == "videos":
        print("\n--- Bắt đầu chạy Pipeline Xử lý Video ---")
        await run_videos_pipeline(video_url_for_videos) 

    elif args.process == "fuse": # Lựa chọn mới để chỉ chạy bước gộp
        print("\n--- Bắt đầu chạy bước Gộp dữ liệu (Data Fusion) ---")
        fuse_data_pipelines(comments_embedded_filepath, video_scripts_for_fusion_dir, output_fused_filepath)


    print("\n--- Toàn bộ các Pipeline xử lý dữ liệu đã hoàn tất ---")

if __name__ == "__main__":
    asyncio.run(main())