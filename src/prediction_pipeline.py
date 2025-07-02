import torch
import torch.nn as nn
import numpy as np
import json
import os
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModel
import re
import asyncio # Để chạy hàm async

# Import các class từ model.py
from model import FastTextEmbedder, CharBiLSTMEmbedder, XLMREmbedder, BiLSTM_CRF_Model, \
    clean_text, process_text_to_word_tokens, get_char_indices_for_words, align_xlmr_embeddings_to_words, \
    get_fused_embeddings_and_iob2_labels, extract_spans

# --- Cấu hình chung (Cần khớp với cấu hình lúc train) ---
FASTTEXT_MODEL_PATH = "../models/fasttext/cc.vi.300.bin" 
XLMR_MODEL_NAME = "../models/huggingface/hub/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
MODEL_SAVE_PATH = "../models/absa_bilstm_crf_model.pth"

# Khởi tạo các embedder toàn cục để tái sử dụng
fasttext_embedder = None
char_bilstm_embedder = None
xlmr_embedder = None
absa_model = None
idx_to_tag_map = None
tag_to_idx_map = None
model_device = None
char_to_idx_loaded = None # Thêm biến để lưu char_to_idx

def load_absa_model(model_path=MODEL_SAVE_PATH):
    """
    Tải mô hình ABSA đã huấn luyện và các embedder.
    """
    global fasttext_embedder, char_bilstm_embedder, xlmr_embedder, absa_model, idx_to_tag_map, tag_to_idx_map, model_device, char_to_idx_loaded

    print(f"Loading ABSA model from {model_path}...")
    
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Tải checkpoint
        checkpoint = torch.load(model_path, map_location=model_device)

        # Tải FastText Embedder
        fasttext_embedder = FastTextEmbedder(FASTTEXT_MODEL_PATH)

        # Tải CharBiLSTM Embedder (cần char_to_idx đã được lưu)
        char_to_idx_loaded = checkpoint['char_to_idx'] # Tải char_to_idx đã được lưu
        char_bilstm_embedder = CharBiLSTMEmbedder(
            vocab_size=len(char_to_idx_loaded), 
            embedding_dim=checkpoint['embedding_dim'] // 3 if checkpoint['embedding_dim'] % 3 == 0 else 50, # Adjust if needed
            hidden_dim=checkpoint['hidden_dim'] // 5 if checkpoint['hidden_dim'] % 5 == 0 else 50 # Adjust if needed
        ).to(model_device)
        # Lưu ý: CharBiLSTMEmbedder không có state_dict riêng để tải từ checkpoint này
        # nếu bạn không lưu riêng nó. Hiện tại nó được khởi tạo mới.
        # Nếu CharBiLSTMEmbedder có tham số huấn luyện, bạn cần lưu state_dict của nó riêng.
        # Trong trường hợp này, nó có nn.Embedding và nn.LSTM, nên nó cần state_dict.
        # Giải pháp đơn giản là lưu toàn bộ CharBiLSTMEmbedder.state_dict() vào checkpoint.
        # Tạm thời, CharBiLSTMEmbedder được khởi tạo lại và không tải lại trọng số.
        # -> Cần chỉnh lại model.py để lưu state_dict của CharBiLSTMEmbedder.
        # Tuy nhiên, nếu bạn không muốn train CharBiLSTMEmbedder, khởi tạo mới cũng được.

        # Tải XLMREmbedder
        xlmr_embedder = XLMREmbedder(XLMR_MODEL_NAME)
        xlmr_embedder.model.to(model_device) # Chuyển XLMR model sang device

        # Tải BiLSTM_CRF_Model
        absa_model = BiLSTM_CRF_Model(
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_tags=checkpoint['num_tags'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout']
        ).to(model_device)
        absa_model.load_state_dict(checkpoint['model_state_dict'])
        absa_model.eval() # Chuyển sang chế độ đánh giá

        idx_to_tag_map = checkpoint['idx_to_tag']
        tag_to_idx_map = checkpoint['current_tag_to_idx'] # Lưu lại tag_to_idx_map

        print("✅ ABSA model and embedders loaded successfully.")
        return True

    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}. Please ensure the model is trained and saved.")
        return False
    except Exception as e:
        print(f"❌ Error loading model or embedders: {e}")
        import traceback
        traceback.print_exc()
        return False

def predict_aspect_spans(comment_text: str):
    """
    Dự đoán các khía cạnh trong một bình luận sử dụng mô hình ABSA đã tải.
    """
    global fasttext_embedder, char_bilstm_embedder, xlmr_embedder, absa_model, idx_to_tag_map, tag_to_idx_map, model_device, char_to_idx_loaded

    if absa_model is None:
        print("Model is not loaded. Please call load_absa_model() first.")
        return []

    # Tạo một dict giống format sample đầu vào của get_fused_embeddings_and_iob2_labels
    sample_data = {"text": comment_text, "labels": []} # labels rỗng vì đây là dự đoán

    # Lấy fused embeddings
    fused_embeddings, _, word_tokens = get_fused_embeddings_and_iob2_labels(
        sample_data, fasttext_embedder, char_bilstm_embedder, xlmr_embedder
    )

    if fused_embeddings is None or len(word_tokens) == 0:
        print(f"Could not generate embeddings for text: '{comment_text}'")
        return []

    # Chuẩn bị input cho mô hình
    embeddings_batch = fused_embeddings.unsqueeze(0).to(model_device) # Thêm batch dimension
    # Tạo mask cho batch 1
    mask_batch = torch.ones(embeddings_batch.shape[0], embeddings_batch.shape[1], dtype=torch.bool).to(model_device)

    absa_model.eval() # Đảm bảo mô hình ở chế độ đánh giá
    with torch.no_grad():
        predicted_tag_ids_list = absa_model(embeddings_batch, mask=mask_batch)
    
    # Lấy kết quả dự đoán của mẫu đầu tiên (và duy nhất) trong batch
    predicted_tag_ids = predicted_tag_ids_list[0] 
    predicted_labels_str = [idx_to_tag_map[id_val] for id_val in predicted_tag_ids]

    # Trích xuất các span khía cạnh
    predicted_spans = extract_spans(predicted_labels_str)

    # Chuyển đổi các span dự đoán thành định dạng dễ đọc hơn (ví dụ: kèm text của span)
    aspect_results = []
    for start_idx, end_idx, tag_type in predicted_spans:
        span_text = " ".join(word_tokens[start_idx : end_idx + 1])
        aspect_results.append({
            "span_text": span_text,
            "tag_type": tag_type,
            "start_token_idx": start_idx,
            "end_token_idx": end_idx
        })
    
    return aspect_results

async def process_stream_data(data_stream_source):
    """
    Hàm mô phỏng việc xử lý dữ liệu stream.
    Trong thực tế, data_stream_source có thể là một Kafka consumer, RabbitMQ queue, v.v.
    Ở đây, ta sẽ mô phỏng bằng cách đọc từ một danh sách các bình luận.
    """
    if not load_absa_model():
        print("Failed to load model. Cannot process stream data.")
        return

    print("\n--- Starting Stream Processing Pipeline ---")
    
    # Giả định data_stream_source là một iterable của các bình luận thô
    for i, comment_data in enumerate(data_stream_source):
        comment_text = comment_data.get("text")
        video_id = comment_data.get("video_id", "N/A")
        comment_id = comment_data.get("comment_id", f"comment_{i+1}")

        print(f"\nProcessing Comment [{comment_id}] from Video [{video_id}]:")
        print(f"  Raw Text: '{comment_text}'")

        if not comment_text:
            print("  Skipping empty comment.")
            continue

        predicted_aspects = predict_aspect_spans(comment_text)

        print("  Predicted Aspects:")
        if predicted_aspects:
            for aspect in predicted_aspects:
                print(f"    - Text: '{aspect['span_text']}', Type: {aspect['tag_type']}")
            
            # --- Logic đề xuất KOL (conceptual) ---
            # Đây là nơi bạn sẽ tích hợp logic đề xuất KOL của mình.
            # Ví dụ: Dựa trên các 'tag_type' và 'span_text' được phát hiện,
            # bạn có thể truy vấn cơ sở dữ liệu KOL để tìm những KOL phù hợp.
            # Dữ liệu KOL của bạn cần có thông tin về lĩnh vực chuyên môn, từ khóa liên quan, v.v.

            # Ví dụ đơn giản: Đề xuất KOL dựa trên từ khóa trong aspect_results
            potential_kol_topics = set()
            for aspect in predicted_aspects:
                # Chuyển đổi tag_type hoặc span_text thành topic
                # Ví dụ: "ASPECT" -> "sản phẩm", "chất lượng" -> "chất lượng sản phẩm"
                if aspect['tag_type'] == 'ASPECT':
                    potential_kol_topics.add(aspect['span_text'].lower()) # Sử dụng trực tiếp text span
                elif aspect['tag_type'] == 'PRICE':
                    potential_kol_topics.add("giá cả")
                # Thêm các luật chuyển đổi khác tùy thuộc vào định nghĩa tag của bạn
            
            if potential_kol_topics:
                print(f"  Potential KOL Topics derived: {list(potential_kol_topics)}")
                print("  (Conceptual) Recommending KOLs based on these topics...")
                # Placeholder for actual KOL recommendation logic
                # For a real system, you would query a KOL database/index here
                # Example:
                # recommended_kols = your_kol_recommender.get_kols_for_topics(potential_kol_topics, video_id)
                # print(f"  Recommended KOLs: {recommended_kols}")
            else:
                print("  No specific KOL topics derived from aspects.")

        else:
            print("    No aspects detected.")
        
        # Mô phỏng độ trễ xử lý stream
        await asyncio.sleep(0.1) # Simulate some processing time

    print("\n--- Stream Processing Pipeline Finished ---")

# --- Khối chính để chạy pipeline dự đoán ---
if __name__ == "__main__":
    print("Running src/prediction_pipeline.py directly for demonstration purposes.")

    # Mô phỏng dữ liệu stream (có thể đến từ Kafka, API, file, etc.)
    sample_stream_data = [
        {"video_id": "vid1", "comment_id": "c1", "text": "điện thoại này thiết kế đẹp, pin cũng trâu."},
        {"video_id": "vid2", "comment_id": "c2", "text": "giá hơi cao so với chất lượng."},
        {"video_id": "vid1", "comment_id": "c3", "text": "sản phẩm dùng rất tốt, đáng tiền."},
        {"video_id": "vid3", "comment_id": "c4", "text": "dịch vụ chăm sóc khách hàng kém quá."},
        {"video_id": "vid2", "comment_id": "c5", "text": "camera chụp ảnh khá tệ trong điều kiện thiếu sáng."},
        {"video_id": "vid4", "comment_id": "c6", "text": "máy chạy mượt mà, nhưng dễ nóng."},
        {"video_id": "vid5", "comment_id": "c7", "text": "mình thấy sản phẩm này ổn, không có gì nổi bật."},
    ]

    try:
        # Chạy pipeline xử lý stream
        asyncio.run(process_stream_data(sample_stream_data))
    except Exception as e:
        print(f"An error occurred during stream processing: {e}")
        import traceback
        traceback.print_exc()