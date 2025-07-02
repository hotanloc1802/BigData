import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, LongType
import torch
import numpy as np # Để xử lý numpy array từ comment['embeddings']
from datetime import datetime

# Import các hàm và lớp cần thiết từ model_loader
from model_loader import load_all_absa_prediction_models, _model_cache, get_fused_embeddings_for_udf


# Định nghĩa schema cho đầu ra của UDF
# Mỗi hàng sẽ trả về một comment_id và danh sách các span dự đoán
# Đây là một ví dụ, bạn có thể điều chỉnh để phù hợp với định dạng đầu ra mong muốn
prediction_udf_schema = ArrayType(StructType([
    StructField("span_text", StringType(), True),
    StructField("tag_type", StringType(), True),
    StructField("start_token_idx", LongType(), True),
    StructField("end_token_idx", LongType(), True)
]))

# Helper để trích xuất span từ IOB2 tags
def _extract_spans_from_iob2(word_tokens, iob2_labels):
    spans = []
    current_span = None
    for i, tag in enumerate(iob2_labels):
        if tag.startswith('B-'):
            if current_span:
                spans.append(current_span)
            current_span = {"span_text": word_tokens[i], "tag_type": tag[2:], "start_token_idx": i, "end_token_idx": i}
        elif tag.startswith('I-'):
            if current_span and current_span["tag_type"] == tag[2:]:
                current_span["span_text"] += " " + word_tokens[i]
                current_span["end_token_idx"] = i
            else: # Invalid I-tag or type mismatch
                if current_span: spans.append(current_span)
                current_span = None # Reset
        else: # 'O' tag
            if current_span:
                spans.append(current_span)
            current_span = None
    if current_span: # Add last span if it exists
        spans.append(current_span)
    return spans


@pandas_udf(prediction_udf_schema, PandasUDFType.SCALAR_ITER) # Sử dụng SCALAR_ITER cho pandas UDFs phức tạp
def predict_absa_spans_udf(comments_batch_df_iter: pd.DataFrame) -> pd.Series: # Nhận DataFrame Iterator
    # load_all_absa_prediction_models() sẽ được gọi một lần trên mỗi worker
    # và kết quả sẽ được cache trong _model_cache
    load_all_absa_prediction_models() 
    
    absa_model = _model_cache.get('absa_model')
    tag_to_idx_map = _model_cache.get('tag_to_idx_map')
    idx_to_tag_map = _model_cache.get('idx_to_tag_map')
    fasttext_embedder = _model_cache.get('fasttext_embedder')
    char_bilstm_embedder = _model_cache.get('char_bilstm_embedder')
    xlmr_embedder = _model_cache.get('xlmr_embedder')
    device = _model_cache.get('device')

    if absa_model == "LOAD_ERROR" or absa_model is None:
        # Nếu model không tải được, trả về Series lỗi
        # Cần biết số lượng rows trong batch để trả về Series cùng kích thước
        # pandas_udf SCALAR_ITER nhận Series Iterator, chứ không phải DataFrame Iterator trực tiếp
        # nên chúng ta cần điều chỉnh cách này nếu đầu vào là một Series Iterator
        # Tuy nhiên, trong PySpark 3.x, SCALAR_ITER nhận iterator of pandas.Series, không phải DataFrame
        # For a single column UDF, it is Series to Series
        # Re-thinking: if it's SCALAR_ITER, it receives an iterator of pandas Series, each Series is a batch
        # So the input is `comments_batch_series_iter: Iterator[pd.Series]`
        for comments_batch_series in comments_batch_df_iter: # This is how SCALAR_ITER receives Series batches
             yield pd.Series([[]] * len(comments_batch_series)) # Empty list of spans if model unavailable
        return # Exit the UDF
    
    # Process each Series (batch) from the iterator
    for comment_series_batch in comments_batch_df_iter:
        results_list = []
        
        if comment_series_batch.empty:
            yield pd.Series([], dtype=prediction_udf_schema) # Return empty Series for empty input
            continue

        for _, row in comment_series_batch.iterrows(): # Iterate over rows of the DataFrame batch
            comment_text = row['comment_text'] # Sửa tên cột nếu cần (từ Kafka)
            comment_id = row['comment_id'] # Giả sử có comment_id
            video_id = row['video_id'] # Giả sử có video_id

            if not pd.notnull(comment_text) or not isinstance(comment_text, str) or not comment_text.strip():
                results_list.append([]) # Return empty list for invalid text
                continue
            
            try:
                # Tiền xử lý và tạo embeddings (tương tự như get_fused_embeddings_and_iob2_labels nhưng không cần labels)
                # Đảm bảo các hàm này có thể được import hoặc định nghĩa lại trong prediction_udf.py/model_loader.py
                fused_embeddings, tokens = get_fused_embeddings_for_udf(
                    comment_text, fasttext_embedder, char_bilstm_embedder, xlmr_embedder
                )

                if fused_embeddings is None:
                    results_list.append([]) # No spans if no embeddings
                    continue
                
                # Model inference
                fused_embeddings = fused_embeddings.unsqueeze(0).to(device) # Add batch dimension and move to device
                mask = (torch.ones(fused_embeddings.shape[0], fused_embeddings.shape[1], dtype=torch.bool)).to(device) # Mask all true for now (no padding in batch=1)
                
                predicted_tag_ids_list = absa_model(fused_embeddings, mask=mask)
                predicted_tag_ids = predicted_tag_ids_list[0]
                predicted_iob_labels = [idx_to_tag_map[tag_id] for tag_id in predicted_tag_ids]

                # Trích xuất spans
                extracted_spans = _extract_spans_from_iob2(tokens, predicted_iob_labels)
                results_list.append(extracted_spans)

            except Exception as e:
                print(f"Worker process: Error processing comment '{comment_id}': {e}")
                results_list.append([]) # Trả về rỗng nếu có lỗi xử lý

        yield pd.Series(results_list, dtype=prediction_udf_schema) # Trả về Series kết quả cho batch