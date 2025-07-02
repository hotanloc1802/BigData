import pandas as pd
import torch
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType

from model_loader import (
    load_all_absa_prediction_models,
    _model_cache,
    get_fused_embeddings_for_udf
)

# Schema kết quả
output_schema = StructType([
    StructField("video_id", StringType()),
    StructField("comment_id", StringType()),
    StructField("comment_text", StringType()),
    StructField("extracted_spans", ArrayType(
        StructType([
            StructField("span_text", StringType()),
            StructField("tag_type", StringType()),
            StructField("start_token_idx", LongType()),
            StructField("end_token_idx", LongType())
        ])
    ))
])

# Hàm hỗ trợ trích xuất span từ IOB2
def _extract_spans_from_iob2(word_tokens, iob2_labels):
    spans = []
    current_span = None
    for i, tag in enumerate(iob2_labels):
        if tag.startswith('B-'):
            if current_span:
                spans.append(current_span)
            current_span = {
                "span_text": word_tokens[i],
                "tag_type": tag[2:],
                "start_token_idx": i,
                "end_token_idx": i
            }
        elif tag.startswith('I-'):
            if current_span and current_span["tag_type"] == tag[2:]:
                current_span["span_text"] += " " + word_tokens[i]
                current_span["end_token_idx"] = i
            else:
                if current_span:
                    spans.append(current_span)
                current_span = None
        else:
            if current_span:
                spans.append(current_span)
            current_span = None
    if current_span:
        spans.append(current_span)
    return spans

# UDF sử dụng GROUPED_MAP để xử lý batch comment
@pandas_udf(output_schema, functionType=PandasUDFType.GROUPED_MAP)
def predict_absa_spans_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    # Load mô hình và các thành phần cần thiết
    load_all_absa_prediction_models()
    absa_model = _model_cache.get('absa_model')
    tag_to_idx_map = _model_cache.get('tag_to_idx_map')
    idx_to_tag_map = _model_cache.get('idx_to_tag_map')
    fasttext_embedder = _model_cache.get('fasttext_embedder')
    char_bilstm_embedder = _model_cache.get('char_bilstm_embedder')
    xlmr_embedder = _model_cache.get('xlmr_embedder')
    device = _model_cache.get('device')

    results = []

    for _, row in pdf.iterrows():
        comment_text = row.get("comment_text", "")
        comment_id = row.get("comment_id", "")
        video_id = row.get("video_id", "")

        if not isinstance(comment_text, str) or not comment_text.strip():
            results.append((video_id, comment_id, comment_text, []))
            continue

        try:
            # Lấy embedding
            fused_embeddings, tokens = get_fused_embeddings_for_udf(
                comment_text,
                fasttext_embedder,
                char_bilstm_embedder,
                xlmr_embedder
            )

            if fused_embeddings is None:
                results.append((video_id, comment_id, comment_text, []))
                continue

            # Dự đoán nhãn
            fused_embeddings = fused_embeddings.unsqueeze(0).to(device)
            mask = torch.ones(fused_embeddings.shape[:2], dtype=torch.bool).to(device)
            predicted_tag_ids = absa_model(fused_embeddings, mask=mask)[0]
            predicted_iob_labels = [idx_to_tag_map[tag_id] for tag_id in predicted_tag_ids]

            spans = _extract_spans_from_iob2(tokens, predicted_iob_labels)
            results.append((video_id, comment_id, comment_text, spans))

        except Exception as e:
            print(f"[ERROR] comment_id={comment_id}: {e}")
            results.append((video_id, comment_id, comment_text, []))

    return pd.DataFrame(results, columns=["video_id", "comment_id", "comment_text", "extracted_spans"])
