from pyspark.sql.types import StructType, StructField, StringType, ArrayType, LongType, FloatType

# Schema cho dữ liệu comment thô nhận từ Kafka
# Giả định mỗi message Kafka là một JSON string có trường "comment"
comment_raw_schema = StructType([
    StructField("comment_text", StringType(), True),
    # Có thể thêm các trường khác nếu dữ liệu Kafka của bạn có (ví dụ: "video_id", "timestamp")
    StructField("video_id", StringType(), True),
    StructField("comment_id", StringType(), True),
    StructField("created_time", LongType(), True)
])

# Schema cho đầu ra dự đoán từ mô hình ABSA
# Đây là kết quả sau khi UDF xử lý comment
prediction_output_schema = StructType([
    StructField("comment_text", StringType(), True),
    StructField("video_id", StringType(), True),
    StructField("comment_id", StringType(), True),
    StructField("predicted_iob_labels", ArrayType(StringType()), True), # IOB2 tags
    StructField("extracted_spans", ArrayType(StructType([
        StructField("span", StringType(), True),
        StructField("tag_type", StringType(), True),
        StructField("start_token_idx", LongType(), True),
        StructField("end_token_idx", LongType(), True),
        # Có thể thêm sentiment nếu mô hình của bạn dự đoán sentiment riêng
    ])), True),
    StructField("processing_timestamp", StringType(), True)
])

# Schema cho dữ liệu đầu ra tổng hợp nếu bạn muốn đẩy kết quả đã xử lý lên Kafka topic khác
# Hoặc ghi ra storage
# Ví dụ: một comment đã được xử lý
processed_comment_schema = StructType([
    StructField("comment_id", StringType(), True),
    StructField("video_id", StringType(), True),
    StructField("original_text", StringType(), True),
    StructField("predicted_tags", ArrayType(StringType()), True),
    StructField("extracted_aspect_sentiments", ArrayType(StructType([
        StructField("aspect", StringType(), True),
        StructField("sentiment", StringType(), True),
        StructField("span_text", StringType(), True),
    ])), True),
    StructField("processing_time", StringType(), True)
])