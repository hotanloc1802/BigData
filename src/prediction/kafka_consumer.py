import os
import sys
import time
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, expr, lit, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from prediction_udf import predict_absa_spans_udf 
from schema import comment_raw_schema, prediction_output_schema # Import Schemas

# --- A. Thiết lập môi trường Python cho Spark ---
# KIỂM TRA LẠI ĐƯỜNG DẪN NÀY!
# Đảm bảo đường dẫn này trỏ đến Python interpreter trong môi trường Conda của bạn.
# Ví dụ: CONDA_ENV_PYTHON_PATH = r"D:\DevTools\Anacoda\envs\spark311_env\python.exe"
# HOẶC chỉ cần "python" nếu môi trường đã được kích hoạt đúng và Python nằm trong PATH
CONDA_ENV_PYTHON_PATH = "python" # Mặc định là 'python' nếu đã kích hoạt môi trường Conda/venv đúng

os.environ['PYSPARK_PYTHON'] = CONDA_ENV_PYTHON_PATH
os.environ['PYSPARK_DRIVER_PYTHON'] = CONDA_ENV_PYTHON_PATH

print(f"Đã đặt PYSPARK_PYTHON và PYSPARK_DRIVER_PYTHON thành: {CONDA_ENV_PYTHON_PATH}")
print("--- Thông tin Python của Driver Script ---")
print(f"Đường dẫn Python thực thi (Driver): {sys.executable}")
print(f"Phiên bản Python (Driver): {sys.version}")
print("--- Kết thúc thông tin ---")

# --- B0. Khởi tạo SparkSession và Cấu hình ---
KAFKA_CONNECTOR_VERSION = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"
COMMONS_POOL2_VERSION = "org.apache.commons:commons-pool2:2.12.0"

spark = SparkSession.builder \
    .appName("ABSA_Kafka_Prediction_Stream") \
    .config("spark.jars.packages", f"{KAFKA_CONNECTOR_VERSION},{COMMONS_POOL2_VERSION}") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.python.worker.memory", "2g") \
    .config("spark.sql.adaptive.enabled", "false") \
    .config("spark.driver.maxResultSize", "0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("SparkSession đã được khởi tạo và cấu hình.")

# --- B1. Đọc stream từ Kafka ---
KAFKA_TOPIC_INPUT = "youtube_commenttt" # Topic mà Producer gửi dữ liệu tới
KAFKA_TOPIC_OUTPUT_PREDICTIONS = "absa_predictions" # Topic để gửi kết quả dự đoán
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

print(f"Đang thiết lập đọc stream từ Kafka topic: {KAFKA_TOPIC_INPUT} trên server: {KAFKA_BOOTSTRAP_SERVERS}")
kafkaDF = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", KAFKA_TOPIC_INPUT) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()
print("Thiết lập đọc stream từ Kafka thành công.")

# --- B2. Parse JSON và chuẩn bị dữ liệu ---
# Sử dụng comment_raw_schema từ schemas.py
parsedDF = kafkaDF.selectExpr("CAST(value AS STRING) as json_str") \
    .withColumn("data", from_json(col("json_str"), comment_raw_schema)) \
    .select(col("data.comment_text").alias("comment_text"),
            col("data.video_id").alias("video_id"),
            col("data.comment_id").alias("comment_id")) \
    .filter(col("comment_text").isNotNull() & (col("comment_text") != "") & (col("comment_text").cast(StringType()) != "NaN")) 
print("Đã thiết lập parsing JSON và trích xuất cột 'comment_text', 'video_id', 'comment_id'.")


# --- B3. Áp dụng Pandas UDF để dự đoán ABSA ---
# predict_absa_spans_udf đã được định nghĩa trong prediction_udf.py
# Áp dụng UDF để thêm cột 'predicted_spans'
# Truyền các cột cần thiết làm đối số cho UDF
df_with_predictions = parsedDF.withColumn(
    "extracted_spans", predict_absa_spans_udf(col("comment_text"), col("video_id"), col("comment_id")) 
)

# Thêm timestamp cho kết quả
df_final_output = df_with_predictions.withColumn("processing_timestamp", current_timestamp())

print("Đã áp dụng UDF và tạo DataFrame với cột 'extracted_spans' và 'processing_timestamp'.")

# --- B4. Khởi chạy Streaming Query để ghi ra Console (hoặc Kafka topic khác) ---
output_mode = "append" # "append" là phù hợp nhất cho console sink khi thêm cột mới

# Checkpoint location là cần thiết cho Spark Streaming để duy trì trạng thái
checkpoint_dir_consumer = os.path.join("spark_checkpoints", "absa_prediction_console")

print(f"Đang khởi chạy streaming query (ABSA Prediction - Console Output) với output mode: {output_mode} và checkpoint: {checkpoint_dir_consumer}")

# Định dạng output cho console (chỉ các cột cần thiết)
query = df_final_output.select(
        "video_id", 
        "comment_id", 
        "comment_text", 
        "extracted_spans", 
        "processing_timestamp"
    ) \
    .writeStream \
    .format("console") \
    .outputMode(output_mode) \
    .option("truncate", "false") \
    .option("numRows", 20) \
    .option("checkpointLocation", checkpoint_dir_consumer) \
    .trigger(processingTime="10 seconds") \
    .start()

print("Streaming query (ABSA Prediction - Console Output) đã khởi chạy! Chờ dữ liệu từ Kafka...")
print("Kết quả sẽ được in trực tiếp ra console mỗi 10 giây.")

try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("Đã nhận tín hiệu dừng (Ctrl+C). Đang dừng query...")
    if query:
        try:
            query.stop()
        except Exception as e_stop:
            print(f"Lỗi khi dừng query: {e_stop}")
    print("Query đã dừng.")
except Exception as e_stream:
    print(f"Lỗi trong quá trình streaming: {e_stream}")
    if query:
        try:
            query.stop()
        except Exception as e_stop:
            print(f"Lỗi khi dừng query do lỗi streaming: {e_stop}")
    print("Query đã dừng do lỗi.")
finally:
    if 'spark' in locals() and spark.getActiveSession():
        try:
            spark.stop()
        except Exception as e_spark_stop:
            print(f"Lỗi khi đóng SparkSession: {e_spark_stop}")
    print("SparkSession đã đóng (nếu có).")
