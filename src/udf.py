import os
import sys
import json
import re
import time
import torch
import torch.nn as nn
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModel
import fasttext
import pandas as pd # Import pandas here for use in UDF
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, expr, pandas_udf, decode
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, LongType, IntegerType

# --- A. Thiết lập môi trường Python cho Spark ---

# KIỂM TRA LẠI ĐƯỜNG DẪN NÀY!
# Đảm bảo đường dẫn này trỏ đến Python interpreter trong môi trường Conda của bạn.
# Ví dụ: r"C:\Users\YourUser\anaconda3\envs\your_spark_env\python.exe"
# Hoặc trên Linux: "/opt/conda/envs/your_spark_env/bin/python"
CONDA_ENV_PYTHON_PATH = r"D:\DevTools\Anacoda\envs\spark311_env\python.exe" 

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
    .appName("Kafka_ABSA_Prediction_Spark_Streaming") \
    .config("spark.jars.packages", f"{KAFKA_CONNECTOR_VERSION},{COMMONS_POOL2_VERSION}") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.python.worker.memory", "4g") \
    .config("spark.sql.adaptive.enabled", "false") \
    .config("spark.executor.instances", "2") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("SparkSession đã được khởi tạo và cấu hình.")

# --- Đảm bảo model.py và các tài nguyên khác được phân phối tới worker nodes ---
# Assuming model.py is in the same directory as this script.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..')) # Adjust if your project root is different

# Add model.py to the Spark workers' Python path
model_py_path = os.path.join(current_dir, 'model.py') # Assuming model.py is in the same dir
if not os.path.exists(model_py_path):
    print(f"ERROR: model.py not found at {model_py_path}. Please ensure it's copied there.")
    sys.exit(1)
spark.sparkContext.addPyFile(model_py_path)
print(f"Added model.py to SparkContext: {model_py_path}")

# Add FastText model file to Spark workers' cache
FASTTEXT_MODEL_FILENAME = "cc.vi.300.bin"
FASTTEXT_MODEL_PATH_LOCAL = os.path.join(project_root, "models", "fasttext", FASTTEXT_MODEL_FILENAME)

if not os.path.exists(FASTTEXT_MODEL_PATH_LOCAL):
    print(f"ERROR: FastText model not found at {FASTTEXT_MODEL_PATH_LOCAL}. Please ensure it's copied there.")
    sys.exit(1)
spark.sparkContext.addFile(FASTTEXT_MODEL_PATH_LOCAL)
print(f"Added FastText model to SparkContext: {FASTTEXT_MODEL_PATH_LOCAL} (accessible via SparkFiles.get('{FASTTEXT_MODEL_FILENAME}'))")

# Define XLMR_MODEL_NAME. This path is for the driver to load for initial checks or if transformers handles caching itself.
# IMPORTANT: For workers to access it, ensure it's either pre-downloaded on worker nodes
# in the Hugging Face cache directory, or that workers have internet access to download it.
# Distributing a full Hugging Face model directory via addFile is non-trivial (requires zipping/unzipping).
XLMR_MODEL_NAME = os.path.join(project_root, "models", "huggingface", "hub", "models--xlm-roberta-base", "snapshots", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089")
# If this path needs to be resolved by workers, ensure it's accessible.
# For this example, we assume Hugging Face's pipeline will manage downloads/caching on workers.
print(f"XLM-R Model Path for reference: {XLMR_MODEL_NAME}")

# The path to your saved ABSA BiLSTM-CRF model
MODEL_SAVE_PATH_LOCAL = os.path.join(project_root, "models", "absa_bilstm_crf_model.pth")
if not os.path.exists(MODEL_SAVE_PATH_LOCAL):
    print(f"ERROR: ABSA model not found at {MODEL_SAVE_PATH_LOCAL}. Please ensure it's copied there.")
    sys.exit(1)
spark.sparkContext.addFile(MODEL_SAVE_PATH_LOCAL)
print(f"Added ABSA BiLSTM-CRF model to SparkContext: {MODEL_SAVE_PATH_LOCAL} (accessible via SparkFiles.get('absa_bilstm_crf_model.pth'))")


# --- Global Model Cache for the UDF (on each Spark Worker) ---
_absa_model_cache = {}

# --- B1. Đọc stream từ Kafka ---
kafka_topic = "tiktok_comment2" # Your specified topic
kafka_bootstrap_servers = "localhost:9092"

print(f"Đang thiết lập đọc stream từ Kafka topic: {kafka_topic} trên server: {kafka_bootstrap_servers}")
kafkaDF = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", kafka_topic) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()

kafkaDF = kafkaDF.withColumn("json_str", decode(col("value"), "UTF-8"))
# --- B2. Parse JSON và lấy các trường cần thiết ---
# Schema cho incoming Kafka message (value field)
# Assuming Kafka message has "comment_text", "video_id", "comment_id"
comment_schema = StructType([
    StructField("comment_text", StringType(), True),
    StructField("video_id", StringType(), True),
    StructField("comment_id", StringType(), True)
])

parsedDF = kafkaDF.withColumn("data", from_json(col("json_str"), comment_schema)) \
    .filter(col("data").isNotNull()) \
    .select(
        col("data.comment_text").alias("comment_text"),
        col("data.video_id").alias("video_id"),
        col("data.comment_id").alias("comment_id")
    ) \
    .filter(
        col("comment_text").isNotNull() &
        (col("comment_text") != "") &
        (col("comment_text").cast(StringType()) != "NaN")
    )
 # Lọc kỹ hơn
print("Đã thiết lập parsing JSON và trích xuất cột 'comment_text', 'video_id', 'comment_id'.")

# --- B3. Định nghĩa Pandas UDF cho ABSA Prediction ---
# Define the output schema for the UDF
absa_output_schema = StructType([
    StructField("comment_id", StringType(), True),
    StructField("video_id", StringType(), True),
    StructField("original_comment_text", StringType(), True),
    StructField("predicted_aspects", ArrayType(StructType([
        StructField("span_text", StringType(), True),
        StructField("tag_type", StringType(), True),
        StructField("start_token_idx", IntegerType(), True),
        StructField("end_token_idx", IntegerType(), True)
    ])), True),
    StructField("potential_kol_topics", ArrayType(StringType()), True),
    StructField("error_message", StringType(), True) # For UDF errors
])


@pandas_udf(absa_output_schema)
def predict_absa_udf_lazy_load(
    comment_texts: pd.Series, video_ids: pd.Series, comment_ids: pd.Series
) -> pd.DataFrame:
    """
    Pandas UDF to perform ABSA prediction using models loaded and cached on each worker.
    """
    # Import necessary modules inside the UDF to ensure they are available on the worker
    from pyspark import SparkFiles # For accessing files added via sparkContext.addFile
    # Your model.py functions and classes
    # Assuming model.py is correctly added via sparkContext.addPyFile
    from model import FastTextEmbedder, CharBiLSTMEmbedder, XLMREmbedder, BiLSTM_CRF_Model, \
        clean_text, process_text_to_word_tokens, get_char_indices_for_words, align_xlmr_embeddings_to_words, \
        get_fused_embeddings_and_iob2_labels, extract_spans

    global _absa_model_cache

    model_key = "absa_model" # A single key for our combined model components

    if model_key not in _absa_model_cache:
        print(f"Worker process: Initializing ABSA models for the first time...")
        
        try:
            # Determine device (CPU for most Spark deployments unless GPU cluster)
            device = torch.device("cpu") # Keep it on CPU for general Spark deployments
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Uncomment for GPU-enabled workers

            # Get paths to distributed files
            fasttext_model_path_worker = SparkFiles.get(FASTTEXT_MODEL_FILENAME)
            absa_model_save_path_worker = SparkFiles.get("absa_bilstm_crf_model.pth")

            # Load checkpoint to get model parameters and mappings
            checkpoint = torch.load(absa_model_save_path_worker, map_location=device)
            
            # Initialize Embedders
            fasttext_embedder = FastTextEmbedder(fasttext_model_path_worker)
            
            char_to_idx_loaded = checkpoint.get('char_to_idx')
            if char_to_idx_loaded is None:
                raise ValueError("'char_to_idx' not found in checkpoint. Cannot initialize CharBiLSTMEmbedder.")
            char_bilstm_embedder = CharBiLSTMEmbedder(
                vocab_size=len(char_to_idx_loaded),
                embedding_dim=checkpoint.get('char_embedding_dim', 50),
                hidden_dim=checkpoint.get('char_lstm_hidden_dim', 50)
            ).to(device)
            if 'char_bilstm_state_dict' in checkpoint:
                char_bilstm_embedder.load_state_dict(checkpoint['char_bilstm_state_dict'])
            char_bilstm_embedder.eval()

            # XLMR Embedder - rely on Hugging Face's caching/download if not locally present on worker
            # Note: XLMR_MODEL_NAME here refers to the original path used by AutoTokenizer/AutoModel.
            # If workers don't have this cached, they will attempt to download.
            xlmr_embedder = XLMREmbedder(XLMR_MODEL_NAME)
            xlmr_embedder.model.to(device)
            xlmr_embedder.model.eval()

            # Initialize BiLSTM_CRF_Model
            absa_model = BiLSTM_CRF_Model(
                embedding_dim=checkpoint['embedding_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                num_tags=checkpoint['num_tags'],
                num_layers=checkpoint['num_layers'],
                dropout=checkpoint['dropout']
            ).to(device)
            absa_model.load_state_dict(checkpoint['model_state_dict'])
            absa_model.eval()

            # Store in cache
            _absa_model_cache[model_key] = {
                "fasttext": fasttext_embedder,
                "char_bilstm": char_bilstm_embedder,
                "xlmr": xlmr_embedder,
                "absa_model": absa_model,
                "idx_to_tag": checkpoint['idx_to_tag'],
                "tag_to_idx": checkpoint['current_tag_to_idx'], # Might not be needed for prediction but good to store
                "device": device,
                "char_to_idx": char_to_idx_loaded
            }
            print(f"Worker process: ABSA models loaded and cached successfully.")

        except Exception as e_load:
            print(f"Worker process: ERROR loading ABSA models: {e_load}")
            import traceback
            traceback.print_exc()
            _absa_model_cache[model_key] = {"error": str(e_load)} # Mark as error

    # Retrieve models from cache
    model_components = _absa_model_cache.get(model_key)

    if "error" in model_components:
        # If model loading failed, return error for all rows in the batch
        results = []
        for i in range(len(comment_texts)):
            results.append({
                "comment_id": comment_ids.iloc[i] if i < len(comment_ids) else None,
                "video_id": video_ids.iloc[i] if i < len(video_ids) else None,
                "original_comment_text": comment_texts.iloc[i] if i < len(comment_texts) else None,
                "predicted_aspects": [],
                "potential_kol_topics": [],
                "error_message": f"MODEL_LOAD_ERROR: {model_components['error']}"
            })
        return pd.DataFrame(results, columns=absa_output_schema.names)

    fasttext_embedder = model_components["fasttext"]
    char_bilstm_embedder = model_components["char_bilstm"]
    xlmr_embedder = model_components["xlmr"]
    absa_model = model_components["absa_model"]
    idx_to_tag_map = model_components["idx_to_tag"]
    device = model_components["device"]

    results = []

    for i in range(len(comment_texts)):
        comment_text = comment_texts.iloc[i]
        video_id = video_ids.iloc[i]
        comment_id = comment_ids.iloc[i]
        
        current_result = {
            "comment_id": comment_id,
            "video_id": video_id,
            "original_comment_text": comment_text,
            "predicted_aspects": [],
            "potential_kol_topics": [],
            "error_message": None
        }

        if not pd.notnull(comment_text) or not isinstance(comment_text, str) or not comment_text.strip():
            current_result["error_message"] = "SKIPPED_INVALID_INPUT"
            results.append(current_result)
            continue

        try:
            # Create a dict matching the input sample format for get_fused_embeddings_and_iob2_labels
            sample_data = {"text": comment_text, "labels": []} # empty labels as it's prediction

            # Get fused embeddings
            fused_embeddings, _, word_tokens = get_fused_embeddings_and_iob2_labels(
                sample_data, fasttext_embedder, char_bilstm_embedder, xlmr_embedder
            )

            if fused_embeddings is None or len(word_tokens) == 0:
                current_result["error_message"] = "EMBEDDING_GENERATION_FAILED"
                results.append(current_result)
                continue

            # Prepare input for the model
            embeddings_batch = fused_embeddings.unsqueeze(0).to(device)
            mask_batch = torch.ones(embeddings_batch.shape[0], embeddings_batch.shape[1], dtype=torch.bool).to(device)

            with torch.no_grad():
                predicted_tag_ids_list = absa_model(embeddings_batch, mask=mask_batch)

            predicted_tag_ids = predicted_tag_ids_list[0]
            predicted_labels_str = [idx_to_tag_map[id_val] for id_val in predicted_tag_ids]

            predicted_spans = extract_spans(predicted_labels_str)

            aspect_results = []
            potential_kol_topics = set()

            for start_idx, end_idx, tag_type in predicted_spans:
                span_text = " ".join(word_tokens[start_idx : end_idx + 1])
                aspect_results.append({
                    "span_text": span_text,
                    "tag_type": tag_type,
                    "start_token_idx": start_idx,
                    "end_token_idx": end_idx
                })
                
                # KOL Recommendation Logic
                if tag_type == 'ASPECT':
                    potential_kol_topics.add(span_text.lower())
                elif tag_type == 'PRICE':
                    potential_kol_topics.add("giá cả")
                # Add more rules as needed based on your tag types

            current_result["predicted_aspects"] = aspect_results
            current_result["potential_kol_topics"] = list(potential_kol_topics)

        except Exception as e_pred:
            current_result["error_message"] = f"PREDICTION_ERROR: {str(e_pred)[:200]}" # Limit error message length
            print(f"Worker process: Prediction error for comment_id {comment_id}: {e_pred}")
            import traceback
            traceback.print_exc()

        results.append(current_result)
    
    return pd.DataFrame(results, columns=absa_output_schema.names)

print("Đã định nghĩa Pandas UDF (lazy load) cho ABSA prediction.")

# --- B4. Áp dụng UDF và chuẩn bị DataFrame để ghi ra console ---
# Áp dụng UDF để thêm cột với các kết quả ABSA
df_with_absa_results = parsedDF.withColumn(
    "absa_results", 
    predict_absa_udf_lazy_load(
        col("comment_text"), 
        col("video_id"), 
        col("comment_id")
    )
).select(
    col("absa_results.comment_id"),
    col("absa_results.video_id"),
    col("absa_results.original_comment_text"),
    col("absa_results.predicted_aspects"),
    col("absa_results.potential_kol_topics"),
    col("absa_results.error_message")
)


print("Đã áp dụng UDF và tạo DataFrame với các cột kết quả ABSA.")

# --- B5. Khởi chạy Streaming Query để ghi ra Console ---
output_mode = "append" 
checkpoint_dir = "/tmp/streaming_ABSA_spark_checkpoint_console" # Unique checkpoint directory

print(f"Đang khởi chạy streaming query (ABSA Prediction Spark Streaming - Console Output) với output mode: {output_mode} và checkpoint: {checkpoint_dir}")

query = df_with_absa_results.writeStream \
    .format("console") \
    .outputMode(output_mode) \
    .option("truncate", "false") \
    .option("numRows", 20) \
    .option("checkpointLocation", checkpoint_dir) \
    .trigger(processingTime="30 seconds") \
    .start()

print("Streaming query (ABSA Prediction Spark Streaming - Console Output) đã khởi chạy! Chờ dữ liệu từ Kafka...")
print(f"Bạn có thể gửi message JSON (ví dụ: {{'comment_text':'điện thoại này pin trâu nhưng camera hơi tệ', 'video_id':'vid123', 'comment_id':'cmt456'}}) vào topic '{kafka_topic}'.")

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