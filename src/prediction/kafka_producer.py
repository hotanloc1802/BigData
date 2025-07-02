import time
import json
from kafka import KafkaProducer # Cần cài đặt: pip install kafka-python
import random
from datetime import datetime

# Cấu hình Kafka Producer
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC_RAW_COMMENTS = "tiktok_comment" # Topic cho dữ liệu thô (tên mẫu từ script của bạn)

producer = None

def get_kafka_producer():
    """Khởi tạo và trả về Kafka Producer (sẽ được cache)."""
    global producer
    if producer is None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
                api_version=(0, 10), # Đảm bảo tương thích Kafka
                retries=5 # Thử lại nếu gửi không thành công
            )
            print(f"✅ Kafka Producer đã khởi tạo cho {KAFKA_BOOTSTRAP_SERVERS}")
        except Exception as e:
            print(f"❌ Lỗi khi khởi tạo Kafka Producer: {e}")
            producer = "ERROR" # Đánh dấu lỗi
    return producer

def send_comment_to_kafka(comment_text: str, video_id: str = "test_video", comment_id: str = None):
    """
    Gửi một comment dưới dạng JSON tới Kafka topic.
    """
    current_producer = get_kafka_producer()
    if current_producer == "ERROR":
        print("❌ Không thể gửi tin nhắn: Kafka Producer không khả dụng.")
        return False

    if comment_id is None:
        comment_id = f"comment_{video_id}_{int(time.time() * 1000)}_{random.randint(0, 9999)}"

    message = {
        "comment_text": comment_text,
        "video_id": video_id,
        "comment_id": comment_id,
        "created_time": int(datetime.now().timestamp())
    }

    try:
        future = current_producer.send(KAFKA_TOPIC_RAW_COMMENTS, value=message)
        record_metadata = future.get(timeout=10) # Chờ 10 giây để xác nhận
        print(f"✅ Gửi thành công: Topic={record_metadata.topic}, Partition={record_metadata.partition}, Offset={record_metadata.offset} - Comment: '{comment_text[:50]}...'")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi gửi tin nhắn Kafka: {e}")
        return False

if __name__ == "__main__":
    # Ví dụ cách sử dụng Producer
    print("Chạy Kafka Producer trực tiếp (ví dụ).")
    
    # Giả lập dữ liệu cào được từ các file
    sample_comments = [
        "Sản phẩm này dùng rất tốt, giao hàng nhanh chóng.",
        "Giá hơi cao so với chất lượng, không hài lòng lắm.",
        "Màu son đẹp nhưng nhanh trôi quá, cần cải thiện.",
        "Nhân viên tư vấn nhiệt tình, sẽ ủng hộ shop dài dài.",
        "Chất lượng sản phẩm không như quảng cáo, thất vọng.",
        "Đóng gói cẩn thận, sản phẩm nguyên vẹn khi nhận."
    ]

    for i, comment in enumerate(sample_comments):
        send_comment_to_kafka(comment, video_id=f"tiktok_video_abc_{i%2}", comment_id=f"cid_{i}")
        time.sleep(1) # Chờ một chút giữa các tin nhắn
    
    # Đảm bảo tất cả tin nhắn đã được gửi trước khi thoát
    if producer and producer != "ERROR":
        producer.flush()
        print("Producer flushed. Tất cả tin nhắn đã được gửi.")
    print("Hoàn tất ví dụ Producer.")