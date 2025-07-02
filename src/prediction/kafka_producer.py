import time
import json
from kafka import KafkaProducer
import random
from datetime import datetime
from confluent_kafka.admin import AdminClient, NewTopic

# Cấu hình Kafka
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC_RAW_COMMENTS = "tiktok_comment2"

producer = None

def create_topic_if_not_exists(topic_name, num_partitions=1, replication_factor=1):
    """Tạo Kafka topic nếu chưa tồn tại."""
    admin_client = AdminClient({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})
    metadata = admin_client.list_topics(timeout=5)

    if topic_name in metadata.topics:
        print(f"⚠️ Topic '{topic_name}' đã tồn tại.")
        return

    new_topic = [NewTopic(topic_name, num_partitions=num_partitions, replication_factor=replication_factor)]
    fs = admin_client.create_topics(new_topic)
    
    for topic, f in fs.items():
        try:
            f.result()
            print(f"✅ Topic '{topic}' đã được tạo.")
        except Exception as e:
            print(f"❌ Lỗi khi tạo topic '{topic}': {e}")

def get_kafka_producer():
    global producer
    if producer is None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
                api_version=(0, 10),
                retries=5
            )
            print(f"✅ Kafka Producer đã khởi tạo cho {KAFKA_BOOTSTRAP_SERVERS}")
        except Exception as e:
            print(f"❌ Lỗi khi khởi tạo Kafka Producer: {e}")
            producer = "ERROR"
    return producer

def send_structured_comment_to_kafka(video_data: dict, comment_data: dict):
    current_producer = get_kafka_producer()
    if current_producer == "ERROR":
        print("❌ Không thể gửi tin nhắn: Kafka Producer không khả dụng.")
        return False

    video_url = video_data.get("video_url", "")
    video_id_from_url = video_url.split('/')[-1] if '/' in video_url else "unknown_video"

    consumer_message = {
        "comment_text": comment_data.get("text", ""),
        "video_id": video_id_from_url,
        "comment_id": comment_data.get("id", f"cid_{int(time.time()*1000)}"),
        "created_time": comment_data.get("created_time", int(datetime.now().timestamp())) * 1000
    }

    try:
        future = current_producer.send(KAFKA_TOPIC_RAW_COMMENTS, value=consumer_message)
        record_metadata = future.get(timeout=10)
        print(f"✅ Gửi thành công: Topic={record_metadata.topic}, Partition={record_metadata.partition}, Offset={record_metadata.offset} - Comment: '{consumer_message['comment_text'][:50]}...'")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi gửi tin nhắn Kafka: {e}")
        return False

if __name__ == "__main__":
    print("▶️ Chạy Kafka Producer (ví dụ).")

    # B1: Tạo topic nếu chưa tồn tại
    create_topic_if_not_exists(KAFKA_TOPIC_RAW_COMMENTS)

    # B2: Dữ liệu mẫu
    sample_data = [
        {
            "video_url": "https://www.tiktok.com/@chouchinchan/video/7519112491503226119",
            "comments": [
                {
                    "id": "7519115810519548680",
                    "text": "Cái mặt chị mộc quá mộc, nhìn như không trang điểm gì luôn á.", 
                    "likes_count": 2324,
                    "author": { "user_id": "7030455703039951899", "username": "lnmp_1708" },
                    "created_time": 1750680582,
                    "replies": []
                },
                {
                    "id": "7519115810519548681",
                    "text": "Chị ơi, chị có thể cho em biết tên sản phẩm này không?", 
                    "likes_count": 123,
                    "author": { "user_id": "user1", "username": "user1_name" },
                    "created_time": 1750680600,
                    "replies": []
                }
            ]
        }
    ]

    for video_entry in sample_data:
        for comment_entry in video_entry["comments"]:
            send_structured_comment_to_kafka(video_entry, comment_entry)
            time.sleep(1)

    if producer and producer != "ERROR":
        producer.flush()
        print("✅ Producer flushed. Tất cả tin nhắn đã được gửi.")

    print("✅ Hoàn tất Producer.")
