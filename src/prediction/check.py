from kafka import KafkaConsumer
import json

# Cau hinh Kafka Consumer
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC_INPUT = 'tiktok_comment' # Topic ma consumer Spark dang lang nghe

print(f"Dang khoi tao KafkaConsumer cho topic '{KAFKA_TOPIC_INPUT}'...")
try:
    consumer = KafkaConsumer(
        KAFKA_TOPIC_INPUT,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest', # Bat dau tu dau topic neu khong co offset nao duoc luu
        enable_auto_commit=True,
        group_id='my-test-group', # Mot group ID de theo doi offset
        value_deserializer=lambda x: json.loads(x.decode('utf-8')) if x else None
    )
    print("KafkaConsumer da duoc khoi tao thanh cong. Dang cho tin nhan...")

    for message in consumer:
        print(f"Nhan tin nhan:")
        print(f"  Topic: {message.topic}")
        print(f"  Partition: {message.partition}")
        print(f"  Offset: {message.offset}")
        print(f"  Key: {message.key}")
        print(f"  Value: {message.value}")
        print("-" * 30)

except Exception as e:
    print(f"Loi khi khoi tao hoac nhan tin nhan tu Kafka: {e}")

finally:
    if 'consumer' in locals() and consumer is not None:
        consumer.close()
        print("KafkaConsumer da dong.")