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
from confluent_kafka import Consumer, KafkaException, KafkaError

# Assuming model.py contains the following classes/functions:
# FastTextEmbedder, CharBiLSTMEmbedder, XLMREmbedder, BiLSTM_CRF_Model,
# clean_text, process_text_to_word_tokens, get_char_indices_for_words,
# align_xlmr_embeddings_to_words, get_fused_embeddings_and_iob2_labels, extract_spans
# You need to ensure 'model.py' is in the same directory or accessible via PYTHONPATH.
try:
    from model import FastTextEmbedder, CharBiLSTMEmbedder, XLMREmbedder, BiLSTM_CRF_Model, \
        clean_text, process_text_to_word_tokens, get_char_indices_for_words, align_xlmr_embeddings_to_words, \
        get_fused_embeddings_and_iob2_labels, extract_spans
except ImportError as e:
    print(f"Error importing from model.py: {e}")
    print("Please ensure 'model.py' is in the same directory or accessible via sys.path.")
    sys.exit(1)


# --- Configuration (Should match training configuration) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

FASTTEXT_MODEL_PATH = os.path.join(project_root, "models", "fasttext", "cc.vi.300.bin")
XLMR_MODEL_NAME = os.path.join(project_root, "models", "huggingface", "hub", "models--xlm-roberta-base", "snapshots", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089")
MODEL_SAVE_PATH = os.path.join(project_root, "models", "absa_bilstm_crf_model.pth")

# --- Global Model Cache for the Consumer ---
# These will be initialized once when load_absa_model is called
_global_fasttext_embedder = None
_global_char_bilstm_embedder = None
_global_xlmr_embedder = None
_global_absa_model = None
_global_idx_to_tag_map = None
_global_tag_to_idx_map = None
_global_model_device = None
_global_char_to_idx_loaded = None

def load_absa_model_for_consumer(model_path=MODEL_SAVE_PATH):
    """
    Loads the ABSA model and embedders.
    Uses global variables to cache the loaded models.
    """
    global _global_fasttext_embedder, _global_char_bilstm_embedder, _global_xlmr_embedder, \
           _global_absa_model, _global_idx_to_tag_map, _global_tag_to_idx_map, \
           _global_model_device, _global_char_to_idx_loaded

    # Check if already loaded
    if _global_absa_model is not None:
        print("Model and embedders already loaded.")
        return True

    print(f"Loading ABSA model from {model_path} for consumer...")

    _global_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_global_model_device}")

    try:
        checkpoint = torch.load(model_path, map_location=_global_model_device)

        _global_fasttext_embedder = FastTextEmbedder(FASTTEXT_MODEL_PATH)
        print("FastTextEmbedder loaded.")

        _global_char_to_idx_loaded = checkpoint.get('char_to_idx')
        if _global_char_to_idx_loaded is None:
            raise ValueError("'char_to_idx' not found in checkpoint. Cannot initialize CharBiLSTMEmbedder.")

        _global_char_bilstm_embedder = CharBiLSTMEmbedder(
            vocab_size=len(_global_char_to_idx_loaded),
            embedding_dim=checkpoint['char_embedding_dim'] if 'char_embedding_dim' in checkpoint else 50,
            hidden_dim=checkpoint['char_lstm_hidden_dim'] if 'char_lstm_hidden_dim' in checkpoint else 50
        ).to(_global_model_device)
        if 'char_bilstm_state_dict' in checkpoint:
            _global_char_bilstm_embedder.load_state_dict(checkpoint['char_bilstm_state_dict'])
            print("CharBiLSTMEmbedder state_dict loaded.")
        else:
            print("WARNING: 'char_bilstm_state_dict' not found in checkpoint. CharBiLSTMEmbedder initialized with random weights.")
        _global_char_bilstm_embedder.eval() # Set to eval mode

        _global_xlmr_embedder = XLMREmbedder(XLMR_MODEL_NAME)
        _global_xlmr_embedder.model.to(_global_model_device)
        _global_xlmr_embedder.model.eval() # Set to eval mode
        print("XLMREmbedder loaded.")

        _global_absa_model = BiLSTM_CRF_Model(
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_tags=checkpoint['num_tags'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout']
        ).to(_global_model_device)
        _global_absa_model.load_state_dict(checkpoint['model_state_dict'])
        _global_absa_model.eval() # Set to evaluation mode
        print("BiLSTM_CRF_Model loaded.")

        _global_idx_to_tag_map = checkpoint['idx_to_tag']
        _global_tag_to_idx_map = checkpoint['current_tag_to_idx']

        print("✅ ABSA model and embedders loaded successfully for consumer.")
        return True

    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}. Please ensure the model is trained and saved.")
        return False
    except Exception as e:
        print(f"❌ Error loading model or embedders: {e}")
        import traceback
        traceback.print_exc()
        return False

def predict_aspect_spans_for_consumer(comment_text: str):
    """
    Predicts aspect spans for a given comment text using the globally loaded model.
    """
    if _global_absa_model is None:
        print("Model is not loaded. Cannot perform prediction.")
        return []

    # Create a dict matching the input sample format for get_fused_embeddings_and_iob2_labels
    sample_data = {"text": comment_text, "labels": []} # empty labels as it's prediction

    # Get fused embeddings
    fused_embeddings, _, word_tokens = get_fused_embeddings_and_iob2_labels(
        sample_data, _global_fasttext_embedder, _global_char_bilstm_embedder, _global_xlmr_embedder
    )

    if fused_embeddings is None or len(word_tokens) == 0:
        # print(f"Could not generate embeddings for text: '{comment_text}'")
        return []

    # Prepare input for the model
    # Add a batch dimension (batch_size=1)
    embeddings_batch = fused_embeddings.unsqueeze(0).to(_global_model_device)
    # Create mask for batch 1 (all tokens are real for a single sequence)
    mask_batch = torch.ones(embeddings_batch.shape[0], embeddings_batch.shape[1], dtype=torch.bool).to(_global_model_device)

    with torch.no_grad():
        predicted_tag_ids_list = _global_absa_model(embeddings_batch, mask=mask_batch)

    # Get prediction for the first (and only) sample in the batch
    predicted_tag_ids = predicted_tag_ids_list[0]
    predicted_labels_str = [_global_idx_to_tag_map[id_val] for id_val in predicted_tag_ids]

    # Extract aspect spans
    predicted_spans = extract_spans(predicted_labels_str)

    # Convert predicted spans to a more readable format (e.g., with span text)
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

def start_kafka_consumer():
    """
    Starts the Kafka consumer to read messages, predict aspects, and print results.
    """
    KAFKA_TOPIC_INPUT = "tiktok_comment2"
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
    CONSUMER_GROUP_ID = "absa_prediction_group" # Unique group ID for this consumer

    # Kafka consumer configuration
    conf = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': CONSUMER_GROUP_ID,
        'auto.offset.reset': 'earliest',  # Start reading from the beginning if no committed offset
        'enable.auto.commit': False,      # Manual commit for more control
        'heartbeat.interval.ms': 3000,    # How often to send heartbeats to the consumer coordinator
        'session.timeout.ms': 10000,      # Consumer session timeout
        'max.poll.interval.ms': 300000,   # Max time between polls before consumer is considered failed
    }

    consumer = Consumer(conf)

    try:
        consumer.subscribe([KAFKA_TOPIC_INPUT])
        print(f"Subscribed to Kafka topic: {KAFKA_TOPIC_INPUT}")
        print("Waiting for messages... Press Ctrl+C to exit.")

        while True:
            msg = consumer.poll(timeout=1.0) # Poll for messages, with a 1-second timeout

            if msg is None:
                # print("No message received within timeout.")
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event - not an error
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    sys.stderr.write(f"Unknown topic or partition: {msg.error()}\n")
                elif msg.error().code() == KafkaError._NO_OFFSET:
                    sys.stderr.write(f"No offset available for partition: {msg.error()}\n")
                elif msg.error():
                    raise KafkaException(msg.error())
                continue

            try:
                # Decode message value from bytes to string
                message_value = msg.value().decode('utf-8')
                data = json.loads(message_value)

                comment_text = data.get("comment_text")
                video_id = data.get("video_id", "N/A")
                comment_id = data.get("comment_id", "N/A") # Default to N/A if not present

                print(f"\n--- Received Message ---")
                print(f"  Topic: {msg.topic()}, Partition: {msg.partition()}, Offset: {msg.offset()}")
                print(f"  Video ID: {video_id}, Comment ID: {comment_id}")
                print(f"  Comment Text: '{comment_text[:100]}...'")

                if not comment_text or not isinstance(comment_text, str) or not comment_text.strip():
                    print("  Skipping invalid/empty comment text.")
                    consumer.commit(message=msg) # Commit offset for skipped message
                    continue

                # Perform prediction
                predicted_aspects = predict_aspect_spans_for_consumer(comment_text)

                print("  Predicted Aspects:")
                if predicted_aspects:
                    for aspect in predicted_aspects:
                        print(f"    - Text: '{aspect['span_text']}', Type: {aspect['tag_type']}")
                    
                    # --- Conceptual KOL Recommendation Logic ---
                    potential_kol_topics = set()
                    for aspect in predicted_aspects:
                        if aspect['tag_type'] == 'ASPECT':
                            potential_kol_topics.add(aspect['span_text'].lower())
                        elif aspect['tag_type'] == 'PRICE':
                            potential_kol_topics.add("giá cả")
                        # Add more rules as needed based on your tag types
                    
                    if potential_kol_topics:
                        print(f"  Potential KOL Topics derived: {list(potential_kol_topics)}")
                        print("  (Conceptual) Recommending KOLs based on these topics...")
                        # In a real system, you'd integrate with a KOL recommendation service/DB here
                    else:
                        print("  No specific KOL topics derived from aspects.")

                else:
                    print("    No aspects detected.")
                
                consumer.commit(message=msg) # Manually commit the offset after successful processing

            except json.JSONDecodeError as e:
                print(f"  Error decoding JSON from message value: {e}")
                print(f"  Raw message value: {message_value}")
                # Consider committing here as well if you want to skip malformed messages
                consumer.commit(message=msg)
            except Exception as e:
                print(f"  An error occurred during message processing: {e}")
                import traceback
                traceback.print_exc()
                # Depending on your error handling policy, you might or might not commit here.
                # For simplicity, let's commit to avoid reprocessing a message that caused an error.
                consumer.commit(message=msg)

    except KeyboardInterrupt:
        sys.stderr.write('%% Aborted by user\n')
    finally:
        # Close down consumer to commit final offsets.
        consumer.close()
        print("Kafka Consumer closed.")

# --- Main execution block ---
if __name__ == "__main__":
    print("Starting standalone Kafka Consumer for ABSA Prediction.")

    # Load the model once when the consumer starts
    if not load_absa_model_for_consumer():
        print("Exiting due to model loading failure.")
        sys.exit(1)

    # Start consuming Kafka messages
    start_kafka_consumer()