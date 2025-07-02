from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json

# Ngăn Hugging Face tự động tải .safetensors
os.environ["HUGGINGFACE_HUB_USE_SAFETENSORS"] = "0"

# --- Cấu hình ---
MODEL_NAME = "VietAI/vit5-large-vietnews-summarization"
INPUT_FOLDER = "../data/script"           # 🔧 Đường dẫn cố định tới thư mục chứa các file .txt
OUTPUT_FOLDER = "../data/script/"   # 🔧 Thư mục lưu kết quả tóm tắt

# --- Tải mô hình và tokenizer ---
print(f"Đang tải tokenizer và mô hình: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        revision="main",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        use_safetensors=False
    )
    print("Tải thành công.")
except Exception as e:
    print(f"Lỗi khi tải mô hình hoặc tokenizer: {e}")
    exit()

# --- Hàm đọc file .txt ---
def read_script_from_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Lỗi khi đọc file '{file_path}': {e}")
        return ""

# --- Hàm tóm tắt bằng ViT5 ---
def get_script_context_vit5(script_text: str, max_length: int = 256, min_length: int = 50) -> str:
    if not script_text:
        return "Không có nội dung script để tóm tắt."

    input_text = "vietnews: " + script_text + " </s>"
    try:
        input_ids = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).input_ids

        summary_ids = model.generate(
            input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=5,
            early_stopping=True
        )

        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"Lỗi khi tóm tắt: {e}"

# --- Thực thi ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    txt_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]

    if not txt_files:
        print(f"Không tìm thấy file .txt nào trong thư mục: {INPUT_FOLDER}")
        exit()

    for txt_file in txt_files:
        full_input_path = os.path.join(INPUT_FOLDER, txt_file)
        print(f"\n--- Đang xử lý: {txt_file} ---")
        script_content = read_script_from_file(full_input_path)

        if script_content:
            summary = get_script_context_vit5(script_content)
            output_data = {
                "original_script_path": full_input_path,
                "summarized_context": summary
            }

            base_name = os.path.splitext(txt_file)[0]
            output_filename = f"{base_name}_summary.json"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)
                print(f"✔ Đã lưu: {output_path}")
            except Exception as e:
                print(f"Lỗi khi lưu {output_path}: {e}")
        else:
            print(f"Bỏ qua vì không đọc được nội dung: {txt_file}")
