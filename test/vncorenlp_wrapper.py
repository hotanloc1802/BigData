import subprocess
import json
import os
import uuid

class VnCoreNLPWrapper:
    def __init__(self, jar_path, annotators="wseg,pos,ner", models_dir="VnCoreNLP/models", max_heap_size="-Xmx2g"):
        self.jar_path = jar_path
        self.annotators = annotators
        self.models_dir = models_dir
        self.max_heap_size = max_heap_size

        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"Không tìm thấy file jar: {jar_path}")
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục models: {models_dir}")

    def annotate_batch(self, texts):
        """
        Annotates a list of texts using VnCoreNLP in a single batch.

        Args:
            texts (list): A list of strings, where each string is a comment/text to be annotated.

        Returns:
            list: A list of lists of dictionaries. Each inner list corresponds to an input text
                  and contains dictionaries representing the annotated tokens for that text.
                  Returns an empty list if an error occurs or if no annotations are found.
        """
        if not texts:
            return []

        tmp_input = f"temp_input_{uuid.uuid4().hex[:8]}.txt"
        tmp_output = f"temp_output_{uuid.uuid4().hex[:8]}.txt"

        # Write each text to a new line in the input file
        with open(tmp_input, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text.strip() + "\n") # Ensure each text is on a new line

        cmd = [
            "java", self.max_heap_size,
            "-jar", self.jar_path,
            "-fin", tmp_input,
            "-fout", tmp_output,
            "-annotators", self.annotators,
            "-models", self.models_dir
        ]

        all_annotated_texts = []
        try:
            subprocess.run(cmd, check=True)

            with open(tmp_output, "r", encoding="utf-8") as f:
                lines = f.readlines()

            current_text_annotations = []
            for line in lines:
                line = line.strip()
                if not line: # Empty line indicates end of a sentence/document
                    if current_text_annotations: # If there were annotations for the previous text
                        all_annotated_texts.append(current_text_annotations)
                    current_text_annotations = [] # Reset for the next text
                    continue
                elif line.startswith("#"): # Skip comment lines
                    continue

                parts = line.split("\t")
                if len(parts) >= 4:
                    current_text_annotations.append({
                        "index": parts[0],
                        "form": parts[1],
                        "pos": parts[2],
                        "ner": parts[3]
                    })
            
            # Add any remaining annotations for the last text
            if current_text_annotations:
                all_annotated_texts.append(current_text_annotations)

            if not all_annotated_texts:
                print("⚠️ Output trống hoặc không đúng định dạng. Nội dung file:")
                for line in lines:
                    print("📝", line.strip())

            return all_annotated_texts

        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi khi gọi VnCoreNLP (mã lỗi {e.returncode}): {e}")
            print(f"Stderr: {e.stderr.decode('utf-8') if e.stderr else 'N/A'}")
            print(f"Stdout: {e.stdout.decode('utf-8') if e.stdout else 'N/A'}")
            return []
        except FileNotFoundError as e:
            print(f"❌ Lỗi: Không tìm thấy lệnh 'java'. Đảm bảo Java đã được cài đặt và có trong PATH: {e}")
            return []
        except Exception as e:
            print("❌ Lỗi khi gọi VnCoreNLP:", str(e))
            return []
        finally:
            for f in [tmp_input, tmp_output]:
                if os.path.exists(f):
                    os.remove(f)