import os
from rag_engine import load_pdf

def test():
    upload_dir = "uploads"
    for filename in os.listdir(upload_dir):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(upload_dir, filename)
            print(f"Loading {filename}...")
            try:
                chunks = load_pdf(file_path)
                print(f"Success! {filename}: {chunks} chunks")
            except Exception as e:
                print(f"FAILED to load {filename}: {e}")

if __name__ == "__main__":
    test()
