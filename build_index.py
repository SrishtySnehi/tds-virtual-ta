#build_index.py

import os
import json
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ------------------------------------------
# 🔧 Initialize the embedding model
# ------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------------------
# 📥 Load JSONL files
# ------------------------------------------
def load_jsonl_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    return lines

# ------------------------------------------
# 📦 Build the vector index
# ------------------------------------------
def build_index(course_path, forum_path):
    print(f"🔍 Checking for files...")
    for path in [course_path, forum_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Missing file: {path}")
        print(f"✅ Found: {os.path.basename(path)} at {path}")

    print("📥 Loading data...")
    course_data = load_jsonl_file(course_path)
    forum_data = load_jsonl_file(forum_path)

    print("📚 Creating documents...")
    documents = []

    skipped = 0

    for item in course_data:
        text = item.get("text") or item.get("content")
        if not text:
            skipped += 1
            continue
        documents.append(Document(
            page_content=text,
            metadata={"source": "course", "title": item.get("title", "")}
        ))

    for item in forum_data:
        text = item.get("text") or item.get("content")
        if not text:
            skipped += 1
            continue
        documents.append(Document(
            page_content=text,
            metadata={"source": "forum", "title": item.get("topic_title", "")}
        ))

    print(f"📄 Loaded {len(documents)} valid documents, skipped {skipped} incomplete ones.")

    if not documents:
        raise ValueError("❌ No valid documents found. Check your JSONL data formatting.")

    print("🧠 Generating embeddings...")
    db = FAISS.from_documents(documents, embeddings)

    os.makedirs("vectorstore", exist_ok=True)
    db.save_local("vectorstore")
    print("✅ FAISS index built and saved to 'vectorstore/'")

# ------------------------------------------
# 🚀 Main execution
# ------------------------------------------
if __name__ == "__main__":
    build_index("data/tds-course-content.jsonl", "data/tds-forum-posts.jsonl")