import os
import shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# === CONFIG ===
together_api_key = "8de3d389ac7d624bce9f9e8dbd7255de80ced7cd6bca300c3ba37b7b2e9f603e"  # 🔑 Replace with your Together.ai API key
os.environ["TOGETHER_API_KEY"] = together_api_key

faiss_path = "vectorstore"
docs_path = "docs"  # Folder with your text documents

# === INITIALIZE EMBEDDINGS ===
print("🔠 Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === LOAD OR BUILD VECTORSTORE ===
if Path(faiss_path).exists():
    print("📦 Loading existing FAISS index...")
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
else:
    print("📄 Creating new FAISS index...")
    docs = []
    for filename in os.listdir(docs_path):
        with open(os.path.join(docs_path, filename), "r", encoding="utf-8") as f:
            content = f.read()
            docs.append(Document(page_content=content, metadata={"source": filename}))
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_path)

    # === BACKUP VECTORSTORE ZIP ===
    shutil.make_archive("vectorstore_backup", "zip", faiss_path)
    print("📁 Backup saved: vectorstore_backup.zip")

# === INITIALIZE LLM (Together.ai) ===
print("🤖 Loading Together.ai model...")
llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    base_url="https://api.together.xyz/v1",
    api_key=together_api_key,
    temperature=0.3
)

# === BUILD RETRIEVAL QA CHAIN ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# === FUNCTION FOR APP (Optional)
def get_answer(question: str) -> str:
    result = qa_chain.invoke(question)
    return result["result"]

# === INTERACTIVE LOOP ===
while True:
    query = input("❓ Enter your question (or 'exit' to quit): ")
    if query.lower() == "exit":
        break
    try:
        result = qa_chain.invoke(query)
        print("✅ Answer:", result["result"])
        print("\n📚 Sources:")
        for i, doc in enumerate(result["source_documents"]):
            print(f" - [{i+1}] {doc.metadata.get('source', 'Unknown')}")
    except Exception as e:
        print("❌ Error:", str(e))
