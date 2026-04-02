# rag.py
import os
import json
import numpy as np
import faiss
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")
DATA_PATH = "processed/data.json"
INDEX_PATH = "embeddings/index.faiss"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

client = InferenceClient(token=HF_TOKEN)

# Load chunks
with open(DATA_PATH) as f:
    data = json.load(f)

texts = [item["content"] for item in data]


def get_embedding(text: str) -> np.ndarray:
    result = client.feature_extraction(text, model=MODEL_NAME)
    return np.array(result, dtype=np.float32).flatten()


def build_index():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    print("Building embeddings...")
    embeddings = np.stack([get_embedding(t) for t in texts])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print("✅ Index saved.")
    return index, embeddings


# Load or build index
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index, _ = build_index()


class RAGIndex:
    def search(self, query: str, k: int = 3) -> list[str]:
        k = min(k, len(texts))
        query_emb = get_embedding(query).reshape(1, -1)
        _, indices = index.search(query_emb, k)
        return [texts[i] for i in indices[0] if i != -1]
