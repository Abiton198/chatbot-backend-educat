import os
import json
import numpy as np
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")
DATA_PATH = os.path.join(os.path.dirname(__file__), "processed/data.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

client = InferenceClient(token=HF_TOKEN)

# Load chunks
with open(DATA_PATH) as f:
    data = json.load(f)

texts = [item["content"] for item in data]


def get_embedding(text: str) -> np.ndarray:
    result = client.feature_extraction(text, model=MODEL_NAME)
    return np.array(result, dtype=np.float32).flatten()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class RAGIndex:
    def __init__(self):
        self._embeddings = None

    def _load_embeddings(self):
        """Lazy load — build embeddings on first search."""
        if self._embeddings is None:
            print("Building embeddings via HuggingFace API...")
            self._embeddings = [get_embedding(t) for t in texts]
            print("✅ Embeddings ready.")

    def search(self, query: str, k: int = 3) -> list:
        self._load_embeddings()
        query_emb = get_embedding(query)
        scores = [cosine_similarity(query_emb, emb) for emb in self._embeddings]
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [texts[i] for i in top_k]