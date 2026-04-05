import os
import json
import numpy as np
import hashlib
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")

BASE_DIR = os.path.dirname(__file__)
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")

EMBEDDINGS_FILE = os.path.join(PROCESSED_FOLDER, "embeddings.npy")
METADATA_FILE = os.path.join(PROCESSED_FOLDER, "metadata.json")
CHUNK_IDS_FILE = os.path.join(PROCESSED_FOLDER, "chunk_ids.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

client = InferenceClient(token=HF_TOKEN)


# =========================
# 🔑 CREATE UNIQUE ID
# =========================
def generate_chunk_id(text):
    return hashlib.md5(text.encode()).hexdigest()


# =========================
# 📥 LOAD ALL CHUNKS
# =========================
def load_all_chunks():
    all_data = []

    for file in os.listdir(PROCESSED_FOLDER):
        if not file.endswith(".json"):
            continue
        if file in ["processed_files.json", "metadata.json", "chunk_ids.json"]:
            continue

        path = os.path.join(PROCESSED_FOLDER, file)

        with open(path) as f:
            data = json.load(f)

            for item in data:
                content = item["content"]

                all_data.append({
                    "content": content,
                    "source": item.get("source", file),
                    "id": generate_chunk_id(content)
                })

    return all_data


# =========================
# 🔢 EMBEDDING
# =========================
def get_embedding(text):
    result = client.feature_extraction(text, model=MODEL_NAME)
    return np.array(result, dtype=np.float32).flatten()


# =========================
# 📐 SIMILARITY
# =========================
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# =========================
# 🧠 RAG INDEX
# =========================
class RAGIndex:
    def __init__(self):
        self.dataset = load_all_chunks()
        self._embeddings = None

    def _load_or_update_embeddings(self):

        # Load existing
        if os.path.exists(EMBEDDINGS_FILE):
            embeddings = np.load(EMBEDDINGS_FILE)
        else:
            embeddings = np.array([])

        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE) as f:
                metadata = json.load(f)
        else:
            metadata = []

        if os.path.exists(CHUNK_IDS_FILE):
            with open(CHUNK_IDS_FILE) as f:
                existing_ids = set(json.load(f))
        else:
            existing_ids = set()

        print(f"📊 Existing chunks: {len(existing_ids)}")

        # Find NEW chunks
        new_chunks = [c for c in self.dataset if c["id"] not in existing_ids]

        print(f"🆕 New chunks detected: {len(new_chunks)}")

        # If no new chunks → load directly
        if not new_chunks and len(embeddings) > 0:
            print("⚡ No updates needed. Using cache.")
            self._embeddings = embeddings
            self.dataset = metadata
            return

        # Build embeddings ONLY for new chunks
        new_embeddings = []

        for chunk in new_chunks:
            emb = get_embedding(chunk["content"])
            new_embeddings.append(emb)

        if len(new_embeddings) > 0:
            new_embeddings = np.array(new_embeddings)

            # Append
            if embeddings.size == 0:
                embeddings = new_embeddings
            else:
                embeddings = np.vstack([embeddings, new_embeddings])

            metadata.extend(new_chunks)
            existing_ids.update([c["id"] for c in new_chunks])

        # Save updated data
        np.save(EMBEDDINGS_FILE, embeddings)

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        with open(CHUNK_IDS_FILE, "w") as f:
            json.dump(list(existing_ids), f, indent=2)

        print("💾 Incremental update complete")

        self._embeddings = embeddings
        self.dataset = metadata

    def search(self, query, k=3):
        self._load_or_update_embeddings()

        query_emb = get_embedding(query)

        scores = [
            cosine_similarity(query_emb, emb)
            for emb in self._embeddings
        ]

        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            {
                "content": self.dataset[i]["content"],
                "source": self.dataset[i]["source"],
                "score": scores[i]
            }
            for i in top_k
        ]