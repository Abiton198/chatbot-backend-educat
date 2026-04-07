import os
import json
import numpy as np
import hashlib
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")

BASE_DIR         = os.path.dirname(__file__)
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")

EMBEDDINGS_FILE = os.path.join(PROCESSED_FOLDER, "embeddings.npy")
METADATA_FILE   = os.path.join(PROCESSED_FOLDER, "metadata.json")
CHUNK_IDS_FILE  = os.path.join(PROCESSED_FOLDER, "chunk_ids.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

client = InferenceClient(token=HF_TOKEN)


# =========================
# 🚫 FILE CLASSIFICATION
# Same keyword logic as process_exams.py so RAG
# never accidentally embeds exam or memo content.
#
# Only files that are NOT exam/memo get embedded —
# i.e. theory books, notes, textbooks, study guides.
# =========================

# Files that are always internal — never content
_INTERNAL_FILES = {
    "processed_files.json",
    "metadata.json",
    "chunk_ids.json",
    "processed_exams.json",
    "processed_memos.json",
    "embeddings.npy",
}

# Keywords that mark a file as an exam question paper
_EXAM_KEYWORDS = [
    "exam", "paper", "question", "theory", "p1", "p2", "p3",
    "nov", "november", "may", "june", "feb", "february",
    "march", "mar", "aug", "august", "sep", "september",
    "oct", "october", "term", "trial", "nsc", "dbe",
]

# Keywords that mark a file as a memo / answer file
_MEMO_KEYWORDS = [
    "memo", "memorandum", "answers", "answer_key", "marking",
]


def _is_rag_eligible(filename):
    """
    Returns True ONLY if the file should be embedded into the RAG index.

    Rules:
      - Skip internal tracking files
      - Skip memo files (answer keys — not teaching content)
      - Skip exam question paper files (questions only, no explanations)
      - Accept everything else (theory books, notes, study guides etc.)
    """
    if filename in _INTERNAL_FILES:
        return False

    lower = filename.lower()

    # Skip memos first (they often also contain exam keywords)
    if any(kw in lower for kw in _MEMO_KEYWORDS):
        return False

    # Skip exam question papers
    if any(kw in lower for kw in _EXAM_KEYWORDS):
        return False

    # Anything else is eligible content for the tutor
    return True


# =========================
# 🔑 UNIQUE CHUNK ID
# =========================
def generate_chunk_id(text):
    return hashlib.md5(text.encode()).hexdigest()


# =========================
# 📥 LOAD ELIGIBLE CHUNKS
# Only loads files that pass _is_rag_eligible()
# =========================
def load_all_chunks():
    if not os.path.exists(PROCESSED_FOLDER):
        print(f"⚠️  Processed folder not found: {PROCESSED_FOLDER}")
        return []

    all_data      = []
    loaded_files  = []
    skipped_files = []

    for filename in sorted(os.listdir(PROCESSED_FOLDER)):

        if not filename.endswith(".json"):
            continue

        if not _is_rag_eligible(filename):
            skipped_files.append(filename)
            continue

        path = os.path.join(PROCESSED_FOLDER, filename)

        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠️  Could not load {filename}: {e}")
            continue

        if not isinstance(data, list):
            skipped_files.append(filename)
            continue

        count = 0
        for item in data:
            content = item.get("content", "").strip()
            if not content:
                continue
            all_data.append({
                "content": content,
                "source":  item.get("source", filename),
                "id":      generate_chunk_id(content)
            })
            count += 1

        if count > 0:
            loaded_files.append(f"{filename} ({count} chunks)")

    print(f"\n📚 RAG content files loaded  : {len(loaded_files)}")
    for f in loaded_files:
        print(f"     ✅ {f}")

    if skipped_files:
        print(f"🚫 Skipped (exam/memo/system): {len(skipped_files)}")
        for f in skipped_files:
            print(f"     → {f}")

    print(f"📦 Total chunks for RAG      : {len(all_data)}\n")

    return all_data


# =========================
# 🔢 EMBEDDING
# =========================
def get_embedding(text):
    result = client.feature_extraction(text, model=MODEL_NAME)
    return np.array(result, dtype=np.float32).flatten()


# =========================
# 📐 COSINE SIMILARITY
# =========================
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# =========================
# 🧠 RAG INDEX
# =========================
class RAGIndex:

    def __init__(self):
        self.dataset     = load_all_chunks()
        self._embeddings = None

        if not self.dataset:
            print("⚠️  RAG: No eligible content files found.")
            print("💡  Add theory books / study guides to the processed/ folder.")
            print("    Files with keywords like: memo, exam, paper, nov, may, term")
            print("    will be excluded — only pure content files are embedded.\n")

    # =========================
    # 🔄 LOAD OR UPDATE EMBEDDINGS
    # Incremental — only embeds NEW chunks
    # =========================
    def _load_or_update_embeddings(self):

        # ── Load existing state ───────────────────────
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

        print(f"📊 Cached chunks  : {len(existing_ids)}")

        # ── Find new chunks not yet embedded ──────────
        new_chunks = [c for c in self.dataset if c["id"] not in existing_ids]
        print(f"🆕 New chunks     : {len(new_chunks)}")

        # ── Use cache if nothing new ──────────────────
        if not new_chunks and embeddings.size > 0:
            print("⚡ No updates needed — using cache.\n")
            self._embeddings = embeddings
            self.dataset     = metadata
            return

        # ── Embed only new chunks ─────────────────────
        if new_chunks:
            print(f"🔢 Embedding {len(new_chunks)} new chunk(s)...")
            new_embeddings = []

            for i, chunk in enumerate(new_chunks, 1):
                emb = get_embedding(chunk["content"])
                new_embeddings.append(emb)
                if i % 10 == 0 or i == len(new_chunks):
                    print(f"   {i}/{len(new_chunks)} embedded")

            new_embeddings = np.array(new_embeddings)

            if embeddings.size == 0:
                embeddings = new_embeddings
            else:
                embeddings = np.vstack([embeddings, new_embeddings])

            metadata.extend(new_chunks)
            existing_ids.update(c["id"] for c in new_chunks)

            # ── Persist ───────────────────────────────
            np.save(EMBEDDINGS_FILE, embeddings)

            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=2)

            with open(CHUNK_IDS_FILE, "w") as f:
                json.dump(list(existing_ids), f, indent=2)

            print(f"💾 Embeddings saved — {len(metadata)} total chunks\n")

        self._embeddings = embeddings
        self.dataset     = metadata

    # =========================
    # 🔍 SEARCH
    # =========================
    def search(self, query, k=3):

        if not self.dataset:
            return []

        self._load_or_update_embeddings()

        if self._embeddings is None or self._embeddings.size == 0:
            return []

        query_emb = get_embedding(query)

        scores = [
            cosine_similarity(query_emb, emb)
            for emb in self._embeddings
        ]

        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            {
                "content": self.dataset[i]["content"],
                "source":  self.dataset[i]["source"],
                "score":   round(scores[i], 4)
            }
            for i in top_k
        ]