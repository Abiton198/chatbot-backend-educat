# 📁 Used for file paths and environment variables
import os

# 🚫 Force CPU usage (important because your GPU is not compatible)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 💾 For reading your dataset (JSON file)
import json

# 🔍 FAISS = fast similarity search (your vector database)
import faiss

# 🔢 Used for handling embeddings as arrays
import numpy as np

# 🧠 Model that converts text → embeddings (numbers AI understands)
from sentence_transformers import SentenceTransformer


# 📂 Path to your processed dataset (created in ingest.py)
PROCESSED_PATH = "processed/data.json"

# 📂 Path where FAISS index (AI memory) will be stored
INDEX_PATH = "embeddings/index.faiss"

# 🧠 Embedding model name (lightweight + fast)
MODEL_NAME = "all-MiniLM-L6-v2"


# 🧠 MAIN CLASS: Handles retrieval (searching your dataset)
class RAGIndex:

    def __init__(self):

        # ❌ If dataset does not exist → stop everything
        if not os.path.exists(PROCESSED_PATH):
            raise FileNotFoundError(f"Run ingest.py first! Missing: {PROCESSED_PATH}")

        # 🧠 Load embedding model (this converts text into vectors)
        self.model = SentenceTransformer(MODEL_NAME)

        # 📥 Load dataset from JSON file
        with open(PROCESSED_PATH) as f:
            data = json.load(f)

        # 🧾 Extract only the text content (ignore metadata)
        self.texts = [item["content"] for item in data]

        # 🔍 Check if embeddings (FAISS index) already exist
        if os.path.exists(INDEX_PATH):

            # ✅ Load existing index (FASTER startup)
            self.index = faiss.read_index(INDEX_PATH)

        else:
            # 🏗️ Build embeddings from scratch
            self._build_index()


    def _build_index(self):
        """Encode texts and build a new FAISS index."""

        # 📁 Create embeddings folder if it doesn’t exist
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

        # 🧠 Convert ALL text chunks into embeddings (vectors)
        embeddings = self.model.encode(self.texts, show_progress_bar=True)

        # 📏 Get size of each embedding vector
        dimension = embeddings.shape[1]

        # 🔍 Create FAISS index (L2 distance = similarity search)
        self.index = faiss.IndexFlatL2(dimension)

        # 📥 Add embeddings into FAISS database
        self.index.add(np.array(embeddings, dtype=np.float32))

        # 💾 Save index to file (so we don’t rebuild every time)
        faiss.write_index(self.index, INDEX_PATH)

        print("✅ Embeddings created and saved.")


    def search(self, query: str, k: int = 3) -> list[str]:
        """Return the top-k most relevant text chunks for a query."""

        # 🔒 Ensure k is not bigger than available data
        k = min(k, len(self.texts))

        # 🧠 Convert user question into embedding
        query_embedding = self.model.encode([query])

        # 🔍 Search FAISS index for closest matches
        # Returns distances + indices
        _, indices = self.index.search(
            np.array(query_embedding, dtype=np.float32),
            k
        )

        # 📤 Return matching text chunks
        return [
            self.texts[i] 
            for i in indices[0] 
            if i != -1  # -1 means no result
        ]