import pdfplumber
import os
import json
import hashlib

print("🔥 SCRIPT STARTED")
print("📁 Current directory:", os.getcwd())


# =========================
# 🚫 BLOCKLIST
# PDFs here will never be processed — no chunks created
# =========================
BLOCKLIST = {
    "Gr12_CAT_Theory Book.pdf",
    "May-memo_ 2025.pdf",
    # Add more here as needed:
    # "some_other_file.pdf",
}


# =========================
# 📂 PATHS
# =========================
DATA_FOLDER      = "data/"
PROCESSED_FOLDER = "processed/"
TRACKER_FILE     = os.path.join(PROCESSED_FOLDER, "processed_files.json")

os.makedirs(PROCESSED_FOLDER, exist_ok=True)


# =========================
# 🔐 FILE HASH (change detection)
# =========================
def get_file_hash(filepath):
    """Return MD5 hash of a file so we can detect if it changed."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================
# 🧠 LOAD / SAVE TRACKER
# =========================
def load_tracker():
    """
    Tracker format:
    {
      "filename.pdf": {
        "hash": "abc123...",
        "chunks_file": "processed/filename.json",
        "chunk_count": 12
      },
      ...
    }
    """
    if not os.path.exists(TRACKER_FILE):
        return {}
    try:
        with open(TRACKER_FILE) as f:
            data = json.load(f)
            # Migrate old list format → new dict format
            if isinstance(data, list):
                print("⚙️  Migrating old tracker format...")
                return {name: {"hash": None, "chunks_file": None, "chunk_count": None} for name in data}
            return data
    except (json.JSONDecodeError, Exception):
        print("⚠️  Could not read tracker — starting fresh.")
        return {}


def save_tracker(tracker):
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)


# =========================
# 🧾 EXTRACT TEXT FROM PDF
# =========================
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# =========================
# ✂️ CHUNK TEXT
# Chunks carry full metadata so extract_questions.py
# has everything it needs to build complete questions.
# =========================
def chunk_text(text, source_file, chunk_size=300):
    """
    Split text into overlapping word-chunks.
    Each chunk dict includes:
      - source       : original PDF filename
      - chunk_index  : position in the document
      - total_chunks : total number of chunks for this file
      - content      : the text of this chunk
    These fields are used downstream by extract_questions.py
    to correctly reconstruct section/question structure.
    """
    words = text.split()
    raw_chunks = []

    for i in range(0, len(words), chunk_size):
        raw_chunks.append(" ".join(words[i:i + chunk_size]))

    total = len(raw_chunks)

    chunks = []
    for idx, chunk in enumerate(raw_chunks):
        chunks.append({
            "source":       source_file,
            "chunk_index":  idx,
            "total_chunks": total,
            "content":      chunk
        })

    return chunks


# =========================
# 🔀 MERGE CHUNKS
# If a PDF was re-processed (changed), merge new chunks
# into the existing file rather than overwriting blindly.
# =========================
def merge_chunks(existing_chunks, new_chunks):
    """
    Replace chunks from the same source with the new version.
    Keeps chunks from OTHER sources untouched (future: multi-source files).
    """
    if not existing_chunks:
        return new_chunks

    source = new_chunks[0]["source"] if new_chunks else None
    kept = [c for c in existing_chunks if c.get("source") != source]
    return kept + new_chunks


# =========================
# 🚀 PROCESS FILES
# =========================
def process_files():
    tracker = load_tracker()

    print("\n📌 Already tracked files:", list(tracker.keys()) or "none")

    if not os.path.exists(DATA_FOLDER):
        print("❌ data/ folder NOT found!")
        return

    pdf_files = [f for f in sorted(os.listdir(DATA_FOLDER)) if f.endswith(".pdf")]

    if not pdf_files:
        print("⚠️  No PDF files found in data/")
        return

    # ── Pre-flight summary ────────────────────────────────
    blocked  = [f for f in pdf_files if f in BLOCKLIST]
    eligible = [f for f in pdf_files if f not in BLOCKLIST]

    new_files     = [f for f in eligible if f not in tracker]
    changed_files = [
        f for f in eligible
        if f in tracker and tracker[f].get("hash") != get_file_hash(os.path.join(DATA_FOLDER, f))
    ]
    unchanged_files = [
        f for f in eligible
        if f in tracker and tracker[f].get("hash") == get_file_hash(os.path.join(DATA_FOLDER, f))
    ]

    print(f"\n📂 Total PDFs     : {len(pdf_files)}")
    print(f"🚫 Blocklisted    : {len(blocked)}")
    for b in blocked:
        print(f"     → {b}")
    print(f"⏭️  Unchanged      : {len(unchanged_files)}")
    print(f"🆕 New            : {len(new_files)}")
    print(f"✏️  Changed        : {len(changed_files)}")

    to_process = new_files + changed_files

    if not to_process:
        print("\n✅ All files are up to date. Nothing to do.")
        return

    print(f"\n🔄 Processing {len(to_process)} file(s)...\n")

    new_count     = 0
    updated_count = 0

    for file in to_process:
        is_update = file in tracker
        label     = "✏️  UPDATE" if is_update else "🆕 NEW"

        print(f"{label}: {file}")

        pdf_path = os.path.join(DATA_FOLDER, file)

        # 📥 Extract text
        text = extract_text_from_pdf(pdf_path)
        print(f"  📝 Extracted {len(text)} characters")

        if not text.strip():
            print(f"  ⚠️  No text extracted — skipping.\n")
            continue

        # ✂️ Chunk text
        new_chunks = chunk_text(text, source_file=file)
        print(f"  🔹 {len(new_chunks)} chunk(s) created")

        # 📦 Output path
        output_path = os.path.join(PROCESSED_FOLDER, file.replace(".pdf", ".json"))

        # 🔀 Merge if updating an existing file
        if is_update and os.path.exists(output_path):
            try:
                with open(output_path) as f:
                    existing_chunks = json.load(f)
                merged = merge_chunks(existing_chunks, new_chunks)
                print(f"  🔀 Merged: {len(existing_chunks)} old + {len(new_chunks)} new → {len(merged)} total chunks")
            except Exception:
                merged = new_chunks
                print(f"  ⚠️  Could not read existing chunks — replacing.")
        else:
            merged = new_chunks

        # 💾 Save chunks
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)

        print(f"  💾 Saved: {output_path}")

        # ✅ Update tracker with new hash
        tracker[file] = {
            "hash":        get_file_hash(pdf_path),
            "chunks_file": output_path,
            "chunk_count": len(merged)
        }
        save_tracker(tracker)

        if is_update:
            updated_count += 1
        else:
            new_count += 1

        print()

    # ── Final summary ─────────────────────────────────────
    print("=" * 45)
    print(f"🆕 New files processed    : {new_count}")
    print(f"✏️  Updated files          : {updated_count}")
    print(f"⏭️  Skipped (unchanged)    : {len(unchanged_files)}")
    print(f"🚫 Skipped (blocklisted)  : {len(blocked)}")
    print(f"📦 Total tracked          : {len(tracker)}")
    print("✅ DONE")


# =========================
# ▶️ RUN
# =========================
if __name__ == "__main__":
    process_files()