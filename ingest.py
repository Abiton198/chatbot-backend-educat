# 📚 Library to extract text from PDF files
import pdfplumber

# 📁 Used to work with folders and file paths
import os

# 💾 Used to save data in JSON format
import json


print("🔥 SCRIPT STARTED")
print("📁 Current directory:", os.getcwd())


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
# =========================
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


# =========================
# 📂 PATHS
# =========================
data_folder = "data/"
processed_folder = "processed/"
tracker_file = os.path.join(processed_folder, "processed_files.json")

# Ensure processed folder exists
os.makedirs(processed_folder, exist_ok=True)


# =========================
# 🧠 LOAD TRACKER
# =========================
if os.path.exists(tracker_file):
    with open(tracker_file, "r") as f:
        processed_files = json.load(f)
else:
    processed_files = []

print("📌 Already processed files:", processed_files)


# =========================
# 🚀 PROCESS FILES
# =========================
new_files_processed = 0

if not os.path.exists(data_folder):
    print("❌ data folder NOT found!")

else:
    for file in os.listdir(data_folder):

        if not file.endswith(".pdf"):
            continue

        print("\n📄 Found:", file)

        # ⛔ SKIP if already processed
        if file in processed_files:
            print("⏭️ Skipping (already processed)")
            continue

        print("✅ Processing NEW file:", file)

        pdf_path = os.path.join(data_folder, file)

        # 📥 Extract text
        text = extract_text_from_pdf(pdf_path)
        print("📝 Extracted length:", len(text))

        # ✂️ Chunk text
        chunks = chunk_text(text)
        print("🔹 Number of chunks:", len(chunks))

        # 📦 Prepare dataset
        file_chunks = []
        for chunk in chunks:
            file_chunks.append({
                "source": file,
                "content": chunk
            })

        # 💾 Save per-file JSON
        output_file = os.path.join(processed_folder, file.replace(".pdf", ".json"))

        with open(output_file, "w") as f:
            json.dump(file_chunks, f, indent=2)

        print("💾 Saved:", output_file)

        # ✅ Mark as processed
        processed_files.append(file)
        new_files_processed += 1


# =========================
# 💾 SAVE TRACKER
# =========================
with open(tracker_file, "w") as f:
    json.dump(processed_files, f, indent=2)


# =========================
# ✅ FINAL OUTPUT
# =========================
print("\n📊 New files processed:", new_files_processed)
print("📦 Total processed files:", len(processed_files))
print("✅ DONE")