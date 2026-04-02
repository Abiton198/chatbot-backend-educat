# 📚 Library to extract text from PDF files
import pdfplumber

# 📁 Used to work with folders and file paths
import os

# 💾 Used to save data in JSON format
import json


# 🚀 Debug: confirms script is running
print("🔥 SCRIPT STARTED")

# 📁 Shows where Python is currently running from
# (IMPORTANT for fixing folder path issues)
print("📁 Current directory:", os.getcwd())


# 🧾 FUNCTION: Extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""

    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:

        # Loop through each page in the PDF
        for page in pdf.pages:

            # Extract text from the page
            page_text = page.extract_text()

            # ⚠️ Some pages may return None → avoid errors
            if page_text:
                text += page_text + "\n"

    return text


# ✂️ FUNCTION: Break long text into smaller chunks
# This is VERY IMPORTANT for AI retrieval later
def chunk_text(text, chunk_size=300):

    # Split text into words
    words = text.split()

    chunks = []

    # Loop through words in steps of chunk_size
    for i in range(0, len(words), chunk_size):

        # Join words back into a chunk
        chunk = " ".join(words[i:i+chunk_size])

        chunks.append(chunk)

    return chunks


# 📂 Folder where your PDFs are stored
data_folder = "data/"

# 🧠 This will store ALL chunks from ALL PDFs
all_chunks = []


print("🚀 Starting ingestion...")


# ❌ Check if data folder exists
if not os.path.exists(data_folder):
    print("❌ data folder NOT found!")

else:
    # 📄 Loop through all files in the data folder
    for file in os.listdir(data_folder):

        print("📄 Found file:", file)

        # ✅ Only process PDF files
        if file.endswith(".pdf"):

            print("✅ Processing:", file)

            # 📥 Step 1: Extract text from PDF
            text = extract_text_from_pdf(os.path.join(data_folder, file))

            print("📝 Extracted length:", len(text))

            # ✂️ Step 2: Break text into chunks
            chunks = chunk_text(text)

            print("🔹 Number of chunks:", len(chunks))

            # 📦 Step 3: Store chunks in dataset format
            for chunk in chunks:
                all_chunks.append({
                    "source": file,   # which book it came from
                    "content": chunk # actual text content
                })


# 📊 Show total chunks collected
print("📦 Total chunks:", len(all_chunks))


# 💾 STEP 4: Save dataset to JSON file

# Create "processed" folder if it doesn’t exist
os.makedirs("processed", exist_ok=True)

# Save all chunks into a JSON file
with open("processed/data.json", "w") as f:
    json.dump(all_chunks, f, indent=2)


# ✅ Final confirmation
print("✅ Dataset created and saved!")