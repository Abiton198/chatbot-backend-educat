import os
import json
import time
from groq import Groq, RateLimitError, AuthenticationError
from dotenv import load_dotenv

# =========================
# 🔑 LOAD ENV & CLIENT
# =========================
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY is not set.")

client = Groq(api_key=api_key)

# =========================
# 📁 PATHS
# =========================
PROCESSED_FOLDER = "processed"
OUTPUT_FOLDER = "exams"
TRACK_FILE = "processed_exams.json"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Create tracking file if not exists
if not os.path.exists(TRACK_FILE):
    with open(TRACK_FILE, "w") as f:
        json.dump([], f)

with open(TRACK_FILE) as f:
    processed_files = set(json.load(f))


# =========================
# 🧠 AI EXTRACT FUNCTION
# =========================
def extract_questions(text):

    prompt = f"""
You are an expert exam parser.

Extract ALL questions and memo answers from THIS TEXT ONLY.

Return ONLY valid JSON (no markdown):

[
  {{
    "question": "question text",
    "marks": number,
    "memo": "answer text"
  }}
]

Rules:
- Extract EVERY question in this chunk
- Do NOT skip anything
- If marks missing → assume 1
- Keep answers aligned to the correct question
- Return ONLY JSON

TEXT:
{text}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
    except AuthenticationError:
        print("❌ Invalid API key")
        return []
    except RateLimitError:
        print("⚠️ Rate limit hit, waiting...")
        time.sleep(2)
        return []
    except Exception as e:
        print(f"❌ API error: {e}")
        return []

    content = response.choices[0].message.content.strip()

    # Clean markdown if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, list) else []
    except Exception as e:
        print("❌ JSON parse failed")
        print(content[:300])
        return []


# =========================
# 🔄 MAIN PROCESSOR
# =========================
def process_files():

    if not os.path.exists(PROCESSED_FOLDER):
        print(f"❌ Folder '{PROCESSED_FOLDER}' not found.")
        return

    json_files = [
        f for f in os.listdir(PROCESSED_FOLDER)
        if f.endswith(".json") and f not in ["metadata.json", "chunk_ids.json", "processed_files.json"]
    ]

    if not json_files:
        print("⚠️ No files to process.")
        return

    for file in json_files:

        # 🚫 Skip already processed
        if file in processed_files:
            print(f"⏭️ Skipping {file} (already processed)")
            continue

        print(f"\n📄 Processing: {file}")

        path = os.path.join(PROCESSED_FOLDER, file)

        try:
            with open(path) as f:
                chunks = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")
            continue

        all_questions = []
        seen_questions = set()
        question_id = 1

        for i, chunk in enumerate(chunks):
            text = chunk.get("content", "").strip()
            if not text:
                continue
            extracted = extract_questions(text)
            for q in extracted:
                q_text = q.get("question", "").strip()
                if not q_text or q_text in seen_questions:
                    continue
                seen_questions.add(q_text)
                q["id"] = question_id
                question_id += 1
                all_questions.append(q)

                question_id += 1

            print(f"📊 Total so far: {len(all_questions)}")

            # 🛑 Prevent rate limit
            time.sleep(1)

        if not all_questions:
            print(f"⚠️ Nothing extracted from {file}")
            continue

        # 💾 Save result
        output_path = os.path.join(
            OUTPUT_FOLDER,
            file.replace(".json", "_exam.json")
        )

        with open(output_path, "w") as f:
            json.dump({
                "source": file,
                "total_questions": len(all_questions),
                "questions": all_questions
            }, f, indent=2)

        print(f"💾 Saved: {output_path}")
        print(f"🔥 FINAL COUNT: {len(all_questions)}")

        # ✅ Mark as processed
        processed_files.add(file)

        with open(TRACK_FILE, "w") as f:
            json.dump(list(processed_files), f, indent=2)


# =========================
# ▶️ RUN
# =========================
if __name__ == "__main__":
    process_files()