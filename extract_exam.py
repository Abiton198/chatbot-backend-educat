import os
import re
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
OUTPUT_FOLDER    = "exams"
TRACK_FILE       = "processed_exams.json"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# 🗂️ FILE CLASSIFICATION
#
# Files are accepted ONLY if they match exam OR memo patterns.
# Everything else (theory books, notes, textbooks etc.) is ignored.
#
# EXAM keywords  — words that indicate a question paper
# MEMO keywords  — words that indicate a memo/answer file
# NOISE words    — stripped before keyword matching
# =========================

EXAM_KEYWORDS = [
    "exam", "paper", "question", "theory", "p1", "p2", "p3",
    "nov", "november", "may", "june", "feb", "february",
    "march", "mar", "aug", "august", "sep", "september",
    "oct", "october", "term", "trial", "nsc", "dbe", "cat",
]

MEMO_KEYWORDS = [
    "memo", "memorandum", "answers", "answer_key", "marking",
]

NOISE_WORDS = {
    "memo", "memorandum", "answers", "answer", "marking", "key",
    "theory", "exam", "paper", "nsc", "dbe", "grade", "gr",
    "cat", "caps", "p1", "p2", "p3", "question", "chunks",
    "nov", "november", "oct", "october", "jun", "june",
    "feb", "february", "mar", "march", "aug", "august",
    "sep", "september", "jan", "january", "jul", "july",
    "apr", "april", "dec", "december",
}

MONTH_CANONICAL = {
    "jan": "january",   "january": "january",
    "feb": "february",  "february": "february",
    "mar": "march",     "march": "march",
    "apr": "april",     "april": "april",
    "may": "may",
    "jun": "june",      "june": "june",
    "jul": "july",      "july": "july",
    "aug": "august",    "august": "august",
    "sep": "september", "september": "september",
    "oct": "october",   "october": "october",
    "nov": "november",  "november": "november",
    "dec": "december",  "december": "december",
}


def classify_file(filename):
    """
    Returns: "exam" | "memo" | "skip"

    Logic:
    - If filename contains a MEMO keyword → "memo"
    - Elif filename contains an EXAM keyword → "exam"
    - Else → "skip" (theory book, notes, unknown etc.)
    """
    lower = filename.lower()

    # Check memo first (memo files often also contain exam keywords)
    if any(kw in lower for kw in MEMO_KEYWORDS):
        return "memo"

    if any(kw in lower for kw in EXAM_KEYWORDS):
        return "exam"

    return "skip"


def extract_keywords(filename):
    """
    Pull year, month, term, paper number from a filename.
    Strips noise words so matching is based on signal only.
    e.g. "may_memo_2025.json"       → {"may", "2025"}
         "May_Theory_2025.json"     → {"may", "2025"}
         "Nov_Theory_2024_exam.json"→ {"november", "2024"}
    """
    name = filename.lower()
    name = re.sub(r"\.(json|pdf)$", "", name)
    name = re.sub(r"_(exam|chunks)$", "", name)
    tokens = re.split(r"[^a-z0-9]+", name)

    keywords = set()
    for token in tokens:
        if not token:
            continue
        if token in MONTH_CANONICAL:
            keywords.add(MONTH_CANONICAL[token])
            continue
        if re.match(r"^\d{4}$", token):
            keywords.add(token)
            continue
        if re.match(r"^(term|t)\d$", token):
            keywords.add(token)
            continue
        if re.match(r"^p\d$", token):
            keywords.add(token)
            continue
        if token in NOISE_WORDS:
            continue
        if len(token) >= 2:
            keywords.add(token)

    return keywords


# =========================
# 📋 TRACKING HELPERS
# Track processed pairs so re-running skips done work
# Format: { "exam_file.json": { "exam_done": true, "memo_merged": true, "memo_source": "..." } }
# =========================
def load_tracker():
    if not os.path.exists(TRACK_FILE):
        return {}
    try:
        with open(TRACK_FILE) as f:
            data = json.load(f)
            # Migrate old list format
            if isinstance(data, list):
                print("⚙️  Migrating old tracker format...")
                return {name: {"exam_done": False, "memo_merged": False} for name in data}
            return data
    except Exception:
        print(f"⚠️  Could not read {TRACK_FILE}, starting fresh.")
        return {}

def save_tracker(tracker):
    with open(TRACK_FILE, "w") as f:
        json.dump(tracker, f, indent=2)

def output_path_for(exam_chunk_file):
    return os.path.join(OUTPUT_FOLDER, exam_chunk_file.replace(".json", "_exam.json"))

def exam_output_exists(exam_chunk_file):
    return os.path.exists(output_path_for(exam_chunk_file))


# =========================
# 🔍 MATCH MEMO → EXAM
# =========================
def find_matching_exam(memo_filename, exam_chunk_files):
    """
    Match a memo file to its exam chunk file using shared keywords.
    Returns (best_match_filename, shared_keywords, score) or (None, set(), 0)
    """
    memo_kw = extract_keywords(memo_filename)
    if not memo_kw:
        return None, set(), 0

    best_file   = None
    best_shared = set()
    best_score  = 0

    for exam_file in exam_chunk_files:
        exam_kw = extract_keywords(exam_file)
        shared  = memo_kw & exam_kw
        if not shared:
            continue
        score = len(shared) / len(memo_kw | exam_kw)
        if score > best_score:
            best_score  = score
            best_shared = shared
            best_file   = exam_file

    if best_file and best_score > 0:
        return best_file, best_shared, best_score

    return None, set(), 0


# =========================
# 🧠 EXTRACT QUESTIONS
# =========================
def extract_questions(text):
    prompt = f"""
You are an expert parser for South African NSC (National Senior Certificate) CAT exam papers.

Extract the FULL structure EXACTLY as it appears. Do not skip any questions.

PAPER STRUCTURE:
  SECTION A (Q1–3): MCQ, Matching, True/False
  SECTION B (Q4–8): Open sub-questions e.g. 4.1, 4.7.1
  SECTION C (Q9–10): Scenario sub-questions e.g. 9.3.2

Return ONLY valid JSON (no markdown, no backticks):

[
  {{
    "section": "A",
    "section_title": "SECTION A",
    "section_instructions": "Answer ALL the questions.",
    "total_marks": 25,
    "questions": [
      {{
        "id": 1,
        "question_number": "1.1",
        "parent_question": "QUESTION 1: MULTIPLE-CHOICE QUESTIONS",
        "parent_context": null,
        "question": "Full question text exactly as written",
        "type": "mcq",
        "options": {{"A": "text", "B": "text", "C": "text", "D": "text"}},
        "marks": 1,
        "memo": ""
      }},
      {{
        "id": 2,
        "question_number": "2",
        "parent_question": "QUESTION 2: MATCHING ITEMS",
        "parent_context": "Choose a term from COLUMN B that matches COLUMN A.",
        "question": "Match all items in COLUMN A to COLUMN B",
        "type": "matching",
        "column_a": ["2.1 description one", "2.2 description two"],
        "column_b": ["A. term1", "B. term2", "R. term3"],
        "marks": 10,
        "memo": {{}}
      }},
      {{
        "id": 3,
        "question_number": "3.1",
        "parent_question": "QUESTION 3: TRUE/FALSE ITEMS",
        "parent_context": "Write TRUE or FALSE. If FALSE, correct the underlined word.",
        "question": "The CPU is responsible for processing instructions.",
        "type": "true_false",
        "options": null,
        "marks": 1,
        "memo": ""
      }},
      {{
        "id": 4,
        "question_number": "4.1",
        "parent_question": "QUESTION 4: SYSTEMS TECHNOLOGIES",
        "parent_context": null,
        "question": "State TWO disadvantages of using a wireless mouse.",
        "type": "open",
        "options": null,
        "marks": 2,
        "memo": ""
      }},
      {{
        "id": 5,
        "question_number": "4.7.1",
        "parent_question": "QUESTION 4: SYSTEMS TECHNOLOGIES",
        "parent_context": "One of the functions of an operating system is to manage programs and users.",
        "question": "Explain the difference between a single-user system and a multi-user system.",
        "type": "open",
        "options": null,
        "marks": 2,
        "memo": ""
      }}
    ]
  }}
]

RULES:
- question_number = exact label as printed (1.1, 4.7.1, 9.3.2)
- parent_question = heading of the parent question block
- parent_context  = scenario/context sentence before sub-questions, or null
- marks = number in (brackets), default 1
- memo  = ALWAYS leave as "" or {{}} — filled later from memo file
- NEVER invent memo answers

SECTION MAPPING:
  Q1–3  → section "A"
  Q4–8  → section "B"
  Q9–10 → section "C"

TYPES:
  mcq        → Q1: options dict A/B/C/D
  matching   → Q2: ONE question, column_a and column_b lists, memo = {{}}
  true_false → Q3: one per 3.x number
  open       → all Section B and C sub-questions

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
        print("❌ Invalid API key"); return []
    except RateLimitError:
        print("⚠️ Rate limit hit, waiting 10s..."); time.sleep(10); return []
    except Exception as e:
        print(f"❌ API error: {e}"); return []

    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse failed: {e}")
        print(f"📝 Preview: {content[:300]}")
        return []


# =========================
# 🧠 EXTRACT MEMO ANSWERS
# =========================
def extract_memo_answers(text):
    prompt = f"""
You are an expert parser for South African NSC CAT exam MEMO (marking guideline) files.

Extract ALL answers. Return ONLY valid JSON (no markdown):

{{
  "1.1": "C",
  "1.2": "A",
  "2.1": "R",
  "2.2": "E",
  "3.1": "True",
  "3.2": "False — secondary memory",
  "4.1": "Full answer text / alternative answer",
  "4.7.1": "Full answer text"
}}

RULES:
- Key   = question number exactly as printed (1.1, 4.7.1, 9.3.2)
- MCQ (Q1)       : value = correct letter only e.g. "C"
- Matching (Q2)  : extract each separately — "2.1": "R", "2.2": "E" etc.
- True/False (Q3): value = "True" OR "False — corrected word"
- Open (Q4–10)   : value = full expected answer, join marking points with " / "
- Extract EVERY answer — do NOT skip any
- Return ONLY the JSON object

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
        print("❌ Invalid API key"); return {}
    except RateLimitError:
        print("⚠️ Rate limit hit, waiting 10s..."); time.sleep(10); return {}
    except Exception as e:
        print(f"❌ API error: {e}"); return {}

    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse failed: {e}")
        print(f"📝 Preview: {content[:300]}")
        return {}


# =========================
# 🔗 MERGE SECTIONS
# =========================
def merge_sections(all_sections):
    merged = {}
    order  = {"A": 0, "B": 1, "C": 2}
    for section in all_sections:
        label = section.get("section", "A").strip().upper()
        if label not in merged:
            merged[label] = {
                "section":              label,
                "section_title":        section.get("section_title"),
                "section_instructions": section.get("section_instructions"),
                "total_marks":          section.get("total_marks"),
                "questions":            []
            }
        else:
            for field in ["section_title", "section_instructions", "total_marks"]:
                if not merged[label][field] and section.get(field):
                    merged[label][field] = section.get(field)
        merged[label]["questions"].extend(section.get("questions", []))
    return sorted(merged.values(), key=lambda s: order.get(s["section"], 99))


# =========================
# 🧹 DEDUPLICATE & RE-ID
# =========================
def deduplicate_and_renumber(sections):
    seen = set()
    qid  = 1
    for section in sections:
        clean = []
        for q in section.get("questions", []):
            key = f"{q.get('question_number','')}|{q.get('question','')[:80]}"
            if not q.get("question") or key in seen:
                continue
            seen.add(key)
            q["id"] = qid
            qid += 1
            clean.append(q)
        section["questions"] = clean
    return sections, qid - 1


# =========================
# 🔗 INJECT MEMO ANSWERS
# =========================
def inject_memo_answers(exam_data, memo_answers):
    matched   = 0
    unmatched = []

    # Build sub-number → full column_a label map for matching questions
    col_a_map = {}
    for section in exam_data.get("sections", []):
        for q in section.get("questions", []):
            if q.get("type") == "matching":
                for item in q.get("column_a", []):
                    parts = item.strip().split(" ", 1)
                    if parts:
                        col_a_map[parts[0].rstrip(".")] = item

    for section in exam_data.get("sections", []):
        for q in section.get("questions", []):
            q_num  = q.get("question_number", "").strip()
            q_type = q.get("type", "open")

            if q_type == "matching":
                memo_dict = {
                    label: memo_answers[num]
                    for num, label in col_a_map.items()
                    if num in memo_answers
                }
                if memo_dict:
                    q["memo"] = memo_dict
                    matched += len(memo_dict)
                else:
                    unmatched.append(q_num)
            else:
                if q_num in memo_answers:
                    q["memo"] = memo_answers[q_num]
                    matched += 1
                else:
                    unmatched.append(q_num)

    return exam_data, matched, unmatched


# =========================
# 📊 COUNT TYPES
# =========================
def count_types(sections):
    counts = {"mcq": 0, "matching": 0, "true_false": 0, "open": 0}
    for section in sections:
        for q in section.get("questions", []):
            t = q.get("type", "open")
            counts[t] = counts.get(t, 0) + 1
    return counts


# =========================
# 📥 LOAD CHUNKS
# =========================
def load_chunks(filename):
    path = os.path.join(PROCESSED_FOLDER, filename)
    try:
        with open(path) as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"  ❌ Failed to load {filename}: {e}")
        return []


# =========================
# 🔄 MAIN PROCESSOR
# =========================
def process():

    if not os.path.exists(PROCESSED_FOLDER):
        print(f"❌ Folder '{PROCESSED_FOLDER}' not found.")
        return

    tracker = load_tracker()

    all_json = [
        f for f in sorted(os.listdir(PROCESSED_FOLDER))
        if f.endswith(".json")
        and f not in ["metadata.json", "chunk_ids.json", "processed_files.json", "processed_exams.json"]
    ]

    # ── Classify every file ───────────────────────────────
    exam_files = []
    memo_files = []
    skipped    = []

    for f in all_json:
        kind = classify_file(f)
        if kind == "exam":
            exam_files.append(f)
        elif kind == "memo":
            memo_files.append(f)
        else:
            skipped.append(f)

    # ── Print pre-flight summary ──────────────────────────
    print(f"\n{'='*50}")
    print(f"📂 Files in processed/ : {len(all_json)}")
    print(f"📄 Exam papers         : {len(exam_files)}")
    for f in exam_files:
        status = "✅ done" if tracker.get(f, {}).get("exam_done") and exam_output_exists(f) else "🔄 pending"
        print(f"     {status}  → {f}")
    print(f"📝 Memo files          : {len(memo_files)}")
    for f in memo_files:
        status = "✅ merged" if tracker.get(f, {}).get("memo_merged") else "🔄 pending"
        print(f"     {status}  → {f}")
    print(f"🚫 Ignored (not exam)  : {len(skipped)}")
    for f in skipped:
        print(f"     → {f}")
    print(f"{'='*50}\n")

    if not exam_files and not memo_files:
        print("⚠️  No exam or memo files found.")
        print("💡 Name your files with keywords like:")
        print("   Exam : nov, may, june, theory, paper, exam, p1, p2, term, nsc, dbe, cat")
        print("   Memo : memo, memorandum, answers, marking")
        return

    # ══════════════════════════════════════════════════════
    # STEP 1 — Extract questions from exam papers
    # ══════════════════════════════════════════════════════
    pending_exams = [
        f for f in exam_files
        if not (tracker.get(f, {}).get("exam_done") and exam_output_exists(f))
    ]

    if pending_exams:
        print(f"📄 STEP 1: Extracting questions from {len(pending_exams)} exam paper(s)...\n")
    else:
        print("📄 STEP 1: All exam papers already extracted.\n")

    for idx, exam_file in enumerate(pending_exams, 1):
        print(f"  [{idx}/{len(pending_exams)}] 📄 {exam_file}")

        chunks = load_chunks(exam_file)
        if not chunks:
            print(f"    ⚠️  Empty or invalid — skipping.\n")
            continue

        all_raw_sections = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("content", "").strip()
            if not text:
                continue
            print(f"    🔍 Chunk {i + 1}/{len(chunks)}...")
            extracted = extract_questions(text)
            if extracted:
                all_raw_sections.extend(extracted)
            time.sleep(1.5)

        if not all_raw_sections:
            print(f"    ⚠️  Nothing extracted — will retry next run.\n")
            continue

        sections, total_q = deduplicate_and_renumber(merge_sections(all_raw_sections))
        type_counts        = count_types(sections)

        out_path = output_path_for(exam_file)
        with open(out_path, "w") as f:
            json.dump({
                "source":          exam_file,
                "total_questions": total_q,
                "type_breakdown":  type_counts,
                "memo_merged":     False,
                "memo_source":     None,
                "sections":        sections
            }, f, indent=2)

        print(f"    💾 Saved  : {out_path}")
        print(f"    📊 {total_q} questions | "
              f"MCQ: {type_counts['mcq']} | Matching: {type_counts['matching']} | "
              f"T/F: {type_counts['true_false']} | Open: {type_counts['open']}\n")

        if exam_file not in tracker:
            tracker[exam_file] = {}
        tracker[exam_file]["exam_done"]   = True
        tracker[exam_file]["memo_merged"] = False
        save_tracker(tracker)

    # ══════════════════════════════════════════════════════
    # STEP 2 — Extract & merge memo answers
    # ══════════════════════════════════════════════════════
    pending_memos = [
        f for f in memo_files
        if not tracker.get(f, {}).get("memo_merged")
    ]

    if pending_memos:
        print(f"\n📝 STEP 2: Merging {len(pending_memos)} memo file(s) into exams...\n")
    else:
        print("📝 STEP 2: All memos already merged.\n")

    # Available exam output files for matching
    available_exam_outputs = [
        f for f in os.listdir(OUTPUT_FOLDER)
        if f.endswith("_exam.json")
    ]

    for idx, memo_file in enumerate(pending_memos, 1):
        print(f"  [{idx}/{len(pending_memos)}] 📝 {memo_file}")

        memo_kw = extract_keywords(memo_file)
        print(f"    🔑 Keywords : {sorted(memo_kw)}")

        matched_exam, shared_kw, score = find_matching_exam(memo_file, exam_files)

        if not matched_exam:
            print(f"    ⚠️  No matching exam found.")
            print(f"    💡 Available exam files: {exam_files}")
            print(f"    💡 Make sure the exam and memo share year/month in their filenames.\n")
            continue

        exam_output = output_path_for(matched_exam)

        if not os.path.exists(exam_output):
            print(f"    ⚠️  Exam output not found: {exam_output}")
            print(f"    💡 Run this script first without the memo to extract the exam.\n")
            continue

        print(f"    🔗 Matched  : {matched_exam}")
        print(f"    🤝 Shared   : {sorted(shared_kw)}  ({score:.0%} match)")

        # ── Extract memo answers ──────────────────────────
        memo_chunks = load_chunks(memo_file)
        if not memo_chunks:
            print(f"    ⚠️  Memo empty or invalid — skipping.\n")
            continue

        all_memo_answers = {}
        for i, chunk in enumerate(memo_chunks):
            text = chunk.get("content", "").strip()
            if not text:
                continue
            print(f"    🔍 Memo chunk {i + 1}/{len(memo_chunks)}...")
            answers = extract_memo_answers(text)
            all_memo_answers.update(answers)
            time.sleep(1.5)

        if not all_memo_answers:
            print(f"    ⚠️  No answers extracted from memo — skipping.\n")
            continue

        print(f"    ✅ {len(all_memo_answers)} answers extracted")

        # ── Load exam, inject, save ───────────────────────
        with open(exam_output) as f:
            exam_data = json.load(f)

        updated_exam, matched_count, unmatched = inject_memo_answers(exam_data, all_memo_answers)

        updated_exam["memo_merged"]        = True
        updated_exam["memo_source"]        = memo_file
        updated_exam["memo_answers_total"] = len(all_memo_answers)
        updated_exam["memo_matched"]       = matched_count
        updated_exam["memo_unmatched"]     = unmatched

        with open(exam_output, "w") as f:
            json.dump(updated_exam, f, indent=2)

        print(f"    💾 Updated  : {exam_output}")
        print(f"    📊 Merged   : {matched_count} answers into exam")
        if unmatched:
            print(f"    ⚠️  Unmatched: {len(unmatched)} — {unmatched[:10]}"
                  f"{'...' if len(unmatched) > 10 else ''}")
        print()

        if memo_file not in tracker:
            tracker[memo_file] = {}
        tracker[memo_file]["memo_merged"] = True
        tracker[memo_file]["memo_source"] = matched_exam
        save_tracker(tracker)

        # Also mark on the exam entry
        if matched_exam not in tracker:
            tracker[matched_exam] = {}
        tracker[matched_exam]["memo_merged"] = True
        tracker[matched_exam]["memo_source"] = memo_file
        save_tracker(tracker)

    print("✅ All done.")


# =========================
# ▶️ RUN
# =========================
if __name__ == "__main__":
    process()