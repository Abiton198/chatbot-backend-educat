import os
import re
import json
import time
from groq import Groq, RateLimitError, AuthenticationError
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY is not set.")
client = Groq(api_key=api_key)

PROCESSED_FOLDER = "processed"
OUTPUT_FOLDER    = "exams"
TRACK_FILE       = "processed_exams.json"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

WINDOW_CHARS  = 6000
OVERLAP_CHARS = 500

# ── normalisation ────────────────────────────────────────
def normalize_key(filename):
    name = filename.strip().lower()
    name = re.sub(r'\s+\.', '.', name)
    return name

# ── classification ───────────────────────────────────────
EXAM_KEYWORDS = ["exam","paper","question","theory","p1","p2","p3",
                 "nov","november","may","june","feb","february","march","mar",
                 "aug","august","sep","september","oct","october","term",
                 "trial","nsc","dbe","cat"]
MEMO_KEYWORDS = ["memo","memorandum","answers","answer_key","marking"]
NOISE_WORDS   = {"memo","memorandum","answers","answer","marking","key",
                 "theory","exam","paper","nsc","dbe","grade","gr","cat",
                 "caps","p1","p2","p3","question","chunks","nov","november",
                 "oct","october","jun","june","feb","february","mar","march",
                 "aug","august","sep","september","jan","january","jul","july",
                 "apr","april","dec","december"}
MONTH_CANONICAL = {
    "jan":"january","january":"january","feb":"february","february":"february",
    "mar":"march","march":"march","apr":"april","april":"april","may":"may",
    "jun":"june","june":"june","jul":"july","july":"july",
    "aug":"august","august":"august","sep":"september","september":"september",
    "oct":"october","october":"october","nov":"november","november":"november",
    "dec":"december","december":"december"}

def classify_file(filename):
    lower = filename.lower()
    if any(kw in lower for kw in MEMO_KEYWORDS): return "memo"
    if any(kw in lower for kw in EXAM_KEYWORDS): return "exam"
    return "skip"

def extract_keywords(filename):
    name = filename.lower().strip()
    name = re.sub(r'\s+\.', '.', name)
    name = re.sub(r"\.(json|pdf)$", "", name)
    name = re.sub(r"_(exam|chunks)$", "", name)
    tokens = re.split(r"[^a-z0-9]+", name)
    keywords = set()
    for token in tokens:
        if not token: continue
        if token in MONTH_CANONICAL: keywords.add(MONTH_CANONICAL[token]); continue
        if re.match(r"^\d{4}$", token): keywords.add(token); continue
        if re.match(r"^(term|t)\d$", token): keywords.add(token); continue
        if re.match(r"^p\d$", token): keywords.add(token); continue
        if token in NOISE_WORDS: continue
        if len(token) >= 2: keywords.add(token)
    return keywords

# ── tracker ──────────────────────────────────────────────
def load_tracker():
    if not os.path.exists(TRACK_FILE): return {}
    try:
        with open(TRACK_FILE) as f: data = json.load(f)
    except Exception:
        print("⚠️  Could not read tracker, starting fresh."); return {}
    if isinstance(data, list):
        data = {n: {"exam_done": False, "memo_merged": False} for n in data}
    normalised = {}
    for raw_key, value in data.items():
        nk = normalize_key(raw_key)
        if nk not in normalised:
            normalised[nk] = {"exam_done": False, "memo_merged": False, "memo_source": None}
        if value.get("exam_done"):    normalised[nk]["exam_done"]   = True
        if value.get("memo_merged"):  normalised[nk]["memo_merged"] = True
        if value.get("memo_source"):  normalised[nk]["memo_source"] = value["memo_source"]
    if normalised != data:
        print(f"⚙️  Normalised tracker: {len(data)} raw → {len(normalised)} unique keys")
        with open(TRACK_FILE, "w") as f: json.dump(normalised, f, indent=2)
    return normalised

def save_tracker(tracker):
    with open(TRACK_FILE, "w") as f: json.dump(tracker, f, indent=2)

def tracker_get(tracker, filename): return tracker.get(normalize_key(filename), {})
def tracker_set(tracker, filename, key, value):
    nk = normalize_key(filename)
    if nk not in tracker: tracker[nk] = {}
    tracker[nk][key] = value

def output_path_for(exam_chunk_file):
    stem = re.sub(r"\.json$", "", normalize_key(exam_chunk_file))
    return os.path.join(OUTPUT_FOLDER, stem + "_exam.json")

def exam_output_exists(f): return os.path.exists(output_path_for(f))

# ── matching ─────────────────────────────────────────────
def find_matching_exam(memo_filename, exam_chunk_files):
    memo_kw = extract_keywords(memo_filename)
    if not memo_kw: return None, set(), 0
    best_file, best_shared, best_score = None, set(), 0
    for exam_file in exam_chunk_files:
        exam_kw = extract_keywords(exam_file)
        shared  = memo_kw & exam_kw
        if not shared: continue
        score = len(shared) / len(memo_kw | exam_kw)
        if score > best_score:
            best_score, best_shared, best_file = score, shared, exam_file
    return (best_file, best_shared, best_score) if best_file else (None, set(), 0)

# ── chunk stitching ──────────────────────────────────────
def stitch_chunks(chunks):
    return "\n".join(c.get("content","").strip() for c in chunks if c.get("content","").strip())

def sliding_windows(text, window=WINDOW_CHARS, overlap=OVERLAP_CHARS):
    start = 0
    while start < len(text):
        end = start + window
        yield text[start:end]
        if end >= len(text): break
        start = end - overlap

# ── LLM: extract questions ───────────────────────────────
def extract_questions(text):
    prompt = f"""You are an expert parser for South African NSC CAT exam papers.
Extract all questions. Return ONLY a valid JSON array, no markdown, no backticks.

Format:
[
  {{
    "section": "A",
    "section_title": "SECTION A",
    "section_instructions": "Answer ALL questions.",
    "total_marks": 25,
    "questions": [
      {{
        "id": 1,
        "question_number": "1.1",
        "parent_question": "QUESTION 1: MULTIPLE-CHOICE QUESTIONS",
        "parent_context": null,
        "question": "Exact question text",
        "type": "mcq",
        "options": {{"A": "Trackball", "B": "Stylus", "C": "Touchpad", "D": "Mouse"}},
        "marks": 1,
        "memo": ""
      }},
      {{
        "id": 2,
        "question_number": "2",
        "parent_question": "QUESTION 2: MATCHING ITEMS",
        "parent_context": "Choose from COLUMN B.",
        "question": "Match COLUMN A to COLUMN B",
        "type": "matching",
        "column_a": ["2.1 Integration of two or more technologies"],
        "column_b": ["A. #Value!", "R. convergence"],
        "marks": 10,
        "memo": {{}}
      }},
      {{
        "id": 3,
        "question_number": "3.1",
        "parent_question": "QUESTION 3: TRUE/FALSE ITEMS",
        "parent_context": "Write TRUE or FALSE.",
        "question": "The CPU processes instructions.",
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
        "question": "State TWO disadvantages of a wireless mouse.",
        "type": "open",
        "options": null,
        "marks": 2,
        "memo": ""
      }}
    ]
  }}
]

RULES:
- question_number = exact label (1.1, 2.1, 3.2, 4.7.1, 9.3.2, 10.3.2)
- MCQ options MUST be dict with keys "A","B","C","D" and full text values
- matching: ONE object, column_a = ["2.1 description", ...], column_b = ["A. term", ...]
- true_false: one object per 3.x sub-number
- marks = integer from (N) at end, default 1
- memo = always "" (non-matching) or {{}} (matching), NEVER fill answers
- Section mapping: Q1-3 → "A", Q4-8 → "B", Q9-10 → "C"

TEXT:
{text}"""
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            temperature=0)
    except AuthenticationError: print("❌ Invalid API key"); return []
    except RateLimitError: print("⚠️ Rate limit, waiting 15s..."); time.sleep(15); return []
    except Exception as e: print(f"❌ API error: {e}"); return []
    content = r.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"): content = content[4:]
        content = content.strip()
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse failed: {e}\n📝 {content[:300]}"); return []

# ── LLM: extract memo ────────────────────────────────────
def extract_memo_answers(text):
    prompt = f"""You are an expert parser for South African NSC CAT exam MEMO files.
Extract ALL answers. Return ONLY a valid JSON object, no markdown, no backticks.

Format:
{{
  "1.1": "C",
  "1.2": "C",
  "2.1": "R",
  "3.1": "True",
  "3.2": "False - Braille",
  "4.1": "Can be lost/stolen / Batteries need replacing / Signal interference",
  "4.7.1": "Single-user: one user at a time / Multi-user: multiple users simultaneously"
}}

STRICT RULES:
- Key = question number EXACTLY as printed (1.1, 2.3, 3.5, 4.7.1, 9.3.2, 10.3.2)
- Q1.x MCQ answers    : single letter only — "C" not "C Touchpad"
- Q2.x matching      : single letter only — "R" not "R convergence"  
- Q3.x true/false    : "True" OR "False - corrected word"
- Q4-10 open answers : full text, join alternatives with " / "
- Extract EVERY answer, do NOT skip any
- Return ONLY the JSON object

TEXT:
{text}"""
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            temperature=0)
    except AuthenticationError: print("❌ Invalid API key"); return {}
    except RateLimitError: print("⚠️ Rate limit, waiting 15s..."); time.sleep(15); return {}
    except Exception as e: print(f"❌ API error: {e}"); return {}
    content = r.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"): content = content[4:]
        content = content.strip()
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse failed: {e}\n📝 {content[:300]}"); return {}

# ── merge & dedup ─────────────────────────────────────────
def merge_sections(all_sections):
    merged = {}
    order  = {"A": 0, "B": 1, "C": 2}
    for section in all_sections:
        label = section.get("section", "A").strip().upper()
        if label not in merged:
            merged[label] = {**section, "questions": []}
        for field in ["section_title", "section_instructions", "total_marks"]:
            if not merged[label].get(field) and section.get(field):
                merged[label][field] = section[field]
        merged[label]["questions"].extend(section.get("questions", []))
    return sorted(merged.values(), key=lambda s: order.get(s["section"], 99))

def question_completeness(q):
    score = len(q.get("question") or "")
    if isinstance(q.get("options"), dict): score += len(q["options"]) * 20
    if q.get("column_a"): score += len(q["column_a"]) * 10
    return score

def deduplicate_and_renumber(sections):
    best = {}
    for section in sections:
        for q in section.get("questions", []):
            qn = q.get("question_number", "").strip()
            if not qn or not q.get("question"): continue
            if qn not in best or question_completeness(q) > question_completeness(best[qn]):
                best[qn] = dict(q)
                best[qn]["_section"] = section["section"]

    section_map = {}
    qid = 1
    def sort_key(x):
        parts = x.split(".")
        try: return [int(p) for p in parts]
        except: return parts
    for qn in sorted(best.keys(), key=sort_key):
        q = best[qn]
        sec = q.pop("_section", "A")
        q["id"] = qid; qid += 1
        section_map.setdefault(sec, []).append(q)

    order = {"A": 0, "B": 1, "C": 2}
    sec_meta = {s["section"]: s for s in sections}
    rebuilt = []
    for sec_label in sorted(section_map.keys(), key=lambda x: order.get(x, 99)):
        meta = sec_meta.get(sec_label, {})
        rebuilt.append({
            "section":              sec_label,
            "section_title":        meta.get("section_title"),
            "section_instructions": meta.get("section_instructions"),
            "total_marks":          meta.get("total_marks"),
            "questions":            section_map[sec_label]
        })
    return rebuilt, qid - 1

# ── inject memo ───────────────────────────────────────────
def inject_memo_answers(exam_data, memo_answers):
    matched, unmatched = 0, []
    for section in exam_data.get("sections", []):
        for q in section.get("questions", []):
            q_num  = q.get("question_number", "").strip()
            q_type = q.get("type", "open")
            if q_type == "matching":
                memo_dict = {}
                for item in q.get("column_a", []):
                    m = re.match(r"^(\d+\.\d+)", item.strip())
                    if m:
                        sub = m.group(1)
                        if sub in memo_answers:
                            memo_dict[item] = memo_answers[sub]
                            matched += 1
                if memo_dict: q["memo"] = memo_dict
                else: unmatched.append(q_num)
            else:
                if q_num in memo_answers:
                    q["memo"] = memo_answers[q_num]; matched += 1
                else:
                    unmatched.append(q_num)
    return exam_data, matched, unmatched

def validate_memo_injection(exam_data, memo_answers):
    print("\n    📋 MEMO VALIDATION — Section A spot-check:")
    for section in exam_data.get("sections", []):
        if section["section"] != "A": continue
        for q in section.get("questions", []):
            qn   = q.get("question_number","")
            memo = q.get("memo","")
            raw  = memo_answers.get(qn, "NOT IN MEMO")
            ok   = "✅" if memo and memo == raw else "⚠️ "
            print(f"      {ok} {qn}: stored={repr(str(memo)):30s} raw={repr(raw)}")

def count_types(sections):
    counts = {"mcq": 0, "matching": 0, "true_false": 0, "open": 0}
    for section in sections:
        for q in section.get("questions", []):
            t = q.get("type","open"); counts[t] = counts.get(t,0) + 1
    return counts

def load_chunks(filename):
    path = os.path.join(PROCESSED_FOLDER, filename)
    try:
        with open(path) as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"  ❌ Failed to load {filename}: {e}"); return []

# ── main ──────────────────────────────────────────────────
def process():
    if not os.path.exists(PROCESSED_FOLDER):
        print(f"❌ Folder '{PROCESSED_FOLDER}' not found."); return

    tracker = load_tracker()
    SKIP = {"metadata.json","chunk_ids.json","processed_files.json","processed_exams.json"}
    all_json = [f for f in sorted(os.listdir(PROCESSED_FOLDER))
                if f.endswith(".json") and f not in SKIP]

    exam_files, memo_files, skipped = [], [], []
    for f in all_json:
        kind = classify_file(f)
        if kind == "exam": exam_files.append(f)
        elif kind == "memo": memo_files.append(f)
        else: skipped.append(f)

    print(f"\n{'='*55}")
    print(f"📂 Files: {len(all_json)}  |  Exams: {len(exam_files)}  |  Memos: {len(memo_files)}  |  Skipped: {len(skipped)}")
    for f in exam_files:
        e = tracker_get(tracker, f)
        s = "✅+memo" if e.get("exam_done") and e.get("memo_merged") else "✅ done" if e.get("exam_done") else "🔄"
        print(f"  {s}  {f}")
    for f in memo_files:
        s = "✅ merged" if tracker_get(tracker,f).get("memo_merged") else "🔄"
        print(f"  {s}  {f}")
    print(f"{'='*55}\n")

    # STEP 1
    pending_exams = [f for f in exam_files
                     if not (tracker_get(tracker,f).get("exam_done") and exam_output_exists(f))]
    print(f"📄 STEP 1: {len(pending_exams)} exam(s) to extract\n")

    for idx, exam_file in enumerate(pending_exams, 1):
        print(f"  [{idx}/{len(pending_exams)}] {exam_file}")
        chunks = load_chunks(exam_file)
        if not chunks: print("    ⚠️  Empty\n"); continue

        full_text = stitch_chunks(chunks)
        windows   = list(sliding_windows(full_text))
        print(f"    🔗 {len(chunks)} chunks → {len(full_text)} chars → {len(windows)} windows")

        all_raw = []
        for i, window in enumerate(windows):
            print(f"    🔍 Window {i+1}/{len(windows)}...")
            extracted = extract_questions(window)
            if extracted: all_raw.extend(extracted)
            time.sleep(1.5)

        if not all_raw: print("    ⚠️  Nothing extracted\n"); continue

        sections, total_q = deduplicate_and_renumber(merge_sections(all_raw))
        type_counts = count_types(sections)
        out_path = output_path_for(exam_file)
        with open(out_path, "w") as f:
            json.dump({"source":exam_file,"total_questions":total_q,
                       "type_breakdown":type_counts,"memo_merged":False,
                       "memo_source":None,"sections":sections}, f, indent=2)
        print(f"    💾 {out_path}  |  {total_q}q  MCQ:{type_counts['mcq']} Match:{type_counts['matching']} T/F:{type_counts['true_false']} Open:{type_counts['open']}\n")
        tracker_set(tracker, exam_file, "exam_done",   True)
        tracker_set(tracker, exam_file, "memo_merged", False)
        save_tracker(tracker)

    # STEP 2
    pending_memos = [f for f in memo_files if not tracker_get(tracker,f).get("memo_merged")]
    print(f"\n📝 STEP 2: {len(pending_memos)} memo(s) to merge\n")

    for idx, memo_file in enumerate(pending_memos, 1):
        print(f"  [{idx}/{len(pending_memos)}] {memo_file}")
        memo_kw = extract_keywords(memo_file)
        print(f"    🔑 {sorted(memo_kw)}")

        matched_exam, shared_kw, score = find_matching_exam(memo_file, exam_files)
        if not matched_exam: print(f"    ⚠️  No match. Exams: {exam_files}\n"); continue

        exam_output = output_path_for(matched_exam)
        if not os.path.exists(exam_output): print(f"    ⚠️  Missing: {exam_output}\n"); continue
        if not tracker_get(tracker, matched_exam).get("exam_done"):
            print(f"    ⚠️  Exam not yet extracted\n"); continue

        print(f"    🔗 → {matched_exam}  ({score:.0%} match)")

        memo_chunks = load_chunks(memo_file)
        if not memo_chunks: print("    ⚠️  Memo empty\n"); continue

        full_memo = stitch_chunks(memo_chunks)
        memo_wins = list(sliding_windows(full_memo))
        print(f"    🔗 {len(memo_chunks)} chunks → {len(memo_wins)} windows")

        all_memo_answers = {}
        for i, window in enumerate(memo_wins):
            print(f"    🔍 Memo window {i+1}/{len(memo_wins)}...")
            answers = extract_memo_answers(window)
            for k, v in answers.items():
                if k not in all_memo_answers:  # first window wins
                    all_memo_answers[k] = v
            time.sleep(1.5)

        if not all_memo_answers: print("    ⚠️  No answers\n"); continue
        print(f"    ✅ {len(all_memo_answers)} answers extracted")

        sec_a = {k:v for k,v in all_memo_answers.items() if re.match(r"^[123]\.", k)}
        print(f"    🔎 Section A: {sec_a}")

        with open(exam_output) as f: exam_data = json.load(f)
        updated, matched_count, unmatched = inject_memo_answers(exam_data, all_memo_answers)
        validate_memo_injection(updated, all_memo_answers)

        updated["memo_merged"]        = True
        updated["memo_source"]        = memo_file
        updated["memo_answers_total"] = len(all_memo_answers)
        updated["memo_matched"]       = matched_count
        updated["memo_unmatched"]     = unmatched

        with open(exam_output, "w") as f: json.dump(updated, f, indent=2)
        print(f"\n    💾 {exam_output}  |  merged {matched_count}/{len(all_memo_answers)}")
        if unmatched: print(f"    ⚠️  Unmatched: {unmatched[:15]}{'...' if len(unmatched)>15 else ''}")
        print()

        tracker_set(tracker, memo_file,    "memo_merged", True)
        tracker_set(tracker, memo_file,    "memo_source", matched_exam)
        tracker_set(tracker, matched_exam, "memo_merged", True)
        tracker_set(tracker, matched_exam, "memo_source", memo_file)
        save_tracker(tracker)

    print("✅ All done.")

if __name__ == "__main__":
    process()