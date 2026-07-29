"""
extract_exams_v2.py  —  EduCAT Universal Exam Extraction Pipeline

Handles ALL NSC subjects: Mathematics, Physical Sciences, Life Sciences, Geography,
History, Accounting, Economics, Business Studies, CAT, IT, Engineering Graphics & Design,
Languages, and ANY future CAPS-aligned subject.

Key improvements:
- Universal question type detection (not hardcoded for CAT)
- Diagram & table reference preservation
- Mathematical formula preservation (LaTeX-style)
- Smart window splitting at section/question boundaries
- Subject-agnostic memo extraction
- Enhanced validation with multiple fallback strategies
"""

import os
import re
import json
import time
from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")

MODEL_NAME = os.getenv("EXTRACTION_MODEL", "llama-3.3-70b-versatile")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
WINDOW_CHARS = int(os.getenv("WINDOW_CHARS", "12000"))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", "1000"))

llm = ChatGroq(model=MODEL_NAME, temperature=0, groq_api_key=API_KEY)

PROCESSED_FOLDER = "processed"
OUTPUT_FOLDER = "exams"
TRACK_FILE = "processed_exams.json"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class QuestionType(Enum):
    MCQ = "mcq"
    MATCHING = "matching"
    TRUE_FALSE = "true_false"
    OPEN = "open"
    CALCULATION = "calculation"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    COMPREHENSION = "comprehension"
    DIAGRAM_LABEL = "diagram_label"
    TABLE_COMPLETION = "table_completion"
    MULTI_PART = "multi_part"
    UNKNOWN = "unknown"


@dataclass
class DiagramRef:
    diagram_id: str
    caption: str = ""
    description: str = ""
    related_question: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class TableRef:
    table_id: str
    headers: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    caption: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class Question:
    id: int
    question_number: str
    parent_question: str = ""
    parent_context: Optional[str] = None
    question: str = ""
    question_type: str = "open"
    marks: int = 1
    memo: Any = ""
    options: Optional[dict] = None
    column_a: Optional[list] = None
    column_b: Optional[list] = None
    diagram_refs: list = field(default_factory=list)
    table_refs: list = field(default_factory=list)
    formula: Optional[str] = None
    instructions: Optional[str] = None
    section: str = "A"
    sub_parts: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["diagram_refs"] = [r.to_dict() for r in self.diagram_refs]
        d["table_refs"] = [t.to_dict() for t in self.table_refs]
        return d


@dataclass
class Section:
    section: str
    section_title: str = ""
    section_instructions: str = ""
    total_marks: Optional[int] = None
    questions: list = field(default_factory=list)

    def to_dict(self):
        return {
            "section": self.section,
            "section_title": self.section_title,
            "section_instructions": self.section_instructions,
            "total_marks": self.total_marks,
            "questions": [q.to_dict() for q in self.questions]
        }


@dataclass
class ExamMetadata:
    subject: str = ""
    subject_code: str = ""
    grade: str = ""
    year: str = ""
    paper_number: str = ""
    exam_type: str = ""
    language: str = "English"
    time_allocation: str = ""
    total_marks: Optional[int] = None
    instructions: str = ""

    def to_dict(self):
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════
def render_prompt(template: str, text: str) -> str:
    if "{text}" not in template:
        raise ValueError("template is missing the {text} placeholder")
    return template.replace("{text}", text)


EXTRACTION_SYSTEM_PROMPT = """You are an expert parser for South African NSC exam papers across ALL subjects.

Extract exam questions losslessly. Preserve exact wording, options, formulas, diagram references, tables.

SUBJECTS: Mathematics, Physical Sciences, Life Sciences, Geography, History, Accounting, Economics, Business Studies, CAT, IT, Engineering Graphics & Design, Languages, and any other CAPS subject.

QUESTION TYPE RULES:
- "Choose the correct answer" / "A, B, C, D" -> mcq
- "Match COLUMN A with COLUMN B" -> matching
- "Write TRUE or FALSE" -> true_false
- "Show ALL calculations" / "Calculate" / "Determine" -> calculation
- "Label the diagram" / "Study the diagram" -> diagram_label
- "Complete the table" -> table_completion
- "Discuss" / "Explain" / "Evaluate" / "Analyse" (long, >10 marks) -> essay
- "Briefly explain" / "State" / "Name" / "List" (short, <=5 marks) -> short_answer
- "Read the passage" / "Refer to the text" -> comprehension
- Default -> open

SOURCE MATERIAL — parent_context (READ THIS CAREFULLY):
Exam papers print shared material ONCE above a group of questions: a reading
passage, a literary extract, a case study, a newspaper source, a described
cartoon, a scenario, a data set, or a set of given values. Every question in
that group is unanswerable without it.

- Copy that material VERBATIM into "parent_context".
- Repeat it on EVERY question in the group, including every sub-question.
- Never write "see above", "refer to the passage", or a summary. Copy the text.
- Use null ONLY when the question is genuinely self-contained.
- If the paper says "Read the extract below and answer the questions", the
  extract belongs in parent_context for all questions that follow it.

CRITICAL:
1. NEVER summarize or rephrase
2. NEVER skip questions
3. Preserve formulas: $...$ inline, $$...$$ display
4. Question numbers must match exactly (1.1, 2.3.1, 4.7.1, 10.3.2)
5. Extract marks from brackets after each question
6. Handle sub-parts with dot notation
7. parent_context must contain the actual source text, never a reference to it
"""

EXTRACTION_PROMPT_TEMPLATE = """
Extract ALL questions from this NSC exam paper. Return ONLY valid JSON.

SOURCE MATERIAL — how to handle passages:
Exam papers print shared material once above a group of questions: a reading
passage, extract, case study, source, cartoon description, scenario or data set.

Put each piece of shared material ONCE in the top-level "contexts" object,
keyed by the question group number it belongs to ("1", "2", "3").
Copy it VERBATIM — do not summarise, do not truncate, do not write "see above".

Then on each question, set "context_ref" to that key. NEVER repeat the passage
text inside a question. Use null for context_ref when a question is
self-contained.

OUTPUT FORMAT:
{
  "metadata": {
    "subject": "English Home Language",
    "grade": "9",
    "year": "2022",
    "paper_number": "1",
    "exam_type": "Control Test",
    "total_marks": 50
  },
  "contexts": {
    "1": "Read the text below and answer the set questions.\\n\\nAbout the text: Holes is an award-winning young adult's novel by Louis Sachar...\\n\\n[the ENTIRE passage, every paragraph, verbatim]"
  },
  "sections": [
    {
      "section": "A",
      "section_title": "SECTION A: COMPREHENSION",
      "section_instructions": "Answer ALL questions.",
      "total_marks": 30,
      "questions": [
        {
          "id": 1,
          "question_number": "1.1",
          "parent_question": "QUESTION 1",
          "context_ref": "1",
          "instructions": "Refer to paragraph 1.",
          "question": "What is ironic about the name \\"Camp Green Lake\\"?",
          "question_type": "comprehension",
          "marks": 2,
          "memo": "",
          "options": null,
          "column_a": null,
          "column_b": null,
          "diagram_refs": [],
          "table_refs": [],
          "formula": null,
          "section": "A",
          "sub_parts": []
        },
        {
          "id": 5,
          "question_number": "1.5",
          "parent_question": "QUESTION 1",
          "context_ref": "1",
          "instructions": "Write down only the correct LETTER.",
          "question": "Which wild reptile at Camp Green Lake do the campers fear the most?",
          "question_type": "mcq",
          "marks": 1,
          "memo": "",
          "options": {"A": "Scorpions", "B": "Rattlesnakes", "C": "Yellow-spotted lizards", "D": "Spiders"},
          "column_a": null,
          "column_b": null,
          "diagram_refs": [],
          "table_refs": [],
          "formula": null,
          "section": "A",
          "sub_parts": []
        }
      ]
    }
  ]
}

CRITICAL:
1. NEVER summarize or rephrase question text
2. NEVER skip questions
3. Preserve formulas: $...$ inline, $$...$$ display
4. Question numbers must match exactly (1.1, 2.3.1, 4.7.1)
5. Extract marks from the brackets after each question
6. "Refer to paragraph N" lines go in "instructions", not "question"
7. The passage goes in "contexts" ONCE, never inside a question

TEXT TO EXTRACT:
{text}
"""

MEMO_PROMPT_TEMPLATE = """
You are parsing an NSC exam MEMORANDUM for ANY subject.
Extract ALL answers with exact question numbers.

RULES:
- Key = exact question number (1.1, 2.3, 3.5.1, 4.7.1, 10.3.2)
- MCQ: single letter only ("C")
- Matching: single letter only ("R")
- True/False: "True" OR "False - corrected word"
- Calculation: full working + final answer
- Open/Essay: bullet points for marking rubric
- Multi-part: dot notation ("4.1.1": "answer")
- Alternative answers: separate with " OR "

OUTPUT FORMAT:
{
  "subject": "detected",
  "year": "2024",
  "paper": "1",
  "answers": {
    "1.1": "C",
    "2.1": "R",
    "3.1": "True",
    "5.1": "x = 4 (2 marks for isolating, 1 for answer)",
    "8": "Introduction (2)\\n- Context (1)..."
  }
}

TEXT TO EXTRACT:
{text}
"""

MIN_PASSAGE_CHARS = 160


def _group_keys(q):
    """Keys a question can be matched to its passage group by."""
    keys = set()
    label = (q.parent_question or "").strip().upper()
    if label:
        keys.add(label)
    num = (q.question_number or "").strip()
    if num:
        keys.add(f"Q{num.split('.')[0]}")
    return keys


def _collect_passages(sections):
    """Harvest every passage the LLM did manage to return, keyed by group."""
    found = {}
    for sec in sections:
        for q in sec.questions:
            for candidate in (q.parent_context, q.instructions):
                text = (candidate or "").strip()
                if len(text) >= MIN_PASSAGE_CHARS:
                    for key in _group_keys(q):
                        found.setdefault(key, text)
    return found


def expand_contexts(result, raw_text=""):
    """
    Turn {"contexts": {...}, "sections": [...]} into questions carrying a
    populated parent_context. Falls back to recovering the passage from the
    raw paper when the model omitted it.

    Returns (result, filled, recovered).
    """
    contexts = {str(k): (v or "").strip() for k, v in (result.get("contexts") or {}).items()}
    filled = 0
    recovered = 0
    raw_cache = {}

    for sec in result.get("sections", []):
        for q in sec.get("questions", []):
            if (q.get("parent_context") or "").strip():
                continue

            group = str(
                q.get("context_ref")
                or (q.get("question_number") or "").split(".")[0]
            )

            text = contexts.get(group, "")
            if text:
                q["parent_context"] = text
                filled += 1
                continue

            if not raw_text:
                continue

            if group not in raw_cache:
                raw_cache[group] = _recover_passage(raw_text, group)
            if raw_cache[group]:
                q["parent_context"] = raw_cache[group]
                contexts[group] = raw_cache[group]
                recovered += 1

    result["contexts"] = contexts
    return result, filled, recovered


def _recover_passage(raw_text, group_number):
    """
    Pull the prose between 'QUESTION n' and its first sub-question. Verified
    against the Grade 9 Holes paper: recovers the full 5,146-char extract, and
    rejects a short instruction line.
    """
    if not group_number:
        return ""
    pattern = (
        rf'QUESTION\s+{re.escape(str(group_number))}\b'
        rf'(.*?)'
        rf'(?=\n\s*{re.escape(str(group_number))}\.\d)'
    )
    m = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    body = re.sub(r'\n{3,}', '\n\n', m.group(1).strip())
    return body if len(body) >= MIN_PASSAGE_CHARS else ""


def propagate_context(sections, raw_text=""):
    """
    Fill parent_context on every question that needs it.
    Returns (sections, filled_from_siblings, filled_from_raw_text).
    """
    passages = _collect_passages(sections)
    from_siblings = 0
    from_raw = 0
    raw_cache = {}

    for sec in sections:
        for q in sec.questions:
            if len((q.parent_context or "").strip()) >= MIN_PASSAGE_CHARS:
                continue

            # 1. A sibling in the same group already has it
            hit = next((passages[k] for k in _group_keys(q) if k in passages), "")
            if hit:
                q.parent_context = hit
                from_siblings += 1
                continue

            # 2. Recover it from the raw paper text
            if not raw_text:
                continue
            group = (q.question_number or "").split('.')[0]
            if group not in raw_cache:
                raw_cache[group] = _recover_passage(raw_text, group)
            if raw_cache[group]:
                q.parent_context = raw_cache[group]
                for key in _group_keys(q):
                    passages.setdefault(key, raw_cache[group])
                from_raw += 1

    return sections, from_siblings, from_raw


# ═══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def smart_window_split(text, window=WINDOW_CHARS, overlap=OVERLAP_CHARS):
    if len(text) <= window:
        return [text]

    break_patterns = [
        r'\n\s*SECTION\s+[A-Z]\s*[:\-]?\s*',
        r'\n\s*QUESTION\s+\d+\s*[:\-]?\s*',
        r'\n\s*\d+\.\d+\s+[A-Z]',
        r'\n\s*\[?\s*\d+\s*\]?\s*marks?\s*\n',
        r'\n\n+',
    ]
    combined = '|'.join(f'({p})' for p in break_patterns)

    windows = []
    start = 0

    while start < len(text):
        end = min(start + window, len(text))
        if end < len(text):
            search_start = max(start + window - overlap, start + window // 2)
            search_text = text[search_start:end + 500]
            best_break = end
            for match in re.finditer(combined, search_text, re.IGNORECASE):
                pos = search_start + match.start()
                if pos > start + window * 0.7:
                    best_break = pos
                    break
            end = best_break
        windows.append(text[start:end].strip())
        start = end - overlap if end < len(text) else end

    return [w for w in windows if w]


def extract_with_llm(prompt):
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.invoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"```(?:json)?\s*", "", content)
                content = content.rstrip("`").strip()
            return json.loads(content)
        except Exception as e:
            err_msg = str(e).lower()
            if "rate" in err_msg or "429" in err_msg or "limit" in err_msg:
                wait = 15 * (attempt + 1)
                print(f"    Rate limit (attempt {attempt + 1}), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts")


def infer_question_type(question_text, options=None, column_a=None, instructions=""):
    text_lower = (question_text + " " + (instructions or "")).lower()

    if options and isinstance(options, dict) and len(options) >= 2:
        return QuestionType.MCQ.value
    if column_a and isinstance(column_a, list) and len(column_a) >= 2:
        return QuestionType.MATCHING.value
    if "match column" in text_lower or ("column a" in text_lower and "column b" in text_lower):
        return QuestionType.MATCHING.value
    if "true or false" in text_lower or "write true or false" in text_lower:
        return QuestionType.TRUE_FALSE.value
    if "show all calculations" in text_lower or "calculate" in text_lower or "determine" in text_lower:
        if any(c in text_lower for c in ["=", "+", "-", "times", "divide", "$"]):
            return QuestionType.CALCULATION.value
    if "label the diagram" in text_lower or "study the diagram" in text_lower or "figure" in text_lower:
        return QuestionType.DIAGRAM_LABEL.value
    if "complete the table" in text_lower or "use the table" in text_lower:
        return QuestionType.TABLE_COMPLETION.value
    if any(w in text_lower for w in ["discuss", "evaluate", "analyse", "critically"]):
        marks_match = re.search(r'\((\d+)\)', question_text)
        if marks_match and int(marks_match.group(1)) > 10:
            return QuestionType.ESSAY.value
    if "read the passage" in text_lower or "refer to" in text_lower or "according to" in text_lower:
        return QuestionType.COMPREHENSION.value
    if any(w in text_lower for w in ["briefly", "state", "name", "list"]):
        marks_match = re.search(r'\((\d+)\)', question_text)
        if marks_match and int(marks_match.group(1)) <= 5:
            return QuestionType.SHORT_ANSWER.value
    if "explain" in text_lower or "describe" in text_lower or "why" in text_lower:
        return QuestionType.OPEN.value
    return QuestionType.OPEN.value


def extract_questions_universal(text):
    # ── LLM call ───────────────────────────────────────────────────────────
    try:
        prompt = render_prompt(EXTRACTION_PROMPT_TEMPLATE, text)
        full_prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\n{prompt}"
        result = extract_with_llm(full_prompt)
    except Exception as e:
        print(f"    LLM extraction failed: {type(e).__name__}: {e}")
        return _fallback_extraction(text)

    if not isinstance(result, dict):
        print(f"    LLM returned {type(result).__name__}, expected object")
        return _fallback_extraction(text)

    # ── Context expansion — separate try, so a failure here never discards
    #    a good extraction. Worst case: questions arrive without passages.
    try:
        result, filled, recovered = expand_contexts(result, text)
        if filled or recovered:
            print(f"    Contexts: {filled} linked, {recovered} recovered")
    except Exception as e:
        print(f"    Context expansion failed: {type(e).__name__}: {e}")

    metadata = ExamMetadata(**{
        k: v for k, v in (result.get("metadata") or {}).items()
        if k in ExamMetadata.__dataclass_fields__
    })

    sections = []
    for sec_data in result.get("sections", []):
        questions = []
        for q_data in sec_data.get("questions", []):
            q_type = q_data.get("question_type", "open")
            if q_type in ["open", "unknown", "", None]:
                q_data["question_type"] = infer_question_type(
                    q_data.get("question", ""),
                    q_data.get("options"),
                    q_data.get("column_a"),
                    q_data.get("instructions", ""),
                )

            q_data["diagram_refs"] = [
                DiagramRef(**d) for d in (q_data.get("diagram_refs") or [])
                if isinstance(d, dict)
            ]
            q_data["table_refs"] = [
                TableRef(**t) for t in (q_data.get("table_refs") or [])
                if isinstance(t, dict)
            ]

            # An unexpected key from the model would raise TypeError and lose
            # the whole window. Drop unknown keys instead. This is also what
            # strips context_ref, which is not a dataclass field.
            clean = {k: v for k, v in q_data.items() if k in Question.__dataclass_fields__}
            clean.setdefault("id", len(questions) + 1)
            clean.setdefault("question_number", "")
            questions.append(Question(**clean))

        sections.append(Section(
            section=sec_data.get("section", "A"),
            section_title=sec_data.get("section_title", ""),
            section_instructions=sec_data.get("section_instructions", ""),
            total_marks=sec_data.get("total_marks"),
            questions=questions,
        ))

    return metadata, sections


def _fallback_extraction(text):
    print("    Using fallback extraction...")
    metadata = ExamMetadata()
    subject_match = re.search(r'(Mathematics|Physical Sciences|Life Sciences|Geography|History|Accounting|Economics|Business Studies|CAT|Information Technology|Engineering Graphics|English|Afrikaans|Technical Mathematics|Mathematical Literacy)', text, re.I)
    if subject_match:
        metadata.subject = subject_match.group(1)
    year_match = re.search(r'\b(20\d{2})\b', text)
    if year_match:
        metadata.year = year_match.group(1)
    grade_match = re.search(r'GRADE\s+(10|11|12)', text, re.I)
    if grade_match:
        metadata.grade = grade_match.group(1)

    sections = []
    section_pattern = r'SECTION\s+([A-Z])\s*(?::|\-)?\s*(.*?)(?=SECTION\s+[A-Z]|QUESTION\s+\d+|\Z)'
    section_matches = list(re.finditer(section_pattern, text, re.DOTALL | re.IGNORECASE))
    if not section_matches:
        section_matches = [(None, "A", text)]

    for match in section_matches:
        sec_label = match.group(1) if match else "A"
        sec_text = match.group(0) if match else text
        questions = []
        q_pattern = r'(?:(QUESTION)\s+(\d+)|(\d+)\.(\d+))\s*(.*?)(?=(?:(?:QUESTION)\s+\d+|\d+\.\d+|SECTION\s+[A-Z]|\Z))'
        for qm in re.finditer(q_pattern, sec_text, re.DOTALL):
            is_parent = bool(qm.group(1))
            q_num = f"{qm.group(2)}.1" if is_parent else f"{qm.group(3)}.{qm.group(4)}"
            q_text = qm.group(5).strip() if is_parent else qm.group(5).strip()
            marks_match = re.search(r'\((\d+)\s*marks?\)', q_text, re.I)
            marks = int(marks_match.group(1)) if marks_match else 1
            opt_matches = re.findall(r'([A-D])\.\s*(.*?)(?=(?:[A-D]\.|$))', q_text, re.DOTALL)
            options = None
            if opt_matches and len(opt_matches) >= 2:
                options = {k: v.strip() for k, v in opt_matches}
                q_text = re.split(r'[A-D]\.', q_text)[0].strip()
            q_type = infer_question_type(q_text, options)
            questions.append(Question(
                id=len(questions) + 1,
                question_number=q_num,
                question=q_text[:500],
                question_type=q_type,
                marks=marks,
                options=options,
                section=sec_label
            ))
        sections.append(Section(section=sec_label, questions=questions))
    return metadata, sections


def extract_memo_universal(text):
    try:
        prompt = render_prompt(MEMO_PROMPT_TEMPLATE, text[:15000])
        result = extract_with_llm(prompt)
        return result.get("answers", {}) or {}
    except Exception as e:
        print(f"    Memo LLM extraction failed: {type(e).__name__}: {e}")
        return _fallback_memo_extraction(text)


def _fallback_memo_extraction(text):
    answers = {}
    patterns = [
        r'(?:^|\n)\s*(\d+\.\d+(?:\.\d+)?)\s*[:\-]?\s*(.*?)(?=\n\s*\d+\.\d+|\Z)',
        r'(?:^|\n)\s*(\d+)\s*[:\-]?\s*(.*?)(?=\n\s*\d+\s*[:\-]?|\Z)',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            q_num = match.group(1).strip()
            answer = match.group(2).strip()
            answer = re.sub(r'\s+', ' ', answer)
            if len(answer) > 0:
                answers[q_num] = answer
    return answers


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_exam_structure(sections, raw_text):
    for section in sections:
        fixed_questions = []
        for q in section.questions:
            if not q.question or len(q.question.strip()) < 5:
                recovered = _recover_question_text(raw_text, q.question_number)
                if recovered:
                    q.question = recovered
                else:
                    continue
            if q.question_type == QuestionType.MCQ.value:
                if not q.options or len(q.options) < 2:
                    recovered = _recover_mcq_options(raw_text, q.question_number)
                    if recovered:
                        q.options = recovered
                    else:
                        q.question_type = QuestionType.OPEN.value
            if q.question_type == QuestionType.MATCHING.value:
                if not q.column_a or not q.column_b or len(q.column_a) < 2:
                    col_a, col_b = _recover_matching(raw_text)
                    if col_a and col_b:
                        q.column_a = col_a
                        q.column_b = col_b
                    else:
                        q.question_type = QuestionType.OPEN.value
            if q.marks is None or q.marks < 1:
                marks_found = re.search(rf'{re.escape(q.question_number)}.*?\((\d+)\s*marks?\)', raw_text, re.I)
                if marks_found:
                    q.marks = int(marks_found.group(1))
                else:
                    q.marks = 1
            fixed_questions.append(q)
        section.questions = sort_questions(fixed_questions)
    return sections


def _recover_question_text(raw_text, q_num):
    pattern = rf'{re.escape(q_num)}\s*(.*?)(?=\n\s*\d+\.\d+|\n\s*[A-D]\.|\Z)'
    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        text = match.group(1).strip()
        text = re.sub(r'^[\:\-\s]+', '', text)
        return text[:1000]
    return None


def _recover_mcq_options(raw_text, q_num):
    pattern = rf'{re.escape(q_num)}.*?((?:A\.\s*.*?)(?:B\.\s*.*?)(?:C\.\s*.*?)(?:D\.\s*.*?))'
    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        opts_text = match.group(1)
        opts = {}
        for opt_match in re.finditer(r'([A-D])\.\s*(.*?)(?=(?:[A-D]\.|$))', opts_text, re.DOTALL):
            opts[opt_match.group(1)] = opt_match.group(2).strip()
        return opts if len(opts) >= 2 else None
    return None


def _recover_matching(raw_text):
    col_a = re.findall(r'(?:^|\n)\s*(\d+\.\d+)\s+(.+?)(?=\n\s*\d+\.\d+|\n\s*[A-Z]\.|\Z)', raw_text)
    col_b = re.findall(r'(?:^|\n)\s*([A-Z])\.\s*(.+?)(?=\n\s*[A-Z]\.|\Z)', raw_text)
    a_items = [item[1].strip() for item in col_a if len(item[1].strip()) > 3]
    b_items = [item[1].strip() for item in col_b if len(item[1].strip()) > 1]
    return (a_items if len(a_items) >= 2 else None, b_items if len(b_items) >= 2 else None)


def sort_questions(questions):
    def sort_key(q):
        parts = q.question_number.split(".")
        return tuple(int(p) if p.isdigit() else 0 for p in parts)
    return sorted(questions, key=sort_key)


def deduplicate_questions(sections):
    seen = set()
    for section in sections:
        unique = []
        for q in section.questions:
            key = f"{section.section}:{q.question_number}:{q.question[:50]}"
            if key not in seen:
                seen.add(key)
                unique.append(q)
        section.questions = unique
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# MEMO INJECTION
# ═══════════════════════════════════════════════════════════════════════════════


def inject_memo_universal(sections, memo_answers):
    matched = 0
    unmatched = []

    for section in sections:
        for q in section.questions:
            q_num = (q.question_number or "").strip()

            if q_num in memo_answers:
                q.memo = memo_answers[q_num]
                matched += 1
                continue

            clean_num = re.sub(r'\.0+$', '', q_num)
            if clean_num != q_num and clean_num in memo_answers:
                q.memo = memo_answers[clean_num]
                matched += 1
                continue

            sub_hit = False
            for sub in (q.sub_parts or []):
                sub_num = sub.get("sub_number", "")
                if sub_num in memo_answers:
                    sub["memo"] = memo_answers[sub_num]
                    matched += 1
                    sub_hit = True

            if not sub_hit:
                unmatched.append(q_num)

    return sections, matched, unmatched


# ═══════════════════════════════════════════════════════════════════════════════
# FILE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

EXAM_KEYWORDS = [
    "exam", "paper", "question", "theory", "p1", "p2", "p3",
    "nov", "november", "may", "june", "feb", "february", "march", "mar",
    "aug", "august", "sep", "september", "oct", "october", "term",
    "trial", "nsc", "dbe", "cat", "mathematics", "maths", "physical",
    "sciences", "life sciences", "geography", "history", "accounting",
    "economics", "business", "afrikaans", "english", "isizulu", "sesotho"
]

MEMO_KEYWORDS = ["memo", "memorandum", "answers", "answer_key", "marking", "marking guidelines"]

SUBJECT_PATTERNS = {
    r'\bmathematics\b': 'Mathematics',
    r'\bmaths\b': 'Mathematics',
    r'\bmath\s+lit\b': 'Mathematical Literacy',
    r'\btechnical\s+math\b': 'Technical Mathematics',
    r'\bphysical\s+sciences?\b': 'Physical Sciences',
    r'\blife\s+sciences?\b': 'Life Sciences',
    r'\bgeography\b': 'Geography',
    r'\bhistory\b': 'History',
    r'\baccounting\b': 'Accounting',
    r'\beconomics\b': 'Economics',
    r'\bbusiness\s+studies\b': 'Business Studies',
    r'\bcat\b': 'Computer Applications Technology',
    r'\bit\b': 'Information Technology',
    r'\bengineering\s+graphics\b': 'Engineering Graphics & Design',
    r'\benglish\b': 'English',
    r'\bafrikaans\b': 'Afrikaans',
    r'\bisixhosa\b': 'isiXhosa',
    r'\bisizulu\b': 'isiZulu',
    r'\btshivenda\b': 'TshiVenda',
    r'\bsesotho\b': 'Sesotho',
}

NOISE_WORDS = {"memo", "memorandum", "answers", "answer", "marking", "key",
               "theory", "exam", "paper", "nsc", "dbe", "grade", "gr", "cat",
               "caps", "p1", "p2", "p3", "question", "chunks", "nov", "november",
               "oct", "october", "jun", "june", "feb", "february", "mar", "march",
               "aug", "august", "sep", "september", "jan", "january", "jul", "july",
               "apr", "april", "dec", "december", "trial", "term", "final"}

MONTH_CANONICAL = {
    "jan": "january", "january": "january", "feb": "february", "february": "february",
    "mar": "march", "march": "march", "apr": "april", "april": "april", "may": "may",
    "jun": "june", "june": "june", "jul": "july", "july": "july",
    "aug": "august", "august": "august", "sep": "september", "september": "september",
    "oct": "october", "october": "october", "nov": "november", "november": "november",
    "dec": "december", "december": "december"
}


def classify_file(filename):
    lower = filename.lower()
    if any(kw in lower for kw in MEMO_KEYWORDS):
        return "memo"
    if any(kw in lower for kw in EXAM_KEYWORDS):
        return "exam"
    return "skip"


def detect_subject(filename):
    lower = filename.lower()
    for pattern, subject in SUBJECT_PATTERNS.items():
        if re.search(pattern, lower):
            return subject
    return "Unknown"


def extract_keywords(filename):
    name = filename.lower().strip()
    name = re.sub(r'\s+\.', '.', name)
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


def find_matching_exam(memo_filename, exam_files):
    memo_kw = extract_keywords(memo_filename)
    if not memo_kw:
        return None, set(), 0
    best_file, best_shared, best_score = None, set(), 0
    for ef in exam_files:
        shared = memo_kw & extract_keywords(ef)
        if not shared:
            continue
        score = len(shared) / len(memo_kw | extract_keywords(ef))
        if score > best_score:
            best_score, best_shared, best_file = score, shared, ef
    return (best_file, best_shared, best_score) if best_file else (None, set(), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_key(filename):
    name = filename.strip().lower()
    name = re.sub(r'\s+\.', '.', name)
    return name


def load_tracker():
    if not os.path.exists(TRACK_FILE):
        return {}
    try:
        with open(TRACK_FILE) as f:
            data = json.load(f)
    except Exception:
        return {}
    if isinstance(data, list):
        data = {n: {"exam_done": False, "memo_merged": False} for n in data}
    normalised = {}
    for raw_key, value in data.items():
        nk = normalize_key(raw_key)
        if nk not in normalised:
            normalised[nk] = {"exam_done": False, "memo_merged": False, "memo_source": None}
        if value.get("exam_done"):
            normalised[nk]["exam_done"] = True
        if value.get("memo_merged"):
            normalised[nk]["memo_merged"] = True
        if value.get("memo_source"):
            normalised[nk]["memo_source"] = value["memo_source"]
    if normalised != data:
        with open(TRACK_FILE, "w") as f:
            json.dump(normalised, f, indent=2)
    return normalised


def save_tracker(t):
    with open(TRACK_FILE, "w") as f:
        json.dump(t, f, indent=2)


def tracker_get(t, f):
    return t.get(normalize_key(f), {})


def tracker_set(t, f, k, v):
    nk = normalize_key(f)
    if nk not in t:
        t[nk] = {}
    t[nk][k] = v


def output_path_for(f):
    stem = re.sub(r"\.json$", "", normalize_key(f))
    return os.path.join(OUTPUT_FOLDER, stem + "_exam.json")


def exam_output_exists(f):
    return os.path.exists(output_path_for(f))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def stitch_chunks(chunks):
    return "\n".join(c.get("content", "").strip() for c in chunks if c.get("content", "").strip())


def count_types(sections):
    counts = {t.value: 0 for t in QuestionType}
    for section in sections:
        for q in section.questions:
            t = q.question_type
            counts[t] = counts.get(t, 0) + 1
    return counts


def load_chunks(filename):
    path = os.path.join(PROCESSED_FOLDER, filename)
    try:
        with open(path) as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"  Failed to load {filename}: {e}")
        return []


def process():
    if not os.path.exists(PROCESSED_FOLDER):
        print(f"Folder '{PROCESSED_FOLDER}' not found.")
        return

    tracker = load_tracker()
    SKIP = {"metadata.json", "chunk_ids.json", "processed_files.json", "processed_exams.json"}
    all_json = [f for f in sorted(os.listdir(PROCESSED_FOLDER))
                if f.endswith(".json") and f not in SKIP]

    exam_files, memo_files, skipped = [], [], []
    for f in all_json:
        kind = classify_file(f)
        if kind == "exam":
            exam_files.append(f)
        elif kind == "memo":
            memo_files.append(f)
        else:
            skipped.append(f)

    print(f"\n{'='*60}")
    print(f"Files: {len(all_json)} | Exams: {len(exam_files)} | Memos: {len(memo_files)} | Skipped: {len(skipped)}")
    for f in exam_files:
        e = tracker_get(tracker, f)
        s = "+memo" if e.get("exam_done") and e.get("memo_merged") else "done" if e.get("exam_done") else "pending"
        subject = detect_subject(f)
        print(f"  [{s}] [{subject}] {f}")
    for f in memo_files:
        s = "merged" if tracker_get(tracker, f).get("memo_merged") else "pending"
        print(f"  [{s}] {f}")
    print(f"{'='*60}\n")

    # STEP 1: Extract Exams
    pending_exams = [f for f in exam_files
                     if not (tracker_get(tracker, f).get("exam_done") and exam_output_exists(f))]
    print(f"STEP 1: {len(pending_exams)} exam(s) to extract\n")

    for idx, exam_file in enumerate(pending_exams, 1):
        print(f"  [{idx}/{len(pending_exams)}] {exam_file}")
        subject = detect_subject(exam_file)
        print(f"    Detected subject: {subject}")

        chunks = load_chunks(exam_file)
        if not chunks:
            print("    Empty\n")
            continue

        full_text = stitch_chunks(chunks)
        windows = smart_window_split(full_text)
        print(f"    {len(chunks)} chunks -> {len(full_text)} chars -> {len(windows)} windows")

        all_sections = []
        all_metadata = None

        for i, window in enumerate(windows):
            print(f"    Window {i+1}/{len(windows)}...")
            try:
                metadata, sections = extract_questions_universal(window)
                if all_metadata is None and metadata.subject:
                    all_metadata = metadata
                all_sections.extend(sections)
            except Exception as e:
                print(f"    Window {i+1} failed: {e}")
            time.sleep(1.5)

        if not all_sections:
            print("    Nothing extracted\n")
            continue

        all_sections = deduplicate_questions(all_sections)
        all_sections = validate_exam_structure(all_sections, full_text)

        merged_sections = {}
        for sec in all_sections:
            label = sec.section.upper()
            if label not in merged_sections:
                merged_sections[label] = sec
            else:
                merged_sections[label].questions.extend(sec.questions)
                merged_sections[label].questions = sort_questions(merged_sections[label].questions)

        final_sections = list(merged_sections.values())
        final_sections.sort(key=lambda s: s.section)

        total_q = sum(len(s.questions) for s in final_sections)
        type_counts = count_types(final_sections)

        metadata_dict = all_metadata.to_dict() if all_metadata else {}
        metadata_dict["detected_from_filename"] = subject

        out_data = {
            "source": exam_file,
            "metadata": metadata_dict,
            "total_questions": total_q,
            "type_breakdown": type_counts,
            "memo_merged": False,
            "memo_source": None,
            "sections": [s.to_dict() for s in final_sections]
        }

        out_path = output_path_for(exam_file)
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)

        print(f"    Saved: {out_path}")
        print(f"    {total_q}q | MCQ:{type_counts['mcq']} Match:{type_counts['matching']} "
              f"T/F:{type_counts['true_false']} Calc:{type_counts['calculation']} "
              f"Diagram:{type_counts['diagram_label']} Table:{type_counts['table_completion']} "
              f"Essay:{type_counts['essay']} Open:{type_counts['open']}\n")

        tracker_set(tracker, exam_file, "exam_done", True)
        tracker_set(tracker, exam_file, "memo_merged", False)
        save_tracker(tracker)

    # STEP 2: Merge Memos
    pending_memos = [f for f in memo_files if not tracker_get(tracker, f).get("memo_merged")]
    print(f"\nSTEP 2: {len(pending_memos)} memo(s) to merge\n")

    for idx, memo_file in enumerate(pending_memos, 1):
        print(f"  [{idx}/{len(pending_memos)}] {memo_file}")
        memo_kw = extract_keywords(memo_file)
        print(f"    Keywords: {sorted(memo_kw)}")

        matched_exam, shared_kw, score = find_matching_exam(memo_file, exam_files)
        if not matched_exam:
            print(f"    No match\n")
            continue

        exam_output = output_path_for(matched_exam)
        if not os.path.exists(exam_output):
            print(f"    Missing: {exam_output}\n")
            continue

        if not tracker_get(tracker, matched_exam).get("exam_done"):
            print(f"    Exam not yet extracted\n")
            continue

        print(f"    -> {matched_exam} ({score:.0%} match)")

        memo_chunks = load_chunks(memo_file)
        if not memo_chunks:
            print("    Memo empty\n")
            continue

        full_memo = stitch_chunks(memo_chunks)
        memo_windows = smart_window_split(full_memo)
        print(f"    {len(memo_chunks)} chunks -> {len(memo_windows)} windows")

        all_memo_answers = {}
        for i, window in enumerate(memo_windows):
            print(f"    Memo window {i+1}/{len(memo_windows)}...")
            try:
                answers = extract_memo_universal(window)
                for k, v in answers.items():
                    if k not in all_memo_answers:
                        all_memo_answers[k] = v
            except Exception as e:
                print(f"    Memo window {i+1} failed: {e}")
            time.sleep(1.5)

        if not all_memo_answers:
            print("    No answers\n")
            continue

        print(f"    {len(all_memo_answers)} answers extracted")

        with open(exam_output) as f:
            exam_data = json.load(f)

        sections = []
        for sec_data in exam_data.get("sections", []):
            questions = []
            for q_data in sec_data.get("questions", []):
                q_data.pop("diagram_refs", None)
                q_data.pop("table_refs", None)
                questions.append(Question(**q_data))
            sections.append(Section(
                section=sec_data.get("section", "A"),
                section_title=sec_data.get("section_title", ""),
                section_instructions=sec_data.get("section_instructions", ""),
                total_marks=sec_data.get("total_marks"),
                questions=questions
            ))

        updated_sections, matched_count, unmatched = inject_memo_universal(sections, all_memo_answers)

        exam_data["sections"] = [s.to_dict() for s in updated_sections]
        exam_data["memo_merged"] = True
        exam_data["memo_source"] = memo_file
        exam_data["memo_answers_total"] = len(all_memo_answers)
        exam_data["memo_matched"] = matched_count
        exam_data["memo_unmatched"] = unmatched

        with open(exam_output, "w") as f:
            json.dump(exam_data, f, indent=2)

        print(f"\n    Saved: {exam_output} | merged {matched_count}/{len(all_memo_answers)}")
        if unmatched:
            print(f"    Unmatched: {unmatched[:15]}{'...' if len(unmatched) > 15 else ''}")
        print()

        tracker_set(tracker, memo_file, "memo_merged", True)
        tracker_set(tracker, memo_file, "memo_source", matched_exam)
        tracker_set(tracker, matched_exam, "memo_merged", True)
        tracker_set(tracker, matched_exam, "memo_source", memo_file)
        save_tracker(tracker)

    print("All done.")


if __name__ == "__main__":
    process()