import os
import json
from groq import Groq
from dotenv import load_dotenv

# =========================
# 🔑 LOAD ENV
# =========================
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set in environment")

client = Groq(api_key=API_KEY)

# Model used across all functions
MODEL = "llama-3.3-70b-versatile"

# Exams folder — for loading memo answers directly from exam JSON
BASE_DIR     = os.path.dirname(__file__)
EXAMS_FOLDER = os.path.join(BASE_DIR, "exams")


# =========================
# 📂 LOAD MEMO FROM EXAM FILE
# Used by exam marking so it reads the merged memo
# from the processed exam JSON — NOT from the theory book.
# =========================
def load_exam_memo(exam_name):
    """
    Load all memo answers from an exam JSON file.
    Returns a flat dict: { "question_number": "memo_answer" }
    e.g. { "1.1": "C", "4.1": "answer text...", "2.1": "R" }
    """
    path = os.path.join(EXAMS_FOLDER, exam_name)
    if not os.path.exists(path):
        return {}

    try:
        with open(path) as f:
            exam = json.load(f)
    except Exception:
        return {}

    memo_map = {}

    for section in exam.get("sections", []):
        for q in section.get("questions", []):
            q_num = q.get("question_number", "").strip()
            memo  = q.get("memo", "")

            if not q_num:
                continue

            # Matching memo is a dict — convert to readable string
            if isinstance(memo, dict):
                memo_map[q_num] = memo  # keep as dict for structured marking
            elif memo:
                memo_map[q_num] = str(memo).strip()

    return memo_map


# =========================
# 🎓 AI TUTOR (RAG ANSWER)
# Feeds on theory/content retrieved by rag.py
# =========================
def generate_answer(context, question):
    """
    Generates a tutor answer using RAG context from theory books.
    Called by app.py /chat endpoint.
    """
    try:
        prompt = f"""You are a friendly and knowledgeable CAT (Computer Applications Technology) Grade 12 tutor.

Use the context below — extracted from the CAT theory book — to answer the student's question.
If the context does not contain enough information, use your own CAT knowledge to help.

Context (from theory book):
{context if context.strip() else "No specific context found — using general CAT knowledge."}

Student Question:
{question}

Instructions:
- Explain clearly in simple, student-friendly language
- Use examples where helpful
- Be concise but thorough
- Always relate answers to CAT Grade 12 content
- Do NOT refer to exam questions or memo answers

Answer:"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )

        answer = response.choices[0].message.content
        if not answer or not answer.strip():
            return "⚠️ No response generated. Try rephrasing your question."
        return answer.strip()

    except Exception as e:
        return f"⚠️ Error generating answer: {str(e)}"


# =========================
# 🧠 AI EXAM MARKER (SINGLE QUESTION)
# Feeds ONLY on the memo from the exam JSON.
# For open questions with no memo → gives structured feedback only.
# =========================
def mark_answer(question, question_number, q_type, student_answer, memo, marks):
    """
    Marks a single answer using AI.
    - MCQ and True/False: exact match only, no AI needed
    - Matching: per-pair scoring, no AI needed
    - Open: AI compares meaning against memo answer

    Returns: { "score": int, "feedback": str, "status": str }
    """

    # ── MCQ: no AI needed — exact letter match ────────
    if q_type == "mcq":
        correct = str(memo).strip().upper()
        ans     = str(student_answer).strip().upper()
        if not correct:
            return {"score": 0, "feedback": "No memo available for this question.", "status": "no_memo"}
        if ans == correct:
            return {"score": marks, "feedback": f"Correct! The answer is {correct}.", "status": "correct"}
        return {"score": 0, "feedback": f"Incorrect. The correct answer is {correct}.", "status": "incorrect"}

    # ── True/False: near-exact match ─────────────────
    if q_type == "true_false":
        if not memo:
            return {"score": 0, "feedback": "No memo available.", "status": "no_memo"}
        correct_lower = str(memo).strip().lower()
        ans_lower     = str(student_answer).strip().lower()
        if correct_lower == "true" and ans_lower == "true":
            return {"score": marks, "feedback": "Correct!", "status": "correct"}
        if correct_lower.startswith("false") and ans_lower.startswith("false"):
            # Check correction word
            memo_word = correct_lower.replace("false —", "").replace("false-", "").strip()
            ans_word  = ans_lower.replace("false —", "").replace("false-", "").strip()
            if not memo_word or memo_word in ans_word or ans_word in memo_word:
                return {"score": marks, "feedback": "Correct!", "status": "correct"}
            return {
                "score": marks // 2,
                "feedback": f"Correct that it is FALSE, but the correction should be: '{memo_word}'.",
                "status": "partial"
            }
        return {
            "score": 0,
            "feedback": f"Incorrect. The statement is: {memo}",
            "status": "incorrect"
        }

    # ── Matching: per-pair scoring ────────────────────
    if q_type == "matching":
        if not isinstance(memo, dict) or not memo:
            return {"score": 0, "feedback": "No memo available for matching.", "status": "no_memo"}
        try:
            student_map = json.loads(student_answer) if isinstance(student_answer, str) else student_answer
        except Exception:
            student_map = {}
        correct_count = sum(
            1 for k, v in memo.items()
            if str(student_map.get(k, "")).strip().lower() == str(v).strip().lower()
        )
        total_pairs = len(memo)
        earned      = round((correct_count / total_pairs) * marks) if total_pairs else 0
        status      = "correct" if earned == marks else ("partial" if earned > 0 else "incorrect")
        return {
            "score":    earned,
            "feedback": f"{correct_count}/{total_pairs} correct matches.",
            "status":   status
        }

    # ── Open: AI marking against memo ────────────────
    # If no memo → AI gives feedback only, awards partial for effort
    if not student_answer or not student_answer.strip():
        return {"score": 0, "feedback": "No answer provided.", "status": "missing"}

    if not memo or (isinstance(memo, str) and not memo.strip()):
        # No memo — AI gives qualitative feedback, partial marks for attempting
        try:
            prompt = f"""You are a strict but fair CAT Grade 12 examiner.

The student answered the following question but no memo is available.
Evaluate if the answer shows understanding of the topic.

Question ({marks} mark{"s" if marks > 1 else ""}):
{question}

Student Answer:
{student_answer}

Return ONLY valid JSON:
{{
  "score": {marks // 2},
  "feedback": "short constructive feedback explaining what was good and what could improve",
  "status": "partial"
}}

Note: Since no memo is available, award {marks // 2} marks if the answer shows reasonable understanding, else 0."""

            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].lstrip("json").strip()
            return json.loads(content)
        except Exception as e:
            return {"score": marks // 2, "feedback": "Answer noted — no memo to compare against.", "status": "partial"}

    # ── Open with memo: AI compares meaning ──────────
    try:
        memo_text = memo if isinstance(memo, str) else json.dumps(memo)

        prompt = f"""You are a strict but fair CAT Grade 12 examiner marking a student's answer.

Question ({marks} mark{"s" if marks > 1 else ""}):
{question}

Expected Answer (from memo):
{memo_text}

Student's Answer:
{student_answer}

Marking Instructions:
- Compare MEANING, not exact wording
- Award full marks ({marks}) if the answer captures the key facts from the memo
- Award partial marks if some correct understanding is shown
- Award 0 if the answer is wrong or completely off-topic
- Each mark = one distinct correct fact or point
- Be consistent with real NSC exam marking

Return ONLY valid JSON — no markdown, no explanation:
{{
  "score": <number between 0 and {marks}>,
  "feedback": "<short, specific feedback — what was correct and what was missing>",
  "status": "<correct | partial | incorrect>"
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1].lstrip("json").strip()

        result = json.loads(content)

        # Safety clamp: score cannot exceed marks
        result["score"] = max(0, min(int(result.get("score", 0)), marks))
        return result

    except Exception as e:
        return {
            "score":    0,
            "feedback": f"⚠️ Could not evaluate answer: {e}",
            "status":   "incorrect"
        }


# =========================
# 📋 BATCH MARKER
# Marks all questions in a session using memo from exam JSON.
# Called by app.py /submit endpoint.
# =========================
def mark_exam_batch(exam_name, flat_questions, student_answers):
    """
    Marks all student answers for a full exam.

    Args:
        exam_name       : filename of the exam JSON (to load memo)
        flat_questions  : list of question dicts from flatten_exam()
        student_answers : dict { "0": "answer", "1": "answer", ... }

    Returns:
        {
          "results": [...],
          "total_score": int,
          "total_marks": int
        }
    """
    # Load memo directly from exam JSON
    # This ensures marking always uses the extracted memo,
    # not the theory book or any hardcoded values
    exam_memo = load_exam_memo(exam_name)

    if not exam_memo:
        print(f"⚠️  No memo found in {exam_name} — marking will use inline memo fields only.")

    results     = []
    total_score = 0
    total_marks = 0

    for i, q in enumerate(flat_questions):
        student_ans     = student_answers.get(str(i), "").strip()
        q_num           = q.get("question_number", "")
        q_type          = q.get("type", "open")
        marks           = q.get("marks", 1)
        question_text   = q.get("question", "")

        # ── Get memo: prefer loaded exam memo, fallback to inline field ──
        memo = exam_memo.get(q_num) or q.get("memo", "")

        result = mark_answer(
            question        = question_text,
            question_number = q_num,
            q_type          = q_type,
            student_answer  = student_ans,
            memo            = memo,
            marks           = marks
        )

        result["question_number"] = q_num
        result["question"]        = question_text
        result["type"]            = q_type
        result["marks"]           = marks
        result["student_answer"]  = student_ans or "No answer"

        # Format memo for display in results
        if isinstance(memo, dict):
            result["correct_answer"] = " | ".join(f"{k} → {v}" for k, v in memo.items())
        elif memo:
            result["correct_answer"] = str(memo)
        else:
            result["correct_answer"] = "Not available"

        results.append(result)
        total_score += result.get("score", 0)
        total_marks += marks

    return {
        "results":     results,
        "total_score": total_score,
        "total_marks": total_marks,
        "percentage":  round((total_score / total_marks) * 100, 2) if total_marks else 0
    }


# =========================
# 🧾 EXAM FEEDBACK SUMMARY
# Generates overall performance feedback after marking
# =========================
def generate_exam_feedback(results, score, total, percentage):
    """
    Generates a short personalised performance summary.
    """
    try:
        # Build a concise summary for the AI — avoid sending full answers
        summary = [
            {
                "question_number": r.get("question_number"),
                "type":            r.get("type"),
                "marks":           r.get("marks"),
                "score":           r.get("score"),
                "status":          r.get("status")
            }
            for r in results
        ]

        prompt = f"""You are a motivating CAT Grade 12 teacher reviewing a student's exam performance.

Score: {score} / {total} ({percentage}%)

Question breakdown:
{json.dumps(summary, indent=2)}

Write a short performance summary (3–4 sentences) that:
- Acknowledges the score warmly
- Highlights what went well (correct/partial answers)
- Identifies the weakest area to focus on
- Ends with encouragement

Keep it friendly, specific, and motivating."""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"You scored {score}/{total} ({percentage}%). Keep practising! 🚀"