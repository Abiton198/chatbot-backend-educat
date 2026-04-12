import os
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set")

client = Groq(api_key=API_KEY)
MODEL  = "llama-3.3-70b-versatile"


# =========================
# 🎓 AI TUTOR
# =========================
def generate_answer(context, question):
    try:
        prompt = f"""You are a friendly CAT Grade 12 tutor.
Use the context below to answer the student's question.
If context is insufficient, use your own CAT knowledge.

Context:
{context.strip() if context and context.strip() else "No specific context — using general CAT knowledge."}

Student Question: {question}

Answer clearly, concisely, with examples where helpful."""

        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        ans = r.choices[0].message.content
        return ans.strip() if ans and ans.strip() else "⚠️ No response generated."
    except Exception as e:
        return f"⚠️ Error: {e}"


# =========================
# 🧠 MARK SINGLE ANSWER
# memo comes from q["memo"] which was set during extraction.
# We NEVER load memo by question_number from a separate dict —
# that caused the Kickstarter-for-1.1 bug.
# =========================
def mark_answer(question, question_number, q_type, student_answer, memo, marks, options=None):
    student = str(student_answer).strip() if student_answer else ""

    # ── MCQ ──────────────────────────────────────────────
    if q_type == "mcq":
        correct = str(memo).strip().upper()
        ans     = student.upper()
        if not correct:
            return {"score": 0, "feedback": "No memo for this question.", "status": "no_memo"}
        if not ans:
            return {"score": 0, "feedback": f"No answer selected. Correct: {correct}.", "status": "missing"}
        if ans == correct:
            # Find option text for richer feedback
            opt_text = ""
            if options:
                for opt in options:
                    if isinstance(opt, dict) and opt.get("key","").upper() == correct:
                        opt_text = f" ({opt['value']})"
                        break
            return {"score": marks, "feedback": f"Correct! Answer: {correct}{opt_text}.", "status": "correct"}
        return {"score": 0, "feedback": f"Incorrect. You selected {ans}; correct answer is {correct}.", "status": "incorrect"}

    # ── True/False ────────────────────────────────────────
    if q_type == "true_false":
        if not memo:
            return {"score": 0, "feedback": "No memo available.", "status": "no_memo"}
        correct_lower = str(memo).strip().lower()
        ans_lower     = student.lower()
        if not ans_lower:
            return {"score": 0, "feedback": f"No answer. Correct: {memo}", "status": "missing"}

        correct_is_true = correct_lower.startswith("true")
        student_is_true = ans_lower.startswith("true")

        if correct_is_true and student_is_true:
            return {"score": marks, "feedback": "Correct — True.", "status": "correct"}

        if not correct_is_true and not student_is_true:
            # Both False — check correction word
            def extract_correction(s):
                parts = re.split(r"[-—]", s, maxsplit=1)
                return parts[1].strip().lower() if len(parts) > 1 else ""
            memo_word = extract_correction(correct_lower)
            stu_word  = extract_correction(ans_lower)
            if not memo_word or (stu_word and (memo_word in stu_word or stu_word in memo_word)):
                return {"score": marks, "feedback": f"Correct — False, correction: {stu_word or memo_word}.", "status": "correct"}
            return {
                "score":    marks // 2,
                "feedback": f"Correct it's FALSE, but wrong correction. Expected: '{memo_word}', got: '{stu_word or '(none)'}'.",
                "status":   "partial"
            }

        return {"score": 0, "feedback": f"Incorrect. Correct answer: {memo}.", "status": "incorrect"}

    # ── Matching ──────────────────────────────────────────
    if q_type == "matching":
        if not isinstance(memo, dict) or not memo:
            return {"score": 0, "feedback": "No memo for matching.", "status": "no_memo"}
        try:
            student_map = json.loads(student) if student else {}
        except Exception:
            student_map = {}

        correct_count = 0
        details = []
        for col_a_item, correct_letter in memo.items():
            student_val    = student_map.get(col_a_item, "")
            # student_val looks like "R. convergence" — extract letter only
            student_letter = student_val.strip().split(".")[0].strip().upper() if student_val else ""
            correct_clean  = str(correct_letter).strip().upper()
            if student_letter == correct_clean:
                correct_count += 1
                details.append(f"✅ {col_a_item.split()[0]}: {student_letter}")
            else:
                details.append(f"❌ {col_a_item.split()[0]}: got '{student_letter or '—'}', need '{correct_clean}'")

        total_pairs = len(memo)
        earned = round((correct_count / total_pairs) * marks) if total_pairs else 0
        status = "correct" if earned == marks else "partial" if earned > 0 else "incorrect"
        return {
            "score":    earned,
            "feedback": f"{correct_count}/{total_pairs} correct. " + " | ".join(details),
            "status":   status
        }

    # ── Open ──────────────────────────────────────────────
    if not student:
        return {"score": 0, "feedback": "No answer provided.", "status": "missing"}

    memo_text = memo if isinstance(memo, str) else json.dumps(memo) if memo else ""
    has_memo  = bool(memo_text.strip())

    prompt = f"""You are a strict South African NSC CAT examiner.

Question {question_number} ({marks} mark{"s" if marks != 1 else ""}):
{question}

{"Marking guideline:\n" + memo_text if has_memo else "No marking guideline — use CAT Grade 12 knowledge."}

Student's answer:
{student}

Award marks strictly (1 mark per correct fact). Return ONLY valid JSON:
{{
  "score": <integer 0 to {marks}>,
  "feedback": "<specific: what was correct, what was missing>",
  "status": "<correct | partial | incorrect>"
}}"""

    for attempt in range(2):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0
            )
            content = r.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].lstrip("json").strip()
            result = json.loads(content)
            result["score"] = max(0, min(int(result.get("score", 0)), marks))
            return result
        except Exception as e:
            if attempt == 0:
                time.sleep(5)
            else:
                return {"score": 0, "feedback": f"Could not mark: {e}", "status": "incorrect"}


# =========================
# 📋 BATCH MARKER
# IMPORTANT: memo comes from q["memo"] inside each question dict.
# We do NOT call load_exam_memo() or look up by question_number
# from a separate dict — that was the bug causing Kickstarter/1.1 mismatch.
# =========================
def mark_exam_batch(exam_name, flat_questions, student_answers):
    """
    flat_questions: list of question dicts from flatten_exam().
      Each dict already has q["memo"] set correctly for that question.
      app.py also sets q["student_answer"] before calling this.

    student_answers: { "0": "answer", ... } — backup if student_answer not set.
    """
    results     = []
    total_score = 0
    total_marks = 0

    for i, q in enumerate(flat_questions):
        q_num   = q.get("question_number", f"Q{i+1}")
        q_type  = q.get("type", "open").lower()
        marks   = int(q.get("marks", 1))
        q_text  = q.get("question", "")
        options = q.get("options")  # for MCQ option text in feedback

        # ── Get memo directly from the question dict ──────
        # This is the ONLY correct way. Never look up by index or
        # from a separate memo dict — that caused the Kickstarter bug.
        memo = q.get("memo", "")

        # ── Get student answer ────────────────────────────
        # app.py sets student_answer on the dict; fall back to answers dict
        student = q.get("student_answer", student_answers.get(str(i), ""))
        if student is None:
            student = ""
        student = str(student).strip()

        # Debug log — remove after confirming fix
        print(f"  MARKING {q_num} | type={q_type} | memo={repr(str(memo)[:60])} | student={repr(student[:40])}")

        result = mark_answer(
            question        = q_text,
            question_number = q_num,
            q_type          = q_type,
            student_answer  = student,
            memo            = memo,
            marks           = marks,
            options         = options
        )

        # Format correct_answer for display
        if isinstance(memo, dict) and memo:
            correct_display = " | ".join(f"{k.split()[0]} → {v}" for k, v in memo.items())
        elif memo:
            # For MCQ, enrich with option text
            if q_type == "mcq" and options:
                correct_letter = str(memo).strip().upper()
                for opt in (options or []):
                    if isinstance(opt, dict) and opt.get("key","").upper() == correct_letter:
                        correct_display = f"{correct_letter}. {opt['value']}"
                        break
                else:
                    correct_display = str(memo)
            else:
                correct_display = str(memo)
        else:
            correct_display = "Not available"

        result["question_number"] = q_num
        result["question"]        = q_text
        result["type"]            = q_type
        result["marks"]           = marks
        result["student_answer"]  = student or "No answer"
        result["correct_answer"]  = correct_display
        result["earned"]          = result.get("score", 0)

        results.append(result)
        total_score += result["earned"]
        total_marks += marks

    return {
        "results":     results,
        "total_score": total_score,
        "total_marks": total_marks,
        "percentage":  round((total_score / total_marks) * 100, 2) if total_marks else 0
    }


# =========================
# 🧾 EXAM FEEDBACK SUMMARY
# =========================
def generate_exam_feedback(results, score, total, percentage):
    try:
        summary = [
            {"q": r.get("question_number"), "type": r.get("type"),
             "marks": r.get("marks"), "score": r.get("score"), "status": r.get("status")}
            for r in results
        ]
        prompt = f"""You are a motivating CAT Grade 12 teacher.
Score: {score}/{total} ({percentage}%)
Breakdown: {json.dumps(summary)}

Write 3-4 sentences: acknowledge the score, highlight strengths, identify weakest area, end with encouragement."""

        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.4
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"You scored {score}/{total} ({percentage}%). Keep practising! 🚀"