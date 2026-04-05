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


# =========================
# 🎓 AI TUTOR (RAG ANSWER)
# =========================
def generate_answer(context, question):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""
You are a friendly and smart CAT Grade 12 tutor.

Use the context below to answer the student's question.

Context:
{context}

Question:
{question}

Instructions:
- Explain clearly in simple terms
- Use examples if helpful
- Keep it concise but helpful

Answer:
"""
            }],
            max_tokens=400
        )

        # ✅ SAFE EXTRACTION
        answer = response.choices[0].message.content

        if not answer or answer.strip() == "":
            return "⚠️ No response generated. Try rephrasing your question."

        return answer.strip()

    except Exception as e:
        return f"⚠️ Error generating answer: {str(e)}"

# =========================
# 🧠 AI EXAM MARKING
# =========================
def mark_answer(question, student_answer, memo, marks):
    """
    Marks a single student's answer using AI based on meaning,
    returning score, feedback, and status
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""
You are a strict but fair CAT Grade 12 examiner.

Question:
{question}

Memo (Correct Answer):
{memo}

Student Answer:
{student_answer}

Total Marks:
{marks}

Instructions:
- Compare meaning, not exact wording
- Award marks fairly
- Give partial credit if some understanding is shown
- If wrong, give 0
- Be consistent with real exam marking

Return ONLY valid JSON in this format:
{{
  "score": number,
  "feedback": "short helpful feedback",
  "status": "correct | partial | incorrect"
}}
"""
            }],
            max_tokens=200
        )

        content = response.choices[0].message.content.strip()
        return json.loads(content)

    except Exception as e:
        # fallback if JSON fails
        return {
            "score": 0,
            "feedback": f"⚠️ Could not evaluate answer: {e}",
            "status": "incorrect"
        }


# =========================
# 🧾 AI EXAM FEEDBACK (SUMMARY)
# =========================
def generate_exam_feedback(results):
    """
    Generates overall exam feedback after all answers are marked.
    `results` is expected as a list of dicts:
    [{ "question": "...", "student_answer": "...", "score": X, "marks": Y }, ...]
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""
You are a motivating teacher.

Here are a student's exam results:
{json.dumps(results, indent=2)}

Give:
- A short performance summary
- Encouragement
- One improvement tip

Keep it friendly and motivating.
"""
            }],
            max_tokens=200
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Could not generate exam feedback: {e}"


# =========================
# 🔗 HELPER: Batch Marking (Optional)
# =========================
def mark_exam_batch(exam_questions, student_answers):
    """
    Marks all student answers for a full exam.
    Returns detailed results and total score.
    """
    results = []
    total_score = 0
    for i, q in enumerate(exam_questions):
        student_ans = student_answers.get(str(i), "")
        res = mark_answer(
            question=q.get("question", ""),
            student_answer=student_ans,
            memo=q.get("memo", ""),
            marks=q.get("marks", 1)
        )
        res["question"] = q.get("question", "")
        res["student_answer"] = student_ans
        res["marks"] = q.get("marks", 1)
        results.append(res)
        total_score += res.get("score", 0)

    return {
        "results": results,
        "total_score": total_score
    }