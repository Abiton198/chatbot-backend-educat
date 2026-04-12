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


def generate_answer(context, question):
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":f"You are a friendly CAT Grade 12 tutor.\nContext: {context or 'Use general CAT knowledge.'}\nQuestion: {question}\nAnswer:"}],
            max_tokens=500, temperature=0.3)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {e}"


def mark_answer(question, question_number, q_type, student_answer, memo, marks, options=None):
    """
    Grade a single answer. memo comes from q['memo'] — never from an external lookup.
    """
    student = str(student_answer).strip() if student_answer else ""

    if q_type == "mcq":
        correct = str(memo).strip().upper()
        ans     = student.upper()
        if not correct:
            return {"score":0,"feedback":"No memo.","status":"no_memo"}
        if not ans:
            return {"score":0,"feedback":f"No answer selected. Correct: {correct}.","status":"missing"}
        opt_text = ""
        for opt in (options or []):
            if isinstance(opt,dict) and opt.get("key","").upper()==correct:
                opt_text=f" ({opt['value']})"
                break
        if ans == correct:
            return {"score":marks,"feedback":f"Correct! {correct}{opt_text}.","status":"correct"}
        return {"score":0,"feedback":f"Incorrect. You selected {ans}; correct is {correct}{opt_text}.","status":"incorrect"}

    if q_type == "true_false":
        if not memo:
            return {"score":0,"feedback":"No memo.","status":"no_memo"}
        cl = str(memo).strip().lower()
        al = student.lower()
        if not al:
            return {"score":0,"feedback":f"No answer. Correct: {memo}","status":"missing"}
        ct = cl.startswith("true")
        at = al.startswith("true")
        if ct and at:
            return {"score":marks,"feedback":"Correct — True.","status":"correct"}
        if not ct and not at:
            def corr(s): parts=re.split(r"[-—]",s,maxsplit=1); return parts[1].strip().lower() if len(parts)>1 else ""
            mw=corr(cl); aw=corr(al)
            if not mw or (aw and (mw in aw or aw in mw)):
                return {"score":marks,"feedback":f"Correct — False, correction: {aw or mw}.","status":"correct"}
            return {"score":marks//2,"feedback":f"Correct it's FALSE but wrong correction. Expected '{mw}', got '{aw or '(none)'}'.","status":"partial"}
        return {"score":0,"feedback":f"Incorrect. Correct: {memo}.","status":"incorrect"}

    if q_type == "matching":
        if not isinstance(memo,dict) or not memo:
            return {"score":0,"feedback":"No memo for matching.","status":"no_memo"}
        try:
            sm = json.loads(student) if student else {}
        except Exception:
            sm = {}
        correct_count=0; details=[]
        for k,v in memo.items():
            sv=sm.get(k,"")
            sl=sv.strip().split(".")[0].strip().upper() if sv else ""
            cv=str(v).strip().upper()
            if sl==cv:
                correct_count+=1; details.append(f"✅ {k.split()[0]}:{sl}")
            else:
                details.append(f"❌ {k.split()[0]}: got '{sl or '—'}' need '{cv}'")
        total=len(memo)
        earned=round((correct_count/total)*marks) if total else 0
        status="correct" if earned==marks else "partial" if earned>0 else "incorrect"
        return {"score":earned,"feedback":f"{correct_count}/{total} correct. "+" | ".join(details),"status":status}

    # Open
    if not student:
        return {"score":0,"feedback":"No answer provided.","status":"missing"}
    memo_text = memo if isinstance(memo,str) else json.dumps(memo) if memo else ""
    has_memo  = bool(memo_text.strip())
    prompt = f"""You are a strict South African NSC CAT examiner.

Question {question_number} ({marks} mark{"s" if marks!=1 else ""}):
{question}

{"Marking guideline:\n"+memo_text if has_memo else "No guideline — use CAT Grade 12 knowledge."}

Student's answer:
{student}

Award 1 mark per distinct correct fact. Return ONLY valid JSON:
{{"score":<int 0-{marks}>,"feedback":"<specific: what was correct, what was missing>","status":"<correct|partial|incorrect>"}}"""

    for attempt in range(2):
        try:
            r=client.chat.completions.create(model=MODEL,messages=[{"role":"user","content":prompt}],max_tokens=250,temperature=0)
            content=r.choices[0].message.content.strip()
            if content.startswith("```"):
                content=content.split("```")[1].lstrip("json").strip()
            result=json.loads(content)
            result["score"]=max(0,min(int(result.get("score",0)),marks))
            return result
        except Exception as e:
            if attempt==0: time.sleep(5)
            else: return {"score":0,"feedback":f"Could not mark: {e}","status":"incorrect"}


def generate_exam_feedback(results, score, total, percentage):
    try:
        weak=[r["question_number"] for r in results if r["status"]!="correct"]
        prompt=f"""You are a motivating CAT Grade 12 teacher.
Score: {score}/{total} ({percentage}%)
Wrong/partial questions: {", ".join(weak) if weak else "none — perfect score!"}
Write 3-4 sentences of encouraging, specific feedback. Mention topics to revise."""
        r=client.chat.completions.create(model=MODEL,messages=[{"role":"user","content":prompt}],max_tokens=220,temperature=0.4)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"Score: {score}/{total} ({percentage}%). Keep practising! 🚀"