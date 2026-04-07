from dotenv import load_dotenv
load_dotenv()

import os
import json
import uuid
import re
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import generate_answer, mark_exam_batch, generate_exam_feedback
from rag import RAGIndex

app = Flask(__name__)

# =========================
# 🌐 CORS CONFIG
# =========================
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "http://localhost:5176",
    "https://edu-cat.netlify.app"
]}})

# =========================
# 🧠 INIT SYSTEMS
# =========================
rag = RAGIndex()

EXAMS_FOLDER = "exams"
sessions = {}  # ⚠️ In-memory — replace with DB for production


# =========================
# 📁 LOAD EXAM FILE
# =========================
def load_exam(exam_name):
    path = os.path.join(EXAMS_FOLDER, exam_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# =========================
# 🔥 FLATTEN EXAM → QUESTION LIST
# =========================
def flatten_exam(exam):
    flat     = []
    sections = exam.get("sections", [])

    if not sections and "questions" in exam:
        sections = [{
            "section":              "A",
            "section_title":        None,
            "section_instructions": None,
            "total_marks":          None,
            "questions":            exam["questions"]
        }]

    for section in sections:
        sec_label        = section.get("section", "")
        sec_title        = section.get("section_title") or ""
        sec_instructions = section.get("section_instructions") or ""
        sec_marks        = section.get("total_marks")

        for q in section.get("questions", []):
            q_type          = q.get("type", "open").lower()
            q_text          = q.get("question", "").strip()
            marks           = q.get("marks", 1)
            memo            = q.get("memo", "")
            q_id            = q.get("id")
            question_number = q.get("question_number", f"Q{q_id}" if q_id else "")
            parent_question = q.get("parent_question", "")
            parent_context  = q.get("parent_context")

            options  = None
            column_a = column_b = None

            if q_type == "mcq":
                raw_opts = q.get("options")
                if isinstance(raw_opts, dict):
                    options = [{"key": k, "value": v} for k, v in sorted(raw_opts.items()) if str(v).strip()]
                elif isinstance(raw_opts, list):
                    options = [{"key": chr(65+i), "value": str(v).strip()} for i, v in enumerate(raw_opts) if str(v).strip()]
                if not options:
                    q_text, options = extract_mcq_from_text(q_text)

            if q_type == "matching":
                column_a = q.get("column_a", [])
                column_b = q.get("column_b", [])

            flat.append({
                "id":                   q_id,
                "question_number":      question_number,
                "parent_question":      parent_question,
                "parent_context":       parent_context,
                "section":              sec_label,
                "section_title":        sec_title,
                "section_instructions": sec_instructions,
                "section_total_marks":  sec_marks,
                "question":             q_text or "⚠️ Question text missing",
                "type":                 q_type,
                "options":              options,
                "column_a":             column_a,
                "column_b":             column_b,
                "marks":                marks,
                "memo":                 memo,
                "saved_answer":         ""
            })

    return flat


def extract_mcq_from_text(question_text):
    pattern = r"(?:A[\.\)]\s*(.*?)\s*)(?:B[\.\)]\s*(.*?)\s*)(?:C[\.\)]\s*(.*?)\s*)(?:D[\.\)]\s*(.*))"
    match   = re.search(pattern, question_text, re.IGNORECASE | re.DOTALL)
    if match:
        keys    = ["A", "B", "C", "D"]
        options = [{"key": keys[i], "value": opt.strip()} for i, opt in enumerate(match.groups()) if opt and opt.strip()]
        clean   = re.split(r"A[\.\)]", question_text, flags=re.IGNORECASE)[0].strip()
        return clean, options if options else None
    return question_text, None


# =========================
# 🏠 HOME — TEST UI
# =========================
@app.route("/")
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>EduCAT — AI Learning System</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: Arial, sans-serif; max-width: 960px; margin: auto; padding: 20px; background: #f9f9f9; }
        h1 { text-align: center; color: #2c3e50; }
        button { padding: 10px 18px; margin: 5px; cursor: pointer; border: none; border-radius: 6px; background: #3498db; color: white; font-size: 14px; }
        button:hover { opacity: 0.88; }
        button:disabled { background: #bdc3c7; cursor: not-allowed; }
        textarea { width: 100%; height: 130px; margin-top: 10px; padding: 10px; border-radius: 6px; border: 1px solid #ddd; font-size: 14px; resize: vertical; }
        .box { border: 1px solid #ddd; padding: 18px; border-radius: 10px; margin-top: 20px; background: white; }
        .hidden { display: none; }
        select, input[type=text] { padding: 10px; width: 100%; margin-top: 10px; border-radius: 6px; border: 1px solid #ccc; font-size: 14px; }
        .sec-header { background: #eaf0fb; padding: 10px 14px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid #3498db; }
        .sec-header .sec-label { font-weight: bold; font-size: 15px; color: #2c3e50; }
        .sec-header .sec-sub   { font-size: 13px; color: #555; margin-top: 3px; }
        .parent-heading { font-size: 13px; font-weight: bold; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
        .parent-context { background: #fefae0; border-left: 3px solid #f1c40f; padding: 8px 12px; border-radius: 5px; font-size: 13px; color: #555; margin-bottom: 10px; }
        .q-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
        .q-row .q-num  { font-weight: bold; white-space: nowrap; color: #2c3e50; min-width: 40px; }
        .q-row .q-text { flex: 1; font-size: 15px; line-height: 1.5; }
        .q-row .q-mark { white-space: nowrap; font-weight: bold; color: #e74c3c; }
        .option-label { display: block; margin: 8px 0; padding: 9px 14px; border: 1px solid #ddd; border-radius: 6px; cursor: pointer; }
        .option-label:hover { background: #f0f4ff; }
        .tf-label { display: inline-block; margin-right: 20px; padding: 9px 18px; border: 1px solid #ddd; border-radius: 6px; cursor: pointer; }
        .tf-label:hover { background: #f0f4ff; }
        .match-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .match-table th { text-align: left; padding: 8px 10px; border-bottom: 2px solid #ddd; background: #f5f5f5; }
        .match-table td { padding: 8px 10px; border-bottom: 1px solid #eee; vertical-align: middle; }
        .match-table select { width: 100%; padding: 6px; border-radius: 4px; border: 1px solid #ccc; }
        .nav-bar { margin-top: 18px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
        .nav-bar .submit-btn { margin-left: auto; background: #e74c3c; }
        .progress { margin-top: 10px; color: #888; font-size: 13px; }
        .result-card { border: 1px solid #ccc; margin: 10px 0; padding: 12px 16px; border-radius: 8px; font-size: 14px; line-height: 1.6; }
        .feedback-box { background: #eafaf1; border-left: 4px solid #27ae60; padding: 14px; border-radius: 8px; margin-top: 12px; font-size: 14px; }
    </style>
</head>
<body>

<h1>🎓 EduCAT — AI Learning System</h1>

<div class="box">
    <h3>Select Mode</h3>
    <button onclick="setMode('tutor')">🎓 AI Tutor</button>
    <button onclick="setMode('exam')">📝 Exam Mocker</button>
</div>

<div id="tutorBox" class="box hidden">
    <h3>🎓 Ask AI Tutor</h3>
    <input type="text" id="tutorInput" placeholder="Ask anything about CAT..." />
    <button onclick="askTutor()" style="margin-top:10px;">Ask</button>
    <div id="tutorOutput" style="margin-top:12px;"></div>
</div>

<div id="examBox" class="box hidden">
    <h3>📝 Exam Setup</h3>
    <button onclick="loadExams()">🔄 Load Exams</button>
    <select id="examSelect"></select>
    <div id="memoStatus" style="margin-top:8px; font-size:13px; color:#888;"></div>
    <button onclick="startExam()" style="margin-top:10px; background:#27ae60;">▶ Start Exam</button>
    <div id="examArea" class="hidden" style="margin-top:20px;"></div>
</div>

<div id="resultsBox" class="box hidden"></div>

<script>
let mode       = null;
let session_id = null;
let index      = 0;
let total      = 0;

function setMode(m) {
    mode = m;
    document.getElementById("tutorBox").classList.add("hidden");
    document.getElementById("examBox").classList.add("hidden");
    if (m === "tutor") document.getElementById("tutorBox").classList.remove("hidden");
    else               document.getElementById("examBox").classList.remove("hidden");
}

async function askTutor() {
    const q = document.getElementById("tutorInput").value.trim();
    if (!q) return;
    const res  = await fetch("/chat", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({question:q, mode:"tutor"}) });
    const data = await res.json();
    document.getElementById("tutorOutput").innerHTML += `<p><b>Q:</b> ${q}<br><b>AI:</b> ${data.answer || "⚠️ No response"}</p><hr>`;
    document.getElementById("tutorInput").value = "";
}

async function loadExams() {
    const res  = await fetch("/exams");
    const data = await res.json();
    const sel  = document.getElementById("examSelect");
    sel.innerHTML = "";
    (data.exams || []).forEach(e => {
        const opt = document.createElement("option");
        opt.value = e;
        opt.text  = e.replace("_exam.json","").replace(/_/g," ");
        sel.appendChild(opt);
    });
}

async function startExam() {
    const exam = document.getElementById("examSelect").value;
    if (!exam) { alert("Select an exam first."); return; }
    const res  = await fetch("/start-exam", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({exam}) });
    const data = await res.json();
    if (data.error) { alert("Error: " + data.error); return; }
    session_id = data.session_id;
    total      = data.total_questions;
    index      = 0;

    // Show memo status
    const memoEl = document.getElementById("memoStatus");
    memoEl.innerHTML = data.memo_merged
        ? "✅ Memo loaded — AI marking enabled"
        : "⚠️ No memo loaded — AI will give partial feedback only";

    document.getElementById("examArea").classList.remove("hidden");
    document.getElementById("resultsBox").classList.add("hidden");
    loadQuestion();
}

async function loadQuestion() {
    const res = await fetch("/question", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ session_id, index }) });
    const q   = await res.json();

    if (q.error) {
        document.getElementById("examArea").innerHTML = `<p style="color:red">⚠️ ${q.error}</p>`;
        return;
    }

    let html = "";

    // Section header
    html += `
        <div class="sec-header">
            <div class="sec-label">SECTION ${q.section}${q.section_title ? " — " + q.section_title : ""}</div>
            ${q.section_instructions ? `<div class="sec-sub">${q.section_instructions}</div>` : ""}
            ${q.section_total_marks  ? `<div class="sec-sub">Total marks for this section: <b>${q.section_total_marks}</b></div>` : ""}
        </div>`;

    if (q.parent_question) html += `<div class="parent-heading">${q.parent_question}</div>`;
    if (q.parent_context)  html += `<div class="parent-context">📌 ${q.parent_context}</div>`;

    html += `
        <div class="q-row" style="margin-top:10px;">
            <span class="q-num">${q.question_number}.</span>
            <span class="q-text">${q.question}</span>
            <span class="q-mark">(${q.marks})</span>
        </div>`;

    // MCQ
    if (q.type === "mcq" && Array.isArray(q.options) && q.options.length > 0) {
        html += `<div style="margin-top:14px;">`;
        q.options.forEach(opt => {
            const chk = q.saved_answer === opt.key ? "checked" : "";
            html += `<label class="option-label"><input type="radio" name="mcq" value="${opt.key}" ${chk}> <b>${opt.key}.</b>&nbsp; ${opt.value}</label>`;
        });
        html += `</div>`;
    }
    // True/False
    else if (q.type === "true_false") {
        const savedTF = q.saved_answer || "";
        html += `
            <div style="margin-top:14px;">
                <label class="tf-label"><input type="radio" name="tf" value="True"  ${savedTF==="True"?"checked":""}> ✅ True</label>
                <label class="tf-label"><input type="radio" name="tf" value="False" ${savedTF==="False"||savedTF.startsWith("False")?"checked":""}> ❌ False</label>
                <div id="tfCorrection" style="margin-top:12px;${savedTF.startsWith("False")?"":"display:none"}">
                    <label style="font-size:13px;color:#555;">If FALSE — write the corrected word/phrase:</label>
                    <input type="text" id="tfCorrectionBox" placeholder="e.g. secondary memory"
                        value="${savedTF.startsWith("False — ") ? savedTF.replace("False — ","") : ""}">
                </div>
            </div>
            <script>
                document.querySelectorAll('input[name="tf"]').forEach(r => {
                    r.addEventListener("change", () => {
                        document.getElementById("tfCorrection").style.display = r.value==="False"&&r.checked ? "block" : r.value==="True"&&r.checked ? "none" : "";
                    });
                });
            <\\/script>`;
    }
    // Matching
    else if (q.type === "matching" && Array.isArray(q.column_a) && q.column_a.length > 0) {
        let saved = {};
        try { saved = JSON.parse(q.saved_answer || "{}"); } catch(e) {}
        html += `
            <p style="margin-top:12px;font-size:13px;color:#555;"><i>Match each item in COLUMN A to the correct answer in COLUMN B.</i></p>
            <table class="match-table">
                <thead><tr><th style="width:45%">COLUMN A</th><th>COLUMN B — Your Match</th></tr></thead>
                <tbody>`;
        q.column_a.forEach((item, i) => {
            const savedVal = saved[item] || "";
            html += `<tr><td>${item}</td><td><select name="match_${i}" data-item="${encodeURIComponent(item)}">
                <option value="">-- Select --</option>`;
            q.column_b.forEach(b => {
                html += `<option value="${b}" ${savedVal===b?"selected":""}>${b}</option>`;
            });
            html += `</select></td></tr>`;
        });
        html += `</tbody></table>`;
    }
    // Open
    else {
        html += `<textarea id="answerBox" placeholder="Write your answer here..." style="margin-top:14px;">${q.saved_answer || ""}</textarea>`;
    }

    // Nav
    html += `
        <div class="nav-bar">
            <button onclick="saveAnswer()">💾 Save</button>
            <button onclick="prevQ()" ${index===0?"disabled":""}>⬅ Back</button>
            <button onclick="nextQ()" ${index>=total-1?"disabled":""}>Next ➡</button>
            <button class="submit-btn" onclick="submitExam()">✅ Submit Exam</button>
        </div>
        <p class="progress">Question ${index+1} of ${total}</p>`;

    document.getElementById("examArea").innerHTML = html;
}

async function saveAnswer() {
    let answer = "";
    const mcqSel = document.querySelector('input[name="mcq"]:checked');
    if (mcqSel) answer = mcqSel.value;

    const tfSel = document.querySelector('input[name="tf"]:checked');
    if (tfSel) {
        if (tfSel.value === "False") {
            const corr = (document.getElementById("tfCorrectionBox")?.value || "").trim();
            answer = corr ? `False — ${corr}` : "False";
        } else {
            answer = "True";
        }
    }

    const matchSels = document.querySelectorAll('select[name^="match_"]');
    if (matchSels.length > 0) {
        const obj = {};
        matchSels.forEach(s => { if (s.value) obj[decodeURIComponent(s.dataset.item)] = s.value; });
        answer = JSON.stringify(obj);
    }

    const textBox = document.getElementById("answerBox");
    if (textBox) answer = textBox.value;

    const res  = await fetch("/answer", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ session_id, index, answer }) });
    const data = await res.json();
    if (data.status === "saved") {
        const btn = document.querySelector("button[onclick='saveAnswer()']");
        if (btn) { btn.textContent = "✅ Saved!"; setTimeout(() => btn.textContent = "💾 Save", 1500); }
    }
}

function nextQ() { if (index < total-1) { index++; loadQuestion(); } }
function prevQ() { if (index > 0)       { index--; loadQuestion(); } }

async function submitExam() {
    if (!confirm("Submit exam? You cannot change answers after this.")) return;
    const res  = await fetch("/submit", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({session_id}) });
    const data = await res.json();
    if (data.error) { alert("Error: " + data.error); return; }

    let html = `
        <h3>📊 Results</h3>
        <p><b>Score:</b> ${data.score} / ${data.total} &nbsp;|&nbsp; <b>${data.percentage}%</b></p>`;

    if (data.feedback) {
        html += `<div class="feedback-box">🤖 <b>AI Feedback:</b><br>${data.feedback}</div>`;
    }

    data.results.forEach(r => {
        const bg   = r.status==="correct" ? "#d4edda" : r.status==="partial" ? "#fff3cd" : "#f8d7da";
        const icon = r.status==="correct" ? "✅" : r.status==="partial" ? "⚠️" : "❌";
        html += `
            <div class="result-card" style="background:${bg};">
                <b>${icon} ${r.question_number} [${r.marks}]:</b> ${r.question}<br>
                <b>Your Answer:</b> ${r.student_answer || "<i>No answer</i>"}<br>
                <b>Correct Answer:</b> ${r.correct_answer || "<i>Not available</i>"}<br>
                <b>AI Feedback:</b> ${r.feedback || "—"}<br>
                <b>Status:</b> ${r.status} &nbsp;|&nbsp; <b>Earned:</b> ${r.earned}/${r.marks}
            </div>`;
    });

    const box = document.getElementById("resultsBox");
    box.innerHTML = html;
    box.classList.remove("hidden");
    box.scrollIntoView({ behavior: "smooth" });
}
</script>
</body>
</html>
"""


# =========================
# 📚 LIST EXAMS
# =========================
@app.route("/exams", methods=["GET"])
def list_exams():
    try:
        exams = sorted([f for f in os.listdir(EXAMS_FOLDER) if f.endswith("_exam.json")])
        return jsonify({"exams": exams})
    except Exception as e:
        return jsonify({"error": str(e)})


# =========================
# ▶️ START EXAM SESSION
# =========================
@app.route("/start-exam", methods=["POST"])
def start_exam():
    try:
        data      = request.get_json()
        exam_name = data.get("exam")
        exam      = load_exam(exam_name)

        if not exam:
            return jsonify({"error": "❌ Exam not found"})

        flat = flatten_exam(exam)
        if not flat:
            return jsonify({"error": "❌ No questions found in this exam"})

        sid = str(uuid.uuid4())
        sessions[sid] = {
            "exam":      exam_name,
            "questions": flat,
            "answers":   {}
        }

        return jsonify({
            "session_id":      sid,
            "total_questions": len(flat),
            "memo_merged":     exam.get("memo_merged", False)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


# =========================
# ❓ GET QUESTION
# =========================
@app.route("/question", methods=["POST"])
def get_question():
    try:
        data    = request.get_json()
        sid     = data.get("session_id")
        index   = data.get("index", 0)
        session = sessions.get(sid)

        if not session:
            return jsonify({"error": "Invalid session"})

        flat = session["questions"]
        if index < 0 or index >= len(flat):
            return jsonify({"error": "Question index out of range"})

        q = flat[index].copy()
        q["saved_answer"] = session["answers"].get(str(index), "")
        return jsonify(q)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


# =========================
# 💾 SAVE ANSWER
# =========================
@app.route("/answer", methods=["POST"])
def save_answer():
    try:
        data    = request.get_json()
        sid     = data.get("session_id")
        index   = data.get("index")
        answer  = data.get("answer", "")
        session = sessions.get(sid)

        if not session:
            return jsonify({"error": "Invalid session"})

        session["answers"][str(index)] = answer
        return jsonify({"status": "saved"})

    except Exception as e:
        return jsonify({"error": str(e)})


# =========================
# 📝 SUBMIT + AI MARK
# Uses mark_exam_batch from model.py
# which feeds on memo from the exam JSON file
# =========================
@app.route("/submit", methods=["POST"])
def submit_exam():
    try:
        data    = request.get_json()
        sid     = data.get("session_id")
        session = sessions.get(sid)

        if not session:
            return jsonify({"error": "Invalid session"})

        exam_name = session["exam"]
        flat      = session["questions"]
        answers   = session["answers"]

        # ── AI marks all answers using memo from exam JSON ──
        marked = mark_exam_batch(exam_name, flat, answers)

        # Add earned field (same as score per question)
        for r in marked["results"]:
            r["earned"] = r.get("score", 0)

        # ── AI generates overall feedback ──────────────────
        feedback = generate_exam_feedback(
            results    = marked["results"],
            score      = marked["total_score"],
            total      = marked["total_marks"],
            percentage = marked["percentage"]
        )

        return jsonify({
            "score":      marked["total_score"],
            "total":      marked["total_marks"],
            "percentage": marked["percentage"],
            "results":    marked["results"],
            "feedback":   feedback,
            "message":    "✅ Exam submitted! Review your answers below 🚀"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


# =========================
# 🧠 AI TUTOR
# Feeds on RAG context from theory books only
# =========================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data     = request.get_json()
        question = data.get("question", "")
        mode     = data.get("mode", "tutor")

        if not question:
            return jsonify({"answer": "⚠️ Please enter a question."})

        if mode == "tutor":
            chunks  = rag.search(question)
            context = " ".join(
                c["content"] if isinstance(c, dict) and "content" in c else str(c)
                for c in chunks
            )
            answer = generate_answer(context, question)
            return jsonify({"answer": answer})

        return jsonify({"answer": "⚠️ Invalid mode."})

    except Exception as e:
        return jsonify({"answer": f"⚠️ Server error: {str(e)}"})


# =========================
# ▶️ RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True, port=8000)