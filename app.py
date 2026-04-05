from dotenv import load_dotenv
load_dotenv()

import os
import json
import uuid
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import generate_answer
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
sessions = {}  # ⚠️ In-memory (replace with DB later)


# =========================
# 📁 HELPERS
# =========================
def load_exam(exam_name):
    path = os.path.join(EXAMS_FOLDER, exam_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# =========================
# 🏠 HOME (TEST UI)
# =========================
@app.route("/")
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AI Tutor + Exam Mocker</title>
    <style>
        body { font-family: Arial; max-width: 900px; margin: auto; padding: 20px; }
        h1 { text-align: center; }
        button { padding: 10px 15px; margin: 5px; cursor: pointer; }
        textarea { width: 100%; height: 120px; margin-top: 10px; }
        .box { border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-top: 20px; }
        .hidden { display: none; }
        select, input { padding: 10px; width: 100%; margin-top: 10px; }
    </style>
</head>
<body>

<h1>🚀 AI Learning System</h1>

<!-- MODE SWITCH -->
<div class="box">
    <h3>Select Mode</h3>
    <button onclick="setMode('tutor')">🎓 AI Tutor</button>
    <button onclick="setMode('exam')">📝 AI Exam Mocker</button>
</div>

<!-- ================= TUTOR ================= -->
<div id="tutorBox" class="box hidden">
    <h3>🎓 Ask AI Tutor</h3>
    <input id="tutorInput" placeholder="Ask anything..." />
    <button onclick="askTutor()">Ask</button>
    <div id="tutorOutput"></div>
</div>

<!-- ================= EXAM ================= -->
<div id="examBox" class="box hidden">
    <h3>📝 Exam Setup</h3>

    <button onclick="loadExams()">Load Exams</button>
    <select id="examSelect"></select>
    <button onclick="startExam()">Start Exam</button>

    <div id="examArea" class="hidden">
        <h4 id="questionText"></h4>
        <textarea id="answerBox" placeholder="Type your answer..."></textarea>

        <button onclick="saveAnswer()">💾 Save</button>
        <button onclick="prevQ()">⬅ Back</button>
        <button onclick="nextQ()">Next ➡</button>
        <button onclick="submitExam()">✅ Submit</button>

        <p id="progress"></p>
    </div>
</div>

<!-- ================= RESULTS ================= -->
<div id="resultsBox" class="box hidden"></div>

<script>
let mode = null;
let session_id = null;
let index = 0;
let total = 0;

// ================= MODE SWITCH =================
function setMode(m) {
    mode = m;

    document.getElementById("tutorBox").classList.add("hidden");
    document.getElementById("examBox").classList.add("hidden");

    if (m === "tutor") {
        document.getElementById("tutorBox").classList.remove("hidden");
    } else {
        document.getElementById("examBox").classList.remove("hidden");
    }
}

// ================= TUTOR =================
async function askTutor() {
    const q = document.getElementById("tutorInput").value;

    const res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ question: q, mode: "tutor" })
    });

    const data = await res.json();

    // ✅ SAFE ACCESS
    const answer = data.answer || "⚠️ No response from AI";

    document.getElementById("tutorOutput").innerHTML += 
        "<p><b>AI:</b> " + answer + "</p>";
}

// ================= LOAD EXAMS =================
async function loadExams() {
    const res = await fetch("/exams");
    const data = await res.json();

    console.log("API response:", data); // 👈 DEBUG

    const select = document.getElementById("examSelect");
    select.innerHTML = "";

    (data.exams || []).forEach(e => {
        const opt = document.createElement("option");
        opt.value = e;
        opt.text = e;
        select.appendChild(opt);
    });
}

// ================= START EXAM =================
async function startExam() {
    const exam = document.getElementById("examSelect").value;

    const res = await fetch("/start-exam", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ exam })
    });

    const data = await res.json();

    session_id = data.session_id;
    total = data.total_questions;
    index = 0;

    document.getElementById("examArea").classList.remove("hidden");

    loadQuestion();
}

// ================= LOAD QUESTION =================
async function loadQuestion() {

    
    const res = await fetch("/question", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ session_id, index })
    });

    const q = await res.json();
    const section = q.section || "SECTION";
    const number = q.number || `Q${index + 1}`;
    const question = q.question || "⚠️ Question missing";
    const marks = q.marks || 1;
    
    let html = `
    <h3>${section}</h3>

    <div style="display:flex; justify-content:space-between;">
        <h4>${number} ${question}</h4>
        <span><b>[${marks}]</b></span>
    </div>
`;

    // ================= MCQ =================
    console.log("QUESTION FULL:", q);
console.log("OPTIONS TYPE:", typeof q.options);
console.log("OPTIONS VALUE:", q.options);if (Array.isArray(q.options) && q.options.length > 0) {
    const letters = ["A", "B", "C", "D", "E"];

    html += `<div style="margin-top:10px;">`;

    q.options.forEach((opt, i) => {
        html += `
            <label style="display:block; margin:6px 0;">
                <input type="radio" name="mcq" value="${opt}">
                <b>${letters[i]}.</b> ${opt}
            </label>
        `;
    });

    html += `</div>`;
}

    
   
    // ================= TEXT =================
    else {
        html += `
            <textarea id="answerBox" 
                style="width:100%; height:120px; margin-top:10px;"
                placeholder="Write your answer here..."></textarea>
        `;
    }

    // ================= NAV =================
    html += `
        <div style="margin-top:15px;">
            <button onclick="saveAnswer()">💾 Save</button>
            <button onclick="prevQ()">⬅ Back</button>
            <button onclick="nextQ()">Next ➡</button>
            <button onclick="submitExam()">✅ Submit</button>
        </div>

        <p>Question ${index + 1} of ${total}</p>
    `;

    document.getElementById("examArea").innerHTML = html;

    // ================= LOAD SAVED =================
    if (q.options && q.saved_answer) {
        document.querySelectorAll('input[name="mcq"]').forEach(r => {
            if (r.value === q.saved_answer) {
                r.checked = true;
            }
        });
    } else if (q.saved_answer) {
        document.getElementById("answerBox").value = q.saved_answer;
    }
}

// ================= SAVE ANSWER =================
async function saveAnswer() {
    let answer = "";

    const selected = document.querySelector('input[name="mcq"]:checked');
    if (selected) {
        answer = selected.value;
    }

    const textBox = document.getElementById("answerBox");
    if (textBox) {
        answer = textBox.value;
    }

    await fetch("/answer", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            session_id,
            index,
            answer
        })
    });

    alert("Saved!");
}

// ================= NAVIGATION =================
function nextQ() {
    if (index < total - 1) {
        index++;
        loadQuestion();
    }
}

function prevQ() {
    if (index > 0) {
        index--;
        loadQuestion();
    }
}

// ================= SUBMIT =================
async function submitExam() {
    const res = await fetch("/submit", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ session_id })
    });

    const data = await res.json();

    let html = "<h3>Results</h3>";
    html += "<p><b>Score:</b> " + data.score + " / " + data.total + " (" + data.percentage + "%)</p>";

    data.results.forEach((r, i) => {
        html += `
            <div style="border:1px solid #ccc; margin:10px; padding:10px;">
                <b>Q${i+1}:</b> ${r.question}<br>
                <b>Your Answer:</b> ${r.student_answer}<br>
                <b>Correct Answer:</b> ${r.correct_answer}<br>
                <b>Status:</b> ${r.status}<br>
            </div>
        `;
    });

    document.getElementById("resultsBox").innerHTML = html;
    document.getElementById("resultsBox").classList.remove("hidden");
}
</script>

</body>
</html>
"""


# =========================
# 📚 GET AVAILABLE EXAMS
# =========================
@app.route("/exams", methods=["GET"])
def list_exams():
    try:
        print("EXAMS_FOLDER:", EXAMS_FOLDER)  # 👈 DEBUG

        exams = [
            f for f in os.listdir(EXAMS_FOLDER)
            if f.endswith("_exam.json")
        ]

        return jsonify({"exams": exams})

    except Exception as e:
        print("ERROR:", str(e))  # 👈 DEBUG
        return jsonify({"error": str(e)})

# =========================
# ▶️ START EXAM SESSION
# =========================
@app.route("/start-exam", methods=["POST"])
def start_exam():
    try:
        data = request.get_json()
        exam_name = data.get("exam")

        exam = load_exam(exam_name)

        if not exam:
            return jsonify({"error": "❌ Exam not found"})

        session_id = str(uuid.uuid4())

        sessions[session_id] = {
            "exam": exam_name,
            "answers": {},
        }

        return jsonify({
            "session_id": session_id,
            "total_questions": len(exam["questions"])
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# =========================
# ❓ GET QUESTION (NEXT/BACK)
# =========================
@app.route("/question", methods=["POST"])
def get_question():
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        index = data.get("index", 0)

        session = sessions.get(session_id)
        if not session:
            return jsonify({"error": "Invalid session"})

        exam = load_exam(session["exam"])

        flat_questions = []

        # =========================
        # 🔥 OPTION NORMALIZER
        # =========================
        def normalize_options(options):
            if not options:
                return None

            # already list
            if isinstance(options, list):
                return [str(o).strip() for o in options if str(o).strip()]

            # string → split
            if isinstance(options, str):
                if "\n" in options:
                    parts = options.split("\n")
                elif "," in options:
                    parts = options.split(",")
                else:
                    parts = [options]

                return [p.strip() for p in parts if p.strip()]

            # dict → values
            if isinstance(options, dict):
                return [str(v).strip() for v in options.values()]

            return None

        # =========================
        # 🔥 MCQ EXTRACTOR FROM TEXT
        # =========================
        import re

        def extract_mcq_from_text(question_text):
            """
            Extract options if question contains:
            A. xxx B. xxx C. xxx D. xxx
            """
            pattern = r"(?:A[\.\)]\s*(.*?)\s*)(?:B[\.\)]\s*(.*?)\s*)(?:C[\.\)]\s*(.*?)\s*)(?:D[\.\)]\s*(.*))"
            match = re.search(pattern, question_text, re.IGNORECASE)

            if match:
                options = [opt.strip() for opt in match.groups() if opt.strip()]

                # remove options from question text
                clean_question = re.split(r"A[\.\)]", question_text)[0].strip()

                return clean_question, options

            return question_text, None

        # =========================
        # 🔥 BUILD QUESTIONS
        # =========================
        if "sections" in exam:
            for section in exam.get("sections", []):
                section_title = section.get("title", "SECTION")

                for q in section.get("questions", []):
                    question_text = q.get("question", "No question text")

                    # normalize existing options
                    options = normalize_options(q.get("options"))

                    # 🔥 if no options → try extract from question text
                    if not options:
                        question_text, extracted_options = extract_mcq_from_text(question_text)
                        options = extracted_options

                    flat_questions.append({
                        "question": question_text,
                        "number": q.get("number"),
                        "section": section_title,
                        "options": options,
                        "marks": q.get("marks", 1)
                    })
        else:
            for i, q in enumerate(exam.get("questions", [])):
                question_text = q.get("question", "No question text")

                options = normalize_options(q.get("options"))

                if not options:
                    question_text, extracted_options = extract_mcq_from_text(question_text)
                    options = extracted_options

                flat_questions.append({
                    "question": question_text,
                    "number": f"Q{i+1}",
                    "section": "General",
                    "options": options,
                    "marks": q.get("marks", 1)
                })

        # =========================
        # 🔥 SAFE INDEX
        # =========================
        if index < 0 or index >= len(flat_questions):
            return jsonify({"error": "Out of range"})

        q = flat_questions[index]

        print("FINAL QUESTION:", q)  # debug

        return jsonify({
            "question": q["question"] or "⚠️ Missing question",
            "number": q["number"] or f"Q{index+1}",
            "section": q["section"] or "SECTION",
            "options": q["options"] if q["options"] else [],
            "is_mcq": True if q["options"] else False,
            "marks": q["marks"] or 1,
            "index": index,
            "saved_answer": session["answers"].get(str(index), "")
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "question": "⚠️ Error loading question",
            "number": "",
            "section": "",
            "marks": 0,
            "options": [],
            "error": str(e)
        })


# =========================
# 💾 SAVE ANSWER (AUTO SLOT)
# =========================
@app.route("/answer", methods=["POST"])
def save_answer():
    try:
        data = request.get_json()

        session_id = data.get("session_id")
        index = data.get("index")
        answer = data.get("answer")

        session = sessions.get(session_id)

        if not session:
            return jsonify({"error": "Invalid session"})

        session["answers"][str(index)] = answer

        return jsonify({"status": "saved"})

    except Exception as e:
        return jsonify({"error": str(e)})


# =========================
# 📝 SUBMIT + MARK EXAM
# =========================
@app.route("/submit", methods=["POST"])
def submit_exam():
    try:
        data = request.get_json()

        session_id = data.get("session_id")
        session = sessions.get(session_id)

        if not session:
            return jsonify({"error": "Invalid session"})

        exam = load_exam(session["exam"])
        questions = exam["questions"]

        results = []
        score = 0
        total = 0

        for i, q in enumerate(questions):

            student_answer = session["answers"].get(str(i), "").strip()
            memo = q.get("memo", "").strip()
            marks = q.get("marks", 1)

            total += marks

            # 🧠 BASIC MARKING (upgrade later to AI)
            if student_answer.lower() == memo.lower():
                status = "correct"
                earned = marks
            elif student_answer:
                status = "partial"
                earned = marks // 2
            else:
                status = "missing"
                earned = 0

            score += earned

            results.append({
                "question": q["question"],
                "marks": marks,
                "student_answer": student_answer,
                "correct_answer": memo,
                "status": status,
                "earned": earned
            })

        return jsonify({
            "score": score,
            "total": total,
            "percentage": round((score / total) * 100, 2) if total else 0,
            "results": results,
            "message": "✅ Exam completed! Review your performance and try again to improve 🚀"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


# =========================
# 🧠 AI TUTOR (RAG CHAT)
# =========================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        question = data.get("question", "")
        mode = data.get("mode", "tutor")

        if not question:
            return jsonify({
                "answer": "⚠️ Please enter a question."
            })

        if mode == "tutor":
            chunks = rag.search(question)
            context_parts = []

            for chunk in chunks:
                if isinstance(chunk, dict) and "content" in chunk:
                    context_parts.append(chunk["content"])
                elif isinstance(chunk, str):
                    context_parts.append(chunk)

            context = " ".join(context_parts)

            answer = generate_answer(context, question)

            return jsonify({
                "answer": answer
            })

        return jsonify({
            "answer": "⚠️ Invalid mode selected."
        })

    except Exception as e:
        return jsonify({
            "answer": f"⚠️ Server error: {str(e)}"
        })

# =========================
# ▶️ RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True, port=8000)