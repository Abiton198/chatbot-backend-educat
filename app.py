from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import generate_answer
from rag import RAGIndex
import traceback

app = Flask(__name__)
CORS(app)
rag = RAGIndex()

@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>CAT AI Tutor</title></head>
    <body style="font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px;">
        <h1>📚 CAT AI Tutor</h1>
        <input id="question" type="text" placeholder="Ask a question..."
               style="width: 100%; padding: 10px; font-size: 16px;" />
        <button onclick="ask()"
                style="margin-top: 10px; padding: 10px 20px; font-size: 16px;">Ask</button>
        <div id="answer" style="margin-top: 20px; padding: 15px; background: #f0f0f0;
                                 border-radius: 8px; display:none;"></div>
        <script>
            async function ask() {
                const question = document.getElementById("question").value;
                if (!question.trim()) return;
                document.getElementById("answer").style.display = "block";
                document.getElementById("answer").innerText = "Thinking...";
                try {
                    const res = await fetch("/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question: question })
                    });
                    const data = await res.json();
                    document.getElementById("answer").innerText = data.answer ?? "No answer returned";
                } catch (err) {
                    document.getElementById("answer").innerText = "Request failed: " + err.message;
                }
            }
            document.getElementById("question").addEventListener("keypress", function(e) {
                if (e.key === "Enter") ask();
            });
        </script>
    </body>
    </html>
    """

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        question = data.get("question", "")
        chunks = rag.search(question)
        context = " ".join(chunks)
        answer = generate_answer(context, question)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"question": "", "answer": f"Error: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True, port=8000)