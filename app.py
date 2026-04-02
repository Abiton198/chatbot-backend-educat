# app.py
from dotenv import load_dotenv
load_dotenv()  # Must be first before any other imports

import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from model import generate_answer
from rag import RAGIndex
import traceback

app = FastAPI()
rag = RAGIndex()

class ChatRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
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

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        chunks = rag.search(request.question)  # Fixed: was search(), now rag.search()
        context = " ".join(chunks)
        answer = generate_answer(context, request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        traceback.print_exc()
        return {"question": request.question, "answer": f"Error: {str(e)}"}