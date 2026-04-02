from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag import RAGIndex
from model import generate_answer
import traceback

app = FastAPI()
rag = RAGIndex()

class ChatRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def home():
    return "AI Tutor Available"

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        chunks = rag.search(request.question)
        context = " ".join(chunks)
        answer = generate_answer(context, request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        traceback.print_exc()
        return {"question": request.question, "answer": f"Error: {str(e)}"}