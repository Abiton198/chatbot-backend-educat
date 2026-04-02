import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(context, question):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""You are a CAT Grade 12 tutor.

Context: {context}

Question: {question}

Answer clearly for a student:"""
        }],
        max_tokens=300
    )
    return response.choices[0].message.content