from fastapi import FastAPI
import requests

from app.rag import search, init_collection, add_document

app = FastAPI()

OLLAMA_URL = "http://ollama:11434/api/generate"


# -----------------------
# HEALTH CHECK
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------
# PURE LLM (NO RAG)
# -----------------------
@app.post("/chat")
def chat(prompt: str):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()


# -----------------------
# RAG PIPELINE (REAL AI)
# -----------------------
@app.post("/ask")
def ask(question: str):

    # retrieve context from Qdrant
    context = search(question)

    # build grounded prompt
    prompt = f"""
You are a precise QA assistant.

Use ONLY the context below:

Context:
{context}

Question:
{question}

Answer clearly and technically.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return {
        "question": question,
        "context": context,
        "answer": response.json()["response"]
    }