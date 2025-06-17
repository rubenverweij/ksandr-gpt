"""
TAALMODEL API KSANDR
"""

from onprem import LLM
from fastapi import FastAPI

app = FastAPI()

llm = LLM(default_model="llama")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(prompt: str):
    return llm.ask(prompt)


@app.post("/chat")
def chat_with_llm(prompt: str):
    return llm.chat(prompt)
