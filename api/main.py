"""
TAALMODEL API KSANDR
"""

from onprem import LLM
from fastapi import FastAPI

app = FastAPI()

llm = LLM(default_model="llama")
llm.ingest("/root/ksandr_texts")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/ask/")
def prompt(prompt: str):
    return llm.ask(prompt)
