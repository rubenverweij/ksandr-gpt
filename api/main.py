"""
TAALMODEL API KSANDR
"""

from onprem import LLM
from fastapi import FastAPI

app = FastAPI()

llm = LLM(default_model="llama")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/ask/")
def prompt(prompt: str):
    return llm.ask(prompt)
