"""
TAALMODEL API KSANDR
"""

from onprem import LLM
from fastapi import FastAPI

app = FastAPI()

llm = LLM(n_gpu_layers=-1)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(prompt: str, permission: list[int] | None, aad: list[int] | None):
    """Perform a query with optional permission and aad filters.

    Args:
        prompt (str): The users prompt.
        permission (list[int] | None): List of permission ids.
        aad (list[int] | None): List of aad ids.

    Returns:
        dict | None: The filter object, or None if no filters.
    """
    filters = []
    if permission:
        filters.append({"permission": {"$in": permission}})
    if aad:
        filters.append({"aad": {"$in": aad}})

    if not filters:
        filter_obj = {"table": False}
    elif len(filters) == 1:
        filter_obj = filters[0]
    else:
        # Both present
        filter_obj = {"$and": filters}
    return llm._ask(prompt, filters=filter_obj, table_k=0)


@app.post("/chat")
def chat_with_llm(prompt: str):
    return llm.chat(prompt)
