"""
TAALMODEL API KSANDR
"""

from typing import List, Optional
from pydantic import BaseModel
from onprem import LLM
from fastapi import FastAPI

app = FastAPI()

llm = LLM(n_gpu_layers=-1)


class AskRequest(BaseModel):
    prompt: str
    permission: Optional[List[int]] = None
    aad: Optional[List[int]] = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    """Perform a query with optional permission and aad filters."""
    permission = request.permission
    aad = request.aad

    filters = []
    if permission:
        filters.append({"permission": {"$in": permission}})

    if aad:
        filters.append({"aad": {"$in": aad}})

    if not filters:
        filter_obj = {"table": True}
    elif len(filters) == 1:
        filter_obj = filters[0]
    else:
        filter_obj = {"$and": filters}
    return llm._ask(prompt=request.prompt, filters=filter_obj, table_k=0)


@app.post("/chat")
def chat(request: AskRequest):
    """Perform a query with optional permission and aad filters."""
    permission = request.permission
    aad = request.aad

    filters = []
    if permission:
        filters.append({"permission": {"$in": permission}})

    if aad:
        filters.append({"aad": {"$in": aad}})

    if not filters:
        filter_obj = {"table": True}
    elif len(filters) == 1:
        filter_obj = filters[0]
    else:
        filter_obj = {"$and": filters}

    return llm.chat(prompt=request.prompt, filters=filter_obj, table_k=0)
