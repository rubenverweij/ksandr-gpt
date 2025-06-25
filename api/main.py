"""
TAALMODEL API KSANDR
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from onprem import LLM
from fastapi import FastAPI

app = FastAPI()
llm = LLM(n_gpu_layers=-1)


DEFAULT_QA_PROMPT = """Use the following pieces of context delimited by three backticks to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Always respond in Dutch.

```{context}```

Question: {question}
Nuttig antwoord:"""


class AskRequest(BaseModel):
    prompt: str
    permission: Optional[Dict[str, Dict[str, List[int]]]] = None
    aad: Optional[List[int]] = None


def _build_filter(
    permission_data: Optional[Dict[str, Dict[str, List[int]]]],
) -> Dict[str, Any]:
    if not permission_data:
        return {"table": False}
    permissions = []
    for source in permission_data:
        for category, ids in permission_data[source].items():
            for id_ in ids:
                permissions.append(f"{category}_{id_}")
    if not permissions:
        return {"table": False}
    return {"permission": {"$in": permissions}}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    filter_obj = _build_filter(request.permission)
    try:
        return llm._ask(
            question=request.prompt,
            filters=filter_obj,
            table_k=0,
            qa_template=DEFAULT_QA_PROMPT,
        )
    except Exception as e:
        return {"error": str(e), "filter": filter_obj}


@app.post("/chat")
def chat(request: AskRequest):
    filter_obj = _build_filter(request.permission)
    try:
        return llm.chat(prompt=request.prompt, filters=filter_obj, table_k=0)
    except Exception as e:
        return {"error": str(e), "filter": filter_obj}
