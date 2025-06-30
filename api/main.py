"""
TAALMODEL API KSANDR
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from onprem import LLM
from fastapi import FastAPI

app = FastAPI()
llm = LLM(n_gpu_layers=-1)


DEFAULT_QA_PROMPT = """

Je bent een behulpzame en feitelijke assistent. Gebruik uitsluitend de onderstaande context om de vraag van de gebruiker te beantwoorden. 

Geef altijd een antwoord in het Nederlands. Als het antwoord niet expliciet in de context staat, geef dan aan: "Het antwoord is niet beschikbaar in de aangeleverde context."

[CONTEXT]
{context}

[VRAAG]
{question}

[ANTWOORD]

"""


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
    return {"permission_and_type": {"$in": permissions}}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    filter_obj = _build_filter(request.permission)
    source_max = getattr(request, "source_max", None)
    score_threshold = getattr(request, "score_threshold", None)
    try:
        return llm._ask(
            question=request.prompt,
            filters=filter_obj,
            table_k=0,
            k=source_max,
            score_threshold=score_threshold,
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
