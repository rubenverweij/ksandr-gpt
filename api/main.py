"""
TAALMODEL API KSANDR
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from onprem import LLM
from onprem.pipelines import Summarizer
from fastapi import FastAPI

app = FastAPI()
llm = LLM(
    n_gpu_layers=-1, embedding_model_kwargs={"device": "cuda"}, store_type="sparse"
)


DEFAULT_QA_PROMPT = """
Je bent een behulpzame en feitelijke assistent die vragen beantwoordt over documenten op het Ksandr-platform.

Ksandr is het collectieve kennisplatform van de Nederlandse netbeheerders. Door kennis over netcomponenten te borgen, ontwikkelen en delen, helpt Ksandr de netbeheerders om de kwaliteit van hun netten op minimaal het maatschappelijk gewenste niveau te houden.

De meeste vragen gaan over zogenoemde componenten in Ageing Asset Dossiers (AAD’s). Deze dossiers bevatten onderhouds- en conditie-informatie over de 20 meest relevante netcomponenten. Ze worden jaarlijks geactualiseerd op basis van faalinformatie, storingen en andere relevante inzichten. Beheerteams stellen op basis daarvan een verschilanalyse op, waarmee netbeheerders van elkaar kunnen leren. Toegang tot deze dossiers verloopt via een speciaal portaal op de Ksandr-website.

🟡 **Belangrijke instructies:**
- Gebruik **uitsluitend de onderstaande context** om de vraag te beantwoorden.
- Geef het antwoord **altijd in het Nederlands**.
- Als het antwoord niet expliciet in de context staat, zeg dan:  
  **"Het antwoord is niet beschikbaar in de aangeleverde context."**

[CONTEXT]
{context}

[VRAAG]
{question}

[ANTWOORD]
"""

ZEPHYR_PROMPT_TEMPLATE = """<|system|>
Je bent een behulpzame en feitelijke assistent die vragen beantwoordt over documenten op het Ksandr-platform.

Ksandr is het collectieve kennisplatform van de Nederlandse netbeheerders. Door kennis over netcomponenten te borgen, ontwikkelen en delen, helpt Ksandr de netbeheerders om de kwaliteit van hun netten op minimaal het maatschappelijk gewenste niveau te houden.

De meeste vragen gaan over zogenoemde componenten in Ageing Asset Dossiers (AAD’s). Deze dossiers bevatten onderhouds- en conditie-informatie over de 20 meest relevante netcomponenten. Ze worden jaarlijks geactualiseerd op basis van faalinformatie, storingen en andere relevante inzichten. Beheerteams stellen op basis daarvan een verschilanalyse op, waarmee netbeheerders van elkaar kunnen leren. Toegang tot deze dossiers verloopt via een speciaal portaal op de Ksandr-website.

Belangrijke instructies:
- Gebruik uitsluitend de onderstaande context om de vraag te beantwoorden.
- Geef het antwoord altijd in het Nederlands.
- Als het antwoord niet expliciet in de context staat, zeg dan: "Het antwoord is niet beschikbaar in de aangeleverde context."
</s>
<|user|>
[CONTEXT]
{context}

[VRAAG]
{question}
</s>
<|assistant|>
"""


class AskRequest(BaseModel):
    prompt: str
    permission: Optional[Dict[str, Dict[str, List[int]]]] = None
    aad: Optional[List[int]] = None


class SummaryRequest(BaseModel):
    doc_path: str
    concept: str


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
        return llm.chat(
            prompt=request.prompt,
            filters=filter_obj,
            table_k=0,
            prompt_template=ZEPHYR_PROMPT_TEMPLATE,
        )
    except Exception as e:
        return {"error": str(e), "filter": filter_obj}


@app.post("/summarise")
def summarise(request: SummaryRequest):
    try:
        summ = Summarizer(llm)
        summary, sources = summ.summarize_by_concept(
            request.doc_path, concept_description=request.concept
        )
        return summary
    except Exception as e:
        return {"error": str(e)}
