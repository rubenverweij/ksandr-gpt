"""
TAALMODEL API KSANDR
"""

import io
from starlette.responses import StreamingResponse
import asyncio
import sys
import os
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel
from onprem import LLM
from onprem.pipelines import Summarizer
from fastapi import FastAPI

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))
SOURCE_MAX = int(os.getenv("SOURCE_MAX", 2))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.6))
STORE_TYPE = os.getenv("STORE_TYPE", "sparse")

app = FastAPI()
llm = LLM(
    model_url="Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
    model_download_path="/root/.cache/huggingface/hub/models--unsloth--Qwen3-30B-A3B-Instruct-2507-GGUF/snapshots/eea7b2be5805a5f151f8847ede8e5f9a9284bf77",
    n_gpu_layers=-1,
    embedding_model_kwargs={"device": "cuda"},
    temperature=TEMPERATURE,
    rag_num_source_docs=SOURCE_MAX,
    rag_score_threshold=SCORE_THRESHOLD,
    store_type=STORE_TYPE,
    verbose=False,
)


DEFAULT_QA_PROMPT = """
Je bent een behulpzame en feitelijke assistent die vragen beantwoordt over documenten op het Ksandr-platform.

Ksandr is het collectieve kennisplatform van de Nederlandse netbeheerders. Door kennis over netcomponenten te borgen, ontwikkelen en delen, helpt Ksandr de netbeheerders om de kwaliteit van hun netten op minimaal het maatschappelijk gewenste niveau te houden.

De meeste vragen gaan over zogenoemde componenten in Ageing Asset Dossiers (AAD’s). Deze dossiers bevatten onderhouds- en conditie-informatie over de 20 meest relevante netcomponenten. Ze worden jaarlijks geactualiseerd op basis van faalinformatie, storingen en andere relevante inzichten. Beheerteams stellen op basis daarvan een verschilanalyse op, waarmee netbeheerders van elkaar kunnen leren. Toegang tot deze dossiers verloopt via een speciaal portaal op de Ksandr-website.

🟡 **Belangrijke instructies:**
- Verbeter spelling en grammatica, schrijf in correct Nederlands.
- Gebruik uitsluitend de onderstaande context. Voeg geen externe kennis of aannames toe.
- Beantwoord kort en bondig in maximaal 3 zinnen.
- Als de context geen relevant antwoord bevat, zeg **precies één keer**: **"Ik weet het antwoord niet."**.
- Herhaal de vraag niet en gebruik het label [ANTWOORD] **niet** in je antwoord.

[CONTEXT]
{context}

[VRAAG]
{question}

Antwoord hieronder:
"""


SUMMARY_PROMPT = """Wat zegt de volgende context in het Nederlands met betrekking tot "{concept_description}"? \n\nCONTEXT:\n{text}"""


class AskRequest(BaseModel):
    prompt: str
    permission: Optional[Dict[str, Union[Dict[str, List[int]], List[int], bool]]] = None


class SummaryRequest(BaseModel):
    doc_path: str
    concept: str


def _build_filter(
    permission_data: Optional[Dict[str, Union[Dict[str, List[int]], List[int], bool]]],
) -> Dict[str, Any]:
    if not permission_data:
        return {"table": False}
    permissions = []
    for source, value in permission_data.items():
        if isinstance(value, dict):
            for category, ids in value.items():
                for id_ in ids:
                    permissions.append(f"{id_}_{category}")
        elif isinstance(value, list):
            for id_ in value:
                permissions.append(f"{id_}_{source}")
        elif isinstance(value, bool):
            if value:
                permissions.append(f"true_{source}")
            else:
                permissions.append(f"false_{source}")
    return {"permission_and_type_k": {"$in": permissions}}


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
            # filters=filter_obj, FIXME filter object aanpassen aan sparse db
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
            # filters=filter_obj,
            table_k=0,
            prompt_template=DEFAULT_QA_PROMPT,
        )
    except Exception as e:
        return {"error": str(e), "filter": filter_obj}


@app.post("/summarise")
def summarise(request: SummaryRequest):
    try:
        summ = Summarizer(llm)
        summary, _ = summ.summarize_by_concept(
            request.doc_path,
            concept_description=request.concept,
            summary_prompt=SUMMARY_PROMPT,
        )
        return summary
    except Exception as e:
        return {"error": str(e)}


class StreamPrinter(io.StringIO):
    def __init__(self):
        super().__init__()
        self._buffer = asyncio.Queue()

    def write(self, s):
        # Called when something is printed
        self._buffer.put_nowait(s)

    def flush(self):
        pass  # optional depending on how print() is used

    async def stream(self):
        while True:
            s = await self._buffer.get()
            yield s


@app.post("/ask_stream")
async def ask_stream(request: AskRequest):
    async def stream():
        original_stdout = sys.stdout
        printer = StreamPrinter()
        sys.stdout = printer  # redirect prints to our buffer
        try:
            loop = asyncio.get_running_loop()
            task = loop.run_in_executor(
                None,
                lambda: llm._ask(
                    question=request.prompt,
                    filters=_build_filter(request.permission),
                    table_k=0,
                    k=getattr(request, "source_max", None),
                    score_threshold=getattr(request, "score_threshold", None),
                    qa_template=DEFAULT_QA_PROMPT,
                ),
            )
            while not task.done():
                try:
                    yield await asyncio.wait_for(printer._buffer.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
            while not printer._buffer.empty():
                yield await printer._buffer.get()
        finally:
            sys.stdout = original_stdout

    return StreamingResponse(stream(), media_type="text/plain")
