"""
TAALMODEL API KSANDR
"""

import io
from starlette.responses import StreamingResponse
import asyncio
import sys
import argparse

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel
from onprem import LLM
from onprem.pipelines import Summarizer
from fastapi import FastAPI


def parse_args():
    parser = argparse.ArgumentParser(description="TAALMODEL API Configurations")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Parameter which influences the creativity of the model",
    )
    parser.add_argument(
        "--source_max",
        type=int,
        default=6,
        help="Maximum number of source documents for RAG",
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.8, help="Threshold for RAG score"
    )
    parser.add_argument(
        "--store_type",
        type=str,
        choices=["dense", "sparse"],
        default="sparse",
        help="Data store type",
    )
    args = parser.parse_args()
    return args


args = parse_args()
app = FastAPI()
llm = LLM(
    n_gpu_layers=-1,
    embedding_model_kwargs={"device": "cuda"},
    temperature=args.temperature,
    rag_num_source_docs=args.source_max,
    rag_score_threshold=args.score_threshold,
    store_type=args.store_type,
    verbose=False,
)


DEFAULT_QA_PROMPT = """
Je bent een behulpzame en feitelijke assistent die vragen beantwoordt over documenten op het Ksandr-platform.

Ksandr is het collectieve kennisplatform van de Nederlandse netbeheerders. Door kennis over netcomponenten te borgen, ontwikkelen en delen, helpt Ksandr de netbeheerders om de kwaliteit van hun netten op minimaal het maatschappelijk gewenste niveau te houden.

De meeste vragen gaan over zogenoemde componenten in Ageing Asset Dossiers (AAD’s). Deze dossiers bevatten onderhouds- en conditie-informatie over de 20 meest relevante netcomponenten. Ze worden jaarlijks geactualiseerd op basis van faalinformatie, storingen en andere relevante inzichten. Beheerteams stellen op basis daarvan een verschilanalyse op, waarmee netbeheerders van elkaar kunnen leren. Toegang tot deze dossiers verloopt via een speciaal portaal op de Ksandr-website.

🟡 **Belangrijke instructies:**
- Geef kort en bondig het antwoord in het Nederlands.
- Als het antwoord niet duidelijk in de context staat, zeg dan:  
  **"Ik weet het antwoord niet."**
- Gebruik uitsluitend de onderstaande context om de vraag te beantwoorden:

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
            prompt_template=ZEPHYR_PROMPT_TEMPLATE,
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
