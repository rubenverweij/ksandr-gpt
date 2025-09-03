"""
TAALMODEL API KSANDR
"""

import asyncio
from fastapi import FastAPI
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel
from onprem import LLM
import os
import uuid


# Wachtrijen voor gelijktijdige verwerking van verzoeken
request_queue = asyncio.Queue()
semaphore = asyncio.Semaphore(5)
app = FastAPI()
request_responses = {}

# Configuratie variabelen
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))
SOURCE_MAX = int(os.getenv("SOURCE_MAX", 2))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.6))
STORE_TYPE = os.getenv("STORE_TYPE", "sparse")

# Initialisatie van het taalmodel
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
<|im_start|>system

Je bent een behulpzame en feitelijke assistent die vragen beantwoordt over documenten op het Ksandr-platform.

Ksandr is het collectieve kennisplatform van de Nederlandse netbeheerders. Door kennis over netcomponenten te borgen, ontwikkelen en delen, helpt Ksandr de netbeheerders om de kwaliteit van hun netten op het gewenste maatschappelijk niveau te houden.

De meeste vragen gaan over zogenoemde componenten in 'Ageing Asset Dossiers' (AAD’s). Deze dossiers bevatten onderhouds- en conditie-informatie van relevante netcomponenten. Ze worden jaarlijks geactualiseerd op basis van faalinformatie, storingen en andere relevante inzichten. Beheerteams stellen op basis daarvan een verschilanalyse op, waarmee netbeheerders van elkaar kunnen leren. Toegang tot deze dossiers verloopt via een speciaal portaal op de Ksandr-website.

**Belangrijke instructies bij de beantwoording:**
- Verbeter spelling en grammatica.
- Gebruik correct en helder Nederlands.
- Wees kort en bondig.
- Vermijd herhaling.
- Als de onderstaande context geen tekst bevat zeg dan: "Ik weet het antwoord niet."
- Beantwoord alleen de gestelde vraag. Negeer andere vragen in de context. Gebruik uitsluitend de context. Maak geen aannames.

<|im_end|>
<|im_start|>

Context:
{context}
Vraag:
{question}
<|im_end|>
"""


class AskRequest(BaseModel):
    prompt: str
    permission: Optional[Dict[str, Union[Dict[str, List[int]], List[int], bool]]] = None

    class Config:
        extra = "allow"


async def process_request(request: AskRequest):
    """Function that processes the request. This simulates the async task processing."""
    # filter_obj = _build_filter(request.permission)
    source_max = getattr(request, "source_max", None)
    score_threshold = getattr(request, "score_threshold", None)

    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: llm._ask(
                question=request.prompt,
                # filters=filter_obj, FIXME filter object aanpassen aan sparse db
                table_k=0,
                k=source_max,
                score_threshold=score_threshold,
                qa_template=DEFAULT_QA_PROMPT,
            ),
        )
        if not response.get("source_documents"):
            response["answer"] = (
                "Ik weet het antwoord helaas niet, misschien kan je de vraag anders formuleren?"
            )
        return response
    except Exception as e:
        return {"error": str(e)}


async def request_worker():
    """Worker to process requests one by one."""
    while True:
        request = await request_queue.get()
        async with semaphore:
            response = await process_request(request)
            request["response"] = response


@app.post("/ask")
async def ask(request: AskRequest):
    """Handles incoming requests."""
    request_id = str(uuid.uuid4())
    request["id"] = request_id
    await request_queue.put(request)
    request_responses[request_id] = {"status": "processing"}
    return {"message": "Request is being processed", "request_id": request_id}


@app.get("/status/{request_id}")
async def get_status(request_id: str):
    """Check the status of a request."""
    if request_id in request_responses:
        return request_responses[request_id]
    return {"message": "Request not found"}


@app.on_event("startup")
async def startup():
    """Start the worker to process requests sequentially"""
    asyncio.create_task(request_worker())


def _build_filter(
    permission_data: Optional[Dict[str, Union[Dict[str, List[int]], List[int], bool]]],
) -> Dict[str, Any]:
    """Bouwt een filter op basis van de opgegeven permissies."""
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
            permissions.append(f"{'true' if value else 'false'}_{source}")
    return {"permission_and_type_k": {"$in": permissions}}
