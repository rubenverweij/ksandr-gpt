import asyncio
import time
import uuid
import os
from datetime import datetime
from components import vind_relevante_componenten, COMPONENTS
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional, Union, List, Any
from onprem import LLM

# Configuratie voor gelijktijdige verwerking van verzoeken
request_queue = asyncio.Queue()
semaphore = asyncio.Semaphore(5)
app = FastAPI()
request_responses = {}

# Configuratievariabelen
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))
SOURCE_MAX = int(os.getenv("SOURCE_MAX", 2))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.6))
STORE_TYPE = os.getenv("STORE_TYPE", "sparse")
INCLUDE_FILTER = os.getenv("INCLUDE_FILTER", False)

# Initialisatie van het taalmodel
llm = LLM(
    model_url="Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
    model_download_path="/root/.cache/huggingface/hub/models--unsloth--Qwen3-30B-A3B-Instruct-2507-GGUF/snapshots/eea7b2be5805a5f151f8847ede8e5f9a9284bf77",
    n_gpu_layers=-1,
    max_tokens=700,
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

Componenten met een AAD dossier zijn: 1) LK ELA12 schakelinstallatie 2) ABB VD4 vaccuum vermogensschakelaar 3) Eaton L-SEP installatie 4) Siemens NXplusC schakelaar 5) Siemens 8DJH schakelaar 6) Eaton FMX schakelinstallatie 7) Merlin Gerin RM6 schakelaar 8) Hazemeijer CONEL schakelinstallatie 9) Eaton 10 kV COQ schakelaar 10) Eaton Capitole schakelaar 11) Eaton Xiria schakelinstallatie 12) Eaton Holec SVS schakelaar 13) MS/LS distributie transformator 14) Eaton Magnefix MD MF schakelinstallatie 15) ABB DR12 schakelaar 16) ABB Safe schakelinstallatie 17) kabelmoffen 18) Eaton MMS schakelinstallatie 19) ABB BBC DB10 schakelaar 20) HS MS vermogens transformator 21)

**Belangrijke instructies bij de beantwoording:**
- Verbeter spelling en grammatica.
- Gebruik correct en helder Nederlands.
- Wees kort en bondig.
- Vermijd herhaling.
- Als de onderstaande context geen tekst bevat zeg dan: "Ik weet het antwoord niet."
- Beantwoord alleen de gestelde vraag. Negeer andere vragen in de context. Maak geen aannames.

<|im_end|>
<|im_start|>

Context:
{context}
Vraag:
{question}
<|im_end|>
"""


# Vraagmodel
class AskRequest(BaseModel):
    prompt: str
    permission: Optional[Dict[str, Union[Dict[str, List[int]], List[int], bool]]] = None

    class Config:
        extra = "allow"  # Sta extra velden toe


# Verwerkt het verzoek en haalt de reactie op
async def process_request(request: AskRequest):
    """Verwerkt een verzoek asynchroon."""
    source_max = getattr(request, "source_max", None)
    score_threshold = getattr(request, "score_threshold", None)

    if INCLUDE_FILTER:
        active_filter = {
            "type_id": vind_relevante_componenten(
                vraag=request.prompt, componenten_dict=COMPONENTS
            )
        }
    else:
        active_filter = None

    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: llm._ask(
                question=request.prompt,
                filters=active_filter,
                table_k=0,
                k=source_max,
                score_threshold=score_threshold,
                qa_template=DEFAULT_QA_PROMPT,
            ),
        )
        if not response.get("source_documents"):
            response["answer"] = (
                "Ik weet het antwoord helaas niet, probeer je vraag anders te formuleren."
            )
        return response
    except Exception as e:
        return {"error": str(e), "filter": active_filter}


# Worker voor het verwerken van verzoeken in de wachtrij
async def request_worker():
    """Verwerkt verzoeken één voor één."""
    while True:
        request = await request_queue.get()
        async with semaphore:
            response = await process_request(request)
            end_time = time.time()
            duration = end_time - request_responses[request.id]["start_time"]
            request_responses[request.id].update(
                {
                    "status": "completed",
                    "response": response,
                    "end_time": end_time,
                    "time_duration": duration,
                }
            )


@app.post("/ask")
async def ask(request: AskRequest):
    """Verwerkt binnenkomende verzoeken."""
    request.id = str(uuid.uuid4())  # Genereer een uniek ID
    await request_queue.put(request)  # Voeg het verzoek toe aan de wachtrij
    start_time = time.time()
    in_queue = request_queue.qsize()
    request_responses[request.id] = {
        "status": "processing",
        "start_time": start_time,
        "in_queue_start": in_queue,
        "start_time_formatted": datetime.fromtimestamp(start_time).strftime(
            "%H:%M:%S %d-%m-%Y"
        ),
    }
    return {
        "message": "Verzoek wordt verwerkt",
        "request_id": request.id,
        "in_queue_start": in_queue,
        "start_time": datetime.fromtimestamp(start_time).strftime("%H:%M:%S %d-%m-%Y"),
    }


@app.get("/status/{request_id}")
async def get_status(request_id: str):
    """Check de status en het resultaat van een verzoek."""
    if request_id in request_responses:
        response_data = request_responses[request_id]
        if response_data["status"] == "completed":
            return request_responses[request_id]
        return {
            "status": "processing",
            "start_time_formatted": response_data["start_time_formatted"],
            "in_queue_start": response_data["in_queue_start"],
            "in_queue_current": await get_request_position_in_queue(
                request_id=request_id
            ),
        }
    return {"message": "Verzoek niet gevonden"}


@app.on_event("startup")
async def startup():
    """Start de worker om verzoeken sequentieel te verwerken."""
    asyncio.create_task(request_worker())


async def get_request_position_in_queue(request_id: str) -> int:
    """Calculate the real-time position of the request in the queue."""
    queue_list = list(request_queue._queue)
    for index, queued_request in enumerate(queue_list):
        if queued_request.id == request_id:
            return index + 1
    return 0


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
    return {"permission_and_type": {"$in": permissions}}
