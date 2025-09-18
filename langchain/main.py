import asyncio
import time
import uuid
import os
from datetime import datetime
from helpers import (
    vind_relevante_componenten,
    COMPONENTS,
    uniek_antwoord,
    get_embedding_function,
    find_relevant_context,
    similarity_search_with_nouns,
)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional, Union, List, Any
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import BaseCallbackHandler


# Configuratie voor gelijktijdige verwerking van verzoeken
request_queue = asyncio.Queue()
semaphore = asyncio.Semaphore(5)
app = FastAPI()
request_responses = {}

# Configuratievariabelen
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
SOURCE_MAX = int(os.getenv("SOURCE_MAX", 10))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 1.1))
STORE_TYPE = os.getenv("STORE_TYPE", "sparse")
INCLUDE_FILTER = int(os.getenv("INCLUDE_FILTER", 1))
DEFAULT_MODEL_PATH = "/root/.cache/huggingface/hub/models--unsloth--Qwen3-30B-A3B-Instruct-2507-GGUF/snapshots/eea7b2be5805a5f151f8847ede8e5f9a9284bf77/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
CHROMA_PATH = "/root/onprem_data/chroma"

# Initialisatie van het taalmodel
LLM = LlamaCpp(
    model_path=DEFAULT_MODEL_PATH,
    max_tokens=2000,
    n_gpu_layers=-1,
    n_ctx=32768,
    verbose=False,
    streaming=True,
    temperature=TEMPERATURE,
    top_p=0.9,
)
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


print(
    f"Starting container with temperature: {TEMPERATURE}, source_max: {SOURCE_MAX}, score_theshold: {SCORE_THRESHOLD}, store: {STORE_TYPE} and filter: {INCLUDE_FILTER}"
)

DEFAULT_QA_PROMPT = """
<|im_start|>system

Je bent een behulpzame en feitelijke assistent die vragen beantwoordt over documenten op het Ksandr-platform.

Ksandr is het collectieve kennisplatform van de Nederlandse netbeheerders. Door kennis over netcomponenten te borgen, ontwikkelen en delen, helpt Ksandr de netbeheerders om de kwaliteit van hun netten op het gewenste maatschappelijk niveau te houden.

De meeste vragen gaan over zogenoemde componenten in 'Ageing Asset Dossiers' (AAD’s). Deze dossiers bevatten onderhouds- en conditie-informatie van relevante netcomponenten. Ze worden jaarlijks geactualiseerd op basis van faalinformatie, storingen en andere relevante inzichten. Beheerteams stellen op basis daarvan een verschilanalyse op, waarmee netbeheerders van elkaar kunnen leren. Toegang tot deze dossiers verloopt via een speciaal portaal op de Ksandr-website.

Componenten met een AAD dossier zijn: 1) LK ELA12 schakelinstallatie 2) ABB VD4 vaccuum vermogensschakelaar 3) Eaton L-SEP installatie 4) Siemens NXplusC schakelaar 5) Siemens 8DJH schakelaar 6) Eaton FMX schakelinstallatie 7) Merlin Gerin RM6 schakelaar 8) Hazemeijer CONEL schakelinstallatie 9) Eaton 10 kV COQ schakelaar 10) Eaton Capitole schakelaar 11) Eaton Xiria schakelinstallatie 12) Eaton Holec SVS schakelaar 13) MS/LS distributie transformator 14) Eaton Magnefix MD MF schakelinstallatie 15) ABB DR12 schakelaar 16) ABB Safe schakelinstallatie 17) kabelmoffen 18) Eaton MMS schakelinstallatie 19) ABB BBC DB10 schakelaar 20) HS MS vermogens transformator

**Belangrijke instructies bij de beantwoording:**
- Verbeter spelling en grammatica.
- Gebruik correct en helder Nederlands.
- Wees volledig, maar als het kan kort en bondig.
- Herhaal het antwoord niet.
- Betrek geen onnodige details bij een algemene vraag.
- Als het antwoord niet duidelijk blijkt uit de context zeg dan: "Ik weet het antwoord niet."

<|im_end|>
<|im_start|>user

context:
{context}

Vraag:
{question}

<|im_end|>
<|im_start|>assistant
"""

DEFAULT_QA_PROMPT_SIMPLE = """
<|im_start|>system

Je bent een behulpzame en feitelijke assistent die vragen beantwoordt over documenten op het Ksandr-platform.
Ksandr is het collectieve kennisplatform van de Nederlandse netbeheerders. Door kennis over netcomponenten te borgen, ontwikkelen en delen, helpt Ksandr de netbeheerders om de kwaliteit van hun netten op het gewenste maatschappelijk niveau te houden.
De meeste vragen gaan over zogenoemde componenten in 'Ageing Asset Dossiers' (AAD’s). Deze dossiers bevatten onderhouds- en conditie-informatie van relevante netcomponenten. Ze worden jaarlijks geactualiseerd op basis van faalinformatie, storingen en andere relevante inzichten. Beheerteams stellen op basis daarvan een verschilanalyse op, waarmee netbeheerders van elkaar kunnen leren. Toegang tot deze dossiers verloopt via een speciaal portaal op de Ksandr-website.
Componenten met een AAD dossier zijn: 1) LK ELA12 schakelinstallatie 2) ABB VD4 vaccuum vermogensschakelaar 3) Eaton L-SEP installatie 4) Siemens NXplusC schakelaar 5) Siemens 8DJH schakelaar 6) Eaton FMX schakelinstallatie 7) Merlin Gerin RM6 schakelaar 8) Hazemeijer CONEL schakelinstallatie 9) Eaton 10 kV COQ schakelaar 10) Eaton Capitole schakelaar 11) Eaton Xiria schakelinstallatie 12) Eaton Holec SVS schakelaar 13) MS/LS distributie transformator 14) Eaton Magnefix MD MF schakelinstallatie 15) ABB DR12 schakelaar 16) ABB Safe schakelinstallatie 17) kabelmoffen 18) Eaton MMS schakelinstallatie 19) ABB BBC DB10 schakelaar 20) HS MS vermogens transformator

**Belangrijke instructies bij de beantwoording:**
- Verbeter spelling en grammatica.
- Gebruik correct en helder Nederlands.
- Wees volledig, maar als het kan kort en bondig.
- Herhaal het antwoord niet.

<|im_end|>
<|im_start|>user

Vraag:
{question}

<|im_end|>
<|im_start|>assistant
"""


# Streaming handler
class StreamingResponseCallback(BaseCallbackHandler):
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.partial_response = ""

    def on_llm_new_token(self, token: str, **kwargs):
        """Callback method for when a new token is generated by the model."""
        self.partial_response += token
        if self.request_id in request_responses:
            request_responses[self.request_id]["partial_response"] = (
                self.partial_response
            )

    def on_llm_end(self, output: str, **kwargs):
        """Callback method for when the model has finished generating."""
        if self.request_id in request_responses:
            request_responses[self.request_id]["status"] = "completed"
            request_responses[self.request_id]["response"] = self.partial_response


# Vraagmodel
class AskRequest(BaseModel):
    prompt: str
    permission: Optional[Dict[str, Union[Dict[str, List[int]], List[int], bool]]] = None
    user_id: Optional[str] = "123"
    rag: Optional[int] = 1

    class Config:
        extra = "allow"  # Sta extra velden toe


def ask_llm(prompt: str, filter: Optional[Dict | None], model: LlamaCpp, rag: int):
    if rag:
        document_search = similarity_search_with_nouns(query=prompt)
        context_text, results, summary = find_relevant_context(
            prompt=prompt,
            filter_chroma=filter,
            db=db,
            source_max=SOURCE_MAX,
            score_threshold=SCORE_THRESHOLD,
        )
        results_new_schema = []
        for doc, score in results:
            doc_dict = {
                "id": doc.id,
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "type": doc.type,
            }
            doc_dict["metadata"]["score"] = score
            results_new_schema.append(doc_dict)
        prompt_with_template = DEFAULT_QA_PROMPT.format(
            context=context_text, question=prompt
        )
    else:
        prompt_with_template = DEFAULT_QA_PROMPT_SIMPLE.format(question=prompt)
        results_new_schema = None
        document_search = None
    return {
        "question": prompt,
        "answer": model.invoke(prompt_with_template),
        "source_documents": results_new_schema,
        "where_document": document_search,
        "summary": summary,
    }


# Verwerkt het verzoek en haalt de reactie op
async def process_request(request: AskRequest):
    """Process a request asynchronously and stream the result."""
    if INCLUDE_FILTER:
        active_filter = vind_relevante_componenten(
            vraag=request.prompt, componenten_dict=COMPONENTS
        )
    else:
        active_filter = None

    try:
        # Pass request_id for tracking the streaming response
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: ask_llm(
                prompt=request.prompt,
                model=LLM,
                filter=active_filter,
                rag=request.rag,
            ),
        )
        response["active_filter"] = str(active_filter)
        response["answer"] = uniek_antwoord(response["answer"])
        if request.rag:
            if not response.get("source_documents"):
                response["answer"] = (
                    "Op basis van de informatie die ik tot mijn beschikking heb, weet ik het antwoord helaas niet."
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
    """Check the status and the result of a request."""
    if request_id in request_responses:
        response_data = request_responses[request_id]
        if response_data["status"] == "completed":
            return response_data
        elif "partial_response" in response_data:
            return {
                "status": "processing",
                "start_time_formatted": response_data["start_time_formatted"],
                "in_queue_start": response_data["in_queue_start"],
                "partial_response": response_data["partial_response"],
                "in_queue_current": await get_request_position_in_queue(
                    request_id=request_id
                ),
            }
        return {
            "status": "processing",
            "start_time_formatted": response_data["start_time_formatted"],
            "in_queue_start": response_data["in_queue_start"],
            "in_queue_current": await get_request_position_in_queue(
                request_id=request_id
            ),
        }
    return {"message": "Request not found", "status": "not_found"}


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
