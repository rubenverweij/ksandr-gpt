import asyncio
import time
import uuid
import os
import re
from datetime import datetime
from pathlib import Path
from templates import TEMPLATES, SYSTEM_PROMPT, dynamische_prompt_elementen
from llm import LLMManager, RecursiveSummarizer
from refs import replace_patterns
from graph import build_cypher_query, check_for_nbs, match_query_by_tags
from helpers import (
    maak_metadata_filter,
    COMPONENTS,
    uniek_antwoord,
    get_embedding_function,
    vind_relevante_context,
    maak_chroma_filter,
    trim_context_to_fit,
    haal_dossiers_op,
    detect_aad,
    source_document_dummy,
    is_valid_sentence,
    clean_text_with_dup_detection,
    summary_request,
    get_summary,
)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Union, List
from langchain_chroma import Chroma
from langchain_core.callbacks import BaseCallbackHandler
from langchain_neo4j import Neo4jGraph

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
GRAPH = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

# Configuratie voor gelijktijdige verwerking van verzoeken
request_queue = asyncio.Queue()
semaphore = asyncio.Semaphore(5)
app = FastAPI()
request_responses = {}

# Configuratievariabelen
CONFIG = {
    "TEMPERATURE": float(os.getenv("TEMPERATURE", 0.2)),
    "SOURCE_MAX": int(os.getenv("SOURCE_MAX", 10)),
    "SOURCE_MAX_RERANKER": int(os.getenv("SOURCE_MAX_RERANKER", 0)),
    "SCORE_THRESHOLD": float(os.getenv("SCORE_THRESHOLD", 1.1)),
    "INCLUDE_FILTER": int(os.getenv("INCLUDE_FILTER", 1)),
    "MAX_TOKENS": int(os.getenv("MAX_TOKENS", 750)),
    "MAX_CTX": int(os.getenv("MAX_CTX", 8000)),
    "INCLUDE_SUMMARY": int(os.getenv("INCLUDE_SUMMARY", 0)),
    "INCLUDE_KEYWORDS": int(os.getenv("INCLUDE_KEYWORDS", 0)),
    "DEFAULT_MODEL_PATH": str(
        os.getenv(
            "DEFAULT_MODEL_PATH",
            "/root/huggingface/hub/models--unsloth--Qwen3-30B-A3B-Instruct-2507-GGUF/snapshots/eea7b2be5805a5f151f8847ede8e5f9a9284bf77/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
        )
    ),
    "INCLUDE_PERMISSION": int(os.getenv("INCLUDE_PERMISSION", 0)),
    "CHROMA_PATH": os.getenv("CHROMA_PATH", "/root/onprem_data/chroma"),
    "CHROMA_PATH_CYPHER": os.getenv(
        "CHROMA_PATH_CYPHER", "/root/onprem_data/chroma_cypher"
    ),
}

model = os.path.basename(CONFIG["DEFAULT_MODEL_PATH"])
DEFAULT_QA_PROMPT = TEMPLATES[model]["DEFAULT_QA_PROMPT"]
CYPHER_PROMPT = TEMPLATES[model]["CYPHER_PROMPT"]
DEFAULT_QA_PROMPT_SIMPLE = TEMPLATES[model]["DEFAULT_QA_PROMPT_SIMPLE"]
SUMMARY_PROMPT = TEMPLATES[model]["SUMMARY_PROMPT"]

# Initialisatie van het taalmodel
LLM_MANAGER = LLMManager(
    model_path=CONFIG["DEFAULT_MODEL_PATH"],
    max_tokens=CONFIG["MAX_TOKENS"],
    n_gpu_layers=-1,
    temperature=CONFIG["TEMPERATURE"],
    top_p=0.9,
)
LLM_MANAGER.load_llm(n_ctx=CONFIG["MAX_CTX"])


embedding_function = get_embedding_function()
db = Chroma(
    persist_directory=CONFIG["CHROMA_PATH"], embedding_function=embedding_function
)
db_cypher = Chroma(
    persist_directory=CONFIG["CHROMA_PATH_CYPHER"],
    embedding_function=embedding_function,
)
print(f"Starting container with {CONFIG}")


# Streaming handler voor het streamen van antwoorden
class StreamingResponseCallback(BaseCallbackHandler):
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.partial_response = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.partial_response += token
        # Store partial result
        if self.request_id in request_responses:
            request_responses[self.request_id]["partial_response"] = (
                self.partial_response
            )


class LLMRequest(BaseModel):
    n_ctx: int


class FileRequest(BaseModel):
    file_path: str
    summary_file_path: str = (
        None  # optional, default will be file_path + "_summary.txt"
    )


class AskRequest(BaseModel):
    prompt: str
    permission: Optional[Dict[str, Union[Dict[str, List[int]], List[int], bool]]] = None
    user_id: Optional[str] = "123"
    rag: Optional[int] = 1

    class Config:
        extra = "allow"  # Allow extra fields


class ContextRequest(BaseModel):
    prompt: str


def retrieve_answer_from_vector_store(
    prompt: str, chroma_filter: Optional[Dict | None]
):
    time_start = time.time()
    document_search = maak_chroma_filter(
        question=prompt, include_nouns=CONFIG["INCLUDE_KEYWORDS"]
    )
    time_doc_search = time.time()
    neo_context_text = None
    time_stages = {}
    if not neo_context_text:
        context_text, results, time_stages = vind_relevante_context(
            prompt=prompt,
            filter_chroma=chroma_filter,
            db=db,
            source_max_reranker=CONFIG["SOURCE_MAX_RERANKER"],
            source_max_dense=CONFIG["SOURCE_MAX"],
            score_threshold=CONFIG["SCORE_THRESHOLD"],
            where_document=document_search,
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
    else:
        context_text = neo_context_text
    logging.info("Done building context")
    time_build_context = time.time()
    _, trimmed_context_text = trim_context_to_fit(
        model=LLM_MANAGER.get_llm().client,
        template=DEFAULT_QA_PROMPT,
        context_text=context_text,
        question=prompt,
        n_ctx=CONFIG["MAX_CTX"],
        max_tokens=CONFIG["MAX_TOKENS"],
    )
    if len(trimmed_context_text) < 10:
        trimmed_context_text = "Er is geen informatie gevonden die gebruikt kan worden bij de beantwoording."

    time_reranker_trimming = time.time()
    prompt_with_template = DEFAULT_QA_PROMPT.format(
        system_prompt=SYSTEM_PROMPT,
        context=trimmed_context_text,
        question=prompt,
    )
    time_stages.update(
        {
            "maak_chroma_filter": time_doc_search - time_start,
            "vind_relevante_context": time_build_context - time_doc_search,
            "trim_context_to_fit": time_reranker_trimming - time_build_context,
        }
    )
    return prompt_with_template, results_new_schema, time_stages


def build_prompt_template(prompt: str, chroma_filter: Optional[Dict | None], rag: int):
    reference_documents = None
    time_stages = {}
    if rag:
        if detect_aad(prompt):
            neo4j_result = validate_structured_query(prompt)
            if len(neo4j_result) > 0:
                logging.info(f"Start LLM on neo4j: {neo4j_result}")
                return (
                    retrieve_neo_answer(prompt, neo4j_result),
                    source_document_dummy(),
                    {},
                )
            else:
                logging.info(f"Closest query: {neo4j_result}")
                prompt_with_template, reference_documents, time_stages = (
                    retrieve_answer_from_vector_store(prompt, chroma_filter)
                )
        else:
            neo4j_result = validate_structured_query_embedding(prompt)
            if len(neo4j_result) > 0:
                logging.info(f"Start LLM on neo4j: {neo4j_result}")
                return (
                    retrieve_neo_answer(prompt, neo4j_result),
                    source_document_dummy(),
                    {},
                )
            else:
                prompt_with_template, reference_documents, time_stages = (
                    retrieve_answer_from_vector_store(prompt, chroma_filter)
                )
    else:
        prompt_with_template = DEFAULT_QA_PROMPT_SIMPLE.format(
            system_prompt=SYSTEM_PROMPT, question=prompt
        )
    return prompt_with_template, reference_documents, time_stages


async def async_stream_generator(sync_gen):
    """
    Fully async wrapper for a synchronous generator.
    Yields items without using threads.
    """
    for item in sync_gen:
        await asyncio.sleep(0.001)  # yield control to event loop
        yield item


async def process_request(request: AskRequest):
    """Verwerkt een verzoek en streamt partial responses."""

    if summary_request(request.prompt):
        return {
            "question": request.prompt,
            "answer": get_summary(request.prompt),
            "prompt": "",
            "active_filter": "",
            "source_documents": [],
            "time_stages": {},
        }

    database_filter = (
        maak_metadata_filter(request, COMPONENTS, CONFIG["INCLUDE_PERMISSION"])
        if CONFIG["INCLUDE_FILTER"]
        else None
    )
    callback = StreamingResponseCallback(request.id)
    if request.prompt.startswith("!"):
        request.rag = 0

    # Retrieve the correct template and reference docs
    prompt_with_template, reference_docs, time_stages = build_prompt_template(
        chroma_filter=database_filter, prompt=request.prompt, rag=request.rag
    )

    stream = LLM_MANAGER.get_llm().client(
        prompt_with_template, stream=True, max_tokens=1500
    )
    full_answer = ""
    buffer = ""
    sentence_end_re = re.compile(r"[.!?]")
    seen_sentences = set()
    async for chunk in async_stream_generator(stream):
        token = chunk["choices"][0]["text"]
        full_answer += token
        buffer += token
        callback.on_llm_new_token(token)

        if not sentence_end_re.search(token):
            continue

        # Only check the last sentence is a replication
        sentences = re.split(r"(?<=[.!?])\s*", buffer)
        completed = sentences[:-1]
        buffer = sentences[-1]
        if not completed:
            continue

        # We only check the LAST completed sentence
        last_sentence = completed[-1].strip()
        if (
            last_sentence
            and is_valid_sentence(last_sentence)
            and last_sentence in seen_sentences
        ):
            logging.info(f"Detected duplicate sentence: {last_sentence}")
            final_answer = replace_patterns(uniek_antwoord(full_answer))
            return {
                "question": request.prompt,
                "answer": final_answer,
                "prompt": prompt_with_template,
                "active_filter": str(database_filter),
                "source_documents": reference_docs,
                "time_stages": time_stages,
            }
        seen_sentences.add(last_sentence)

    # Generator klaar, final answer
    final_answer = replace_patterns(uniek_antwoord(full_answer))

    # TODO
    # Nacontrole

    return {
        "question": request.prompt,
        "answer": final_answer,
        "prompt": prompt_with_template,
        "active_filter": str(database_filter),
        "source_documents": reference_docs,
        "time_stages": time_stages,
    }


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


def validate_structured_query(question):
    aads = haal_dossiers_op(question)
    if len(aads) > 0:
        where_clause = "WHERE d.aad_id IN $aad_ids"
    else:
        where_clause = ""
    cypher_to_run = build_cypher_query(question, clause=where_clause)
    cypher_to_run = cypher_to_run.format(where_clause=where_clause)
    logging.info(f"Closest query: {cypher_to_run}")
    parameters = {"aad_ids": aads}
    return GRAPH.query(cypher_to_run, params=parameters)


def validate_structured_query_embedding(question):
    aads = haal_dossiers_op(question)
    nbs = check_for_nbs(question)
    results = db_cypher.similarity_search_with_relevance_scores(question, k=20)
    # NOTE: doc[0] = actual query info and doc[1] = sim score
    tag_filtered_results = [
        doc
        for doc in results
        if match_query_by_tags(question=question, query=doc[0].metadata)
        and doc[1] > doc[0].metadata["threshold"]
    ]
    if len(tag_filtered_results) > 0:
        top_doc, score = tag_filtered_results[0]
        cypher_to_run = top_doc.metadata["cypher"]
        logging.info(f"Closest query: {cypher_to_run} with score {score}")
        parameters = {"aad_ids": aads, "netbeheerders": nbs}
        logging.info(f"Parameters found: {parameters}")
        result = GRAPH.query(cypher_to_run, params=parameters)
        logging.info(f"Neo4j results: {result}")
        return result
    else:
        return []


def retrieve_neo_answer(question, neo4j_result):
    """Verwerk NEO4J verzoeken."""
    _, trimmed_neo4j_result = trim_context_to_fit(
        model=LLM_MANAGER.get_llm().client,
        template=DEFAULT_QA_PROMPT,
        context_text=str(neo4j_result),
        question=question,
        n_ctx=CONFIG["MAX_CTX"],
        max_tokens=CONFIG["MAX_TOKENS"],
    )
    logging.info(
        f"Trimmed neo4j result: {len(str(neo4j_result))} to {len(trimmed_neo4j_result)}"
    )
    return CYPHER_PROMPT.format(
        prompt_elementen=dynamische_prompt_elementen(),
        result=trimmed_neo4j_result,
        question=question,
    )


def get_image_name() -> str:
    return os.getenv("IMAGE_NAME", "Unknown")


@app.post("/summarize")
def summarize(req: FileRequest):
    file_path = Path(req.file_path)

    # Check file exists
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Read text
    text = file_path.read_text(encoding="utf-8")
    logging.info(f"Done reading text file {req.file_path}")
    summarizer = RecursiveSummarizer(
        llm_manager=LLM_MANAGER, template=SUMMARY_PROMPT, text=text
    )
    summary = summarizer.summarize()
    summary_cleaned = clean_text_with_dup_detection(summary)
    return {
        "status": "ok",
        "summary_cleaned": summary_cleaned,
        "summary_raw": summary,
        "summary_length": len(summary_cleaned.split()),
    }


@app.post("/set-context")
def set_context(req: LLMRequest):
    LLM_MANAGER.load_llm(req.n_ctx)
    return {"status": "ok", "new_context": req.n_ctx}


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

        if "partial_response" in response_data:
            return {
                "status": "processing",
                "start_time_formatted": response_data["start_time_formatted"],
                "in_queue_start": response_data["in_queue_start"],
                "response": {
                    "answer": response_data["partial_response"],
                    "source_documents": None,
                },
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


@app.post("/context")
def context(req: ContextRequest):
    return {
        "answer": LLM_MANAGER.get_llm().invoke(req.prompt),
    }


@app.get("/metadata")
def get_metadata():
    CONFIG["image_name"] = get_image_name()
    return CONFIG


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
