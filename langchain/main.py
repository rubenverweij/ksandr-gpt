from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List, Tuple
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.schema import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from get_embedding_function import get_embedding_function

# 🔧 Configuratie
REDIS_URL = "redis://redis-server:6379"
CHROMA_PATH = "/root/onprem_data/chroma"
DEFAULT_MODEL_PATH = "/root/onprem_data/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# 📄 Prompt template voor de assistent
PROMPT_TEMPLATE = """
Je bent een behulpzame, nauwkeurige en feitelijke assistent van het Ksandr data platform. Je taak is om vragen te beantwoorden over documenten die beschikbaar zijn op het Ksandr-platform. 
Documenten worden in de context aangeduid met aadDocumentId en documentId. De 'url' is de locatie van het document, bijvoorbeeld '/aad-document/10' of '/download-aad-file-system-document/12803'. 
Je mag verwijzen naar de url-waarden die je tegenkomt in de context.
Gebruik uitsluitend de onderstaande context om de vraag te beantwoorden en herhaal niet. Als het antwoord niet in de context staat, zeg dan eerlijk dat je het niet weet.

Context:
{context}

Vraag:
{question}

Antwoord:
"""

app = FastAPI()


# Pydantic model voor de inkomende verzoek body
class QueryRequest(BaseModel):
    query_text: str
    user_id: str
    model_path: str = DEFAULT_MODEL_PATH  # Optioneel, standaard model pad


# Beperkte Redis geschiedenis om de laatste N berichten (alleen vragen) op te slaan
class LimitedRedisChatMessageHistory(RedisChatMessageHistory):
    def __init__(self, session_id: str, url: str, max_messages: int = 2):
        super().__init__(session_id=session_id, url=url)
        self.max_messages = max_messages

    def add_message(self, message: dict):
        self.redis_client.lpush(self.key, message)
        self.redis_client.ltrim(self.key, 0, self.max_messages - 1)

    def load_memory_variables(self, inputs: dict):
        messages = self.redis_client.lrange(self.key, 0, self.max_messages - 1)
        return {"history": [msg.decode("utf-8") for msg in messages]}


# Haal gebruikersgeheugen op (alleen de laatste twee vragen)
def get_user_memory(user_id: str) -> ConversationBufferMemory:
    redis_history = LimitedRedisChatMessageHistory(
        session_id=user_id, url=REDIS_URL, max_messages=2
    )
    return ConversationBufferMemory(
        memory_key="history",
        chat_memory=redis_history,
        return_messages=True,
    )


# 🤖 Initialiseer het taalmodel met streaming
def load_model(model_path: str) -> LlamaCpp:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modelbestand niet gevonden: {model_path}")

    return LlamaCpp(
        model_path=model_path,
        max_tokens=500,
        n_gpu_layers=-1,
        n_ctx=8096,
        max_new_tokens=150,
        verbose=False,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0.4,
        top_p=0.85,
    )


# 🧠+📚 Combineer context uit vector DB met geheugen om vraag te beantwoorden
def query_rag(query_text: str, user_id: str, model: LlamaCpp) -> str:
    # Haal het gebruikersgeheugen op (alleen de laatste twee vragen)
    memory = get_user_memory(user_id)

    # Zoek naar relevante documenten via Chroma
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results: List[Tuple[Document, float]] = db.similarity_search_with_score(
        query_text, k=10
    )

    if not results:
        print("⚠️ Geen relevante documenten gevonden in de database.")
        return "Geen relevante bronnen gevonden, probeer de vraag te verduidelijken."

    # Bouw de context op uit de gevonden documenten
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Format de prompt met context en vraag
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Start een conversatie met geheugen
    conversation = RunnableWithMessageHistory(
        llm=model,
        memory=memory,
        verbose=False,
    )

    # Stream en retourneer het antwoord
    response = conversation.predict(input=prompt)
    return response


# Laad het model bij de opstart van de FastAPI-app
@app.on_event("startup")
async def startup_event():
    model_path = DEFAULT_MODEL_PATH
    app.state.model = load_model(model_path)
    print("Model succesvol geladen!")


@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        model = app.state.model
        response = query_rag(
            query_text=request.query_text, user_id=request.user_id, model=model
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Er is een fout opgetreden: {str(e)}"
        )
