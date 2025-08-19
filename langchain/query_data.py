import argparse
import os
from typing import List, Tuple

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.chains import ConversationChain

from get_embedding_function import get_embedding_function

# 🔧 Configuratie
REDIS_URL = "redis://localhost:6379"
CHROMA_PATH = "/root/onprem_data/chroma"
DEFAULT_MODEL_PATH = "/root/onprem_data/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# 📄 Prompt template voor de assistent
PROMPT_TEMPLATE = """
Je bent een behulpzame, nauwkeurige en feitelijke assistent van het Ksandr data platform. Je taak is om vragen te beantwoorden over documenten die beschikbaar zijn op het Ksandr-platform. 
Documenten worden in de context aangeduid met aadDocumentId en documentId. De 'url' is de locatie van het document, bijvoorbeeld '/aad-document/10' of '/download-aad-file-system-document/12803'. 
Je mag verwijzen naar de url-waarden die je tegenkomt in de context.
Gebruik uitsluitend de onderstaande context om de vraag te beantwoorden. Als het antwoord niet in de context staat, zeg dan eerlijk dat je het niet weet.

Context:
{context}

Vraag:
{question}

Antwoord:
"""


# 🧠 Haal gebruikersspecifiek geheugen op (of maak aan)
def get_user_memory(user_id: str) -> ConversationBufferMemory:
    redis_history = RedisChatMessageHistory(
        session_id=user_id,
        url=REDIS_URL,
    )
    return ConversationBufferMemory(
        memory_key="history",
        chat_memory=redis_history,
        return_messages=True,
    )


# 🤖 Initialiseert het taalmodel met streaming
def load_model(model_path: str) -> LlamaCpp:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modelbestand niet gevonden: {model_path}")

    return LlamaCpp(
        model_path=model_path,
        max_tokens=500,
        n_gpu_layers=-1,
        n_ctx=8096,
        verbose=False,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )


# 🧠+📚 Combineer context uit vector DB met geheugen om vraag te beantwoorden
def query_rag(query_text: str, user_id: str, model: LlamaCpp) -> str:
    # Haal gebruikersgeheugen op
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
    conversation = ConversationChain(
        llm=model,
        memory=memory,
        verbose=False,
    )

    # Stream en retourneer het antwoord
    response = conversation.predict(input=prompt)
    return response


# 🚀 Entry point van het script
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stel een vraag aan het Ksandr-platform via RAG + geheugen."
    )
    parser.add_argument("query_text", type=str, help="De gebruikersvraag.")
    parser.add_argument(
        "--user-id", type=str, required=True, help="Unieke gebruikers- of sessie-ID."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Pad naar het LLaMA-model.",
    )

    args = parser.parse_args()

    # Initialiseert het model
    model = load_model(args.model_path)

    # Voer de query uit
    query_rag(query_text=args.query_text, user_id=args.user_id, model=model)


if __name__ == "__main__":
    main()
