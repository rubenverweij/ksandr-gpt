import argparse
import os
from typing import List, Tuple

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.schema import Document

from get_embedding_function import get_embedding_function


CHROMA_PATH = "chroma"

DEFAULT_MODEL_PATH = "/root/onprem_data/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

PROMPT_TEMPLATE = """
Je bent een behulpzame, nauwkeurige en feitelijke assistent van het Ksandr data platform. Je taak is om vragen te beantwoorden over documenten die beschikbaar zijn op het Ksandr-platform. 
Documenten worden in de context aangeduid aadDocumentId en documentId, de 'url' is de locatie van het document bijvoorbeeld '/aad-document/10' en /download-aad-file-system-document/12803. Je kan verwijzen naar de url waarden die je tegenkomt in de context.
Gebruik voor specifieke vragen uitsluitend de onderstaande context om de vraag te beantwoorden. Als het antwoord niet in de context staat, zeg dan eerlijk dat je het niet weet.

Context:
{context}

Vraag:
{question}

Antwoord:
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Query ChromaDB with RAG pipeline.")
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the LLaMA model file.",
    )
    args = parser.parse_args()

    query_rag(args.query_text, model_path=args.model_path)


def query_rag(query_text: str, model_path: str) -> str:
    """Query ChromaDB with a text question and return an LLM-generated answer."""

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory="chroma/", embedding_function=embedding_function)

    # Search the DB.
    results: List[Tuple[Document, float]] = db.similarity_search_with_score(
        query_text, k=10
    )
    if not results:
        print("⚠️ No relevant documents found in the database.")
        return "No relevant context found."

    # Build context from search results.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Build prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize LLM
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    model = LlamaCpp(
        model_path=model_path,
        max_tokens=500,
        n_gpu_layers=-1,
        n_ctx=8096,
        verbose=False,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    return model.invoke(prompt), [
        doc.metadata.get("id") for doc, _ in results if doc.metadata.get("id")
    ]


if __name__ == "__main__":
    main()
