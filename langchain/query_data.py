import argparse
import os
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.schema import Document

from get_embedding_function import get_embedding_function


CHROMA_PATH = "chroma"

DEFAULT_MODEL_PATH = "/root/onprem_data/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
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
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results: List[Tuple[Document, float]] = db.similarity_search_with_score(
        query_text, k=5
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
        max_tokens=2500,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
    )

    # Run inference
    response_text = model.invoke(prompt)

    # Gather sources
    sources = [doc.metadata.get("id") for doc, _ in results if doc.metadata.get("id")]

    # Output
    formatted_response = f"📝 Response:\n{response_text}\n\n📚 Sources:\n{sources}"
    print(formatted_response)

    return response_text


if __name__ == "__main__":
    main()
