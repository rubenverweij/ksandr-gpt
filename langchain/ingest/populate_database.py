"""
This script populates the Chroma vector database with documents for retrieval-augmented generation and search.

Key functionalities:
- Loads and splits documents (primarily JSON files) from a specified directory.
- Cleans, prepares, and enriches document chunks with metadata.
- Inserts processed documents into the Chroma persistent vectorstore using a specified embedding function.
- Includes helper routines for file parsing, validity checks, and chunk formatting.

Typical usage:
    python populate_database.py --directory /path/to/data --chroma_path /path/to/chroma

Dependencies:
- langchain libraries for document splitting and Chroma vectorstore management
- Local helpers for file extraction, cleaning, and embeddings

Intended for ingestion pipeline automation in the Ksandr platform.
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from helpers import (
    extract_file_data,
    looks_like_clean_text,
    prepare_text_for_vector_store,
    clean_html,
)
from embeddings import get_embedding_function

VALID_PERMISSIONS = {"cat-1", "cat-2"}


def load_terms(terms, file_path):
    terms_list = []
    for idx, question in enumerate(terms):
        if len(question["instruction"]) < 15:
            question["instruction"] = f"Wat is {question['instruction']}"
        tekst = f"Vraag: {question['instruction']} \nAntwoord: {question['output']}"
        metadata = {
            "file_path": file_path.as_posix(),
            "chunk": idx,
            "char_length": len(str(tekst)),
            "useful": 1,
            "source": file_path.as_posix(),
            "source_search": file_path.as_posix(),
            "key": "terms",
            "extension": "json",
        }
        metadata.update(extract_file_data(file_path.as_posix()))
        terms_list.append(
            Document(
                page_content=tekst,
                metadata=metadata,
            )
        )
    return terms_list


def load_documents(directory: Path) -> List[Document]:
    """Laad alle JSON-documenten uit een map, splits ze op en verrijk met metadata."""
    documenten: List[Document] = []
    for file_path in directory.rglob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = json.load(f)
                content = clean_html(content)
                if file_path.name == "terms.json":
                    print("Ingesting terms")
                    documenten = documenten + load_terms(content, file_path)
                else:
                    splitter = RecursiveJsonSplitter(min_chunk_size=MIN_CHUNK_SIZE_JSON)
                    chunks = splitter.split_text(json_data=content, convert_lists=True)
                    for idx, chunk in enumerate(chunks):
                        if not looks_like_clean_text(chunk):
                            continue
                        try:
                            parsed = json.loads(chunk)  # parse back into dict
                            main_key = (
                                list(parsed.keys())[0]
                                if isinstance(parsed, dict)
                                else None
                            )
                        except Exception:
                            main_key = None
                        chunk_cleaned = str(chunk)
                        chunk_cleaned = chunk_cleaned.replace("'", '"')
                        chunk_cleaned = chunk_cleaned.replace("\\'", "'")
                        chunk_cleaned = chunk_cleaned.replace('"', "")
                        chunk_cleaned = re.sub(r'[{}"\[\]]', "", chunk_cleaned)
                        metadata = {
                            "file_path": file_path.as_posix(),
                            "chunk": idx,
                            "char_length": len(str(chunk_cleaned)),
                            "useful": 0 if len(chunk_cleaned) < 150 else 1,
                            "source": file_path.as_posix(),
                            "source_search": file_path.as_posix(),
                            "key": main_key,
                            "extension": "json",
                        }
                        metadata.update(extract_file_data(file_path.as_posix()))
                        documenten.append(
                            Document(
                                page_content=chunk_cleaned,
                                metadata=metadata,
                            )
                        )
        except json.JSONDecodeError:
            print(f"❌ Ongeldig JSON-bestand: {file_path}")
        except Exception as e:
            print(f"❌ Fout bij lezen van {file_path}: {e}")

    if INCLUDE_TEXT:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=MIN_CHUNK_SIZE_TEXT,
            chunk_overlap=100,
        )
        for file_path in directory.rglob("*.txt"):
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    content = f.read()
                    content_cleaned = prepare_text_for_vector_store(content)
                    chunks = splitter.split_text(content_cleaned)
                    for idx, chunk in enumerate(chunks):
                        if not looks_like_clean_text(chunk):
                            continue
                        chunk_cleaned = str(chunk)
                        chunk_cleaned = chunk_cleaned.replace("'", '"')
                        chunk_cleaned = chunk_cleaned.replace("\\'", "'")
                        chunk_cleaned = re.sub(r'[{}"\[\]]', "", chunk_cleaned)
                        metadata = {
                            "file_path": file_path.as_posix(),
                            "chunk": idx,
                            "char_length": len(chunk_cleaned),
                            "useful": 0 if len(chunk_cleaned) < 150 else 1,
                            "source": file_path.as_posix(),
                            "source_search": file_path.as_posix(),
                            "key": "na",
                            "extension": ".txt",
                        }
                        metadata.update(extract_file_data(file_path.as_posix()))
                        documenten.append(
                            Document(
                                page_content=chunk_cleaned,
                                metadata=metadata,
                            )
                        )
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")

    print(f"📄 {len(documenten)} stukken geladen.")
    return documenten


def add_to_chroma(chunks: List[Document]) -> None:
    """Voeg stukken toe aan ChromaDB in batches."""
    batch_size = 4000
    db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=get_embedding_function(),
    )
    if chunks:
        print(
            f"👉 Toevoegen van {len(chunks)} documenten in batches van {batch_size}..."
        )
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            db.add_documents(batch)
            print(f"   ✅ Batch {i // batch_size + 1} toegevoegd ({len(batch)} docs)")
        print("✅ Database bijgewerkt.")
    else:
        print("✅ Geen nieuwe documenten om toe te voegen.")


def clear_database() -> None:
    """Verwijder de ChromaDB-directory en reset de database."""
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
        print("🗑️ Database verwijderd.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verwerk JSON-documenten en sla ze op in ChromaDB."
    )
    parser.add_argument(
        "-chroma",
        type=str,
    )
    parser.add_argument("-source", type=str)
    parser.add_argument(
        "-min_chunk_size_json",
        type=int,
    )
    parser.add_argument(
        "-min_chunk_size_text",
        type=int,
    )
    parser.add_argument("-include_text", type=int, default=1)
    args = parser.parse_args()
    CHROMA_PATH = Path(args.chroma)
    SOURCE_DIR = Path(args.source)
    MIN_CHUNK_SIZE_JSON = args.min_chunk_size_json
    MIN_CHUNK_SIZE_TEXT = args.min_chunk_size_text
    INCLUDE_TEXT = args.include_text
    clear_database()
    chunks = load_documents(SOURCE_DIR)
    if not chunks:
        print("⚠️ Geen documenten gevonden. Stoppen.")
    else:
        add_to_chroma(chunks)
