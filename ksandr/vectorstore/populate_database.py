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
import logging
from pathlib import Path
from typing import List
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from ksandr.vectorstore.helpers import (
    extract_file_data,
    looks_like_clean_text,
    prepare_text_for_vector_store,
    clean_html,
)
from ksandr.embeddings.embeddings import get_embedding_function
from ksandr.vectorstore.config import (
    CHROMA_DB_PATH,
    RAW_DATA_SOURCES,
    running_inside_docker,
)


# Compile patterns once (efficient)
PATTERNS = [
    # /root/ksandr_files/aads/{id}/cat-x/documents/{doc}.txt
    (
        re.compile(
            r"^/root/ksandr_files/aads/(?P<aad_id>\d+)/cat-\d+/documents/(?P<doc_id>\d+)\.txt$"
        ),
        lambda m: f"/aad/{m.group('aad_id')}/document/{m.group('doc_id')}",
    ),
    # /root/ksandr_files/aads/{id}/cat-x/main.json
    (
        re.compile(r"^/root/ksandr_files/aads/(?P<aad_id>\d+)/cat-\d+/main\.json$"),
        lambda m: f"/aad/{m.group('aad_id')}/dossier",
    ),
    # /root/ksandr_files/documents/{id}/{id}.txt
    (
        re.compile(r"^/root/ksandr_files/documents/(?P<doc_id>\d+)/\d+\.txt$"),
        lambda m: f"/private-document/{m.group('doc_id')}",
    ),
    # /root/ksandr_files/expert-group/{group}/documents/{doc}.txt
    (
        re.compile(
            r"^/root/ksandr_files/expert-group/(?P<group_id>\d+)/documents/(?P<doc_id>\d+)\.txt$"
        ),
        lambda m: f"/expert-group/{m.group('group_id')}/document/{m.group('doc_id')}",
    ),
    # /root/ksandr_files/project-group/{group}/documents/{doc}.txt
    (
        re.compile(
            r"^/root/ksandr_files/project-group/(?P<group_id>\d+)/documents/(?P<doc_id>\d+)\.txt$"
        ),
        lambda m: f"/project-group/{m.group('group_id')}/document/{m.group('doc_id')}",
    ),
]


def convert_filepath(path: str) -> str | None:
    """
    Convert internal filepath to public URL structure.
    Returns None if no pattern matches.
    """
    for pattern, transformer in PATTERNS:
        match = pattern.match(path)
        if match:
            return transformer(match)
    return None


def load_terms(terms: list, file_path: Path):
    """
    Convert question-answer pairs from a terms.json file into Document objects for vectorstore ingestion.

    Args:
        terms (list): List of dicts with "instruction" and "output" keys, typically loaded from 'terms.json'.
        file_path (Path): Path to the JSON file for metadata.

    Returns:
        List[Document]: LangChain Document objects with content and enriched metadata for each Q&A pair.
    """
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


def load_users(users: list, file_path: Path):
    """
    Convert user records from a users.json file into Document objects for vectorstore ingestion.

    Args:
        users (list): List of dicts, each representing a user with attributes such as "Naam", "E-mail", "Bedrijf", and others.
        file_path (Path): Path to the JSON file for metadata.

    Returns:
        List[Document]: LangChain Document objects where each document represents a user and associated metadata.
    """

    terms_list = []
    for idx, user in enumerate(users):
        if len(user.get("Naam", "")) == 0:
            continue
        tekst = f"Wie is {user.get('Naam', '')}?. Beschikbare gegevens over KSANDR deelnemer {user.get('Naam', '')}: {user}"
        metadata = {
            "file_path": file_path.as_posix(),
            "chunk": idx,
            "char_length": len(str(tekst)),
            "useful": 1,
            "source": file_path.as_posix(),
            "source_search": file_path.as_posix(),
            "key": "users",
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


def change_patterns_file_path(filepath: Path) -> str:
    """
    Convert a given file path (string or Path) to a standardized/documentation-ready format used for linking/grouping.

    - If the path is within a group directory (containing "groups"), it replaces "groups" with a specific group type
      such as "project-group" or "expert-group", determined from the group's main.json metadata.
    - The returned path always uses the converted format via convert_filepath().

    Args:
        filepath (Union[str, Path]): The file path to be reformatted (can be str or pathlib.Path).

    Returns:
        str: The converted file path in the new format.
    """

    # Ensure input is a Path object (in docstring, Union[str, Path], but function signature uses Path, so clarify)
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    group_link = None
    # "groups" is always checked as substring of str(filepath) below, but then filepath is mutated later...
    # Consistency: always work with string for replacements, and only convert back to Path if necessary.
    filepath_str = str(filepath)

    if "groups" in filepath_str:
        try:
            group_dir = filepath.parents[2]  # points to /root/ksandr_files/groups/832/
            main_json_path = group_dir / "main.json"
            with main_json_path.open("r", encoding="utf-8") as f:
                group_info = json.load(f)
                group_type = group_info.get("type", None)
            if group_type == "Projectgroep":
                group_link = "project-group"
            elif group_type == "Expertgroep":
                group_link = "expert-group"
            else:
                group_link = group_type if group_type else "groups"
        except Exception:
            # Defensive: fallback to "groups" if anything goes wrong
            group_link = "groups"

        # Consistency: Only replace "groups" with group_link if group_link is set and non-empty
        if group_link:
            filepath_str = filepath_str.replace(
                "groups", group_link, 1
            )  # only replace first occurrence

    # Always use convert_filepath on the path string
    return convert_filepath(filepath_str)


def load_documents(env: str) -> List[Document]:
    """
    Load and process all JSON documents for a given environment, splitting them into meaningful, metadata-rich chunks for vectorstore ingestion.

    This function recursively searches the data source directory for JSON files.
    For each discovered file:
        - If the filename is 'terms.json', it extracts question-answer pairs and converts them to Document objects designed for Q&A retrieval tasks.
        - For other JSON files, HTML tags are cleaned, and the file is split into smaller JSON-like text chunks using RecursiveJsonSplitter.
        - Each chunk is cleaned and filtered using heuristics for text quality.
        - Rich metadata is added, including file path, chunk position, source type, character length, and inferred document group or permissions.

    Args:
        env (str): Environment identifier to select which DATA_SOURCE directory to process (e.g., 'production' or 'staging').

    Returns:
        List[Document]: List of LangChain Document objects, each containing chunk content and associated metadata, suitable for vectorstore ingestion.
    """
    include_text = 1
    documenten: List[Document] = []
    directory = Path(RAW_DATA_SOURCES.get(env).get(running_inside_docker()))

    if not directory.exists():
        logging.error(f"Directory does not exist: {directory}")

    for file_path in directory.rglob("*.json"):
        link = change_patterns_file_path(file_path)
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = json.load(f)
                content = clean_html(content)
                if file_path.name == "terms.json":
                    logging.info("Ingesting terms")
                    documenten = documenten + load_terms(content, file_path)
                if file_path.name == "users.json":
                    logging.info("Ingesting users")
                    documenten = documenten + load_users(content, file_path)
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
                        logging.info(
                            f"Ingesting json {file_path.as_posix()} with link {link}"
                        )
                        metadata = {
                            "file_path": file_path.as_posix(),
                            "chunk": idx,
                            "char_length": len(str(chunk_cleaned)),
                            "useful": 0 if len(chunk_cleaned) < 150 else 1,
                            "source": file_path.as_posix(),
                            "link": link,
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
            logging.error(f"❌ Ongeldig JSON-bestand: {file_path}")
        except Exception as e:
            logging.error(f"❌ Fout bij lezen van {file_path}: {e}")

    if include_text:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=MIN_CHUNK_SIZE_TEXT,
            chunk_overlap=100,
        )
        for file_path in directory.rglob("*.txt"):
            link = change_patterns_file_path(file_path)
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
                        logging.info(
                            f"Ingesting file {file_path.as_posix()} with link {link}"
                        )
                        metadata = {
                            "file_path": file_path.as_posix(),
                            "chunk": idx,
                            "char_length": len(chunk_cleaned),
                            "useful": 0 if len(chunk_cleaned) < 150 else 1,
                            "source": file_path.as_posix(),
                            "source_search": file_path.as_posix(),
                            "link": link,
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
                logging.error(f"❌ Error reading {file_path}: {e}")

    logging.info(f"📄 {len(documenten)} stukken geladen.")
    return documenten


def add_to_chroma(chunks: List[Document], env: str) -> None:
    """
    Add a list of Document chunks to the Chroma vector store.

    Args:
        chunks (List[Document]): The list of Document objects (text chunks + metadata) to add.
        env: environment to fill

    This function creates (or loads) a Chroma vector database at the specified persist directory,
    and adds the provided Document objects to it in batches for efficiency. Each batch is processed
    and saved. Progress and batch status messages are printed to the console.

    If the input `chunks` list is empty, a message is printed and no action is taken.
    """
    batch_size = 4000
    db = Chroma(
        persist_directory=Path(CHROMA_DB_PATH.get(env).get(running_inside_docker())),
        embedding_function=get_embedding_function(),
    )
    if chunks:
        logging.info(
            f"👉 Toevoegen van {len(chunks)} documenten in batches van {batch_size}..."
        )
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            db.add_documents(batch)
            logging.info(
                f"   ✅ Batch {i // batch_size + 1} toegevoegd ({len(batch)} docs)"
            )
        logging.info("✅ Database bijgewerkt.")
    else:
        logging.info("✅ Geen nieuwe documenten om toe te voegen.")


def clear_database(env: str) -> None:
    """
    Remove the existing Chroma vector store database by deleting the persist directory.

    This function deletes the directory at CHROMA_PATH (if it exists), effectively clearing all
    data from the current vector store instance. Use this before re-populating the database,
    for example to ensure no old or duplicate data remains.

    Side effects:
        - Deletes all files and subdirectories at CHROMA_PATH.
        - Prints a confirmation message upon successful removal.
    """
    path_to_database = Path(CHROMA_DB_PATH.get(env).get(running_inside_docker()))
    if path_to_database.exists():
        shutil.rmtree(path_to_database)
        logging.info(f"Database {path_to_database.as_posix()} removed.")
    else:
        logging.info(f"Database does not yet exist at {path_to_database.as_posix()}.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Process JSON documents and store them in ChromaDB."
    )
    parser.add_argument(
        "-min_chunk_size_json",
        type=int,
        default=400,
        required=False,
    )
    parser.add_argument(
        "-min_chunk_size_text",
        type=int,
        default=600,
        required=False,
    )
    parser.add_argument(
        "-env",
        type=str,
        choices=["production", "staging"],
        required=False,
        default="production",
        help="Kies de omgeving: 'production' of 'staging' (standaard: production)",
    )
    args = parser.parse_args()

    MIN_CHUNK_SIZE_JSON = args.min_chunk_size_json
    MIN_CHUNK_SIZE_TEXT = args.min_chunk_size_text

    # env variable now comes from os.environ, with fallback to argparse default
    env = os.environ.get("ENV", args.env)

    logging.info(f"🛠️ Omgeving geselecteerd: {env}")

    clear_database(env)
    chunks = load_documents(env)
    if not chunks:
        logging.warning("⚠️ Geen documenten gevonden. Stoppen.")
    else:
        add_to_chroma(chunks, env)
