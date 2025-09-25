import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Union

from bs4 import BeautifulSoup
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from helpers import get_embedding_function


VALID_PERMISSIONS = {"cat-1", "cat-2"}


def extract_file_data(file_path: str) -> Dict[str, Union[int, str]]:
    """Extraheer type, permissie en bestandsnaam uit het pad."""
    parts = file_path.strip("/").split("/")
    result = {
        "type": "",
        "type_id": "na",
        "permission": "",
        "filename": "",
        "permission_and_type": "",
    }

    data_groups = ["aads", "general", "documents", "groups", "ese", "esg", "rmd", "dga"]

    if len(parts) >= 3:
        derde_deel = parts[2]
        if derde_deel in data_groups:
            result["type"] = derde_deel
            try:
                if derde_deel == "aads":
                    aads_index = parts.index("aads")
                    result["permission"] = (
                        parts[aads_index + 2]
                        if len(parts) > aads_index + 2
                        else "cat-3"
                    )
                    result["type_id"] = parts[aads_index + 1]
                    result["permission_and_type"] = (
                        f"{result['permission']}_{result['type_id']}"
                    )
                elif derde_deel in ["documents", "groups", "rmd", "dga"]:
                    result["permission"] = parts[parts.index(derde_deel) + 1]
                    result["permission_and_type"] = (
                        f"{result['permission']}_{result['type']}"
                    )
                elif derde_deel in ["ese", "esg", "general"]:
                    result["permission"] = "true"
                    result["permission_and_type"] = (
                        f"{result['permission']}_{result['type']}"
                    )
            except (IndexError, ValueError):
                pass

    if not result["type"]:
        result["type"] = "unknown"
    if not result["permission"]:
        result["permission"] = "undefined"

    result["filename"] = os.path.splitext(os.path.basename(file_path))[0]
    return result


def clean_html(value):
    """Maak HTML-schoon en verwijder tags/lege velden."""
    if value in (None, "", [], {}):
        return None
    if isinstance(value, str):
        # Verwijder HTML-tags
        cleaned = BeautifulSoup(value, "html.parser").get_text(separator=" ")
        # Verwijder overtollige witruimte
        return re.sub(r"\s+", " ", cleaned).strip()
    if isinstance(value, dict):
        value.pop("tags", None)
        return {k: clean_html(v) for k, v in value.items() if clean_html(v) is not None}
    if isinstance(value, list):
        lijst = [clean_html(v) for v in value if clean_html(v) is not None]
        return lijst if lijst else None
    return value


def prepare_text_for_vector_store(text: str) -> str:
    # Normalize escaped and real newlines
    text = text.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    # Replace 3+ newlines with just 2 (keeps paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing spaces on each line
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Collapse multiple spaces inside lines
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def load_documents(directory: Path) -> List[Document]:
    """Laad alle JSON-documenten uit een map, splits ze op en verrijk met metadata."""
    documenten: List[Document] = []
    for file_path in directory.rglob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = json.load(f)
                content = clean_html(content)
                splitter = RecursiveJsonSplitter(max_chunk_size=CHUNK_SIZE)
                chunks = splitter.split_text(json_data=content, convert_lists=True)
                for idx, chunk in enumerate(chunks):
                    try:
                        parsed = json.loads(chunk)  # parse back into dict
                        main_key = (
                            list(parsed.keys())[0] if isinstance(parsed, dict) else None
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
            chunk_size=CHUNK_SIZE,
            chunk_overlap=100,
        )
        for file_path in directory.rglob("*.txt"):
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    content = f.read()
                    content_cleaned = prepare_text_for_vector_store(content)
                    chunks = splitter.split_text(content_cleaned)
                    for idx, chunk in enumerate(chunks):
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
        "-chunk_size",
        type=int,
    )
    parser.add_argument("-include_text", type=int, default=1)

    args = parser.parse_args()
    CHROMA_PATH = Path(args.chroma)
    SOURCE_DIR = Path(args.source)
    CHUNK_SIZE = args.chunk_size
    INCLUDE_TEXT = args.include_text
    clear_database()
    chunks = load_documents(SOURCE_DIR)
    if not chunks:
        print("⚠️ Geen documenten gevonden. Stoppen.")
    else:
        add_to_chroma(chunks)
