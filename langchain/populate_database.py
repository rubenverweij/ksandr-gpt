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
from langchain_chroma import Chroma

from get_embedding_function import get_embedding_function


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


def load_documents(directory: Path) -> List[Document]:
    """Laad alle JSON-documenten uit een map, splits ze op en verrijk met metadata."""
    documenten: List[Document] = []
    for file_path in directory.rglob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = json.load(f)
                content = clean_html(content)
                splitter = RecursiveJsonSplitter()
                chunks = splitter.split_json(content)
                for idx, chunk in enumerate(chunks):
                    chunk_cleaned = str(chunk)
                    chunk_cleaned = chunk_cleaned.replace("'", '"')
                    chunk_cleaned = chunk_cleaned.replace("\\'", "'")
                    chunk_cleaned = chunk_cleaned.replace('"', "")
                    chunk_cleaned = re.sub(r'[{}"\[\]]', "", chunk_cleaned)
                    metadata = {
                        "file_path": file_path.as_posix(),
                        "chunk": idx,
                        "char_length": len(str(chunk)),
                        "source": file_path.as_posix(),
                        "source_search": file_path.as_posix(),
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

    args = parser.parse_args()
    CHROMA_PATH = Path(args.chroma)
    SOURCE_DIR = Path(args.source)
    clear_database()
    chunks = load_documents(SOURCE_DIR)
    if not chunks:
        print("⚠️ Geen documenten gevonden. Stoppen.")
    else:
        add_to_chroma(chunks)
