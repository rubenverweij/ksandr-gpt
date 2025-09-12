import argparse
import json
import shutil
from pathlib import Path
from typing import List
import os

from typing import Dict, Union
import re

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

from get_embedding_function import get_embedding_function


CHROMA_PATH = Path("/root/onprem_data/chroma")
SOURCE_DIR = Path("/root/ksandr_files")
# CHROMA_PATH = Path("chroma")
# SOURCE_DIR = Path("langchain/docs/")
VALID_PERMISSIONS = {"cat-1", "cat-2"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process JSON documents into ChromaDB."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database before adding documents.",
    )
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    chunks = load_documents(SOURCE_DIR)
    if not chunks:
        print("⚠️ No documents found. Exiting.")
        return
    add_to_chroma(chunks)


def extract_file_data(file_path: str) -> Dict[str, Union[int, str]]:
    """Extract 'type', 'permission', and 'filename' from a path."""
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
        third_part = parts[2]
        if third_part in data_groups:
            result["type"] = third_part
            try:
                if third_part == "aads":
                    aads_index = parts.index("aads")
                    result["permission"] = (
                        parts[aads_index + 2]
                        if len(parts) > aads_index + 2
                        else "cat-3"
                    )
                    result["permission_and_type"] = (
                        f"{result['permission']}_{parts[aads_index + 1]}"
                    )
                    result["type_id"] = parts[aads_index + 1]
                elif third_part in ["documents", "groups", "rmd", "dga"]:
                    result["permission"] = str(parts[parts.index(third_part) + 1])
                    result["permission_and_type"] = (
                        f"{result['permission']}_{result['type']}"
                    )
                elif third_part in ["ese", "esg", "general"]:
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
    if value is None or value == "" or value == [] or value == {}:
        return None
    if isinstance(value, str):
        # Verwijder HTML-tags
        cleaned_value = BeautifulSoup(value, "html.parser").get_text(separator=" ")
        # Verwijder onnodige spaties (aan begin, eind en tussenin)
        return re.sub(r"\s+", " ", cleaned_value).strip()
    if isinstance(value, dict):
        # Verwijder "tags" sleutel als het aanwezig is
        value.pop("tags", None)
        return {k: clean_html(v) for k, v in value.items() if clean_html(v) is not None}
    if isinstance(value, list):
        cleaned_list = [clean_html(v) for v in value if clean_html(v) is not None]
        return cleaned_list if cleaned_list else None
    return value  # Voor andere types zoals int, float, etc.


def load_documents(directory: Path) -> List[Document]:
    """Recursively load all JSON documents from a directory."""
    documents: List[Document] = []
    for file_path in directory.rglob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = json.load(f)
                content = clean_html(content)
                text_splitter = RecursiveJsonSplitter()
                chunks = text_splitter.split_json(content)
                print(file_path)
                for idx, chunk in enumerate(chunks):
                    file_metadata = {
                        "file_path": file_path.as_posix(),
                        "chunk": idx,
                        "char_length": len(str(chunk)),
                    }
                    file_metadata.update(extract_file_data(file_path.as_posix()))
                    documents.append(
                        Document(
                            page_content=str(chunk),
                            metadata=file_metadata,
                        )
                    )
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON file: {file_path}")
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
    print(f"📄 Loaded {len(documents)} chunks.")
    return documents


def add_to_chroma(chunks: List[Document]) -> None:
    """Add new document chunks to the ChromaDB database."""
    batch_size = 4000
    db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=get_embedding_function(),
    )
    if chunks:
        print(f"👉 Adding {len(chunks)} new documents in batches of {batch_size}...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            db.add_documents(batch)
            print(f"   ✅ Added batch {i // batch_size + 1} ({len(batch)} docs)")
        print("✅ Database updated.")
    else:
        print("✅ No new documents to add.")


def clear_database() -> None:
    """Delete the ChromaDB database directory."""
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
        print("🗑️ Database cleared.")


if __name__ == "__main__":
    main()
