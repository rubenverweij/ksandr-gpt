import argparse
import json
import shutil
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

from get_embedding_function import get_embedding_function


CHROMA_PATH = Path("/root/onprem_data/chroma")
SOURCE_DIR = Path("/root/ksandr_files")


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

    documents = load_documents(SOURCE_DIR)
    if not documents:
        print("⚠️ No documents found. Exiting.")
        return

    chunks = split_documents(documents)
    chunks = calculate_chunk_ids(chunks)
    add_to_chroma(chunks)


def load_documents(directory: Path) -> List[Document]:
    """Recursively load all JSON documents from a directory."""
    documents: List[Document] = []

    for file_path in directory.rglob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = json.load(f)

            # Ensure content is a string, since Document expects text
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, indent=2)

            documents.append(
                Document(page_content=content, metadata={"file_path": str(file_path)})
            )
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON file: {file_path}")
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")

    print(f"📄 Loaded {len(documents)} documents.")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split into {len(chunks)} chunks.")
    return chunks


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """Assign unique IDs to document chunks based on their source and order."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("file_path", "unknown")
        current_page_id = source

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def add_to_chroma(chunks: List[Document]) -> None:
    """Add new document chunks to the ChromaDB database."""
    batch_size = 4000
    db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=get_embedding_function(),
    )

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"📦 Existing documents in DB: {len(existing_ids)}")
    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(
            f"👉 Adding {len(new_chunks)} new documents in batches of {batch_size}..."
        )
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            db.add_documents(batch)
            print(f"   ✅ Added batch {i // batch_size + 1} ({len(batch)} docs)")
        db.persist()
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
