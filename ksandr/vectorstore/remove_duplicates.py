"""
Remove duplicate documents from a Chroma vectorstore by hashing document text, identifying duplicates,
removing them from the database, and saving their metadata for further review.

This script connects to a Chroma database, computes MD5 hashes of stripped document texts to normalize content,
detects duplicate entries (i.e., documents with identical text), removes duplicates from the collection,
and writes details of all identified duplicates (with relevant metadata and text snippets) to a JSON file.

Functions:
- hash_doc: Computes an MD5 hash of document text for efficient deduplication.
- dedup_docs_in_chroma: Main process to scan Chroma for duplicates, remove them, and log metadata.

Example usage:
    python remove_duplicates.py --env staging
"""

import hashlib
import json
import argparse
import logging
from langchain_chroma import Chroma
from ksandr.embeddings.embeddings import get_embedding_function
from ksandr.vectorstore.config import (
    CHROMA_DB_PATH,
    DUPLICATES_DATA_PATH,
    running_inside_docker,
)


def hash_doc(text: str) -> str:
    """
    Compute an MD5 hash of the provided text after stripping leading/trailing whitespace.

    Args:
        text (str): The text string to hash.

    Returns:
        str: Hexadecimal MD5 hash of the normalized text.
    """
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def dedup_docs_in_chroma(chroma_path: str, output_json_path: str):
    """
    Detect duplicate documents in a Chroma vectorstore by content hash, remove duplicates,
    and log their metadata to a JSON file.

    Process:
        1. Connect to the Chroma database at the provided path.
        2. Retrieve all documents, compute a hash of document text.
        3. Identify duplicates (documents sharing a hash).
        4. For each set of duplicates, collect relevant metadata and a text snippet.
        5. Remove duplicate documents from the database, keeping only the first occurrence.
        6. Write metadata of all duplicate groups (groups with >=2 documents) to the output JSON file.

    Args:
        chroma_path (str): Path to the Chroma database directory.
        output_json_path (str): Path to save the JSON file with duplicate metadata.

    Side effects:
        - Removes duplicate documents from Chroma vectorstore.
        - Writes a JSON file with duplicate documents' metadata for inspection.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    results = db.get()
    seen_hashes = {}  # Tracks which hashes have already been encountered; maps hash to first doc id
    to_delete = []  # Stores document IDs of duplicates to be removed
    hash_to_metadata = {}  # Collects metadata summaries of each set of duplicates

    for i, (doc, doc_id, metadata) in enumerate(
        zip(results["documents"], results["ids"], results["metadatas"])
    ):
        doc_hash = hash_doc(doc)
        keys_to_extract = ["source", "chunk"]
        subset = {k: metadata[k] for k in keys_to_extract if k in metadata}
        subset["text"] = doc[:20] + " ... " + doc[-20:]
        if doc_hash not in seen_hashes:
            seen_hashes[doc_hash] = doc_id  # First document with this hash is kept
            hash_to_metadata[doc_hash] = [subset]
        else:
            to_delete.append(doc_id)  # Mark duplicate for deletion
            hash_to_metadata[doc_hash].append(subset)

        if i % 1000 == 0:
            logging.info(
                f"➡️  Checked: {i} documents... {len(to_delete)} duplicates found"
            )
    # Retain only hash groups with >=2 documents (i.e., actual duplicates)
    hash_to_metadata = {k: v for k, v in hash_to_metadata.items() if len(v) >= 2}

    # Write duplicate metadata summaries to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(hash_to_metadata, f, ensure_ascii=False, indent=2)

    logging.info(f"Duplicate metadata saved to: {output_json_path}")

    # Remove duplicate documents from Chroma
    if to_delete:
        logging.info(f"{len(to_delete)} duplicates found. Removing...")
        db.delete(ids=to_delete)
    else:
        logging.info("✅ No duplicates found.")

    logging.info("✅ Deduplication complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Remove duplicate documents from the Chroma vectorstore and log their metadata."
    )
    parser.add_argument(
        "-env",
        choices=["production", "staging"],
        default="production",
        help="Environment: 'production' or 'staging' (default: production)",
    )
    args = parser.parse_args()
    chroma_path = CHROMA_DB_PATH.get(args.env).get(running_inside_docker())
    duplicates_json_path = DUPLICATES_DATA_PATH.get(args.env).get(
        running_inside_docker()
    )
    dedup_docs_in_chroma(chroma_path=chroma_path, output_json_path=duplicates_json_path)
