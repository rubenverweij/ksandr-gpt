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
    Scan a Chroma vectorstore and remove duplicate documents based on their normalized text hash (MD5).
    The process loads documents in batches, compares content hashes, deletes duplicate entries, and logs duplicate
    metadata for further analysis.

    Args:
        chroma_path (str): The path to the Chroma vectorstore database directory.
        output_json_path (str): Path to write a JSON file with metadata of found duplicate groups.

    The deduplication is performed without loading all documents into memory at once
    (suitable for large collections), and metadata for all identified duplicate groups
    (i.e., where at least two documents share the same content) is written to disk for review.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    seen_hashes = {}
    duplicates_metadata = {}
    to_delete_batch = []

    BATCH_SIZE = 500  # Number of documents fetched per read operation
    DELETE_BATCH_SIZE = 500  # Number of duplicate IDs deleted per batch

    offset = 0
    total_checked = 0
    total_duplicates = 0

    logging.info("Starting deduplication process...")

    while True:
        results = db.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas"],
        )

        ids = results["ids"]
        documents = results["documents"]
        metadatas = results["metadatas"]

        if not ids:
            break  # No more documents to process

        for doc_id, doc, metadata in zip(ids, documents, metadatas):
            doc_hash = hash_doc(doc)

            # Extract relevant metadata and a text snippet for review
            keys_to_extract = ["source", "chunk"]
            subset = {k: metadata[k] for k in keys_to_extract if k in metadata}
            subset["text"] = doc[:20] + " ... " + doc[-20:]

            if doc_hash not in seen_hashes:
                seen_hashes[doc_hash] = doc_id
                duplicates_metadata.setdefault(doc_hash, []).append(subset)
            else:
                total_duplicates += 1
                to_delete_batch.append(doc_id)
                duplicates_metadata[doc_hash].append(subset)

            total_checked += 1

            # Log progress every 1000 documents
            if total_checked % 1000 == 0:
                logging.info(
                    f"➡️ Checked: {total_checked} docs | Duplicates: {total_duplicates}"
                )

        # Delete duplicate documents in batches to avoid oversized SQL queries
        if len(to_delete_batch) >= DELETE_BATCH_SIZE:
            db.delete(ids=to_delete_batch)
            to_delete_batch.clear()

        offset += BATCH_SIZE

    # Final deletion for any remaining documents in the batch
    if to_delete_batch:
        db.delete(ids=to_delete_batch)

    # Only retain entries where at least two documents were found (i.e., actual duplicates)
    duplicates_metadata = {k: v for k, v in duplicates_metadata.items() if len(v) >= 2}

    # Save metadata for all duplicate groups
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(duplicates_metadata, f, ensure_ascii=False, indent=2)

    logging.info(f"Duplicate metadata saved to: {output_json_path}")
    logging.info(
        f"✅ Deduplication complete. Checked {total_checked} docs. "
        f"Removed {total_duplicates} duplicates."
    )


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
