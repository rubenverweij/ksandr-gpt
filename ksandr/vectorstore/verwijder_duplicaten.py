"""
This script provides functionality to remove duplicate documents from a Chroma vectorstore
by hashing the document text, identifying duplicates, and compiling their metadata.

It connects to a specified Chroma database, computes MD5 hashes for normalization,
and detects duplicate entries (documents with identical text).
Metadata for duplicates is saved to a JSON file for further review or processing.

Functions:
- hash_doc: Generates an MD5 hash of stripped document text for deduplication.
- dedup_docs_in_chroma: Main routine to process Chroma, find duplicates, and save their metadata.

Example usage:
    python verwijder_duplicaten.py
"""

import hashlib
import json
from langchain_chroma import Chroma
from embeddings import get_embedding_function
from config import CHROMA_PATH, OUTPUT_JSON_PATH


def hash_doc(text: str) -> str:
    """
    Generate an MD5 hash of the provided text after stripping leading and trailing whitespace.

    Args:
        text (str): The text string to be hashed.

    Returns:
        str: The resulting MD5 hash as a hexadecimal string.
    """
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def dedup_docs_in_chroma(
    chroma_path: str = CHROMA_PATH, output_json_path: str = OUTPUT_JSON_PATH
):
    """
    Identify duplicate documents in a Chroma vectorstore and store their metadata in a JSON file.

    Connects to the Chroma database at the given path, loads all documents,
    computes a hash of each document's content, and identifies duplicates
    (documents with the same text hash). For each set of duplicates, a summary
    of relevant metadata and a text snippet is stored. The complete dictionary
    of duplicates' metadata is written to the provided output JSON file.

    Args:
        chroma_path (str): Path to the Chroma database directory.
        output_json_path (str): Path to save the JSON file with duplicate metadata.

    Side effects:
        - Writes a JSON file containing metadata about duplicate documents found in the vectorstore.
        - Optionally (if extended), removes duplicates from the database.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    results = db.get()
    seen_hashes = {}  # Hou bij welke hashes we al hebben gezien
    to_delete = []  # Verzamel ID's van duplicaten om te verwijderen
    hash_to_metadata = {}  # Verzamel metadata van duplicaten

    for i, (doc, doc_id, metadata) in enumerate(
        zip(results["documents"], results["ids"], results["metadatas"])
    ):
        doc_hash = hash_doc(doc)
        keys_to_extract = ["source", "chunk"]
        subset = {k: metadata[k] for k in keys_to_extract if k in metadata}
        subset["text"] = doc[:20] + " ... " + doc[-20:]
        if doc_hash not in seen_hashes:
            seen_hashes[doc_hash] = doc_id  # Bewaar eerste document met deze hash
            hash_to_metadata[doc_hash] = [subset]
        else:
            to_delete.append(doc_id)  # Markeer duplicaat voor verwijdering
            hash_to_metadata[doc_hash].append(subset)
        if i % 1000 == 0:
            print(
                f"➡️  Gecontroleerd: {i} documenten... {len(to_delete)} duplicaten gevonden"
            )
    # Filter alleen duplicaten (meer dan 1 keer voorgekomen)
    hash_to_metadata = {k: v for k, v in hash_to_metadata.items() if len(v) >= 2}

    # Sla duplicaten-metadata op in JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(hash_to_metadata, f, ensure_ascii=False, indent=2)

    print(f"Duplicaten-metadata opgeslagen in: {output_json_path}")

    # Verwijder duplicaten uit Chroma database
    if to_delete:
        print(f"{len(to_delete)} duplicaten gevonden. Verwijderen...")
        db.delete(ids=to_delete)
    else:
        print("✅ Geen duplicaten gevonden.")

    print("✅ Deduplicatie voltooid.")


if __name__ == "__main__":
    dedup_docs_in_chroma()
