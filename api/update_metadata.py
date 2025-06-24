#!/usr/bin/env python3
import os
from typing import Dict, Union
from onprem import LLM


def extract_file_data(file_path: str) -> Dict[str, Union[int, str]]:
    """
    Extract 'aad', 'permission' (cat-1 or cat-2), and 'filename' (without extension) from a path.

    Args:
        file_path (str): The path to the file.

    Returns:
        Dict[str, Union[int, str]]: A dictionary with:
            - 'aad': int - The Aad ID.
            - 'permission': str - Either 'cat-1', 'cat-2', or 'cat-3' if invalid.
            - 'filename': str - The filename without extension.
    """
    parts = file_path.strip("/").split("/")
    result = {}

    if "aads" in parts:
        aads_index = parts.index("aads")
        result["aad"] = int(parts[aads_index + 1])  # e.g., 10547
        permission = parts[aads_index + 2]  # e.g., "cat-2"

        if permission not in ["cat-1", "cat-2"]:
            permission = "cat-3"

        result["permission"] = permission

    filename_with_ext = os.path.basename(file_path)
    result["filename"] = os.path.splitext(filename_with_ext)[0]

    return result


def update_metadata() -> None:
    """
    Updates the metadata for documents in the database by extracting
    aad, permission, and filename information from their source path.
    """
    llm = LLM(n_gpu_layers=-1)
    database = llm.load_vectordb()
    ids = database.get()["ids"]

    for id in ids:
        existing_doc = database.get_by_ids([id])[0]
        metadata = extract_file_data(file_path=existing_doc.metadata["source"])
        print(f"Updating doc {id} with metadata:", metadata)

        existing_doc.metadata.update(metadata)
        database.update_document(document_id=id, document=existing_doc)


if __name__ == "__main__":
    update_metadata()
