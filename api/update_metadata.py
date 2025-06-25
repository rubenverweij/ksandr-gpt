#!/usr/bin/env python3
import os
from typing import Dict, Union
from onprem import LLM

VALID_PERMISSIONS = {"cat-1", "cat-2"}


def extract_file_data(file_path: str) -> Dict[str, Union[int, str]]:
    """Extract 'aad', 'permission', and 'filename' from a path."""
    parts = file_path.strip("/").split("/")
    result = {
        "aad": -1,
        "permission": "cat-3",
        "filename": "",
    }
    if "aads" in parts:
        try:
            aads_index = parts.index("aads")
            result["aad"] = int(parts[aads_index + 1])
            permission = parts[aads_index + 2]
            result["permission"] = (
                permission if permission in VALID_PERMISSIONS else "cat-3"
            )
        except (IndexError, ValueError):
            pass
    result["permission_and_type"] = f"{result['permission']}_{result['aad']}"
    result["filename"] = os.path.splitext(os.path.basename(file_path))[0]
    return result


def update_metadata() -> None:
    """Update metadata for documents by extracting aad, permission, and filename."""
    llm = LLM(n_gpu_layers=-1)
    database = llm.load_vectordb()
    ids = database.get().get("ids", [])
    for index, doc_id in enumerate(ids):
        existing_doc = database.get_by_ids([doc_id])[0]
        metadata = extract_file_data(existing_doc.metadata.get("source", ""))
        print(f"{index}/{len(ids)} updating doc {doc_id} with metadata: {metadata}")
        existing_doc.metadata.update(metadata)
        database.update_document(document_id=doc_id, document=existing_doc)


if __name__ == "__main__":
    update_metadata()
