import argparse
import shutil
from onprem import LLM
import os
from typing import Dict, Union


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents to ksandr vector database"
    )

    # NOTE that this is a volume defined when starting the container
    # most likely: -v /home/ubuntu/onprem_data:/root/onprem_data
    parser.add_argument("-vector_db_path", help="path to vector db", type=str)
    parser.add_argument("-documents_path", help="path to text files", type=str)
    parser.add_argument("-chunk_size", help="ingest chunk size", default=300, type=int)
    parser.add_argument(
        "-chunk_overlap", help="ingest overlap chunk size", default=100, type=int
    )
    args = parser.parse_args()

    STORE_TYPE = "sparse"

    try:
        shutil.rmtree(args.vector_db_path)
        print(f"Directory '{args.vector_db_path}' removed successfully.")
    except OSError as e:
        print(f"Error: {e.strerror} - {e.filename}")

    llm = LLM(
        n_gpu_layers=-1,
        embedding_model_kwargs={"device": "cuda"},
        store_type=STORE_TYPE,
    )
    llm.ingest(
        source_directory=args.documents_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print("Done ingesting documents, start adjusting metadata")

    if STORE_TYPE == "sparse":
        database = llm.load_vectorstore()
        docs = []
        for selected_document in database.get_all_docs():
            metadata = extract_file_data(file_path=selected_document["source"])
            selected_document.update(metadata)
            docs.append(selected_document)
        database.update_documents(doc_dicts=docs)
    else:
        # In case it is dense
        database = llm.load_vectordb()
        ids = database.get()["ids"]
        docs = []
        for index, doc_id in enumerate(ids):
            selected_document = database.get_by_ids([doc_id])[0]
            metadata = extract_file_data(file_path=selected_document.metadata["source"])
            selected_document.metadata.update(metadata)
            docs.append(selected_document)
            if index % 1000 == 0:
                print(
                    f"{index}/{len(ids)} updating doc {doc_id} with metadata: {metadata}"
                )
        database.update_documents(ids=ids, documents=docs)
