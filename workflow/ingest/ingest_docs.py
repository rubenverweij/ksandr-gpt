import argparse
import shutil
from onprem import LLM
import os
from typing import Dict, Union

VALID_PERMISSIONS = {"cat-1", "cat-2"}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents to ksandr vector database"
    )

    # NOTE that this is a volume defined when starting the container
    # most likely: -v /home/ubuntu/onprem_data:/root/onprem_data
    parser.add_argument("-vector_db_path", help="path to vector db", type=str)
    parser.add_argument("-documents_path", help="path to text files", type=str)
    parser.add_argument("-chunk_size", help="ingest chunk size", default=300, type=int)
    parser.add_argument("-store", help="store type", default="sparse", type=str)
    parser.add_argument(
        "-chunk_overlap", help="ingest overlap chunk size", default=100, type=int
    )
    args = parser.parse_args()

    try:
        shutil.rmtree(args.vector_db_path)
        print(f"Directory '{args.vector_db_path}' removed successfully.")
    except OSError as e:
        print(f"Error: {e.strerror} - {e.filename}")

    llm = LLM(
        n_gpu_layers=10,
        embedding_model_kwargs={"device": "cuda"},
        store_type=args.store,
    )
    llm.ingest(
        source_directory=args.documents_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=4000,
    )
    print("Done ingesting documents, start adjusting metadata")

    if args.store == "sparse":
        database = llm.load_vectorstore()
        docs = []
        for selected_document in database.get_all_docs():
            metadata = extract_file_data(file_path=selected_document["source"])
            selected_document.update(metadata)
            docs.append(selected_document)
        database.update_documents(doc_dicts=docs)
    else:
        # In case it is dense
        database = llm.load_vectorstore()
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
