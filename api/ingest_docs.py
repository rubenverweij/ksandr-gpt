import argparse
import shutil
from onprem import LLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents to ksandr vector database"
    )

    # NOTE that this is a volume defined when starting the container
    # most likely: -v /home/ubuntu/onprem_data:/root/onprem_data
    parser.add_argument("-vector_db_path", help="path to vector db", type=str)
    parser.add_argument("-documents_path", help="path to text files", type=str)
    parser.add_argument("-chunk_size", help="ingest chunk size", default=300, type=int)
    args = parser.parse_args()

    try:
        shutil.rmtree(args.vector_db_path)
        print(f"Directory '{args.vector_db_path}' removed successfully.")
    except OSError as e:
        print(f"Error: {e.strerror} - {e.filename}")

    llm = LLM(n_gpu_layers=-1)
    llm.ingest(source_directory=args.documents_path, chunk_size=args.chunk_size)

    print("done ingesting documents")
