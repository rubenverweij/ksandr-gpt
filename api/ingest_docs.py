from onprem import LLM
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents to ksandr vector database"
    )
    parser.add_argument("-path", help="path to text files")
    parser.add_argument("-chunk_size", help="ingest chunk size", default=300)
    args = parser.parse_args()
    llm = LLM(n_gpu_layers=-1)
    llm.ingest(args.path, chunk_size=args.chunk_size)
