from onprem import LLM
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents to ksandr vector database"
    )
    parser.add_argument("-path", help="path to text files")
    args = parser.parse_args()
    llm = LLM(n_gpu_layers=-1)
    llm.ingest(args.path, chunk_size=1500)
