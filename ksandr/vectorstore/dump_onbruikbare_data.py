"""
This script scans all .txt files in the specified directory, segments their content into chunks,
and collects examples of "onbruikbare" (unusable) text chunks—those not considered useful for
vector database ingestion based on heuristic filters. These examples are dumped to a JSON file
for inspection, analysis, or further refinement of ingestion criteria.

Main functionalities:
- Recursively loads .txt files from a directory (up to 200 files for sampling).
- Chunkifies each file's content using RecursiveCharacterTextSplitter.
- Applies `looks_like_clean_text` to identify unusable (onbruikbare) chunks.
- Stores examples (with file path, filename, chunk index, and chunk content) in a JSON output.

Intended usage:
    python dump_onbruikbare_data.py

This helps data engineers spot, inspect, and tune preprocessing for irrelevant or malformed text
before inclusion in the Ksandr vector retrieval pipeline.
"""

from helpers import prepare_text_for_vector_store, looks_like_clean_text
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import LOCAL_DIR_GPU_SERVER, INVALID_DATA_LOCATION
import json
import argparse


def dump_unused_data(directory: Path):
    """
    Scans all .txt files in a given directory, segments their cleaned content into text chunks,
    and collects examples of "onbruikbare" (unusable) text—those chunks that do not meet
    the criteria of `looks_like_clean_text`. The examples are stored as a JSON list with metadata
    (path, filename, chunk index, content) at the output location.

    Args:
        directory (Path): The root directory containing .txt files to scan (recursively).

    Output:
        Dumps a JSON file (location from config.INVALD_DATA_LOCATION) with selected examples.

    Intended for data quality analysis and refining preprocessing rules in the Ksandr ingestion pipeline.
    """
    output_json = f"{LOCAL_DIR_GPU_SERVER}/{INVALID_DATA_LOCATION}"
    examples = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    for index, file_path in enumerate(directory.rglob("*.txt")):
        if index < 200:
            print(f"Start reading nr: {index} filename {file_path}")
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()
                content_cleaned = prepare_text_for_vector_store(content)
                chunks = splitter.split_text(content_cleaned)
                for idx, chunk in enumerate(chunks):
                    if not looks_like_clean_text(chunk):
                        examples.append(
                            {
                                "Pad": file_path.as_posix(),
                                "bestand": file_path.name,
                                "chunk": idx,
                                "inhoud": chunk,
                            }
                        )
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump onbruikbare data uit directory.")
    parser.add_argument(
        "--env",
        choices=["production", "staging"],
        default="production",
        help="Environment: 'production' or 'staging' (default: production)",
    )
    args = parser.parse_args()

    if args.env == "production":
        base_dir = f"{LOCAL_DIR_GPU_SERVER}/ksandr_files_production/"
    elif args.env == "staging":
        base_dir = f"{LOCAL_DIR_GPU_SERVER}/ksandr_files/"
    else:
        raise ValueError("Onbekende omgeving opgegeven.")

    dir = Path(base_dir)
    dump_unused_data(dir)
