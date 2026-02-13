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

from ksandr.vectorstore.helpers import (
    prepare_text_for_vector_store,
    looks_like_clean_text,
)
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ksandr.vectorstore.config import (
    RAW_DATA_SOURCES,
    INVALID_DATA_FILE_LOCATION,
    running_inside_docker,
)
import json
import argparse
import logging


def dump_unused_data(directory: Path, output_json: str):
    """
    Recursively scans all .txt files within the specified directory, processes their content into cleaned text chunks,
    and collects examples of "onbruikbare" (unusable) chunks—i.e., chunks not passing the `looks_like_clean_text` check.
    Each unusable chunk is recorded along with its file path, filename, chunk index, and raw content.

    Args:
        directory (Path): Root directory to recursively scan for .txt files.

    Output:
        Writes a JSON file to the path specified by config.INVALID_DATA_LOCATION containing all collected examples.

    Purpose:
        Facilitates data quality inspection and helps iteratively improve preprocessing/ingestion heuristics
        used in the Ksandr vectorstore pipeline.
    """
    examples = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    logging.info(f"Scanning directory for .txt files: {directory}")
    file_count = 0
    for index, file_path in enumerate(directory.rglob("*.txt")):
        if index < 200:
            logging.info(f"Reading file {index + 1}: {file_path}")
            try:
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
                file_count += 1
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
    logging.info(f"Scanned {file_count} files. Found {len(examples)} unusable chunks.")

    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        logging.info(f"Wrote unusable data examples to {output_json}")
    except Exception as e:
        logging.error(f"Failed to write output JSON file: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(description="Dump onbruikbare data uit directory.")
    parser.add_argument(
        "--env",
        choices=["production", "staging"],
        default="production",
        help="Environment: 'production' or 'staging' (default: production)",
    )
    args = parser.parse_args()
    dir = Path(RAW_DATA_SOURCES.get(args.env).get(running_inside_docker()))
    output_json_file = Path(
        INVALID_DATA_FILE_LOCATION.get(args.env).get(running_inside_docker())
    )
    if not dir.exists():
        logging.error(f"Directory does not exist: {dir}")
    else:
        logging.info(f"Using data source directory: {dir}")
        dump_unused_data(directory=dir, output_json=output_json_file)
