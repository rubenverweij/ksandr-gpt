from helpers import prepare_text_for_vector_store, looks_like_clean_text
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json


def dump_onbruikbare_data(directory: Path):
    output_json = "/home/ubuntu/onprem_data/voorbeelden_onbruikbare_teksten.json"
    examples = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    for index, file_path in enumerate(directory.rglob("*.txt")):
        if index < 20:
            print(f"Start reading nr: {index} filename {file_path}")
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()
                content_cleaned = prepare_text_for_vector_store(content)
                chunks = splitter.split_text(content_cleaned)
                for idx, chunk in enumerate(chunks):
                    if not looks_like_clean_text(chunk):
                        examples.append(
                            {
                                "Pad": file_path.absolute(),
                                "bestand": file_path.name,
                                "chunk": idx,
                                "inhoud": chunk,
                            }
                        )
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    dir = Path("/home/ubuntu/ksandr_files/")
    dump_onbruikbare_data(dir)
