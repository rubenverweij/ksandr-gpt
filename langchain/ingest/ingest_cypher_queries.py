import argparse
from embeddings import get_embedding_function
from typing import List, Dict
from pathlib import Path
import json
from langchain_core.documents import Document
from langchain_chroma import Chroma

predefined_queries = [
    {
        "cypher": "MATCH (c:Component {naam: $naam})-[:HEEFT_FAALTYPE]->(f:Faaltype) RETURN COUNT(f) AS aantalFaalvormen",
        "example_questions": [
            "Hoeveel faalvormen zijn er van de LK ELA?",
            "Aantal Faaltype van component BD10?",
        ],
    },
    {
        "cypher": "MATCH (f:Faaltype)-[:HEEFT_FAALTYPE]->(c:Component) RETURN f.OorzaakGeneriek AS oorzaak, COUNT(f) AS aantalFaalvormen ORDER BY aantalFaalvormen DESC",
        "example_questions": [
            "Wat zijn de meest voorkomende oorzaken van faalvormen?",
            "Top oorzaken van Faaltype",
        ],
    },
]


def ingest_cypher_queries(chroma_path, queries: List[Dict]):
    documenten = []
    db = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=get_embedding_function(),
    )
    for query in queries:
        for question in query["example_questions"]:
            documenten.append(
                Document(page_content=question, metadata={"cypher": query["cypher"]})
            )
    db.add_documents(documenten)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-queries_file",
        type=str,
        required=False,
        default=None,
        help="Path to the JSON file containing generated queries",
    )
    parser.add_argument(
        "-chroma", type=str, required=True, help="Path to Chroma vectorstore"
    )
    args = parser.parse_args()
    CHROMA_PATH = Path(args.chroma)
    if args.queries_file:
        SOURCE_FILE = Path(args.queries_file)
        if not SOURCE_FILE.exists():
            raise FileNotFoundError(f"Queries JSON file not found: {SOURCE_FILE}")
    else:
        SOURCE_FILE = None
    queries_data = predefined_queries
    if SOURCE_FILE:
        with open(SOURCE_FILE, "r", encoding="utf-8") as f:
            queries_data = json.load(f)

    print(f"Loaded {len(queries_data)} queries from JSON.")
    ingest_cypher_queries(CHROMA_PATH=CHROMA_PATH, queries=queries_data)
