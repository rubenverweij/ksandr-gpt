import argparse
from embeddings import get_embedding_function
from typing import List, Dict
from pathlib import Path
import json
import shutil
from langchain_core.documents import Document
from langchain_chroma import Chroma

predefined_queries = [
    {
        "cypher": """
MATCH (a:AAD)-[:HEEFT_COMPONENT]->(c:Component)-[:HEEFT_FAALTYPE]->(f:Faaltype)
{where_clause}
RETURN c.naam AS component, COUNT(f) AS aantalFaalvormen
ORDER BY aantalFaalvormen DESC
""",
        "example_questions": [
            "Hoeveel faalvormen zijn er per component?",
            "Aantal faalvormen per component",
            "Hoeveel faalvormen zijn er in totaal",
        ],
    },
    {
        "cypher": """
MATCH (a:AAD)-[:HEEFT_COMPONENT]->(c:Component)-[:HEEFT_FAALTYPE]->(f:Faaltype)
{where_clause}
RETURN f.OorzaakGeneriek AS oorzaak, COUNT(f) AS aantalFaalvormen
ORDER BY aantalFaalvormen DESC
""",
        "example_questions": [
            "Wat zijn de meest voorkomende oorzaken van faalvormen?",
            "Top oorzaken van faalvormen",
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
    chroma_path = Path(args.chroma)

    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        print("🗑️ Database verwijderd.")

    if args.queries_file:
        source_file = Path(args.queries_file)
        if not source_file.exists():
            raise FileNotFoundError(f"Queries JSON file not found: {source_file}")
    else:
        source_file = None
    queries_data = predefined_queries
    if source_file:
        with open(source_file, "r", encoding="utf-8") as f:
            queries_data = json.load(f)

    print(f"Loaded {len(queries_data)} queries from JSON.")
    ingest_cypher_queries(chroma_path=chroma_path, queries=queries_data)
