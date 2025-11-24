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
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs
        MATCH (d:dossier)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (d)-[:heeft_component]->(c:component)
        RETURN DISTINCT 
            d.aad_id AS dossier_id,
            c.component_id AS component_naam,
            d.publicatiedatum AS publicatiedatum
        ORDER BY publicatiedatum DESC
        """,
        "example_questions": [
            "Wat is de publicatiedatum van AAD",
        ],
        "tags": "publicatiedatum",
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs
        MATCH (d:dossier)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids

        MATCH (nb:netbeheerder)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t))

        MATCH (nb)-[:heeft_populatie]->(p:populatie)
        MATCH (d)-[:heeft_populatie]->(p)
        MATCH (d)-[:heeft_component]->(c:component)

        RETURN DISTINCT 
            nb.naam AS netbeheerder,
            d.aad_id AS dossier_id,
            c.component_id AS component_naam,
            p.populatie AS populatie,
            p.aantal_velden AS aantal_velden
        ORDER BY dossier_id, component_naam
        """,
        "example_questions": [
            "Van welke component heeft netbeheerder het meest?",
        ],
        "tags": "populatiegegevens",
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
                Document(
                    page_content=question,
                    metadata={"cypher": query["cypher"], "tags": query["tags"]},
                )
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
