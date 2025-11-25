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
        MATCH (d)-[:heeft_component]->(c:component)
        RETURN DISTINCT 
            c.component_id AS component_naam,
            d.laatste_update AS laatste_update
        ORDER BY laatste_update DESC
        """,
        "example_questions": [
            "Wat is de laatste update datum van AAD?",
            "Wanneer is het AAD dossier voor het laatst gewijzigd?",
        ],
        "tags": "laatste update;gewijzigd",
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
            c.component_id AS component_naam,
            p.populatie AS populatie,
            p.aantal_velden AS aantal_velden
        ORDER BY dossier_id, component_naam
        """,
        "example_questions": [
            "Van welke component heeft netbeheerder het meest?",
            "Welke asset heeft netbeheerder het meeste?",
            "Hoeveel installaties heeft netbeheerder per component?",
        ],
        "tags": "populatiegegevens;meeste;populatie;aantal velden;hoeveel",
    },
    {
        "cypher": """
        MATCH (d:dossier)-[:heeft_component]->(c:component)
        RETURN 
            d.aad_id AS aad_id,
            c.component_id AS component_id
        ORDER BY aad_id, component_id;
        """,
        "example_questions": ["Geef een overzicht van alle componenten met een aads"],
        "tags": "overzicht;aad",
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids
        MATCH (d:dossier)-[:heeft_beheerteam_lid]->(p:persoon)
        MATCH (d:dossier)-[:heeft_component]->(c:component)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids   
        RETURN 
            c.component_id AS component_id,
            p.naam AS naam,
            p.link AS profiel_link
        ORDER BY component_id, naam;
        """,
        "example_questions": ["Wie zitten in het beheerteam van aad?"],
        "tags": "beheerteam",
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (nb:netbeheerder)-[:heeft_beleid]->(b:beleid)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t))
        MATCH (d)-[:heeft_component]->(c:component)
        WHERE toLower(b.soort) CONTAINS "onderhoudsbeleid"
        RETURN DISTINCT
            nb.naam AS netbeheerder,
            c.component_id AS component_naam,  
            b.soort as soort_beleid,
            b AS beleid
        """,
        "example_questions": [
            "Welk onderhoudsbeleid adviseert de fabrikant voor de installatie?",
        ],
        "tags": "onderhoudsbeleid",
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (nb:netbeheerder)-[:heeft_beleid]->(b:beleid)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t))
        MATCH (d)-[:heeft_component]->(c:component)
        WHERE toLower(b.soort) CONTAINS "vervangingsbeleid"
        RETURN DISTINCT
            nb.naam AS netbeheerder,
            c.component_id AS component_naam,  
            b.soort as soort_beleid,
            b AS beleid
        """,
        "example_questions": [
            "Welk vervangingsbeleid adviseert de fabrikant voor de installatie?",
        ],
        "tags": "vervangingsbeleid",
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (nb:netbeheerder)-[:heeft_beleid]->(b:beleid)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t))
        MATCH (d)-[:heeft_component]->(c:component)
        WHERE toLower(b.soort) CONTAINS "onderhoudsstrategie"
        RETURN DISTINCT
            nb.naam AS netbeheerder,
            c.component_id AS component_naam,  
            b.soort as soort_beleid,
            b AS beleid
        """,
        "example_questions": [
            "Welk onderhoudsstrategie adviseert de fabrikant voor de installatie?",
        ],
        "tags": "onderhoudsstrategie",
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (nb:netbeheerder)-[:heeft_beleid]->(b:beleid)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t))
        MATCH (d)-[:heeft_component]->(c:component)
        WHERE toLower(b.soort) CONTAINS "vervangingscriteria"
        RETURN DISTINCT
            nb.naam AS netbeheerder,
            c.component_id AS component_naam,  
            b.soort as soort_beleid,
            b AS beleid
        """,
        "example_questions": [
            "Welk vervangingscriteria adviseert de fabrikant voor de installatie?",
        ],
        "tags": "vervangingscriteria",
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (nb:netbeheerder)-[:heeft_beleid]->(b:beleid)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t))
        MATCH (d)-[:heeft_component]->(c:component)
        WHERE toLower(b.soort) CONTAINS "onderhoud"
        RETURN DISTINCT
            nb.naam AS netbeheerder,
            c.component_id AS component_naam,  
            b.soort as soort_beleid,
            b AS beleid
        """,
        "example_questions": [
            "Wat voor soort onderhoud gebruikt Rendo voor de SVS?",
        ],
        "tags": "onderhoud",
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
