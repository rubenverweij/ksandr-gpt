import argparse
from embeddings import get_embedding_function
from typing import List, Dict
from pathlib import Path
import json
import shutil
from langchain_core.documents import Document
from langchain_chroma import Chroma

BASE_DOSSIER_QUERY = """
WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs
MATCH (d:dossier)
WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
MATCH (d)-[:heeft_component]->(c:component)
"""

predefined_queries = [
    {
        "cypher": """
        {base_query}
        RETURN DISTINCT 
            d.aad_id as aad_dossier_id,
            c.component_id AS component_naam,
            d.publicatiedatum AS publicatiedatum
        ORDER BY publicatiedatum DESC
        """,
        "example_questions": [
            "Wat is de publicatiedatum van AAD",
            "Geef de publicatiedatum van dossier",
        ],
        "tags": "publicatiedatum",
    },
    {
        "cypher": """
        {base_query}
        RETURN DISTINCT 
            d.aad_id as aad_dossier_id,
            c.component_id AS component_naam,
            c.algemene_productbeschrijving AS productbeschrijving
        """,
        "example_questions": ["Geef productbeschrijving van component A"],
        "tags": "productbeschrijving",
    },
    {
        "cypher": """
        {base_query}
        RETURN DISTINCT 
            d.aad_id as aad_dossier_id,
            c.component_id AS component_naam,
            c.primaire_functie_in_het_net AS primaire_functie_in_netwerk
        """,
        "example_questions": ["Beschrijf de primaire functie van component A"],
        "tags": "primaire functie",
    },
    {
        "cypher": """
        {base_query}
        RETURN DISTINCT 
            d.aad_id as aad_dossier_id,
            c.component_id AS component_naam,
            c.secundaire_functie_in_het_net AS secundaire_functie_in_netwerk
        """,
        "example_questions": ["Beschrijf de secundaire functie van component A"],
        "tags": "secundaire functie",
    },
    {
        "cypher": """
        {base_query}
        RETURN DISTINCT 
            d.aad_id as aad_dossier_id,
            c.component_id AS component_naam,
            c.materiaal_omschrijving AS materiaal_omschrijving
        """,
        "example_questions": ["Geef de materiaal omschrijving van de X"],
        "tags": "materiaal omschrijving",
    },
    {
        "cypher": """
        {base_query}
        RETURN DISTINCT 
            d.aad_id as aad_dossier_id,
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
        WITH 
            nb.naam AS naam_netbeheerder,
            c.component_id AS naam_component,
            SUM(p.populatie) AS totale_populatie_component,
            SUM(p.aantal_velden) AS totaal_aantal_velden
        RETURN
            naam_netbeheerder,
            naam_component,
            totale_populatie_component,
            totaal_aantal_velden
        ORDER BY totale_populatie_component;
        """,
        "example_questions": [
            "Van welke component heeft netbeheerder het meest?",
            "Welke asset heeft netbeheerder het meeste?",
            "Geef een opsomming van de populatiegegevens van Stedin",
            "Geef de populatie componenten van Stedin",
            "Hoeveel installaties heeft netbeheerder per component?",
            "Geef het aantal componenten per netbeheerder",
        ],
        "tags": "populatiegegevens;meeste;populatie;aantal velden;hoeveel;het aantal",
    },
    {
        "cypher": """
        MATCH (d:dossier)-[:heeft_component]->(c:component)
        RETURN 
            d.aad_id AS aad_dossier_id,
            c.component_id AS component_naam
        ORDER BY aad_dossier_id, component_naam;
        """,
        "example_questions": [
            "Geef een opsomming van alle componenten met een AAD",
            "Geef een lijst van alle componenten met een AAD dossier",
            "Geef een lijst van alle AAD dossiers",
            "Welke AAD dossiers zijn er allemaal?",
            "Geef alle AAD dossiers.",
            "Geef een opsomming van alle AAD's",
        ],
        "tags": "lijst;aad",
        "threshold": 0.85,
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids
        MATCH (d:dossier)-[:heeft_beheerteam_lid]->(p:persoon)
        MATCH (d:dossier)-[:heeft_component]->(c:component)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids   
        RETURN 
            d.aad_id AS aad_dossier_id,
            c.component_id AS component_naam,
            p.naam AS naam,
            p.link AS profiel_link
        ORDER BY component_naam, naam;
        """,
        "example_questions": [
            "Wie zitten in het beheerteam van aad?",
            "Welke personen zitten in het beheerteam van de fmx",
        ],
        "tags": "beheerteam",
        "threshold": 0.3,
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
            b AS beleid
        """,
        "example_questions": [
            "Welk vervangingsbeleid adviseert de fabrikant voor de installatie?",
            "Wat is het vervangingsbeleid van X voor de magnefix?",
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
            b AS beleid
        """,
        "example_questions": [
            "Welk onderhoudsstrategie hanteert X voor de installatie?",
            "Wat is het onderhoudsstrategie van X voor de magnefix?",
            "Hoe ziet het onderhoudsstrategie van X voor de Siemens 8DJH eruit?",
        ],
        "tags": "onderhoudsstrategie;onderhoudsbeleid",
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids AND
        toLower(b.soort) = toLower("onderhoud_en_inspectie")
        MATCH (d)-[:heeft_component]->(c:component)
        MATCH (nb:netbeheerder)-[:heeft_beleid]->(b:beleid)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t))
        RETURN DISTINCT
            b.onderwerp as soort_inspectie,
            b.goedkeuringseisen_toelichting as goedkeuringseis_inspectie
        """,
        "example_questions": [
            "Geef een lijst van 5 maatregelen die onderdeel uitmaken van een inspectieronde voor de FMX?",
            "Wordt de werking van de veerspanmotor gecontroleerd bij het inspecteren of onderhouden van de FMX?",
            "Wat zijn de goedkeuringseisen bij het inspecteren van olielekkage bij een COQ installatie?",
        ],
        "tags": "inspectie;inspectieronde;inspecteren;goedkeuringseisen",
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
            b AS beleid
        """,
        "example_questions": [
            "Welk vervangingscriteria gebruikt X voor de SVS?",
            "Wat zijn de vervangingscriteria van X voor de magnefix?",
        ],
        "tags": "vervangingscriteria",
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids 
        MATCH (d)-[:heeft_component]->(c:component)
        WHERE toLower(b.soort) CONTAINS "fabrikant"
        RETURN DISTINCT
            c.component_id AS component_naam,  
            b AS beleid
        """,
        "example_questions": [
            "Welk beleid adviseert de fabrikant voor de installatie?",
        ],
        "tags": "fabrikant",
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (nb:netbeheerder)-[:heeft_beleid]->(b:beleid)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t))
        MATCH (d)-[:heeft_component]->(c:component)
        WHERE toLower(b.soort) CONTAINS "periodiek"
        RETURN DISTINCT
            nb.naam AS netbeheerder,
            c.component_id AS component_naam,  
            b AS beleid
        """,
        "example_questions": [
            "Welk periodiek onderhoud wordt gedaan voor de FMX?",
            "Welk onderhoud wordt periodiek gedaan?",
        ],
        "tags": "po/pi;periodiek",
    },
    {
        "cypher": """
        {base_query}
        MATCH (d:dossier)-[:heeft_document]->(doc:document)
        RETURN DISTINCT 
            d.aad_id as aad_dossier_id,
            doc as document
        """,
        "example_questions": [
            "Waar kan ik de notulen van het beheerteam overleg 2023 voor het AAD 10kv COQ vinden?"
        ],
        "tags": "beheerteam overleg;bestanden;brondocumenten aad",
    },
]


def ingest_cypher_queries(chroma_path, queries: List[Dict]):
    documenten = []
    db = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=get_embedding_function(),
    )
    for query in queries:
        if "{base_query}" in query["cypher"]:
            query["cypher"] = query["cypher"].format(base_query=BASE_DOSSIER_QUERY)
        for question in query["example_questions"]:
            documenten.append(
                Document(
                    page_content=question,
                    metadata={
                        "cypher": query["cypher"],
                        "tags": query["tags"],
                        "threshold": query.get("threshold", 0),
                    },
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
