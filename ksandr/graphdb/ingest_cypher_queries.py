"""
This module defines and provides common Cypher queries for ingesting and retrieving
dossier and component data from a Neo4j graph database.

- Contains parameterized/query templates (BASE_DOSSIER_QUERY, BASE_DOSSIER_QUERY_BASIC).
- Supplies a list of predefined Cypher queries (`predefined_queries`) with sample
  user questions and tag metadata, for use in automated graph question answering and
  NL-to-Cypher translation systems.

Typical usage:
    - Import `predefined_queries` for query selection or LLM prompt templates.
    - Use as a reference or base for populating and querying graph DBs in the Ksandr platform.

Intended for configuration/code sharing in ingestion pipelines and retrieval systems.
"""

import argparse
from ksandr.embeddings.embeddings import get_embedding_function
from ksandr.graphdb.config import CHROMA_DB_PATH, running_inside_docker
from typing import List, Dict
from pathlib import Path
import json
import shutil
from langchain_core.documents import Document
from langchain_chroma import Chroma
import logging

BASE_DOSSIER_QUERY_BASIC = """
WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs, $permissions AS permissions
UNWIND keys(permissions) AS category
UNWIND permissions[category] AS allowed_dossier_id
MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
"""

BASE_DOSSIER_QUERY = """
WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs, $permissions AS permissions
UNWIND keys(permissions) AS category
UNWIND permissions[category] AS allowed_dossier_id
MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
"""

predefined_queries = [
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (d)-[:HAS_PERMISSION]->(:permission {category: category})
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
        "tags": "publicatiedatum;gepubliceerd",
        "tags_list": [
            ["publicatiedatum", "aad"],
            ["is", "gepubliceerd"],
            ["publicatiedatum", "dossier"],
        ],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (d)-[:HAS_PERMISSION]->(:permission {category: category})
        RETURN DISTINCT 
            d.aad_id as aad_dossier_id,
            c.component_id AS component_naam,
            c.algemene_productbeschrijving AS productbeschrijving
        """,
        "example_questions": ["Geef productbeschrijving van component A"],
        "tags": "productbeschrijving",
        "tags_list": [
            ["productbeschrijving"],
            ["omschrijving", "component"],
            ["omschrijving", "installatie"],
        ],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (d)-[:HAS_PERMISSION]->(:permission {category: category})
        RETURN DISTINCT 
            d.aad_id as aad_dossier_id,
            c.component_id AS component_naam,
            c.primaire_functie_in_het_net AS primaire_functie_in_netwerk
        """,
        "example_questions": ["Beschrijf de primaire functie van component A"],
        "tags": "primaire functie",
        "tags_list": [["primaire", "functie"]],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (d)-[:HAS_PERMISSION]->(:permission {category: category})
        RETURN DISTINCT 
            d.aad_id as aad_dossier_id,
            c.component_id AS component_naam,
            c.secundaire_functie_in_het_net AS secundaire_functie_in_netwerk
        """,
        "example_questions": ["Beschrijf de secundaire functie van component A"],
        "tags": "secundaire functie",
        "tags_list": [["secundaire", "functie"]],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (d)-[:HAS_PERMISSION]->(:permission {category: category})
        RETURN DISTINCT 
            d.aad_id as aad_dossier_id,
            c.component_id AS component_naam,
            c.materiaal_omschrijving AS materiaal_omschrijving
        """,
        "example_questions": ["Geef de materiaal omschrijving van de X"],
        "tags": "materiaal omschrijving",
        "tags_list": [["materiaal", "omschrijving"], ["omschrijving", "materiaal"]],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (d)-[:HAS_PERMISSION]->(:permission {category: category})
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
        "tags_list": [["aad", "gewijzigd"], ["laatste", "update", "aad"]],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (nb:netbeheerder)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t))
        MATCH (nb)-[:heeft_populatie]->(p:populatie)
        MATCH (d)-[:heeft_populatie]->(p)
        MATCH (p)-[:HAS_PERMISSION]->(:permission {category: category})
        WITH 
            nb.naam AS naam_netbeheerder,
            c.component_id AS naam_component,
            SUM(p.populatie) AS totale_populatie_component
        RETURN
            naam_netbeheerder,
            naam_component,
            totale_populatie_component
        ORDER BY totale_populatie_component DESC;
        """,
        "example_questions": [
            "Van welke component heeft netbeheerder het meest?",
            "Welke asset heeft netbeheerder het meeste?",
            "Geef een opsomming van de populatiegegevens van Stedin",
            "Geef de populatie componenten van Stedin",
            "Hoeveel installaties heeft netbeheerder per component?",
            "Hoeveel installaties heeft Stedin",
            "Hoeveel heeft Liander",
            "Geef het aantal componenten per netbeheerder",
        ],
        "tags": "populatiegegevens;meeste;populatie;aantal velden;hoeveel;het aantal",
        "tags_list": [
            ["populatiegegevens"],
            ["asset", "populatie"],
            ["populatie", "assets"],
            ["asset", "het", "meeste"],
            ["welke", "het", "meeste"],
            ["component", "het", "meeste"],
            ["hoeveel", "hebben"],
            ["hoeveel", "heeft"],
            ["geef", "aantallen"],
            ["aantal", "installaties"],
            ["hoeveel", "van"],
        ],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (nb:netbeheerder)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t))
        MATCH (nb)-[:heeft_populatie]->(p:populatie)
        MATCH (d)-[:heeft_populatie]->(p)
        MATCH (p)-[:HAS_PERMISSION]->(:permission {category: category})
        WITH 
            c.component_id AS naam_component,
            SUM(p.populatie) AS totale_populatie_component
        RETURN
            naam_component,
            totale_populatie_component
        ORDER BY totale_populatie_component DESC;
        """,
        "example_questions": [
            "Hoeveel installaties van x hebben de netbeheerders in totaal",
            "Hoeveel componenten hebben de netbeheerders samen",
        ],
        "tags": "populatiegegevens;meeste;populatie;aantal velden;hoeveel;het aantal",
        "tags_list": [
            ["geef", "de", "som"],
            ["hoeveel", "hebben", "samen"],
            ["hoeveel", "totaal"],
            ["hebben", "in", "totaal"],
            ["geef", "het", "totaal"],
        ],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (nb:netbeheerder)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t))
        MATCH (nb)-[:heeft_populatie]->(p:populatie)
        MATCH (d)-[:heeft_populatie]->(p)
        MATCH (p)-[:HAS_PERMISSION]->(:permission {category: category})
        WITH 
            c.component_id AS naam_component,
            SUM(p.aantal_velden) AS totaal_aantal_velden
        RETURN
            naam_component,
            totaal_aantal_velden
        ORDER BY totaal_aantal_velden DESC;
        """,
        "example_questions": [
            "Hoeveel velden van x hebben de netbeheerders in totaal",
            "Hoeveel velden hebben de netbeheerders samen",
        ],
        "tags": "populatiegegevens;meeste;populatie;aantal velden;hoeveel;het aantal",
        "tags_list": [
            ["geef", "de", "som", "velden"],
            ["hoeveel", "velden", "hebben", "samen"],
            ["hoeveel", "velden", "totaal"],
            ["velden", "hebben", "in", "totaal"],
            ["geef", "het", "totaal", "velden"],
        ],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (nb:netbeheerder)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t))
        MATCH (nb)-[:heeft_populatie]->(p:populatie)
        MATCH (d)-[:heeft_populatie]->(p)
        MATCH (p)-[:HAS_PERMISSION]->(:permission {category: category})
        WITH 
            nb.naam AS naam_netbeheerder,
            c.component_id AS naam_component,
            SUM(p.aantal_velden) AS totaal_aantal_velden
        RETURN
            naam_netbeheerder,
            naam_component,
            totaal_aantal_velden
        ORDER BY totaal_aantal_velden DESC;
        """,
        "example_questions": [
            "Van welke component heeft netbeheerder het meest velden?",
            "Welke asset heeft netbeheerder het meeste velden?",
            "Geef een opsomming van de velden van Stedin",
            "Geef de populatie velden van Stedin",
            "Hoeveel velden heeft netbeheerder per component?",
            "Geef het aantal velden per netbeheerder",
        ],
        "tags": "populatiegegevens;meeste;populatie;aantal velden;hoeveel;het aantal",
        "tags_list": [
            ["aantal", "velden"],
            ["hoeveel", "velden"],
        ],
    },
    {
        "cypher": """
        MATCH (d:dossier)-[:heeft_component]->(c:component)
        RETURN 
            d.aad_id AS aad_dossier_id,
            c.component_id AS component_naam
        ORDER BY component_naam;
        """,
        "example_questions": [
            "Geef een opsomming van alle componenten met een AAD",
            "Geef een lijst van alle componenten met een AAD dossier",
            "Geef een lijst van alle AAD dossiers",
            "Welke AAD dossiers zijn er allemaal?",
            "Geef alle AAD dossiers.",
            "Geef een opsomming van alle AAD's",
        ],
        "tags": "lijst;aad;aad's",
        "threshold": 0.85,
        "tags_list": [
            ["lijst", "aad"],
            ["lijst", "dossier"],
            ["geef", "dossier"],
            ["geef", "aad"],
        ],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        MATCH (d:dossier)-[:heeft_beheerteam_lid]->(p:persoon)
        MATCH (p)-[:HAS_PERMISSION]->(:permission {category: category})
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
        "tags_list": [["beheerteam"]],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (b)-[:HAS_PERMISSION]->(:permission {category: category})
        WHERE toLower(b.soort) CONTAINS "vervangingsbeleid"
        MATCH (nb:netbeheerder)-[:heeft_beleid]->(b:beleid)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t)) 
        RETURN DISTINCT
            nb.naam AS netbeheerder,
            c.component_id AS component_naam,  
            b AS beleid
        """,
        "example_questions": [
            "Welk vervangingsbeleid hanteert X voor de installatie?",
            "Wat is het vervangingsbeleid van X voor de magnefix?",
        ],
        "tags": "vervangingsbeleid",
        "tags_list": [["vervangingsbeleid"]],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (b)-[:HAS_PERMISSION]->(:permission {category: category})
        WHERE toLower(b.soort) CONTAINS "onderhoudsstrategie"
        MATCH (nb:netbeheerder)-[:heeft_beleid]->(b:beleid)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t)) 
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
        "tags_list": [["onderhoudsbeleid"], ["onderhoudsstrategie"]],
    },
    {
        "cypher": """
                WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (b)-[:HAS_PERMISSION]->(:permission {category: category})
        WHERE toLower(b.soort) = "onderhoud_en_inspectie"
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
        "tags_list": [
            ["inspectie"],
            ["inspecteren"],
            ["goedkeuringseisen"],
            ["inspectieronde"],
        ],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (b)-[:HAS_PERMISSION]->(:permission {category: category})
        WHERE toLower(b.soort) CONTAINS "vervangingscriteria"
        MATCH (nb:netbeheerder)-[:heeft_beleid]->(b:beleid)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t)) 
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
        "tags_list": [["vervangingscriteria"]],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (b)-[:HAS_PERMISSION]->(:permission {category: category})
        WHERE toLower(b.soort) CONTAINS "fabrikant"
        RETURN DISTINCT
            c.component_id AS component_naam,  
            b AS beleid
        """,
        "example_questions": [
            "Welk beleid adviseert de fabrikant voor de installatie?",
        ],
        "tags": "fabrikant",
        "tags_list": [
            ["beleid", "fabrikant"],
            ["fabrikant", "beleid"],
            ["onderhoudsbeleid", "fabrikant"],
        ],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (b)-[:HAS_PERMISSION]->(:permission {category: category})
        WHERE toLower(b.soort) CONTAINS "periodiek"
        MATCH (nb:netbeheerder)-[:heeft_beleid]->(b:beleid)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t)) 
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
        "tags_list": [["periodiek", "onderhoud"], ["onderhoud", "periodiek"]],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:HEEFT_COMPONENT]->(c:component)
        MATCH (d:dossier)-[:heeft_beleid]->(b:beleid)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
        MATCH (b)-[:HAS_PERMISSION]->(:permission {category: category})
        WHERE toLower(b.soort) CONTAINS "onderhoud"
        MATCH (nb:netbeheerder)-[:heeft_beleid]->(b:beleid)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t)) 
        RETURN DISTINCT
            nb.naam AS netbeheerder,
            c.component_id AS component_naam,  
            b AS beleid
        """,
        "example_questions": [
            "Welk onderhoud wordt gedaan voor de FMX?",
            "Welk onderhoud wordt gedaan?",
        ],
        "tags": "onderhoud;",
        "tags_list": [["welk", "onderhoud"], ["wat", "voor", "onderhoud"]],
    },
    {
        "cypher": """
        WITH $aad_ids AS dossier_ids, $permissions AS permissions
        UNWIND keys(permissions) AS category
        UNWIND permissions[category] AS allowed_dossier_id
        MATCH (d:dossier {aad_id: allowed_dossier_id})-[:heeft_document]->(doc:document)
        RETURN DISTINCT 
            d.aad_id as aad_dossier_id,
            doc as document
        """,
        "example_questions": [
            "Waar kan ik de notulen van het beheerteam overleg 2023 voor het AAD 10kv COQ vinden?"
        ],
        "tags": "beheerteam overleg;bestanden;brondocumenten aad",
        "tags_list": [["brondocumenten"]],
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
        if "{base_query_simple}" in query["cypher"]:
            query["cypher"] = query["cypher"].format(
                base_query_simple=BASE_DOSSIER_QUERY_BASIC
            )
        for question in query["example_questions"]:
            documenten.append(
                Document(
                    page_content=question,
                    metadata={
                        "cypher": query["cypher"],
                        "tags": query["tags"],
                        "tags_list": json.dumps(query.get("tags_list", [])),
                        "threshold": query.get("threshold", 0),
                    },
                )
            )
    db.add_documents(documenten)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-queries_file",
        type=str,
        required=False,
        default=None,
        help="Path to the JSON file containing generated queries",
    )
    parser.add_argument(
        "-env",
        type=str,
        required=False,
        default="production",
        choices=["production", "staging"],
        help="Select the environment to run against (production or staging).",
    )
    args = parser.parse_args()
    chroma_path = Path(CHROMA_DB_PATH.get(args.env).get(running_inside_docker()))

    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        logging.info(f"Database {chroma_path.as_posix()} verwijderd.")

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

    logging.info(f"Loaded {len(queries_data)} queries from JSON.")
    ingest_cypher_queries(chroma_path=chroma_path, queries=queries_data)
