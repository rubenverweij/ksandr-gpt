from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

cypher_templates = {
    "alle_faalvormen_per_component": {
        "query": """
            MATCH (a:AAD)-[:HEEFT_COMPONENT]->(c:Component)-[:HEEFT_FAALTYPE]->(f:Faaltype)
            WHERE a.aad_id IN $aad_ids
            RETURN 
                c.naam AS component_naam,
                f.Nummer AS nummer_faalvorm,
                f.Naam AS naam_faalvorm,
                f.Beschrijving AS beschrijving,
                coalesce(f.GemiddeldAantalIncidenten, 'Onbekend') AS aantal_incidenten
            ORDER BY CASE coalesce(f.GemiddeldAantalIncidenten, 'Onbekend')
                        WHEN 'Zeer regelmatig (>5)' THEN 5
                        WHEN 'Regelmatig (3-5)' THEN 4
                        WHEN 'Incidenteel (1-2)' THEN 3
                        WHEN 'Onbekend' THEN 2
                        ELSE 1
                     END DESC
        """,
        "parameters": ["aad_ids"],
    },
    "alle_faalvormen_per_component_clean": {
        "query": """
            MATCH (a:AAD)-[:HEEFT_COMPONENT]->(c:Component)-[:HEEFT_FAALTYPE]->(f:Faaltype)
            WHERE a.aad_id IN $aad_ids
            RETURN 
                c.naam AS component_naam,
                f.Nummer AS nummer,
                f.Naam AS naam_faalvorm,
                f.Beschrijving AS beschrijving,
                f.GemiddeldAantalIncidenten AS aantal_incidenten
        """,
        "parameters": ["aad_ids"],
    },
}


def run_cypher(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]


def query_neo4j(prompt: str, chroma_filter):
    """Haal informatie op uit neo4j database."""
    if "faalvorm" in prompt.lower():
        parameters = {}
        for clause in chroma_filter.get("$and", []):
            if "type_id" in clause:
                parameters["aad_ids"] = clause["type_id"].get("$in", [])
        return neo4j_records_to_context(
            run_cypher(
                query=cypher_templates["alle_faalvormen_per_component"]["query"],
                parameters=parameters,
            )
        )
    return None


def neo4j_records_to_context(records):
    """
    Converteert Neo4j-records naar een tekstuele RAG-context.
    Vervangt None door 'Onbekend' en formatteert per faalvorm.
    """
    context_parts = []
    for r in records:
        component = r.get("component_naam", "Onbekend")
        nummer = r.get("nummer_faalvorm", "Onbekend")
        naam = r.get("naam_faalvorm", "Onbekend")
        beschrijving = r.get("beschrijving", "Onbekend")
        incidenten = r.get("aantal_incidenten") or "Onbekend"
        entry = (
            f"Component: {component}\n"
            f"Faalvorm: {naam} ({nummer})\n"
            f"Beschrijving: {beschrijving}\n"
            f"Aantal incidenten: {incidenten}\n"
        )
        context_parts.append(entry.strip())
    return "\n\n".join(context_parts)
