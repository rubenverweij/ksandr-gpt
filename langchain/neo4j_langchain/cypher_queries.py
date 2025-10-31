from neo4j import GraphDatabase
import textwrap

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

cypher_templates = {
    "alle_faalvormen_per_component": {
        "query": """
          MATCH (a:AAD)-[:HEEFT_COMPONENT]->(c:Component)-[:HEEFT_FAALTYPE]->(f:Faaltype)
        WHERE ($aad_ids IS NULL OR size($aad_ids) = 0 OR a.aad_id IN $aad_ids)
        WITH a, c, f
        ORDER BY toInteger(f.NummerInt) ASC, f.Nummer ASC
        WITH a, c, collect({
            Nummer: f.Nummer,
            Naam: f.Naam,
            GemiddeldAantalIncidenten: coalesce(f.GemiddeldAantalIncidenten, 'Onbekend'),
            NummerInt: f.NummerInt,
            Bestandspad: f.Bestandspad
        }) AS faalvormen
        RETURN 
            a.aad_id AS aad_id,
            c.naam AS component_naam,
            faalvormen
        ORDER BY c.naam
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
    parameters = {"aad_ids": []}  # standaard lege lijst
    if "faalvorm" in prompt.lower():
        for clause in chroma_filter.get("$and", []):
            if "type_id" in clause:
                parameters["aad_ids"] = clause["type_id"].get("$in", [])
        # fallback als filter direct op chroma_filter staat
        if not parameters["aad_ids"] and "type_id" in chroma_filter:
            parameters["aad_ids"] = chroma_filter["type_id"].get("$in", [])
        if not parameters["aad_ids"]:
            return None, None
        return neo4j_records_to_context(
            run_cypher(
                query=cypher_templates["alle_faalvormen_per_component"]["query"],
                parameters=parameters,
            )
        )
    return None, None


def neo4j_records_to_context(records):
    """Haal alle faalvormen per component op en vorm ze om tot tekstcontext voor een prompt."""
    metadata = []
    context_blocks = []
    for record in records:
        component = record["component_naam"]
        aad_id = record["aad_id"]
        faalvormen = record["faalvormen"]
        for faalvorm in faalvormen:
            faalvorm_tekst = "\n".join(
                [
                    f"- {faalvorm['Nummer']}: {faalvorm['Naam']} "
                    f"(Frequentie: {faalvorm['GemiddeldAantalIncidenten']})"
                ]
            )
            metadata.append(
                {
                    "id": faalvorm["Nummer"],
                    "metadata": {
                        "source": faalvorm["Bestandspad"],
                        "source_search": faalvorm["Bestandspad"],
                        "file_path": faalvorm["Bestandspad"],
                        "score": 0.50,
                    },
                    "type": "Document",
                }
            )
        context = textwrap.dedent(
            f"""
        Component: {component} (AAD {aad_id})
        Faalvormen:
        {faalvorm_tekst}
        """
        ).strip()
        context_blocks.append(context)
    return context_blocks, metadata
