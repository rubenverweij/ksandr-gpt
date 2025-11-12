import json
from neo4j import GraphDatabase
from pathlib import Path

# ---- Neo4j connection ----
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


def get_labels(tx):
    """Return all node labels."""
    result = tx.run("CALL db.labels()")
    return [record["label"] for record in result]


def get_relationship_types(tx):
    """Return all relationship types."""
    result = tx.run("CALL db.relationshipTypes()")
    return [record["relationshipType"] for record in result]


def get_properties(tx, label):
    """Get sample properties for a label."""
    query = f"""
    MATCH (n:{label})
    WITH keys(n) AS props
    RETURN apoc.coll.toSet(apoc.coll.flatten(collect(props))) AS properties
    LIMIT 1
    """
    result = tx.run(query)
    record = result.single()
    return record["properties"] if record else []


def get_sample_values(tx, label, prop, limit=3):
    """Get sample values for a specific property."""
    query = f"""
    MATCH (n:{label})
    WHERE exists(n.{prop})
    RETURN DISTINCT n.{prop} AS val
    LIMIT {limit}
    """
    result = tx.run(query)
    return [r["val"] for r in result if r["val"]]


def generate_queries_from_db():
    """Auto-generate queries and example questions."""
    queries = []

    with driver.session() as session:
        labels = session.execute_read(get_labels)
        rels = session.execute_read(get_relationship_types)

        print(f"✅ Found labels: {labels}")
        print(f"✅ Found relationships: {rels}")

        # --- Example: generate for AAD → Component → Faaltype chain ---
        if "AAD" in labels and "Component" in labels and "Faaltype" in labels:
            aad_samples = session.execute_read(get_sample_values, "AAD", "aad_id")
            comp_samples = session.execute_read(get_sample_values, "Component", "naam")

            # 1️⃣ Aantal faalvormen per component
            for comp in comp_samples:
                queries.append(
                    {
                        "cypher": f"""
                    MATCH (c:Component {{naam: "{comp}"}})-[:HEEFT_FAALTYPE]->(f:Faaltype)
                    RETURN COUNT(f) AS aantalFaalvormen
                    """.strip(),
                        "example_questions": [
                            f"Hoeveel faalvormen heeft de component {comp}?",
                            f"Geef het aantal faaltypen van {comp}.",
                        ],
                    }
                )

            # 2️⃣ Alle faalvormen per component
            for comp in comp_samples:
                queries.append(
                    {
                        "cypher": f"""
                    MATCH (c:Component {{naam: "{comp}"}})-[:HEEFT_FAALTYPE]->(f:Faaltype)
                    RETURN f.Naam AS faalvorm, f.OorzaakGeneriek AS oorzaak, f.Beschrijving AS beschrijving
                    """.strip(),
                        "example_questions": [
                            f"Welke faalvormen horen bij component {comp}?",
                            f"Wat zijn de oorzaken van faalvormen bij {comp}?",
                        ],
                    }
                )

            # 3️⃣ Per AAD → aantal faalvormen
            for aad in aad_samples:
                queries.append(
                    {
                        "cypher": f"""
                    MATCH (a:AAD {{aad_id: "{aad}"}})-[:HEEFT_COMPONENT]->(c:Component)-[:HEEFT_FAALTYPE]->(f:Faaltype)
                    RETURN COUNT(f) AS totaalFaalvormen
                    """.strip(),
                        "example_questions": [
                            f"Hoeveel faalvormen heeft AAD {aad}?",
                            f"Geef het totaal aantal faaltypen binnen {aad}.",
                        ],
                    }
                )

        # --- Generic fallbacks ---
        for label in labels:
            props = session.execute_read(get_properties, label)
            queries.append(
                {
                    "cypher": f"MATCH (n:{label}) RETURN n LIMIT 10",
                    "example_questions": [f"Toon alle {label.lower()} nodes."],
                }
            )
            for prop in props:
                queries.append(
                    {
                        "cypher": f"MATCH (n:{label}) RETURN DISTINCT n.{prop} LIMIT 10",
                        "example_questions": [
                            f"Welke waarden heeft het veld {prop} van {label}?"
                        ],
                    }
                )

    return queries


if __name__ == "__main__":
    print("🔍 Scanning Neo4j schema and generating queries...")
    queries = generate_queries_from_db()

    output_path = Path("/home/ubuntu/onprem_data/auto_generated_queries.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(queries)} queries → {output_path}")
