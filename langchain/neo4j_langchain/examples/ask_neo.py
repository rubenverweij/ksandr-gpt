from neo4j import GraphDatabase
import argparse


# Connect to your Neo4j database
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# The query we want to test
query = """
MATCH (a:AAD)-[:HEEFT_COMPONENT]->(c:Component)-[:HEEFT_FAALTYPE]->(f:Faaltype)
WHERE ($aad_ids IS NULL OR size($aad_ids) = 0 OR a.aad_id IN $aad_ids)
RETURN 
    a.aad_id AS aad_id,
    c.naam AS component_naam,
    f.Nummer AS nummer_faalvorm,
    f.NummerInt AS nummer_int,
    f.Naam AS naam_faalvorm,
    coalesce(f.GemiddeldAantalIncidenten, 'Onbekend') AS aantal_incidenten,
    f.Bestandspad AS bestandspad
ORDER BY 
    toInteger(f.NummerInt) ASC,
    f.Nummer ASC
"""


# Run the query and print results
def main(question: str):
    with driver.session() as session:
        result = session.run(query, {"aad_ids": []})  # Empty = all AADs
        print("=== FAALVORMEN RESULTATEN ===")
        for record in result:
            print(
                f"Component: {record['component_naam']:<40} | "
                f"Nummer: {record['nummer_faalvorm']:<6} | "
                f"Int: {record['nummer_int']:<3} | "
                f"Naam: {record['naam_faalvorm']:<40} | "
                f"Incidenten: {record['aantal_incidenten']}"
            )

    driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="echo the string you use here")
    args = parser.parse_args()
    main(args.question)
