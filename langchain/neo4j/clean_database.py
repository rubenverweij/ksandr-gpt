from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))


def clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


clear_database()
print("Database cleared.")
