"""
This script connects to a Neo4j graph database and removes all nodes and their relationships.

Usage:
    - When run, this script will delete every node and relationship in the database specified by the connection string.
    - Useful for resetting the database during development or before a new data ingestion process.

Caution:
    - All data in the graph database will be irreversibly deleted after running this script.
"""

from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))


def clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


clear_database()
print("Database cleared.")
