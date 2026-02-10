# Neo4J

We gebruiken [Neo4j](https://neo4j.com/) als een graph database.
Gebruik het volgende commando voor het starten van de container:

```shell
docker run \
    --network host \
    -d \
    --env=NEO4J_AUTH=none \
    -e NEO4J_PLUGINS='["apoc"]' \
    --volume=/home/ubuntu/neo4j:/data \
    neo4j
```

Voor het ingesten van de neo4j data:
    - Login the GPU server
    - Ga naar `/home/ubuntu/ksandr-gpt`
    - Voer de volgende commando's uit:

```shell
source ./venv/bin/activate # activeer virtuele omgeving gebaseerd op requirements.txt
python3 langchain/graphdb/ingest_cypher_queries.py -chroma /home/ubuntu/onprem_data/chroma_cypher
python3 ingest_aads.py
```

Om vast te stellen dat de neo4j data op correcte wijze is toegevoegd kunnen Cypher queries worden uitgevoerd in de container door:

```shell
docker exec -it <neo4j container id d76306d98bc1> cypher-shell -u neo4j -p jouwpassword
SHOW DATABASES;
```

Voorbeelden van queries:

```cypher
# Geef de labels
CALL db.labels();

# Relaties
CALL db.relationshipTypes();

# Eigenschappen
CALL db.propertyKeys();

# Index
SHOW INDEXES;

# Randvoorwaarden 
SHOW CONSTRAINTS;

# Voorbeelden van de volledige structuur
MATCH (n)
UNWIND labels(n) AS label
UNWIND keys(n) AS key
RETURN label, key, count(*) AS occurrences
ORDER BY label, key;

# Verwijderen data uit database
CALL apoc.schema.assert({}, {});
MATCH (n)
DETACH DELETE n;
```
