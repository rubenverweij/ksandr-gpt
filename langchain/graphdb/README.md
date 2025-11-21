# Neo4J

Voor het starten van de neo4j container:

```shell
docker run \
    --network host \
    -d \
    --env=NEO4J_AUTH=none \
    -e NEO4J_PLUGINS='["apoc"]' \
    --volume=/home/ubuntu/neo4j:/root/neo4j \
    neo4j
```

Voor het ingesten van de neo4j data:

```shell
# Ga naar /home/ubuntu/ksandr-gpt
source ./venv/bin/activate # activeer virtuele omgeving
python3 langchain/graph/ingest_faalvormen.py

# start cypher
docker exec -it e0be8352a672 cypher-shell -u neo4j 
docker exec -it <neo4j container id e0be8352a672> cypher-shell -u neo4j -p jouwpassword
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



Aantal queries:

```cypher

```
