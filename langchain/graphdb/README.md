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
```
