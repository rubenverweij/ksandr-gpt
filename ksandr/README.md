# Docker containers en databases

Acties zijn:

1. Starten en stoppen van de `ksandr-gpt-langchain` container
2. Updaten van de Chroma vectorstore
3. Updaten van de Neo4j database.
4. Het uitvoeren van tests
5. Beheren van configuratiebestanden

## Het starten van de `ksandr-gpt-langchain` container

Er zijn meerdere omgevingsvariabelen die gekozen kunnen worden bij het starten van de container:

```python
# Haalt de temperatuur op uit de omgevingsvariabele. Een hogere waarde maakt het taalmodel creatiever (bijv. minder voorspelbaar).
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))  # standaard 0.7 als niet opgegeven

# Het maximum aantal bronnen (documenten) dat wordt meegenomen in het antwoord.
SOURCE_MAX = int(os.getenv("SOURCE_MAX", 5))  # standaard 10

# Drempelwaarde voor de relevantiescore van een document. Documenten onder deze waarde worden genegeerd.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.9))  # standaard 0.9

# Geeft aan of filtering op metadata (zoals datum of type) moet worden toegepast (1 = ja, 0 = nee).
INCLUDE_FILTER = int(os.getenv("INCLUDE_FILTER", 1))  # standaard 1

# Maximaal aantal tokens dat in het gegenereerde antwoord mag voorkomen.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 400))  # standaard 400

# Maximale contextlengte (in tokens) die naar het taalmodel wordt gestuurd.
MAX_CTX = int(os.getenv("MAX_CTX", 4096))  # standaard 4096

```

Om de container te starten (laat de parameters wat ze zijn):

```shell
image="ksandr-gpt-langchain:0.55"

docker run --network host -d --gpus=all --cap-add SYS_RESOURCE \
-e USE_MLOCK=0 \
-e TEMPERATURE=0.7 \
-e INCLUDE_FILTER=1 \
-e SCORE_THRESHOLD=0.90 \
-e MAX_TOKENS=400 \
-e SOURCE_MAX=5 \
-e MAX_CTX=4096 \
-e IMAGE_NAME=$image \
-v /home/ubuntu/da_data/config/creds.json:/ksandr-gpt/ksandr/creds.json:ro \
-v /home/ubuntu/da_data/nltk_data:/root/nltk_data \
-v /home/ubuntu/da_data/huggingface:/root/huggingface \
-v /home/ubuntu/onprem_data:/root/onprem_data \
-v /home/ubuntu/ksandr_files:/root/ksandr_files \
$image

```

## Updaten van de Chroma vectorstore

Documenten in de vectorstore vernieuwen:

```shell
# Start een terminal in de draaiende LLM container. 
docker ps # kopieer de container id
docker exec -it <id> /bin/bash # start terminal in de container
python3 api/ingest/populate_database.py -chroma /root/onprem_data/chroma/ -source /root/ksandr_files -min_chunk_size_json 400 -min_chunk_size_text 600
```

## Updaten van de Neo4j database

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
    - Activeer de virtuele Python omgeving `source ./venv/bin/activate`
    - Voer de volgende commando's uit:

```shell
python3 langchain/graphdb/ingest_cypher_queries.py -chroma /home/ubuntu/onprem_data/chroma_cypher
python3 langchain/ingest_aads.py
```

## Het uitvoeren van tests

Testvragen toevoegen aan `langchain\tests\testvragen.csv` of locatie.

Test script draaien en analyseren resultaten:

```shell
source ./venv/bin/activate # activeer virtuele omgeving gebaseerd op requirements.txt
python3 langchain/tests/test_model.py --file langchain/tests/testvragen.csv # analyseer de antwoorden
python3 langchain/tests/report.py # controleer de resultaten
```

## Beheren van configuratiebestanden

De volgende configuratiebestanden moeten beheerd worden:

1. langchain/refs.py
2. langchain/config.py
3. 