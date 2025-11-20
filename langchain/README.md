# Start langchain container

Er zijn meerdere omgevingsvariabelen die gekozen kunnen worden bij het starten van de container:

```python
# Haalt de temperatuur op uit de omgevingsvariabele. Een hogere waarde maakt het taalmodel creatiever (bijv. minder voorspelbaar).
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))  # standaard 0.2 als niet opgegeven

# Het maximum aantal bronnen (documenten) dat wordt meegenomen in het antwoord.
SOURCE_MAX = int(os.getenv("SOURCE_MAX", 10))  # standaard 10

# Aantal bronnen dat opnieuw wordt geëvalueerd door een reranker-model, indien ingesteld.
SOURCE_MAX_RERANKER = int(os.getenv("SOURCE_MAX_RERANKER", 0))  # standaard 0 (geen reranking)

# Drempelwaarde voor de relevantiescore van een document. Documenten onder deze waarde worden genegeerd.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 1.1))  # standaard 1.1

# Geeft aan of filtering op metadata (zoals datum of type) moet worden toegepast (1 = ja, 0 = nee).
INCLUDE_FILTER = int(os.getenv("INCLUDE_FILTER", 1))  # standaard 1

# Maximaal aantal tokens dat in het gegenereerde antwoord mag voorkomen.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 750))  # standaard 750

# Maximale contextlengte (in tokens) die naar het taalmodel wordt gestuurd.
MAX_CTX = int(os.getenv("MAX_CTX", 8000))  # standaard 8000

# Geeft aan of er een samenvatting moet worden toegevoegd aan het antwoord (1 = ja, 0 = nee).
INCLUDE_SUMMARY = int(os.getenv("INCLUDE_SUMMARY", 0))  # standaard 0

# Geeft aan of sleutelwoorden uit de bron moeten worden toegevoegd (1 = ja, 0 = nee).
INCLUDE_KEYWORDS = int(os.getenv("INCLUDE_KEYWORDS", 0))  # standaard 0
```

Om de container te starten (laat de parameters wat ze zijn):

```shell
image="ksandr-gpt-langchain:0.44"

docker run --network host -d --gpus=all --cap-add SYS_RESOURCE \
-e USE_MLOCK=0 \
-e TEMPERATURE=0.7 \
-e INCLUDE_FILTER=1 \
-e SCORE_THRESHOLD=0.90 \
-e MAX_TOKENS=1500 \
-e SOURCE_MAX=4 \
-e INCLUDE_SUMMARY=0 \
-e INCLUDE_KEYWORDS=0 \
-e MAX_CTX=4096 \
-e INCLUDE_PERMISSION=0 \
-e IMAGE_NAME=$image \
-v /home/ubuntu/nltk_data:/root/nltk_data \
-v /home/ubuntu/huggingface:/root/huggingface \
-v /home/ubuntu/onprem_data:/root/onprem_data \
-v /home/ubuntu/ksandr_files:/root/ksandr_files \
$image

```

Documenten in de vectorstore vernieuwen:

```shell
# Start een terminal in de draaiende LLM container. 
docker ps # kopieer de container id
docker exec -it <id> /bin/bash # start terminal in de container
python3 api/ingest/populate_database.py -chroma /root/onprem_data/chroma/ -source /root/ksandr_files -min_chunk_size_json 400 -min_chunk_size_text 600
```
