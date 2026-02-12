# Updaten van de Chroma vectorstore

Documenten in de vectorstore vernieuwen:

```shell
# Start een terminal in de draaiende LLM container. 
docker ps # kopieer de container id
docker exec -it <id> /bin/bash # start terminal in de container
python3 -m ksandr.vectorstore.write_list_of_invalid_texts --env staging 

python3 -m ksandr.ingest.populate_database --env staging -chroma /root/onprem_data/chroma/ -source /root/ksandr_files -min_chunk_size_json 400 -min_chunk_size_text 600

python3 api/ingest/populate_database.py --env staging -chroma /root/onprem_data/chroma/ -source /root/ksandr_files -min_chunk_size_json 400 -min_chunk_size_text 600
```
