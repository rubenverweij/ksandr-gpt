# Updaten van de Chroma vectorstore

Documenten in de vectorstore vernieuwen:

```shell
# Start een terminal in de draaiende LLM container. 
docker ps # kopieer de container id
docker exec -it <id> /bin/bash # start terminal in de container
python3 -m ksandr.vectorstore.populate_database -env staging 

docker run -d \
  --network host \
  --gpus all \
  --cap-add SYS_RESOURCE \
  -v /home/ubuntu/ksandr_files_staging:/root/ksandr_files_staging \
  -v /home/ubuntu/da_data/huggingface:/root/huggingface \
  -v /home/ubuntu/da_data/config/creds.json:/ksandr-gpt/ksandr/creds.json:ro \
  ksandr-gpt-langchain:0.56 \
  python3 -m ksandr.vectorstore.populate_database -env staging




-v /home/ubuntu/da_data:/root/da_data \
-v /home/ubuntu/ksandr_files:/root/ksandr_files \


# python3 api/ingest/populate_database.py --env staging -chroma /root/onprem_data/chroma/ -source /root/ksandr_files -min_chunk_size_json 400 -min_chunk_size_text 600
```

Duplicaten uit de database verwijderen.

```shell
python3 -m ksandr.vectorstore.remove_duplicates -env staging 
```

Tekststukken detecteren met onbruikbare kwaliteit.

```shell
python3 -m ksandr.vectorstore.write_list_of_invalid_texts -env staging 
```
