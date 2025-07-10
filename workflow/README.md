# Wekelijkse ingest 

Deze docker container vernieuwd alle embeddings beschikbaar in `/home/ubuntu/ksandr_files`. Aanpassingen kunnen gemaakt worden in de dockerfile.

```dockerfile
RUN echo "0 0 * * * python3 api/ingest_docs.py -vector_db_path /root/onprem_data/vectordb/sparse -documents_path /root/ksandr_files/ -chunk_size 300 -chunk_overlap 100 >> /tmp/cron.log 2>&1 > /etc/cron.d/ksandr-ingest-cron-job
```


