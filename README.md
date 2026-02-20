# Digitale Assistent


- **docker-compose.api.yaml**: Docker Compose configuratie voor de API-service van het taalmodel.
- **docker-compose.vectorstore.yaml**: Docker Compose configuratie voor het ingesten en beheren van documenten (vectorstore en graph database).
- **monitor-compose.sh**: Script dat automatisch Docker container events monitort en logt naar `/var/log/docker-ksandr.log`. Start via `nohup ./monitor-compose.sh &`.
- **ingest_schedule.sh**: Script voor geschedulede ingest van data via cronjobs schedule `0 8 * * 0`.

Belangrijke commando's:

```shell
# Voor staging
docker compose -f docker-compose.vectorstore.yaml --env-file .env.staging up
docker compose -f docker-compose.api.yaml --env-file .env.staging up

# Voor productie
docker compose -f docker-compose.vectorstore.yaml --env-file .env.production up
docker compose -f docker-compose.api.yaml --env-file .env.production up
```


- **doc/**: Bevat uitgebreide documentatie over installatie, troubleshooting en gebruik.
- **ksandr/** Directories met de broncode, onder andere voor taalmodel-integratie, API en het ingest-proces.
