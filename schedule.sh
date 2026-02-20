#!/bin/bash

# Navigate to your compose file directory
cd /home/ubuntu/ksandr-gpt || exit

# Run the docker-compose service
docker compose \
  -f docker-compose.vectorstore.yaml \
  --env-file .env.staging \
  run --rm ingest
