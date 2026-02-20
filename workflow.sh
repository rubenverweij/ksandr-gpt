#!/bin/bash
# python3 -m ksandr.vectorstore.populate_database -env ${APP_ENV}
python3 -m ksandr.vectorstore.remove_duplicates -env ${APP_ENV}
# python3 -m ksandr.graphdb.ingest_cypher_queries -env ${APP_ENV}
# python3 -m ksandr.graphdb.ingest_aads -env ${APP_ENV}
