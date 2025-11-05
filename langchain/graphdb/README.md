# Neo4J

```shell
docker run \
    --network host \
    -d \
    --env=NEO4J_AUTH=none \
    -e NEO4J_PLUGINS='["apoc"]' \
    --volume=/home/ubuntu/neo4j:/root/neo4j \
    neo4j
```

```shell

curl -X POST http://localhost:8080/template \
-H "Content-Type: application/json" \
-d '{
  "question": "Wat zijn de faalvormen van de DR12?",
  "template": ""
}'
```

```shell

curl -X POST http://localhost:8080/parameters \
-H "Content-Type: application/json" \
-d '{
  "question": "Wat zijn de faalvormen van de DR12?",
  "template": "alle_faalvormen_per_component"
}'
```

```shell

curl -X POST http://localhost:8080/context \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Je hebt een set Cypher-query templates:
    ['alle_faalvormen_per_component', 'meest_voorkomende_faalvormen', 'faalvormen_met_indicator', 'gemeenschappelijke_faalvormen_tussen_2_componenten', 'faalvormen_voor_component_voor_oorzaak', 'meest_voorkomende_faalvormen_per_component', 'faalvormen_filter']

    Gebruikersvraag: "Wat zijn de faalvormen van de DR12"

    Kies de meest geschikte template die past bij deze vraag.
    Geef alleen de key van de template terug.",
}'
```


