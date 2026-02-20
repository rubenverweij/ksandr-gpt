#!/bin/bash

LOGFILE="/var/log/docker-ksandr.log"

docker events \
  --filter 'event=start' \
  --filter 'event=die' \
  --format '{{.Time}}|{{.Actor.Attributes.name}}|{{.Actor.Attributes.com.docker.compose.project}}|{{.Actor.Attributes.com.docker.compose.service}}|{{.Status}}|{{.Actor.Attributes.exitCode}}' \
| while IFS="|" read epoch name project service status exitcode; do

    timestamp=$(date -d @"$epoch" '+%Y-%m-%d %H:%M:%S')

    if [ "$status" = "die" ]; then
        if [ "$exitcode" = "0" ]; then
            result="SUCCESS"
        else
            result="FAILED (exit=$exitcode)"
        fi
        echo "$timestamp | project=$project | service=$service | container=$name | STOP | $result" >> "$LOGFILE"
    else
        echo "$timestamp | project=$project | service=$service | container=$name | START" >> "$LOGFILE"
    fi

done