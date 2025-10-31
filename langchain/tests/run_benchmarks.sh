#!/bin/bash
# Automate container spin-up, run performance test, and teardown

TEST_SCRIPT="langchain/tests/test_model.py"
IMAGE="ksandr-gpt-langchain:0.43"

# Define configurations — you can easily add or modify combinations here
CONFIGS=(
    "SCORE_THRESHOLD_JSON=0.6 MAX_CTX=2048 SOURCE_MAX=2"
    "SCORE_THRESHOLD_JSON=0.6 MAX_CTX=3096 SOURCE_MAX=4"
    "SCORE_THRESHOLD_JSON=0.8 MAX_CTX=4096 SOURCE_MAX=4"
    "SCORE_THRESHOLD_JSON=0.9 MAX_CTX=5192 SOURCE_MAX=6"
)

for CONFIG in "${CONFIGS[@]}"; do
    eval $CONFIG

    CONTAINER_NAME="perf_stj${SCORE_THRESHOLD_JSON}_ctx${MAX_CTX}_src${SOURCE_MAX}"

    echo "🚀 Starting container with:"
    echo "   SCORE_THRESHOLD_JSON=$SCORE_THRESHOLD_JSON"
    echo "   MAX_CTX=$MAX_CTX"
    echo "   SOURCE_MAX=$SOURCE_MAX"
    echo

    docker run --network host -d --gpus=all --cap-add SYS_RESOURCE \
        -e USE_MLOCK=0 \
        -e TEMPERATURE=0.7 \
        -e INCLUDE_FILTER=1 \
        -e SCORE_THRESHOLD=1 \
        -e SCORE_THRESHOLD_JSON=$SCORE_THRESHOLD_JSON \
        -e MAX_TOKENS=400 \
        -e SOURCE_MAX=$SOURCE_MAX \
        -e INCLUDE_SUMMARY=0 \
        -e INCLUDE_KEYWORDS=0 \
        -e MAX_CTX=$MAX_CTX \
        -e INCLUDE_PERMISSION=0 \
        -e IMAGE_NAME=$IMAGE \
        -v /home/ubuntu/nltk_data:/root/nltk_data \
        -v /home/ubuntu/huggingface:/root/.cache/huggingface \
        -v /home/ubuntu/onprem_data:/root/onprem_data \
        -v /home/ubuntu/ksandr_files:/root/ksandr_files \
        --name "$CONTAINER_NAME" \
        $IMAGE

    echo "⏳ Waiting 60 seconds for container initialization..."
    sleep 60

    echo "⏱️ Running performance test in $CONTAINER_NAME..."
    docker exec "$CONTAINER_NAME" python $(basename "$TEST_SCRIPT")

    echo "🧹 Stopping and removing container..."
    docker stop "$CONTAINER_NAME" >/dev/null
    sleep 60

    docker rm "$CONTAINER_NAME" >/dev/null

    echo "✅ Completed test for STJ=$SCORE_THRESHOLD_JSON, MAX_CTX=$MAX_CTX, SOURCE_MAX=$SOURCE_MAX"
    echo
done

echo "🎯 All performance tests completed."
