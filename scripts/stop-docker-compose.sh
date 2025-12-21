#!/bin/bash
set -e

SCRIPT_PATH=$(realpath "$0")
SCRIPTS_DIR=$(dirname "$SCRIPT_PATH")
COMPOSE_FILE="$SCRIPTS_DIR/docker-compose.generated.yml"

echo "=== Stopping Docker Compose Cluster ==="

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Error: Compose file not found at $COMPOSE_FILE"
    echo "Is the cluster running?"
    exit 1
fi

echo "Using compose file: $COMPOSE_FILE"
docker compose -f "$COMPOSE_FILE" down -v --remove-orphans

echo "Cluster stopped and volumes removed."
