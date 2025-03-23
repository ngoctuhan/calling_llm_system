#!/bin/bash

# Exit on error
set -e

echo "Setting up Call Center System with Docker..."

# Copy docker environment file if it doesn't exist
if [ ! -f .env ]; then
  echo "Creating .env file from .env.docker template..."
  cp .env.docker .env
  echo "Please update the .env file with your actual API keys and credentials."
fi

# Create Docker volume directories with proper permissions
echo "Creating Docker volumes..."
mkdir -p ./docker-volumes/postgres
mkdir -p ./docker-volumes/redis
mkdir -p ./docker-volumes/elasticsearch
mkdir -p ./docker-volumes/qdrant
mkdir -p ./docker-volumes/neo4j/data
mkdir -p ./docker-volumes/neo4j/logs

# Build and start the services
echo "Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Initialize the database
echo "Initializing the database..."
docker-compose exec app python scripts/init_db.py

echo "Call Center System is up and running!"
echo ""
echo "Services:"
echo "- FastAPI app: http://localhost:8000"
echo "- PostgreSQL: localhost:5432"
echo "- Redis: localhost:6379"
echo "- Elasticsearch: http://localhost:9200"
echo "- Neo4j Browser: http://localhost:7474"
echo "- Qdrant API: http://localhost:6333"
echo ""
echo "To stop the services, run: docker-compose down" 