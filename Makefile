.PHONY: help build up down restart logs clean test install dev

# Default target
help:
	@echo "Available commands:"
	@echo "  make build          - Build Docker image"
	@echo "  make up             - Start containers in detached mode"
	@echo "  make down           - Stop and remove containers"
	@echo "  make restart        - Restart containers"
	@echo "  make logs           - View container logs"
	@echo "  make logs-f         - Follow container logs"
	@echo "  make clean          - Remove containers, volumes, and images"
	@echo "  make clean-db       - Remove database and vector store"
	@echo "  make test           - Run tests inside container"
	@echo "  make install        - Install Python dependencies locally"
	@echo "  make dev            - Run development server locally"
	@echo "  make shell          - Open shell in running container"
	@echo "  make health         - Check container health status"

# Build Docker image
build:
	docker-compose build

# Start containers
up:
	docker-compose up -d

# Stop containers
down:
	docker-compose down

# Restart containers
restart:
	docker-compose restart

# View logs
logs:
	docker-compose logs

# Follow logs
logs-f:
	docker-compose logs -f

# Clean containers and volumes
clean:
	docker-compose down -v
	docker rmi tractian-ml-engineering-llm_app || true

# Clean database and vector store
clean-db:
	rm -rf chroma_db
	rm -f tractian.db

# Run tests inside container
test:
	docker-compose exec app pytest -v

# Install dependencies locally
install:
	pip install -r requirements.txt

# Run development server locally
dev:
	uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Open shell in running container
shell:
	docker-compose exec app /bin/bash

# Check container health
health:
	docker-compose ps
	@echo ""
	@echo "Health check endpoint:"
	curl -f http://localhost:8000/health || echo "Service not healthy"

# Build and start in one command
up-build: build up

# Stop, clean, rebuild, and start
rebuild: down clean build up
