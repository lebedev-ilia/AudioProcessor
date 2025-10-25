# AudioProcessor Makefile

.PHONY: help install dev-install test lint format clean run docker-build docker-run

help: ## Show this help message
	@echo "AudioProcessor - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

dev-install: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-asyncio black flake8 mypy isort

test: ## Run tests
	pytest src/tests/ -v

test-cov: ## Run tests with coverage
	pytest src/tests/ -v --cov=src --cov-report=html --cov-report=term

lint: ## Run linting
	flake8 src/
	mypy src/
	isort --check-only src/

format: ## Format code
	black src/
	isort src/

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

run: ## Run the application locally
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

run-worker: ## Run Celery worker
	celery -A src.celery_app worker --loglevel=info

run-flower: ## Run Flower monitoring
	celery -A src.celery_app flower --port=5555

docker-build: ## Build Docker image
	docker build -t audio-processor:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 audio-processor:latest

docker-compose-up: ## Start all services with Docker Compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

docker-compose-logs: ## View Docker Compose logs
	docker-compose logs -f

setup-dev: ## Setup development environment
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@echo "Development environment setup complete!"
	@echo "Activate with: source venv/bin/activate"

check-health: ## Check service health
	curl -f http://localhost:8000/health || echo "Service is not running"

check-metrics: ## Check Prometheus metrics
	curl -f http://localhost:8000/metrics || echo "Metrics endpoint not available"

check-flower: ## Check Flower monitoring
	curl -f http://localhost:5555 || echo "Flower is not running"

check-grafana: ## Check Grafana dashboard
	curl -f http://localhost:3000 || echo "Grafana is not running"

all-checks: check-health check-metrics check-flower check-grafana ## Run all health checks

# Development workflow
dev-start: docker-compose-up ## Start development environment
	@echo "Development environment started!"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	@echo "Flower: http://localhost:5555"
	@echo "Grafana: http://localhost:3000 (admin/admin)"

dev-stop: docker-compose-down ## Stop development environment
	@echo "Development environment stopped!"

dev-restart: dev-stop dev-start ## Restart development environment

# Testing workflow
test-unit: ## Run unit tests only
	pytest src/tests/ -v -m unit

test-integration: ## Run integration tests only
	pytest src/tests/ -v -m integration

test-fast: ## Run fast tests only
	pytest src/tests/ -v -m "not slow"

# Code quality workflow
quality-check: lint test ## Run all quality checks

quality-fix: format ## Fix code formatting issues

# Deployment helpers
build-prod: ## Build production Docker image
	docker build -t audio-processor:prod -f Dockerfile.prod .

deploy-local: ## Deploy locally with Docker Compose
	docker-compose -f docker-compose.prod.yml up -d

# Monitoring helpers
logs-api: ## View API logs
	docker-compose logs -f audio-processor

logs-worker: ## View worker logs
	docker-compose logs -f audio-worker

logs-all: ## View all logs
	docker-compose logs -f

# Database helpers (if needed)
db-migrate: ## Run database migrations
	@echo "Database migrations not implemented yet"

db-reset: ## Reset database
	@echo "Database reset not implemented yet"
