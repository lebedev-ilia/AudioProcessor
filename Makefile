# AudioProcessor Makefile - GPU-Optimized Async Version

.PHONY: help install dev-install test lint format clean run docker-build docker-run
.PHONY: test-unit test-integration test-performance test-all test-coverage test-lint test-clean
.PHONY: build build-gpu up up-gpu down logs clean-docker test-docker health status

# Variables
DOCKER_COMPOSE = docker-compose
IMAGE_NAME = audio-processor
TAG = latest

help: ## Show this help message
	@echo "AudioProcessor - GPU-Optimized Async Version"
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ===== Installation =====
install: ## Install production dependencies
	pip install -r requirements.txt

dev-install: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-asyncio black flake8 mypy isort

# ===== Testing =====
test: test-unit ## Run unit tests (default)

test-unit: ## Run unit tests only
	pytest src/tests/ -m "unit" -v --tb=short

test-integration: ## Run integration tests only
	pytest src/tests/ -m "integration" -v --tb=short --integration

test-performance: ## Run performance tests only
	pytest src/tests/ -m "performance" -v --tb=short --performance

test-all: ## Run all tests
	pytest src/tests/ -v --tb=short --integration --performance

test-coverage: ## Run tests with coverage
	pytest src/tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-api: ## Run API tests
	pytest src/tests/test_api_endpoints.py -v

test-celery: ## Run Celery tests
	pytest src/tests/test_celery_tasks.py -v

test-s3: ## Run S3 tests
	pytest src/tests/test_s3_client.py -v

test-extractors: ## Run extractor tests
	pytest src/tests/test_extractors.py src/tests/test_extractors_detailed.py -v

test-fast: ## Run fast tests only
	pytest src/tests/ -v -m "not slow"

test-quick: ## Quick test (unit tests only, no coverage)
	pytest src/tests/ -m "unit" -v --tb=short -x

test-file: ## Test specific file (usage: make test-file FILE=src/tests/test_basic.py)
	pytest $(FILE) -v

test-marker: ## Test with specific marker (usage: make test-marker MARKER=gpu)
	pytest src/tests/ -m "$(MARKER)" -v

# ===== Code Quality =====
lint: ## Run linting
	flake8 src/
	mypy src/
	isort --check-only src/

format: ## Format code
	black src/
	isort src/

quality-check: lint test ## Run all quality checks
quality-fix: format ## Fix code formatting issues

# ===== Docker Operations =====
build: ## Build Docker image
	@echo "🔨 Building GPU-optimized Docker image..."
	$(DOCKER_COMPOSE) build --no-cache

up: ## Start all services with Docker Compose
	@echo "🚀 Starting GPU-optimized services..."
	$(DOCKER_COMPOSE) up -d

up-build: ## Start with rebuild
	@echo "🚀 Starting with rebuild..."
	$(DOCKER_COMPOSE) up -d --build

down: ## Stop all services
	@echo "🛑 Stopping services..."
	$(DOCKER_COMPOSE) down

restart: ## Restart services
	@echo "🔄 Restarting services..."
	$(DOCKER_COMPOSE) restart

# ===== Docker Testing =====
test-docker: ## Run tests in Docker
	@echo "🧪 Running tests in Docker..."
	$(DOCKER_COMPOSE) exec audio-processor python -m pytest src/tests/ -v

test-docker-basic: ## Run basic tests in Docker
	@echo "🧪 Running basic tests in Docker..."
	$(DOCKER_COMPOSE) exec audio-processor python -m pytest src/tests/test_basic.py -v

test-unified: ## Run Unified API tests in Docker
	@echo "🚀 Running Unified API tests in Docker..."
	$(DOCKER_COMPOSE) exec audio-processor python src/tests/test_unified_processor.py

# ===== Logs =====
logs: ## Show all logs
	$(DOCKER_COMPOSE) logs -f

logs-api: ## Show API logs
	$(DOCKER_COMPOSE) logs -f audio-processor

logs-worker: ## Show worker logs
	$(DOCKER_COMPOSE) logs -f audio-worker

logs-redis: ## Show Redis logs
	$(DOCKER_COMPOSE) logs -f redis

logs-minio: ## Show MinIO logs
	$(DOCKER_COMPOSE) logs -f minio

# ===== Monitoring =====
status: ## Show service status
	@echo "📊 Service status:"
	$(DOCKER_COMPOSE) ps

health: ## Check service health
	@echo "🏥 Checking service health:"
	@curl -s http://localhost:8000/health | jq . || echo "API unavailable"
	@curl -s http://localhost:8000/unified/config | jq . && echo "🚀 Unified API available" || echo "🚀 Unified API unavailable"
	@curl -s http://localhost:5555 | grep -q "Flower" && echo "Flower available" || echo "Flower unavailable"
	@curl -s http://localhost:9090 | grep -q "Prometheus" && echo "Prometheus available" || echo "Prometheus unavailable"
	@curl -s http://localhost:3000 | grep -q "Grafana" && echo "Grafana available" || echo "Grafana unavailable"

# ===== Development =====
run: ## Run the application locally
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

run-worker: ## Run Celery worker
	celery -A src.celery_app worker --loglevel=info

run-flower: ## Run Flower monitoring
	celery -A src.celery_app flower --port=5555

shell: ## Connect to API container
	$(DOCKER_COMPOSE) exec audio-processor bash

shell-worker: ## Connect to worker container
	$(DOCKER_COMPOSE) exec audio-worker bash

# ===== Cleanup =====
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage .pytest_cache/ .mypy_cache/

test-clean: ## Clean test artifacts
	rm -rf htmlcov/ .coverage .pytest_cache/
	rm -rf src/tests/__pycache__/
	find src/tests/ -name "*.pyc" -delete
	find src/tests/ -name "__pycache__" -type d -exec rm -rf {} +

clean-docker: ## Clean Docker resources
	@echo "🧹 Cleaning Docker resources..."
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f

clean-images: ## Remove Docker images
	@echo "🧹 Removing Docker images..."
	docker rmi $(IMAGE_NAME):$(TAG) 2>/dev/null || true

clean-all: clean test-clean clean-docker clean-images ## Full cleanup

# ===== Development Workflow =====
dev-start: up ## Start development environment
	@echo "Development environment started!"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	@echo "Flower: http://localhost:5555"
	@echo "Grafana: http://localhost:3000 (admin/admin)"

dev-stop: down ## Stop development environment
	@echo "Development environment stopped!"

dev-restart: dev-stop dev-start ## Restart development environment

# ===== System Info =====
info: ## Show system information
	@echo "ℹ️  System information:"
	@echo "Docker version: $$(docker --version)"
	@echo "Docker Compose version: $$(docker-compose --version)"
	@echo "GPU support: $$(docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unavailable')"
	@echo "Available memory: $$(docker system df)"

# ===== Backup =====
backup: ## Create data backup
	@echo "💾 Creating backup..."
	mkdir -p backups
	$(DOCKER_COMPOSE) exec redis redis-cli BGSAVE
	$(DOCKER_COMPOSE) exec minio mc mirror /data backups/minio-$(shell date +%Y%m%d_%H%M%S)