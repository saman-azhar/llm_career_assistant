# === LLM Career Assistant - Makefile ===
# Modern production-ready makefile for the Career Assistant project

PROJECT_NAME=career_assistant
VENV=venv
PYTHON=python3
COMPOSE=docker compose
COMPOSE_DEV=-f docker-compose.dev.yml
COMPOSE_PROD=-f docker-compose.prod.yml

.PHONY: help init install preprocess ingest test dev-up dev-down prod-up prod-down docker-build docker-clean clean lint format

# --- Environment Setup ---
help:
	@echo "========================================"
	@echo "LLM Career Assistant - Makefile Help"
	@echo "========================================"
	@echo ""
	@echo "🔧 SETUP & ENVIRONMENT:"
	@echo "  make init              - Create and setup virtual environment"
	@echo "  make install           - Install dependencies from requirements.txt"
	@echo ""
	@echo "📊 DATA PIPELINE:"
	@echo "  make preprocess        - Run complete data preprocessing (JD + Resume)"
	@echo "  make ingest            - Ingest preprocessed data into Qdrant"
	@echo "  make pipeline          - Run full pipeline (preprocess → ingest → test)"
	@echo ""
	@echo "🧪 TESTING & QUALITY:"
	@echo "  make test              - Run all unit tests"
	@echo "  make test-rag          - Test RAG pipeline specifically"
	@echo "  make test-cov          - Run tests with coverage report"
	@echo "  make lint              - Run code linting (flake8)"
	@echo "  make format            - Format code with black"
	@echo ""
	@echo "🐳 DOCKER DEVELOPMENT:"
	@echo "  make dev-up            - Start dev environment (with hot-reload)"
	@echo "  make dev-down          - Stop dev environment"
	@echo "  make dev-logs          - View dev logs"
	@echo "  make dev-ps            - List running dev containers"
	@echo ""
	@echo "🚀 DOCKER PRODUCTION:"
	@echo "  make prod-up           - Start production environment"
	@echo "  make prod-down         - Stop production environment"
	@echo "  make prod-logs         - View production logs"
	@echo "  make prod-ps           - List running prod containers"
	@echo ""
	@echo "🧹 MAINTENANCE:"
	@echo "  make clean             - Clean temporary files and caches"
	@echo "  make docker-clean      - Remove all containers and volumes"
	@echo ""

# --- Virtual Environment Setup ---
init:
	@echo ">>> Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo ">>> Virtual environment created at ./$(VENV)"
	@echo ">>> To activate: source $(VENV)/bin/activate"

install:
	@echo ">>> Installing dependencies..."
	$(VENV)/bin/pip install --upgrade pip setuptools wheel
	$(VENV)/bin/pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# --- Data Processing Pipeline ---
preprocess:
	@echo ">>> Starting data preprocessing..."
	@echo "  [1/2] Preprocessing job descriptions..."
	$(PYTHON) -c "from $(PROJECT_NAME).preprocessing.preprocessing_jd import preprocess_job_data; preprocess_job_data('$(PROJECT_NAME)/data/raw/glassdoor_jobs.csv', '$(PROJECT_NAME)/data/processed/cleaned_job_data_final.csv')"
	@echo "  [2/2] Preprocessing resumes..."
	$(PYTHON) -c "from $(PROJECT_NAME).preprocessing.preprocessing_cv import preprocess_resumes; preprocess_resumes('$(PROJECT_NAME)/data/raw/UpdatedResumeDataSet.csv', '$(PROJECT_NAME)/data/processed/cleaned_resume_data_final.csv')"
	@echo "✓ Preprocessing completed"

ingest:
	@echo ">>> Ingesting data into Qdrant..."
	$(PYTHON) -c "import mlflow; mlflow.end_run(); from $(PROJECT_NAME).rag_pipeline.ingest import ingest_data; ingest_data(chunking=True)"
	@echo "✓ Data ingestion completed"

pipeline: preprocess ingest test-rag
	@echo "✓ Complete pipeline executed successfully!"

# --- Testing ---
test:
	@echo ">>> Running all unit tests..."
	$(PYTHON) -m pytest $(PROJECT_NAME)/tests -v --tb=short
	@echo "✓ Tests completed"

test-rag:
	@echo ">>> Running RAG pipeline tests..."
	$(PYTHON) -m pytest $(PROJECT_NAME)/tests/test_rag_pipeline.py -v -s

test-cov:
	@echo ">>> Running tests with coverage..."
	$(PYTHON) -m pytest $(PROJECT_NAME)/tests --cov=$(PROJECT_NAME) --cov-report=html --cov-report=term-missing
	@echo "✓ Coverage report generated: htmlcov/index.html"

# --- Code Quality ---
lint:
	@echo ">>> Running linter (flake8)..."
	$(PYTHON) -m flake8 $(PROJECT_NAME) --max-line-length=120 --exclude=__pycache__
	@echo "✓ Linting completed"

format:
	@echo ">>> Formatting code with black..."
	$(PYTHON) -m black $(PROJECT_NAME) --line-length=120
	@echo "✓ Code formatted"

# --- Docker Development Commands ---
dev-up:
	@echo ">>> Starting development environment..."
	$(COMPOSE) $(COMPOSE_DEV) down
	$(COMPOSE) $(COMPOSE_DEV) up -d
	@echo "✓ Development environment started"
	@echo "  API Docs:    http://localhost:8000/docs"
	@echo "  MLflow:      http://localhost:5000"
	@echo "  Qdrant:      http://localhost:6333/dashboard"
	@echo "  PostgreSQL:  localhost:5432"

dev-down:
	@echo ">>> Stopping development environment..."
	$(COMPOSE) $(COMPOSE_DEV) down
	@echo "✓ Development environment stopped"

dev-logs:
	@echo ">>> Development logs (api)..."
	$(COMPOSE) $(COMPOSE_DEV) logs api -f

dev-ps:
	@echo ">>> Development containers..."
	$(COMPOSE) $(COMPOSE_DEV) ps

# --- Docker Production Commands ---
prod-up:
	@echo ">>> Starting production environment..."
	$(COMPOSE) $(COMPOSE_PROD) down
	$(COMPOSE) $(COMPOSE_PROD) up -d
	@echo "✓ Production environment started"
	@echo "  API:         http://localhost:8000/docs"
	@echo "  MLflow:      http://localhost:5000"
	@echo "  Qdrant:      http://localhost:6333"

prod-down:
	@echo ">>> Stopping production environment..."
	$(COMPOSE) $(COMPOSE_PROD) down
	@echo "✓ Production environment stopped"

prod-logs:
	@echo ">>> Production logs (api)..."
	$(COMPOSE) $(COMPOSE_PROD) logs api -f

prod-ps:
	@echo ">>> Production containers..."
	$(COMPOSE) $(COMPOSE_PROD) ps

# --- Cleaning ---
clean:
	@echo ">>> Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Temporary files cleaned"

docker-clean:
	@echo ">>> Removing all Docker containers and volumes..."
	$(COMPOSE) $(COMPOSE_DEV) down -v
	$(COMPOSE) $(COMPOSE_PROD) down -v
	@echo "✓ Docker cleanup completed"

# --- Quick Start Commands ---
start: dev-up
	@echo "✓ Development environment is ready!"

stop: dev-down
	@echo "✓ Development environment stopped"

restart: dev-down dev-up
	@echo "✓ Development environment restarted"
