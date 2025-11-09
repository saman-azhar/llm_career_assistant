# === Project Makefile ===
PROJECT_NAME=career_assistant
VENV=.venv

# --- Setup and Installation ---
init:
	@echo ">>> Setting up environment..."
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

# --- Preprocessing Stage ---
preprocess:
	@echo ">>> Running data preprocessing..."
	python -m $(PROJECT_NAME).preprocessing.preprocessing_cv data/raw/UpdatedResumeDataSet.csv data/processed/cleaned_resume_data_final.csv
	python -m $(PROJECT_NAME).preprocessing.preprocessing_jd data/raw/glassdoor_jobs.csv data/processed/cleaned_job_data_final.csv

# --- Vector Database (Qdrant) ---
vector-db:
	@echo ">>> Starting Qdrant Vector Database..."
	docker run -d --name qdrant \
		-p 6333:6333 -p 6334:6334 \
		-v ./qdrant_storage:/qdrant/storage \
		qdrant/qdrant

# --- Semantic Matching Stage ---
match:
	@echo ">>> Running semantic matching with Qdrant integration..."
	python -m $(PROJECT_NAME).semantic.semantic_matching \
		--resume_csv data/processed/cleaned_resume_data_final.csv \
		--job_csv data/processed/cleaned_job_data_final.csv \
		--collection_name career_vectors \
		--top_n 5

# --- RAG Pipeline Stage ---
rag:
	@echo ">>> Running RAG pipeline..."
	python -m $(PROJECT_NAME).rag_pipeline.main

# --- Docker Management ---
docker-build:
	@echo ">>> Building Docker image..."
	docker build -t $(PROJECT_NAME):latest .

docker-run:
	@echo ">>> Running Docker container..."
	docker run -it --rm -p 8080:8080 $(PROJECT_NAME):latest

docker-down:
	@echo ">>> Stopping and removing Docker containers..."
	docker stop qdrant || true && docker rm qdrant || true

# --- Testing ---
test:
	@echo ">>> Running tests..."
	pytest -v $(PROJECT_NAME)/tests

# --- Clean ---
clean:
	@echo ">>> Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf $(PROJECT_NAME)/data/vector_store/default/*
	rm -rf $(PROJECT_NAME)/artifacts/models/*
	rm -rf ./qdrant_storage

# --- Lint ---
lint:
	flake8 $(PROJECT_NAME)

# --- Help ---
help:
	@echo "Available targets:"
	@echo "  init          - Setup venv and install dependencies"
	@echo "  preprocess    - Run resume and JD preprocessing"
	@echo "  vector-db     - Start Qdrant vector database"
	@echo "  match         - Run semantic matching (Qdrant)"
	@echo "  rag           - Run RAG pipeline"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"
	@echo "  docker-down   - Stop and remove containers"
	@echo "  test          - Run all tests"
	@echo "  clean         - Clean caches, artifacts, and temp files"
	@echo "  lint          - Lint code for style issues"
