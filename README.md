# LLM-Powered Career Assistant

A modular NLP project that analyzes job descriptions, compares them with a CV, identifies missing skills, and generates tailored cover letters using both LLM and template-based approaches. Includes a RAG pipeline, FastAPI endpoints, and MLflow tracking.

---

## Tech Stack

- **Python, Pandas, Scikit-learn**
- **NLP/LLM:** HuggingFace Transformers (`flan-t5-base`), SentenceTransformers (`e5-base-v2`)
- **Vector DB:** Qdrant (RAG retrieval)
- **MLOps:** MLflow, Docker
- **API:** FastAPI

---

## Features

- Semantic matching between CVs and job descriptions
- Missing skills detection and upskilling suggestions
- RAG pipeline with both LLM and template-based cover letter generation
- Production-ready: FastAPI endpoints, MLflow logging, Dockerized workflow

---

## Quick Start

```bash
git clone https://github.com/saman-azhar/llm_career_assistant.git
cd llm_career_assistant
pip install -r requirements.txt
# or: docker-compose up --build
```

See `setup.md` for full setup and usage instructions.

---

## System Overview & Scaling

- **Frontend:** Streamlit app for interactive demo and recruiter-facing UI
- **Backend:** FastAPI for all core endpoints (`/match`, `/compare`, `/health`, `/metrics`)
- **RAG Pipeline:** Qdrant vector DB, e5-base-v2 embeddings, LLM (Flan-T5-base) and template-based generators
- **Experiment Tracking:** MLflow (auto-launched with Docker Compose)
- **Orchestration:** Docker Compose for full-stack local/prod setup

### Main Approaches
- **Template-based:** Fast, deterministic, zero hallucination, ideal for production and demo
- **LLM-based (Flan-T5):** Contextual, flexible, shows AI/ML skills, can be swapped for larger models (Mistral, Llama, etc.)
- **API-based (OpenAI/Anthropic):** For premium/enterprise use, higher cost, best quality

### Scaling Plan (Summary)
- Start with RAG + template for speed and reliability
- Add LLM fine-tuning (LoRA/QLoRA) as data grows for more personalized output
- Use API-based LLMs for premium features or when infra is not available
- See the Scaling Plan tab in the Streamlit app and project_notes.md for full details

---

## Quick Start (Full Stack)

```bash
docker compose -f docker-compose.dev.yml up --build
```
- Streamlit: http://localhost:8501
- FastAPI:   http://localhost:8000
- Qdrant:    http://localhost:6333
- MLflow:    http://localhost:5000

---

## Main API Endpoints
- `/match` (POST): Template-based skill matching and cover letter generation
- `/compare` (POST): Compare LLM vs template output
- `/health` (GET): Health check
- `/metrics` (GET): Metrics (placeholder)
- `/collections` (GET): Qdrant collections
- See http://localhost:8000/docs for full API reference

---

## More
- See `setup.md` for full setup and usage instructions
- See `project_notes.md` for technical journey and design decisions
- For scaling, cost, and architecture, see the Scaling Plan tab in Streamlit

---
