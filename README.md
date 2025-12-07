

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
