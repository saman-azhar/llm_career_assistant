
# LLM-Powered Career Assistant

An end-to-end NLP project that analyzes job descriptions, compares them with a CV, identifies missing skills, and generates improvement suggestions — now enhanced with a RAG pipeline, production-ready FastAPI endpoints, and LLM-driven contextual generation.

---

## Tech Stack

**Core:** Python, Pandas, Scikit-learn
**NLP/LLM:** HuggingFace Transformers, SentenceTransformers (`all-MiniLM-L6-v2` → `e5-base-v2`), Mistral-7B-Instruct-v0.2
**Vector Database:** Qdrant (RAG-style retrieval)
**MLOps:** MLflow, Docker, GitHub
**API & UI:** FastAPI (real-time CV–JD matching), Streamlit (interactive demo)

---

## Project Overview

| Phase  | Focus                         | Deliverable                                                                           |
| ------ | ----------------------------- | ------------------------------------------------------------------------------------- |
| Week 1 | Data cleaning & title mapping | Cleaned dataset, EDA visuals                                                          |
| Week 2 | ML baseline (LR + SVM)        | Classification notebook                                                               |
| Week 3 | Semantic matching             | Embedding-based similarity, missing skills                                            |
| Week 4 | LLM integration               | Job summary + draft cover letters (Flan-T5)                                           |
| Week 5 | RAG pipeline & Production     | `e5-base-v2` embeddings + Mistral-7B + Qdrant, FastAPI endpoints, Dockerized workflow |
| Week 6 | Streamlit demo                | Interactive portfolio app                                                             |

---

## Quick Start

```bash
git clone https://github.com/saman-azhar/llm_career_assistant.git
cd llm_career_assistant
pip install -r requirements.txt
```

* Setup details → [`setup_notes.md`](https://github.com/saman-azhar/llm_career_assistant/blob/main/setup.md)
* Progress notes → [`project_notes.md`](https://github.com/saman-azhar/llm_career_assistant/blob/main/project_notes.md)

---

## Highlights

* End-to-end semantic matching between CVs and job descriptions.
* Missing skills detection for targeted upskilling suggestions.
* RAG-style LLM generation: context-grounded cover letters and insights.
* Production-ready pipeline: FastAPI endpoints, modular scripts, CLI-friendly.
* Embeddings upgraded to `e5-base-v2` for higher semantic fidelity.
* LLM generator upgraded to `Mistral-7B-Instruct` for long-context and instruction-following tasks.
* Vector database (Qdrant) integration for fast retrieval and RAG functionality.
* Fully containerized workflow (Docker) with MLflow tracking for reproducibility.
* Streamlit dashboard for interactive demonstration of CV–JD alignment and insights.

---
