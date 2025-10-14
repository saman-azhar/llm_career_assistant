
# LLM-Powered Career Assistant

An end-to-end NLP project that analyzes job descriptions, compares them with a CV, identifies missing skills, and generates improvement suggestions — packaged with MLOps best practices.

---

## Tech Stack

**Core:** Python, Pandas, Scikit-learn
**NLP/LLM:** HuggingFace Transformers, SentenceTransformers, (optional) OpenAI API
**MLOps:** MLflow, Docker, GitHub
**API & UI:** FastAPI, Streamlit

---

## Project Overview

| Phase  | Focus                         | Deliverable                  |
| ------ | ----------------------------- | ---------------------------- |
| Week 1 | Data cleaning & title mapping | Cleaned dataset, EDA visuals |
| Week 2 | ML baseline (LR + SVM)        | Classification notebook      |
| Week 3 | Semantic matching             | Embedding-based similarity   |
| Week 4 | LLM integration               | Job summary + cover letter   |
| Week 5 | MLOps + Deployment            | MLflow + Docker API          |
| Week 6 | Streamlit demo                | Interactive portfolio app    |

---

## Quick Start

```bash
git clone https://github.com/saman-azhar/llm_career_assistant.git
cd llm_career_assistant
pip install -r requirements.txt
```

Setup details → [`setup_notes.md`](https://github.com/saman-azhar/llm_career_assistant/blob/main/setup.md)
Progress notes → [`project_notes.md`](https://github.com/saman-azhar/llm_career_assistant/blob/main/project_notes.md)

---

## Highlights

* Job–CV semantic matching via Sentence-BERT
* Baseline ML → LLM pipeline progression
* Reproducible with MLflow + Docker
* Streamlit dashboard for presentation

