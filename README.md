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


## Scaling Plan

This section explains how this system could be scaled for real-world, high-volume use.

### Current Bottleneck Analysis

**Current Setup:**
- CPU-only inference (laptop/standard server)
- Memory: ~4GB for model + embeddings
- No batching/caching
- Model: FLAN-T5-base (~0.2B params), quantized for CPU portability
- Inference: 2-5 seconds per request (measured live)
- Throughput: ~1 req/sec (measured live)
- Cost: ~$0.01/request (estimated for EC2 CPU)

**Bottleneck: Where Time is Spent**

| Component        | Time (est) | %   |
|------------------|------------|-----|
| Model Loading    | 1-2s       | 30% |
| Embedding        | 0.5-1s     | 20% |
| LLM Inference    | 0.8-1.5s   | 40% |
| Retrieval        | 0.1-0.3s   | 10% |

**Biggest problem:** LLM inference is slow on CPU (measured live).
**Root cause:** Small model + no acceleration.
**Solution:** Larger model + GPU acceleration (expected to be much faster).

---

### Production Solution

**Infrastructure Changes:**
- GPU Deployment: A100 GPU (80GB VRAM), vLLM for inference optimization, Redis for caching + batching
- Model Stack: Mistral-7B or Llama-2-13B, LoRA fine-tuning on recruitment data, int8 quantization
- Serving: vLLM for auto-batching, FastAPI with async queuing, Load balancing (multiple GPUs)

**Performance Improvements:**

| Metric      | Current           | Production (est)     | Improvement (est)      |
|-------------|-------------------|----------------------|------------------------|
| Latency     | 2-5s (measured)   | 100-200ms (expected) | 20-50x faster (expected)|
| Throughput  | 1 req/s (measured)| 100+ req/s (expected)| 100x faster (expected) |
| Token Cost  | $0.01 (estimated) | $0.0001 (estimated)  | 100x cheaper (expected)|
| Accuracy    | Baseline          | +25% (if fine-tuned) | Better output (expected)|

---

### Implementation Roadmap

**Phase 1: Validate (Done)**
- RAG pipeline architecture
- Semantic matching working
- Template approach reliable
- LLM integration validated
- API endpoints functional

**Result:** Confident in approach, identified bottlenecks

**Phase 2: Optimize (Next)**
- Fine-tune LLM on recruitment data
	1. Collect 500-1000 CV-JD-coverletter triplets
	2. Create LoRA adapter (low-rank updates)
	3. Fine-tune Mistral-7B for 2-3 hours on A100
	4. Evaluate on held-out test set
	5. Expected: +20-30% quality improvement
- Infrastructure: AWS SageMaker (pay-per-hour), Total cost: ~$50-100 for fine-tuning
- Deliverable: Fine-tuned model weights + eval metrics

**Phase 3: Deploy (Final)**
- Production deployment
	1. Package fine-tuned model + vLLM config
	2. Deploy on AWS SageMaker Endpoint
	3. Set up auto-scaling (0-10 replicas)
	4. Add monitoring + alerting
	5. Set up cost controls
- Infrastructure: A100 on-demand: $3-4/hour per replica, Auto-scale based on queue depth, Cost: ~$100-300/month at 100k req/month
- API: Same FastAPI interface, just faster + better

---

### System Design & Scaling: 2025 Best Practices

**1. System Architecture for Scale**
- API Layer: FastAPI (async) behind a load balancer (AWS ALB/GCP LB)
- Model Serving: vLLM or Triton Inference Server for batching and GPU efficiency
- Vector Store: Qdrant or Pinecone for scalable, low-latency semantic search
- Caching: Redis for hot queries and embedding reuse
- Monitoring: Prometheus + Grafana
- Deployment: Kubernetes (EKS/GKE) for auto-scaling

**2. Expected Performance & Cost (late 2025)**
- Model: Mistral-7B or Llama-2-13B, quantized (int8)
- Hardware: NVIDIA A100 (80GB) or H100
- Latency: RAG-only: 150–300ms/request (batching, GPU, vector DB); RAG+LLM: 300–700ms/request (prompt length, batch size)
- Throughput: 100–500 req/sec per GPU (with batching)
- Cost: A100 on-demand: ~$3/hr (AWS/GCP); 1M req/month ≈ $150–$300 infra (1–2 GPUs, managed vector DB); Token cost (OpenAI API): $0.0005–$0.002/request; Self-hosted: Only infra cost

**3. Fine-Tuning vs. RAG**
- RAG: Fast, cheap, robust for structured/fact-based tasks
- Fine-Tuning (LoRA/QLoRA/DPO): Use for nuanced, personalized, or domain-specific generation; LoRA/QLoRA for parameter-efficient fine-tuning; DPO for aligning with recruiter preferences; Cost: $100–$500 per run (A100, 2–4 hours)

**4. Example Production Stack**
- API: FastAPI (async, autoscaled)
- Model Serving: vLLM (A100/H100, batching, quantized)
- Vector DB: Qdrant (managed)
- Cache: Redis
- Monitoring: Prometheus, Grafana, Sentry
- CI/CD: GitHub Actions, Docker, Kubernetes

**5. API-based Models as an Alternative**

API-based LLMs (OpenAI GPT-4/4o, Anthropic Claude 3, etc.)

- Strengths: State-of-the-art language understanding and generation; No infrastructure or MLOps required—just call the API; Instantly scalable, always up-to-date; Best for nuanced, context-rich, or premium features
- Considerations: Higher cost at scale (e.g., $1,000+ per 1M requests); Data privacy: sensitive data sent to third party unless using private endpoints; Limited fine-tuning/customization (prompt engineering and RAG are main tools); Slightly higher latency (500–1000ms typical)
- When to use: When you need the highest quality out-of-the-box; For rapid prototyping, premium features, or when infra is not available; If you lack the data/resources for fine-tuning

**6. Summary Table**

| Approach             | Latency (ms) | Cost (per 1M req) | Personalization | Scaling      |
|----------------------|--------------|-------------------|----------------|-------------|
| RAG-only             | 150–300      | $150–$300         | Low–Medium     | Easy        |
| RAG + Fine-tuned     | 300–700      | $200–$500         | High           | Easy        |
| API-based (OpenAI)   | 500–1000     | $1,000+           | High           | Unlimited   |

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
