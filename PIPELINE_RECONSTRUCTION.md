# Complete Pipeline Reconstruction Guide

## Quick Overview
The pipeline has 4 main stages:
1. **Job Data Preprocessing** - Clean and prepare job descriptions
2. **Resume Preprocessing** - Clean and prepare CV data
3. **Qdrant Ingestion** - Load processed data into vector database with chunking
4. **RAG Pipeline Testing** - Verify everything works end-to-end

---

## Stage 1: Job Data Preprocessing

**Input:** `career_assistant/data/raw/glassdoor_jobs.csv`
**Output:** `career_assistant/data/processed/cleaned_job_data_final.csv`

```bash
cd /home/samwise/projects/llm_career_assistant
python3 -c "
from career_assistant.preprocessing.preprocessing_jd import preprocess_job_data
preprocess_job_data(
    'career_assistant/data/raw/glassdoor_jobs.csv',
    'career_assistant/data/processed/cleaned_job_data_final.csv'
)
print('✓ Job preprocessing completed')
"
```

---

## Stage 2: Resume Preprocessing

**Input:** `career_assistant/data/raw/UpdatedResumeDataSet.csv`
**Output:** `career_assistant/data/processed/cleaned_resume_data_final.csv`

```bash
cd /home/samwise/projects/llm_career_assistant
python3 -c "
from career_assistant.preprocessing.preprocessing_cv import preprocess_resumes
preprocess_resumes(
    'career_assistant/data/raw/UpdatedResumeDataSet.csv',
    'career_assistant/data/processed/cleaned_resume_data_final.csv'
)
print('✓ Resume preprocessing completed')
"
```

---

## Stage 3: Clear Old Qdrant Collection

**Why:** Ensures fresh start with new data format and chunking logic

```bash
cd /home/samwise/projects/llm_career_assistant
python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
try:
    client.delete_collection('career_assistant_qdrant')
    print('✓ Deleted old Qdrant collection')
except:
    print('Collection may not exist yet')
"
```

---

## Stage 4: Ingest Data into Qdrant

**What it does:**
- Reads preprocessed job descriptions and resumes
- Creates text chunks (size=300, overlap=70) for better semantic search
- Embeds chunks using `intfloat/e5-base-v2` model
- Stores embeddings in Qdrant with metadata

```bash
cd /home/samwise/projects/llm_career_assistant
python3 -c "
from career_assistant.rag_pipeline.ingest import ingest_data
import mlflow
mlflow.end_run()  # Clear any active runs
ingest_data(chunking=True)
print('✓ Data ingestion completed')
"
```

**Expected output:**
- ~982 total chunks ingested
- Points stored in Qdrant with doc_id, role/name, source, chunk_idx, and text

---

## Stage 5: Test the Pipeline

```bash
cd /home/samwise/projects/llm_career_assistant
python3 -m pytest career_assistant/tests/test_rag_pipeline.py -s
```

**Expected test results:**
- `test_chunking` - ✓ PASS
- `test_ingest` - ✓ PASS  
- `test_retriever_and_generator` - ✓ PASS

---

## Running Everything at Once

```bash
# With venv activated:
cd /home/samwise/projects/llm_career_assistant && source /home/samwise/projects/venv/bin/activate && python run_full_pipeline.py
```

Or use the provided script:
```bash
python3 run_full_pipeline.py
```

---

## Troubleshooting

### Problem: "Module not found"
**Solution:** Make sure virtual environment is activated
```bash
source /home/samwise/projects/venv/bin/activate
```

### Problem: "Qdrant connection refused"
**Solution:** Ensure Qdrant is running via Docker
```bash
docker-compose -f docker-compose.dev.yml up -d qdrant
```

### Problem: Tests still fail with 0 results retrieved
**Solution:** Verify data was ingested
```bash
python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
info = client.get_collection('career_assistant_qdrant')
print(f'Points in collection: {info.points_count}')
"
```

### Problem: "Job preprocessing failed"
**Solution:** Verify input files exist
```bash
ls -la career_assistant/data/raw/
```

---

## Key Configuration

- **Embedding Model:** `intfloat/e5-base-v2` (768-dimensional vectors)
- **Chunk Size:** 300 characters
- **Chunk Overlap:** 70 characters
- **Vector Distance Metric:** COSINE similarity
- **Collection Name:** `career_assistant_qdrant`

---

## What Gets Stored in Qdrant

Each chunk is stored as a point with:
- **id**: Numeric ID (auto-incrementing)
- **vector**: 768-dimensional embedding
- **metadata**:
  - `doc_id`: Original document index
  - `source`: "JD" (job description) or "CV" (resume)
  - `role`: Job title or category name
  - `chunk_idx`: Which chunk of the document
  - `text`: The actual text content

---

## Next Steps After Reconstruction

1. **Run Full Tests:**
   ```bash
   pytest career_assistant/tests/test_rag_pipeline.py -s
   ```

2. **Start API Server:**
   ```bash
   uvicorn career_assistant.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Run Streamlit Demo:**
   ```bash
   streamlit run notebooks/cover_letter_generation.py
   ```

4. **View MLflow Dashboard:**
   ```bash
   Open http://localhost:5000 in browser
   ```
