# career_assistant/tests/test_rag_pipeline.py
import logging
from career_assistant.rag_pipeline.ingest import ingest_data
from career_assistant.rag_pipeline.rag_pipeline import run_rag_pipeline
from career_assistant.utils.chunking import chunk_text


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def test_chunking():
    sample_text = "This is a sample text for chunking test. " * 50  # long text
    chunks = chunk_text(sample_text, chunk_size=10, overlap=2)
    logger.info(f"Chunking test: {len(chunks)} chunks created.")
    assert len(chunks) > 0
    assert all(len(chunk.split()) <= 10 for chunk in chunks)

def test_ingest(chunking=True):
    logger.info("Running ingest test (small dataset or mocked data recommended)...")
    ingest_data(chunking=chunking)
    logger.info("Ingest completed.")

def test_retriever_and_generator():
    cv_text = "I’m an NLP engineer with 3 years of experience in Python, HuggingFace, and LLM fine-tuning."
    jd_text = "We’re hiring an AI engineer with strong experience in NLP, Transformers, and model deployment."

    logger.info("Running RAG pipeline end-to-end...")
    results = run_rag_pipeline(cv_text, jd_text, top_k=3)

    # Check retrieval
    retrieved_jobs = results.get("retrieved_jobs", [])
    retrieved_cvs = results.get("retrieved_cvs", [])
    logger.info(f"Retrieved {len(retrieved_jobs)} jobs and {len(retrieved_cvs)} CVs")
    assert len(retrieved_jobs) > 0
    assert len(retrieved_cvs) > 0

    # Check cover letter generation
    cover_letter = results.get("cover_letter", "")
    logger.info(f"Generated cover letter length (words): {len(cover_letter.split())}")
    assert len(cover_letter.split()) > 0

if __name__ == "__main__":
    logger.info("=== Running chunking test ===")
    test_chunking()

    logger.info("=== Running ingest test ===")
    test_ingest(chunking=True)

    logger.info("=== Running RAG pipeline test ===")
    test_retriever_and_generator()

    logger.info("All tests completed successfully.")
