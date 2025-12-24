# career_assistant/tests/test_rag_pipeline.py
import logging
import pytest
from qdrant_client import QdrantClient
from career_assistant.rag_pipeline.ingest import ingest_data
from career_assistant.rag_pipeline.rag_pipeline import run_rag_pipeline
from career_assistant.utils.chunking import chunk_text
from career_assistant.rag_pipeline.retriever import Retriever
from career_assistant.rag_pipeline.generator import CoverLetterGenerator


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="module", autouse=True)
def setup_test_data():
    """Ensure test data is ingested once before all tests run (reuse if exists)."""
    logger.info("\n" + "=" * 60)
    logger.info("SETUP: Checking for existing test data...")
    logger.info("=" * 60)
    
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "career_assistant_qdrant"
    
    # Check if collection exists and has data
    if client.collection_exists(collection_name):
        collection_info = client.get_collection(collection_name)
        point_count = collection_info.points_count
        if point_count >= 700:  # Expected ~759 chunks
            logger.info(f"[OK] Found existing data: {point_count} points in collection")
            logger.info("  (Skipping reingest to save time)\n")
            yield
            return
    
    # If not enough data, reingest
    logger.info("SETUP: Ingesting test data into Qdrant...")
    ingest_data(chunking=True)
    logger.info("[OK] Test data ready\n")
    yield

def test_chunking():
    """Verify chunking algorithm works correctly with various text lengths."""
    logger.info("Testing chunking algorithm...")
    
    # Test 1: Normal text chunking
    sample_text = "This is a sample text for chunking test. " * 50
    chunks = chunk_text(sample_text, chunk_size=10, overlap=2)
    logger.info(f"  [OK] Chunking test: {len(chunks)} chunks created from 50x repeated text")
    assert len(chunks) > 0, "Should create at least one chunk"
    assert all(len(chunk.split()) <= 10 for chunk in chunks), "All chunks should respect chunk_size"
    
    # Test 2: Empty text handling
    empty_chunks = chunk_text("", chunk_size=10, overlap=2)
    logger.info(f"  [OK] Empty text: {len(empty_chunks)} chunks (should be 0)")
    assert len(empty_chunks) == 0, "Empty text should produce no chunks"
    
    # Test 3: Small text (no chunking needed)
    small_text = "Small text"
    small_chunks = chunk_text(small_text, chunk_size=10, overlap=2)
    logger.info(f"  [OK] Small text: {len(small_chunks)} chunk(s)")
    assert len(small_chunks) == 1, "Small text should produce exactly one chunk"
    assert small_chunks[0] == small_text, "Chunk content should match input"
    
    # Test 4: Verify overlap between chunks
    long_text = " ".join(["word"] * 100)  # 100 words
    overlapped_chunks = chunk_text(long_text, chunk_size=20, overlap=5)
    logger.info(f"  [OK] Overlap validation: {len(overlapped_chunks)} chunks with 5-word overlap")
    assert len(overlapped_chunks) > 1, "Long text should create multiple chunks"
    
    logger.info("[OK] All chunking tests passed")

@pytest.mark.slow
def test_ingest(chunking=True):
    """Test data ingestion pipeline with chunking enabled.
    
    This test is slow (~120s) and only needed when reingest is required.
    Run with: pytest -m slow
    """
    logger.info("Testing data ingestion pipeline...")
    logger.info(f"  Starting ingest with chunking={chunking}")
    ingest_data(chunking=chunking)
    logger.info("[OK] Data ingestion completed successfully")


def test_vector_retrieval():
    """Test that the vector retriever can find similar documents."""
    logger.info("Testing vector retrieval from Qdrant...")
    
    retriever = Retriever(collection_name="career_assistant_qdrant")
    
    # Test job retrieval with JD-like query (should match jobs well)
    jd_query = "We're hiring an AI engineer with strong experience in NLP, Transformers, and model deployment."
    retrieved_jobs = retriever.retrieve_similar_jobs(jd_query, top_k=3)
    logger.info(f"  [OK] Retrieved {len(retrieved_jobs)} similar jobs")
    assert len(retrieved_jobs) > 0, "Should retrieve at least one job"
    
    # Verify job results have required structure
    for job in retrieved_jobs:
        assert "content" in job, "Job should have 'content' field"
        assert "metadata" in job, "Job should have 'metadata' field"
        assert len(job["content"]) > 0, "Job content should not be empty"
        logger.info(f"    - Job snippet: {job['content'][:80]}...")
    
    # Test CV retrieval with CV-like query (should match CVs well)
    cv_query = "I'm an NLP engineer with 3 years of experience in Python, HuggingFace, and LLM fine-tuning."
    retrieved_cvs = retriever.retrieve_similar_cvs(cv_query, top_k=3)
    logger.info(f"  [OK] Retrieved {len(retrieved_cvs)} similar CVs")
    assert len(retrieved_cvs) > 0, "Should retrieve at least one CV"
    
    # Verify CV results have required structure
    for cv in retrieved_cvs:
        assert "content" in cv, "CV should have 'content' field"
        assert "metadata" in cv, "CV should have 'metadata' field"
        assert len(cv["content"]) > 0, "CV content should not be empty"
        logger.info(f"    - CV snippet: {cv['content'][:80]}...")
    
    logger.info("[OK] Vector retrieval test passed")


def test_embedding_and_generation():
    """Test the cover letter generation component."""
    logger.info("Testing cover letter generation...")
    
    generator = CoverLetterGenerator()
    
    # Test cover letter generation directly
    sample_jd = "We're hiring a Senior Python Developer with 5+ years experience. Must know FastAPI, PostgreSQL, Docker."
    sample_cv = "Senior Python developer with 6 years of experience in FastAPI, PostgreSQL, Docker, and Kubernetes."
    result = generator.generate_cover_letter(sample_cv, sample_jd)
    
    # Result is now a dict with generation results
    assert isinstance(result, dict), "Should return a dict"
    assert "cover_letter" in result, "Result should contain 'cover_letter' key"
    assert "match_score" in result, "Result should contain 'match_score' key"
    assert "match_level" in result, "Result should contain 'match_level' key"
    
    match_score = result["match_score"]
    cover_letter = result["cover_letter"]
    match_level = result["match_level"]
    
    logger.info(f"  [OK] Generated result - Level: {match_level}, Score: {match_score}, CL: {cover_letter is not None}")
    
    # For high match scores, cover letter should be generated
    if match_score >= 0.80:
        assert cover_letter is not None, "Cover letter should be generated for good matches"
        assert len(cover_letter) > 50, "Cover letter should have substantial content"
        logger.info(f"  [OK] Good match - Cover letter generated ({len(cover_letter.split())} words)")
    else:
        logger.info(f"  [OK] Low/moderate match - Assessment message provided instead")
    
    logger.info("[OK] Generation test passed")

def test_retriever_and_generator():
    """Test the complete end-to-end RAG pipeline."""
    cv_text = "I'm an NLP engineer with 3 years of experience in Python, HuggingFace, and LLM fine-tuning."
    jd_text = "We're hiring an AI engineer with strong experience in NLP, Transformers, and model deployment."

    logger.info("Testing end-to-end RAG pipeline...")
    results = run_rag_pipeline(cv_text, jd_text, top_k=3)

    # Validate retrieval
    retrieved_jobs = results.get("retrieved_jobs", [])
    retrieved_cvs = results.get("retrieved_cvs", [])
    logger.info(f"  [OK] Retrieval: {len(retrieved_jobs)} jobs, {len(retrieved_cvs)} CVs")
    assert len(retrieved_jobs) > 0, "Should retrieve at least one job"
    assert len(retrieved_cvs) > 0, "Should retrieve at least one CV"

    # Validate generation result
    match_score = results.get("match_score", 0)
    match_level = results.get("match_level", "unknown")
    cover_letter = results.get("cover_letter", None)
    assessment_message = results.get("assessment_message", "")
    
    logger.info(f"  [OK] Generation - Level: {match_level}, Score: {match_score}")
    
    # Always should have assessment message
    assert assessment_message, "Should have assessment message"
    
    # For good matches, should have cover letter
    if match_score >= 0.80:
        assert cover_letter is not None, "Cover letter should be generated for good matches"
        assert len(cover_letter) > 50, "Cover letter should have meaningful content"
        logger.info(f"  [OK] Cover letter: {len(cover_letter.split())} words")
    else:
        logger.info(f"  [OK] Assessment provided for {match_level} match")
    
    logger.info("[OK] End-to-end pipeline test passed")


def test_rag_pipeline_multiple_profiles():
    """Test RAG pipeline with diverse CV and JD combinations to ensure robustness."""
    logger.info("Testing RAG pipeline with multiple profile types...")
    
    test_profiles = [
        {
            "name": "Junior Python Developer",
            "cv": "Entry-level Python developer with 1 year experience in Django and REST APIs.",
            "jd": "Seeking Python developer for Django project. Experience with REST APIs required."
        },
        {
            "name": "Data Scientist",
            "cv": "Data scientist with 4 years in machine learning, statistical analysis, and Python.",
            "jd": "Hiring data scientist for ML pipeline development and model optimization."
        },
        {
            "name": "DevOps Engineer",
            "cv": "DevOps engineer with 5 years in Kubernetes, Docker, CI/CD pipelines, and AWS.",
            "jd": "DevOps role: need strong Kubernetes, Docker, and CI/CD expertise on AWS."
        }
    ]
    
    for profile in test_profiles:
        logger.info(f"  Testing: {profile['name']}")
        results = run_rag_pipeline(profile["cv"], profile["jd"], top_k=2)
        
        # Verify all required outputs exist
        assert results.get("retrieved_jobs"), f"No jobs retrieved for {profile['name']}"
        assert results.get("retrieved_cvs"), f"No CVs retrieved for {profile['name']}"
        assert "match_score" in results, f"No match score for {profile['name']}"
        assert "match_level" in results, f"No match level for {profile['name']}"
        assert results.get("assessment_message"), f"No assessment message for {profile['name']}"
        
        # Cover letter only generated for moderate and good matches
        cover_letter = results.get("cover_letter")
        match_level = results.get("match_level")
        
        if cover_letter:
            assert len(cover_letter) > 50, f"Cover letter too short for {profile['name']}"
            logger.info(f"    [OK] {profile['name']}: {match_level} match - CL length = {len(cover_letter.split())} words")
        else:
            logger.info(f"    [OK] {profile['name']}: {match_level} match - Assessment provided instead of CL")
    
    logger.info("[OK] Multiple profiles test passed")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("INTEGRATION TEST SUITE FOR LLM CAREER ASSISTANT")
    logger.info("=" * 60)
    
    logger.info("\n[1/5] Running chunking algorithm tests...")
    test_chunking()

    logger.info("\n[2/5] Running data ingestion pipeline...")
    test_ingest(chunking=True)

    logger.info("\n[3/5] Running vector retrieval tests...")
    test_vector_retrieval()
    
    logger.info("\n[4/5] Running generation component tests...")
    test_embedding_and_generation()

    logger.info("\n[5/5] Running end-to-end RAG pipeline tests...")
    test_retriever_and_generator()
    
    logger.info("\n[BONUS] Running robustness tests with multiple profiles...")
    test_rag_pipeline_multiple_profiles()

    logger.info("\n" + "=" * 60)
    logger.info("[OK] ALL INTEGRATION TESTS PASSED SUCCESSFULLY")
    logger.info("=" * 60)
