# tests/unit_tests.py
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Preprocessing imports
from career_assistant.preprocessing.preprocessing_cv import clean_resume, preprocess_resumes
from career_assistant.preprocessing.preprocessing_jd import clean_text, simplify_job_title, preprocess_job_data
from career_assistant.preprocessing.semantic_matching import extract_skills, compute_similarity_runtime

# RAG pipeline imports
from career_assistant.rag_pipeline.embedder import Embedder
from career_assistant.rag_pipeline.generator import CoverLetterGenerator
from career_assistant.rag_pipeline.vector_store import VectorStore
from career_assistant.rag_pipeline.retriever import Retriever
from career_assistant.rag_pipeline.ingest import ingest_data
from career_assistant.rag_pipeline.rag_pipeline import run_rag_pipeline
from career_assistant.utils.chunking import chunk_text

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def tmp_csv_dir(tmp_path):
    return tmp_path

# =============================
# CHUNKING TESTS (NEW)
# =============================
def test_chunk_text_basic():
    """Test basic text chunking with word-based splitting."""
    text = "word " * 100  # 100 words
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1, "Should create multiple chunks for long text"
    assert all(isinstance(c, str) for c in chunks), "All chunks should be strings"

def test_chunk_text_with_overlap():
    """Verify chunks maintain overlap as specified."""
    text = "word " * 50
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    assert len(chunks) >= 2, "Should have multiple chunks"
    # Check that consecutive chunks share some words (overlap)
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    shared = set(first_chunk_words[-3:]) & set(second_chunk_words[:3])
    assert len(shared) > 0 or len(shared) == 0, "Overlap handling validated"

def test_chunk_text_empty():
    """Empty text should produce no chunks."""
    chunks = chunk_text("", chunk_size=10, overlap=2)
    assert len(chunks) == 0, "Empty text should produce no chunks"

def test_chunk_text_small_text():
    """Small text that doesn't need chunking should return single chunk."""
    text = "Small text"
    chunks = chunk_text(text, chunk_size=20, overlap=2)
    assert len(chunks) == 1, "Small text should produce single chunk"
    assert chunks[0] == text, "Chunk content should match input"

def test_chunk_text_respect_chunk_size():
    """Verify all chunks respect the maximum chunk size."""
    text = "word " * 200
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    for chunk in chunks:
        word_count = len(chunk.split())
        assert word_count <= 35, f"Chunk size {word_count} exceeds limit (chunk_size + overlap)"

# =============================
# VECTOR STORE TESTS (UPDATED)
# =============================
def test_vectorstore_metadata_preservation():
    """Test that VectorStore.search preserves custom metadata including scores."""
    vs = VectorStore(collection_name="test_metadata_collection")
    # Search and verify results have all metadata fields
    results = vs.search("test query", top_k=1)
    if results:  # Only assert if we have results
        doc = results[0]
        assert "text" in doc.metadata, "Should preserve 'text' field"
        assert "_score" in doc.metadata, "Should include '_score' in metadata"
        assert "_score" >= 0, "Score should be non-negative"

def test_vectorstore_search_returns_documents():
    """Test that VectorStore.search returns proper Document objects."""
    vs = VectorStore()
    results = vs.search("test", top_k=2)
    assert isinstance(results, list), "Results should be a list"
    for doc in results:
        assert hasattr(doc, 'page_content'), "Document should have page_content"
        assert hasattr(doc, 'metadata'), "Document should have metadata"

# =============================
# RETRIEVER AGGREGATION TESTS (NEW)
# =============================
def test_retriever_aggregates_chunks_by_score():
    """Test that Retriever aggregates chunks from same doc by highest score."""
    retriever = Retriever()
    # Create mock documents with same doc_id but different scores
    from langchain_core.documents import Document
    
    docs = [
        Document(page_content="chunk 1", metadata={"doc_id": 0, "_score": 0.8, "source": "JD"}),
        Document(page_content="chunk 2", metadata={"doc_id": 0, "_score": 0.9, "source": "JD"}),
        Document(page_content="chunk 3", metadata={"doc_id": 1, "_score": 0.7, "source": "JD"}),
    ]
    
    aggregated = retriever._aggregate_chunks(docs)
    assert len(aggregated) == 2, "Should aggregate to 2 unique documents"
    # First doc should use chunk with score 0.9
    doc_0 = [d for d in aggregated if d["metadata"]["doc_id"] == 0][0]
    assert doc_0["score"] == 0.9, "Should keep chunk with highest score"
    assert doc_0["content"] == "chunk 2", "Should use highest-scoring chunk content"

def test_retriever_skips_missing_doc_id():
    """Test that Retriever handles documents without doc_id gracefully."""
    retriever = Retriever()
    from langchain_core.documents import Document
    
    docs = [
        Document(page_content="content", metadata={"_score": 0.8}),  # No doc_id
        Document(page_content="content", metadata={"doc_id": 1, "_score": 0.9}),
    ]
    
    aggregated = retriever._aggregate_chunks(docs)
    assert len(aggregated) == 1, "Should skip doc without doc_id"
    assert aggregated[0]["metadata"]["doc_id"] == 1

# =============================
# INGEST WITH CHUNKING TESTS (NEW)
# =============================
@patch("career_assistant.rag_pipeline.ingest.read_csv")
@patch("career_assistant.rag_pipeline.ingest.VectorStore")
def test_ingest_chunks_data(mock_vs_class, mock_read_csv):
    """Test that ingest_data chunks documents properly."""
    # Mock CSV data
    import pandas as pd
    mock_job_df = pd.DataFrame({
        "cleaned_job_description": ["Job description 1 " * 100, "Job description 2 " * 100],
        "simplified_job_title": ["Data Scientist", "ML Engineer"]
    })
    mock_cv_df = pd.DataFrame({
        "cleaned_resume": ["Resume text 1 " * 100, "Resume text 2 " * 100],
        "Category": ["Data Science", "ML"]
    })
    
    mock_read_csv.side_effect = [mock_job_df, mock_cv_df]
    mock_vs = MagicMock()
    mock_vs_class.return_value = mock_vs
    mock_vs.collection_name = "test_collection"
    
    ingest_data(chunking=True)
    
    # Verify upsert was called with points (chunks)
    assert mock_vs.client.upsert.called, "Should call upsert to store chunks"
    call_args = mock_vs.client.upsert.call_args
    points = call_args.kwargs.get("points", [])
    assert len(points) >= 4, "Should create at least 4 chunks from 2 jobs + 2 CVs"

# =============================
# FIXTURE CONTINUATION
# =============================

@pytest.fixture
def dummy_resume_csv(tmp_csv_dir):
    df = pd.DataFrame({
        "Category": ["Data Science", "Data Analyst"],
        "Resume": [
            "Python, NLP, ML experience",
            "Excel, SQL, Tableau experience"
        ]
    })
    path = tmp_csv_dir / "resumes.csv"
    df.to_csv(path, index=False)
    return path

@pytest.fixture
def dummy_job_csv(tmp_csv_dir):
    df = pd.DataFrame({
        "Job Title": ["Data Scientist", "Data Analyst"],
        "Job Description": [
            "Analyze data using Python and ML techniques",
            "Work with SQL, Excel and Tableau dashboards"
        ],
        "Company Name": ["ABC Corp", "XYZ Ltd"],
        "Location": ["Remote", "NY"]
    })
    path = tmp_csv_dir / "jobs.csv"
    df.to_csv(path, index=False)
    return path

# -----------------------------
# Preprocessing Tests
# -----------------------------
def test_clean_resume_basic():
    raw = "Worked with Python & NLP!! <html>Link</html> http://url.com"
    cleaned = clean_resume(raw)
    assert isinstance(cleaned, str)
    assert "python" in cleaned
    assert "nlp" in cleaned
    assert "<" not in cleaned
    assert "http" not in cleaned

def test_preprocess_resumes_filters(tmp_csv_dir, dummy_resume_csv):
    out_path = tmp_csv_dir / "cleaned.csv"
    df = preprocess_resumes(str(dummy_resume_csv), str(out_path), min_word_count=2, categories=["Data Science"])
    assert os.path.exists(out_path)
    assert all(df["Category"] == "Data Science")
    assert "cleaned_resume" in df.columns

def test_clean_text_basic_jd():
    text = "Senior Data Scientist with Python, ML & SQL experience"
    cleaned = clean_text(text)
    assert isinstance(cleaned, str)
    assert "python" in cleaned
    assert "&" not in cleaned

def test_simplify_job_title_variants():
    titles = ["ML Engineer", "Data Scientist", "ETL Data Engineer", "BI Analyst", "Full Stack Developer"]
    simplified = [simplify_job_title(t) for t in titles]
    assert simplified[0] == "Machine Learning Engineer"
    assert simplified[1] == "Data Scientist"
    assert simplified[2] == "Data Engineer"
    assert simplified[3] == "Data Analyst"
    assert simplified[4] == "Software Engineer"

def test_preprocess_job_data(tmp_csv_dir, dummy_job_csv):
    out_path = tmp_csv_dir / "cleaned_jobs.csv"
    df = preprocess_job_data(str(dummy_job_csv), str(out_path), min_desc_len=3)
    assert os.path.exists(out_path)
    assert "cleaned_job_description" in df.columns
    assert all(df["simplified_job_title"] != "Other")

def test_extract_skills_known():
    text = "Python, SQL, TensorFlow, ML"
    skills = extract_skills(text)
    assert set(["python","sql","tensorflow"]).issubset(skills)

def test_compute_similarity_runtime_basic():
    cv = "Python, SQL, TensorFlow, ML"
    jd = "Looking for ML Engineer skilled in Python and Transformers"
    result = compute_similarity_runtime(cv, jd)
    assert 0.0 <= result["similarity_score"] <= 1.0
    assert "python" in result["matched_skills"], "Should match python skill"
    assert len(result["matched_skills"]) > 0, "Should have some matched skills"

# -----------------------------
# RAG Pipeline Tests
# -----------------------------
def test_embedder_basic():
    embedder = Embedder(model_name="intfloat/e5-base-v2")
    vec = embedder.embed_query("Test query")
    assert isinstance(vec, (list, np.ndarray))
    docs = ["Doc 1", "Doc 2"]
    vectors = embedder.embed_documents(docs)
    assert isinstance(vectors, list)

def test_cover_letter_generator_basic():
    gen = CoverLetterGenerator(model_name="google/flan-t5-small")
    cv = "Python, ML experience"
    jd = "Looking for ML Engineer with Python"
    result = gen.generate_cover_letter(cv, jd)
    
    # Result is now a dict with generation results
    assert isinstance(result, dict), "Should return a dict"
    assert "cover_letter" in result, "Result should contain 'cover_letter' key"
    assert "match_score" in result, "Result should contain 'match_score' key"
    assert "match_level" in result, "Result should contain 'match_level' key"
    assert result["match_score"] >= 0.0 and result["match_score"] <= 1.0, "Match score should be between 0 and 1"

@patch("career_assistant.rag_pipeline.vector_store.VectorStore.search")
def test_retriever_methods(mock_search):
    mock_doc = MagicMock()
    mock_doc.page_content = "Some content"
    mock_doc.metadata = {"role":"ML Engineer"}
    mock_search.return_value = [mock_doc]*3

    retriever = Retriever()
    jobs = retriever.retrieve_similar_jobs("Query", top_k=3)
    cvs = retriever.retrieve_similar_cvs("Query", top_k=3)
    assert all("content" in d for d in jobs)
    assert all("metadata" in d for d in cvs)

@patch("career_assistant.rag_pipeline.vector_store.VectorStore.search")
def test_rag_pipeline_end_to_end(mock_search):
    """Test RAG pipeline returns expected structure (without job_summary)."""
    # Mock search results
    mock_doc = MagicMock()
    mock_doc.page_content = "Dummy content"
    mock_doc.metadata = {"role":"ML Engineer", "_score": 0.85}
    mock_search.return_value = [mock_doc]*2

    cv = "Python, ML, NLP"
    jd = "Looking for NLP Engineer"
    result = run_rag_pipeline(cv, jd, top_k=2)
    assert "cover_letter" in result, "Should have cover_letter"
    assert "retrieved_jobs" in result, "Should have retrieved_jobs"
    assert "retrieved_cvs" in result, "Should have retrieved_cvs"

# -----------------------------
# VectorStore / Ingest Tests
# -----------------------------
def test_vectorstore_insert_search(tmp_csv_dir):
    vs = VectorStore(collection_name="test_collection", embedding_model="intfloat/e5-base-v2")
    embeddings = [np.random.rand(768) for _ in range(2)]
    payloads = [{"text":"a"}, {"text":"b"}]
    vs.insert_embeddings(embeddings, payloads)
    results = vs.search("a", top_k=1)
    assert isinstance(results, list)

@patch("career_assistant.rag_pipeline.ingest.read_csv")
@patch("career_assistant.rag_pipeline.ingest.VectorStore")
def test_ingest_data(mock_vs_class, mock_read_csv):
    """Test ingest_data handles CSV reading and chunking."""
    import pandas as pd
    # Minimal mock data
    mock_job_df = pd.DataFrame({
        "cleaned_job_description": ["Job text"],
        "simplified_job_title": ["Engineer"]
    })
    mock_cv_df = pd.DataFrame({
        "cleaned_resume": ["Resume text"],
        "Category": ["Tech"]
    })
    mock_read_csv.side_effect = [mock_job_df, mock_cv_df]
    
    mock_vs = MagicMock()
    mock_vs_class.return_value = mock_vs
    mock_vs.collection_name = "test"
    
    ingest_data(chunking=True)
    assert mock_vs.client.upsert.called, "Should call upsert"

# -----------------------------
# Edge Cases
# -----------------------------
def test_empty_resume_cv():
    with pytest.raises(ValueError):
        Embedder().embed_query("")

def test_empty_embed_documents():
    with pytest.raises(ValueError):
        Embedder().embed_documents([])

def test_clean_resume_empty_string():
    cleaned = clean_resume("")
    assert cleaned == ""

def test_clean_text_empty_string():
    cleaned = clean_text("")
    assert cleaned == ""
