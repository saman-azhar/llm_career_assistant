# test_rag_pipeline.py
from career_assistant.rag_pipeline.rag_pipeline import run_rag_pipeline

def main():
    # Example inputs (can be from your cleaned CSVs)
    cv_text = """
    Experienced NLP Engineer with 3 years in Python, Transformers, and FastAPI.
    Worked on LLM-based resume parsing, semantic search, and cover letter generation systems.
    """
    
    jd_text = """
    We are seeking a Machine Learning Engineer experienced in NLP, LLMs, and API deployment.
    Must have hands-on experience with vector databases like Qdrant and model serving with FastAPI.
    """

    # Run your RAG pipeline
    result = run_rag_pipeline(cv_text=cv_text, jd_text=jd_text)

    print("\n=== RAG OUTPUT ===")
    print("Cover Letter:\n", result.get("cover_letter", "N/A"))
    print("\nInsights:\n", result.get("insights", "N/A"))
    print("\nRetrieved Context:\n", result.get("retrieved_context", []))

if __name__ == "__main__":
    main()
