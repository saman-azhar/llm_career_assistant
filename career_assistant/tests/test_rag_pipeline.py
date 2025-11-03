from career_assistant.main import run_full_pipeline

def test_rag_pipeline():
    cv_text = """Experienced NLP Engineer skilled in Python, Transformers, FastAPI, and vector databases."""
    job_description = """Looking for a Machine Learning Engineer with experience in NLP, LLMs, FastAPI, and Qdrant."""

    # Run pipeline
    result = run_full_pipeline(cv_text, job_description)

    # Ensure we got a result dictionary
    assert isinstance(result, dict), "Pipeline did not return a dictionary"

    print("\n=== DECISION ===")
    print(result.get("decision", "No decision field — check main.py output structure"))

    print("\n=== MATCHED SKILLS ===")
    print(result.get("matched_skills", []))

    print("\n=== MISSING SKILLS ===")
    print(result.get("missing_skills", []))

    print("\n=== GENERATED COVER LETTER ===")
    print(result.get("cover_letter", "No cover letter generated"))
