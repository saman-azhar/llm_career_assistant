import pytest
from unittest.mock import patch
from career_assistant.main import run_full_pipeline


@pytest.mark.slow
def test_rag_pipeline_basic():
    """Test full RAG pipeline end-to-end with sample CV and JD."""
    cv_text = "Experienced NLP Engineer skilled in Python, Transformers, FastAPI, and vector databases."
    job_description = "Looking for a Machine Learning Engineer with experience in NLP, LLMs, FastAPI, and Qdrant."

    # Patch MLflow logging so we don't actually hit MLflow during tests
    with patch("career_assistant.mlflow_logger.log_params") as mock_log_params, \
         patch("career_assistant.mlflow_logger.log_metrics") as mock_log_metrics:

        result = run_full_pipeline(cv_text, job_description)

        # MLflow logging should be called
        mock_log_params.assert_called()
        mock_log_metrics.assert_called()

    # Basic structure checks
    assert isinstance(result, dict), "Pipeline did not return a dictionary"
    assert "cover_letter" in result
    assert "matched_skills" in result
    assert "missing_skills" in result
    assert "decision" in result

    print("\n=== DECISION ===")
    print(result.get("decision"))

    print("\n=== MATCHED SKILLS ===")
    print(result.get("matched_skills"))

    print("\n=== MISSING SKILLS ===")
    print(result.get("missing_skills"))

    print("\n=== GENERATED COVER LETTER ===")
    print(result.get("cover_letter"))


def test_rag_pipeline_empty_inputs():
    """Test pipeline behavior with empty CV and JD inputs."""
    result = run_full_pipeline("", "")

    # Ensure pipeline still returns expected keys
    assert isinstance(result, dict)
    for key in ["cover_letter", "matched_skills", "missing_skills", "decision"]:
        assert key in result

    # Edge case: matched skills should be empty
    assert result["matched_skills"] == []
    # Edge case: missing skills may be empty or all JD skills
    assert isinstance(result["missing_skills"], list)
    # Cover letter should at least return some string
    assert isinstance(result["cover_letter"], str)
