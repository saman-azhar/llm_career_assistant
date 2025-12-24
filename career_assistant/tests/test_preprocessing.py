import os
import pandas as pd
import pytest
from unittest.mock import patch

from career_assistant.preprocessing.preprocessing_cv import clean_resume, preprocess_resumes
from career_assistant.preprocessing.preprocessing_jd import clean_text, simplify_job_title, preprocess_job_data
from career_assistant.preprocessing.semantic_matching import extract_skills, compute_similarity_runtime


@pytest.fixture(scope="module")
def tmp_csv_dir(tmp_path_factory):
    """Fixture for temporary CSV file directory."""
    return tmp_path_factory.mktemp("data")


# -----------------------------
# Resume Preprocessing Tests
# -----------------------------

def test_clean_resume_removes_noise():
    raw_text = "Experienced AI Engineer!! Worked with Python, FastAPI, and NLP. https://example.com"
    cleaned = clean_resume(raw_text)

    print("\n=== CLEANED RESUME SAMPLE ===")
    print(cleaned)

    assert "http" not in cleaned
    assert "engineer" in cleaned
    assert cleaned.islower()
    assert isinstance(cleaned, str)


def test_clean_resume_empty_string():
    cleaned = clean_resume("")
    assert cleaned == ""


def test_preprocess_resumes(tmp_csv_dir):
    df = pd.DataFrame({
        "Category": ["Data Science", "Marketing"],
        "Resume": [
            "Python developer with experience in NLP, Transformers, and ML projects.",
            "Marketing specialist with campaigns and advertising experience."
        ]
    })
    input_csv = tmp_csv_dir / "input_resumes.csv"
    output_csv = tmp_csv_dir / "cleaned_resumes.csv"
    df.to_csv(input_csv, index=False)

    with patch("career_assistant.preprocessing.preprocessing_cv.log_params") as mock_log_params, \
        patch("career_assistant.preprocessing.preprocessing_cv.log_metrics") as mock_log_metrics:
        result = preprocess_resumes(str(input_csv), str(output_csv), min_word_count=3, categories=["Data Science"])
        mock_log_params.assert_called()

    print("\n=== PREPROCESSED RESUME DF ===")
    print(result.head())

    assert os.path.exists(output_csv)
    assert "cleaned_resume" in result.columns
    assert all(result["Category"] == "Data Science")


# -----------------------------
# Job Description Preprocessing Tests
# -----------------------------

def test_clean_text_basic():
    text = "We need a Machine Learning Engineer to build models using Python & TensorFlow!"
    cleaned = clean_text(text)

    print("\n=== CLEANED JD TEXT ===")
    print(cleaned)

    assert "python" in cleaned
    assert "machine" in cleaned
    assert "!" not in cleaned
    assert isinstance(cleaned, str)


def test_clean_text_empty():
    cleaned = clean_text("")
    assert cleaned == ""


def test_simplify_job_title_variants():
    titles = [
        "Senior ML Engineer",
        "Data Scientist II",
        "ETL Data Engineer",
        "BI Analyst",
        "Full Stack Developer",
        "Research Associate"
    ]
    simplified = [simplify_job_title(t) for t in titles]

    print("\n=== SIMPLIFIED TITLES ===")
    print(simplified)

    assert simplified[0] == "Machine Learning Engineer"
    assert simplified[1] == "Data Scientist"
    assert simplified[2] == "Data Engineer"
    assert simplified[3] == "Data Analyst"
    assert simplified[4] == "Software Engineer"
    assert simplified[-1] == "Other"


def test_preprocess_job_data(tmp_csv_dir):
    df = pd.DataFrame({
        "Job Title": ["Data Scientist", "Marketing Manager"],
        "Job Description": [
            "Analyze data using Python, R, and ML techniques.",
            "Manage brand awareness campaigns and social media."
        ],
        "Company Name": ["TechCorp", "Marketly"],
        "Location": ["Remote", "NY"]
    })
    input_csv = tmp_csv_dir / "input_jobs.csv"
    output_csv = tmp_csv_dir / "cleaned_jobs.csv"
    df.to_csv(input_csv, index=False)

    with patch("career_assistant.preprocessing.preprocessing_jd.log_params") as mock_log_params:
        result = preprocess_job_data(str(input_csv), str(output_csv), min_desc_len=3)
        mock_log_params.assert_called()


    print("\n=== PREPROCESSED JOB DF ===")
    print(result.head())

    assert os.path.exists(output_csv)
    assert "cleaned_job_description" in result.columns
    assert all(result["simplified_job_title"] != "Other")


# -----------------------------
# Semantic Matching Tests
# -----------------------------

def test_extract_skills_detects_keywords():
    text = "Proficient in Python, SQL, and TensorFlow for machine learning."
    skills = extract_skills(text)

    print("\n=== EXTRACTED SKILLS ===")
    print(skills)

    assert isinstance(skills, list)
    assert "python" in skills
    assert "tensorflow" in skills


def test_extract_skills_no_known_skills():
    skills = extract_skills("Marketing and sales experience")
    assert skills == []


def test_compute_similarity_runtime_logic():
    cv = "Python, FastAPI, Transformers, and NLP experience"
    jd = "Looking for an NLP Engineer skilled in Python and model deployment"
    result = compute_similarity_runtime(cv, jd)

    print("\n=== SIMILARITY RESULT ===")
    print(result)

    assert 0.0 <= result["similarity_score"] <= 1.0
    assert isinstance(result["matched_skills"], list)
    assert isinstance(result["missing_skills"], list)
    assert "python" in result["matched_skills"]


def test_compute_similarity_runtime_empty_cv_jd():
    result = compute_similarity_runtime("", "")
    assert result["similarity_score"] >= 0.0
    assert result["matched_skills"] == []
    assert result["missing_skills"] == []

