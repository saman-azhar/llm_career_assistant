from career_assistant.preprocessing.semantic_matching import compute_similarity_runtime
from career_assistant.rag_pipeline.rag_pipeline import run_rag_pipeline

def evaluate_cv_job(cv_text: str, jd_text: str, match_threshold: float = 0.65):
    """
    Evaluate CV against JD, generate verdict and cover letter (if match is good)
    """
    # Step 1: Compute semantic similarity + missing skills
    sim_result = compute_similarity_runtime(cv_text, jd_text)

    similarity_score = sim_result['similarity_score']
    matched_skills = sim_result['matched_skills']
    missing_skills = sim_result['missing_skills']

    # Step 2: Decide verdict
    if similarity_score >= match_threshold:
        verdict = "Good match"
        # Generate cover letter highlighting strengths
        cover_letter_data = run_rag_pipeline(cv_text, jd_text)
        cover_letter = cover_letter_data['cover_letter']
    else:
        verdict = "Poor match"
        cover_letter = None

    return {
        "match_score": similarity_score,
        "verdict": verdict,
        "missing_skills": missing_skills,
        "matched_skills": matched_skills,
        "cover_letter": cover_letter
    }
