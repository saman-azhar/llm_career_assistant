from career_assistant.preprocessing.semantic_matching import compute_similarity_runtime
from career_assistant.rag_pipeline.generator import CoverLetterGenerator

def run_full_pipeline(cv_text, job_description):
    # Step 1: Semantic Matching
    match_results = compute_similarity_runtime(cv_text, job_description)
    score = match_results["similarity_score"]
    matched_skills = match_results["matched_skills"]
    missing_skills = match_results["missing_skills"]

    # Step 2: Generate Cover Letter
    generator = CoverLetterGenerator()
    job_summary = generator.summarize_job(job_description)
    cover_letter = generator.generate_cover_letter(cv_text, job_summary)

    # Step 3: Interpret Match Quality
    if score >= 0.7:
        decision = f"Match Score: {score*100:.1f}% — Strong fit. You should apply."
    elif score >= 0.5:
        decision = f"Match Score: {score*100:.1f}% — Somewhat relevant. Might be worth a shot."
    else:
        decision = f"Match Score: {score*100:.1f}% — Weak fit. Missing key skills: {', '.join(missing_skills[:5])}."

    result = {
        "decision": decision,
        "similarity_score": score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "cover_letter": cover_letter
    }

    return result
