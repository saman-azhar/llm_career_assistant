from career_assistant.preprocessing.semantic_matching import compute_similarity_runtime
from career_assistant.rag_pipeline.generator import CoverLetterGenerator
from career_assistant.utils.logger import get_logger

logger = get_logger(__name__)

MAX_TEXT_LEN = 4000  # truncate inputs to prevent generator overflow

def run_full_pipeline(cv_text, job_description):
    # Step 0: Truncate long inputs
    cv_text = cv_text[:MAX_TEXT_LEN]
    job_description = job_description[:MAX_TEXT_LEN]

    # Step 1: Semantic Matching
    try:
        match_results = compute_similarity_runtime(cv_text, job_description)
        score = match_results["similarity_score"]
        matched_skills = match_results["matched_skills"]
        missing_skills = match_results["missing_skills"]
        logger.info(f"Semantic matching done. Score: {score:.2f}")
    except Exception as e:
        logger.error(f"Error in semantic matching: {e}")
        score = 0.0
        matched_skills = []
        missing_skills = []

    # Step 2: Generate Cover Letter
    try:
        generator = CoverLetterGenerator()
        cover_letter = generator.generate_cover_letter(cv_text, job_description)
        logger.info("Cover letter generated successfully.")
    except Exception as e:
        logger.error(f"Error generating cover letter: {e}")
        cover_letter = ""

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


if __name__ == "__main__":
    # Sample test run
    cv_text = "Experienced NLP Engineer skilled in Python, Transformers, FastAPI, and vector databases."
    jd_text = "Looking for a Machine Learning Engineer with experience in NLP, LLMs, FastAPI, and Qdrant."

    result = run_full_pipeline(cv_text, jd_text)

    logger.info("---- Decision ----")
    logger.info(result["decision"])
    logger.info("---- Matched Skills ----")
    logger.info(result["matched_skills"])
    logger.info("---- Missing Skills ----")
    logger.info(result["missing_skills"])
    logger.info("---- Generated Cover Letter ----")
    logger.info(result["cover_letter"])
