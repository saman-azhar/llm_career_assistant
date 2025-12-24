from career_assistant.preprocessing.semantic_matching import compute_similarity_runtime
from career_assistant.rag_pipeline.rag_pipeline import run_rag_pipeline
from career_assistant.utils.logger import get_logger
from career_assistant.mlflow_logger import start_run, log_params, log_metrics

logger = get_logger(__name__)

def evaluate_cv_job(cv_text: str, jd_text: str, match_threshold: float = 0.65):
    """
    Synchronous CV vs JD evaluation with MLflow and console logging.
    Can be called from async endpoints if needed via run_in_executor.
    """
    logger.info("Starting CV-JD evaluation")
    logger.debug(f"CV length: {len(cv_text)}, JD length: {len(jd_text)}, Threshold: {match_threshold}")

    with start_run(run_name="evaluate_cv_job") as run:
        log_params({
            "cv_length": len(cv_text),
            "jd_length": len(jd_text),
            "match_threshold": match_threshold
        })

        # Step 1: Compute similarity
        sim_result = compute_similarity_runtime(cv_text, jd_text)
        similarity_score = sim_result['similarity_score']
        matched_skills = sim_result['matched_skills']
        missing_skills = sim_result['missing_skills']

        log_metrics({
            "similarity_score": similarity_score,
            "num_matched_skills": len(matched_skills),
            "num_missing_skills": len(missing_skills)
        })

        # Step 2: Decide verdict and generate cover letter
        cover_letter = ""
        if similarity_score >= match_threshold:
            verdict = "Good match"
            try:
                cover_letter_data = run_rag_pipeline(cv_text, jd_text)
                cover_letter = cover_letter_data.get('cover_letter', "")
                logger.info("Cover letter generated successfully")
            except Exception as e:
                cover_letter = f"Error generating cover letter: {str(e)}"
                logger.error(f"Cover letter generation failed: {e}")
        else:
            verdict = "Poor match"
            logger.info(f"CV-JD match below threshold ({similarity_score:.2f})")

        result = {
            "match_score": similarity_score,
            "verdict": verdict,
            "missing_skills": missing_skills,
            "matched_skills": matched_skills,
            "cover_letter": cover_letter
        }

        logger.debug(f"Evaluation result: {result}")
        return result
