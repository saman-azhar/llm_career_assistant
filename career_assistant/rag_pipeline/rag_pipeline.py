# career_assistant/rag_pipeline/rag_pipeline.py
from career_assistant.rag_pipeline.retriever import Retriever
from career_assistant.rag_pipeline.generator import CoverLetterGenerator
from career_assistant.mlflow_logger import start_run, log_params, log_metrics
from career_assistant.utils.logger import get_logger
from career_assistant.utils.config import Config

import mlflow

config = Config(env="dev").load_yaml_config()

mlflow_cfg = config.get("mlflow", {})

mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
mlflow.set_experiment(mlflow_cfg["experiment_name"])


logger = get_logger(__name__)


def run_rag_pipeline(cv_text: str, jd_text: str, top_k: int = 5, max_chunks: int = 3):
    if not cv_text.strip() or not jd_text.strip():
        logger.warning("Empty CV or job description text provided to RAG pipeline")
        raise ValueError("CV text and job description text cannot be empty.")

    logger.info(f"Running RAG pipeline (CV length: {len(cv_text)}, JD length: {len(jd_text)}, top_k: {top_k})")
    with start_run(run_name="rag_pipeline_run") as run_id:
        log_params({
            "cv_text_length": len(cv_text),
            "jd_text_length": len(jd_text),
            "top_k": top_k,
            "max_chunks": max_chunks
        })

        # Step 1: Retrieve chunks from vector DB
        retriever = Retriever(collection_name="career_assistant")
        similar_job_chunks = retriever.retrieve_similar_jobs(jd_text, top_k=top_k)
        similar_cv_chunks = retriever.retrieve_similar_cvs(cv_text, top_k=top_k)
        logger.info(f"Retrieved {len(similar_job_chunks)} job chunks and {len(similar_cv_chunks)} CV chunks")

        # Only take top max_chunks chunks to avoid token overflow
        jd_combined = " ".join([chunk["content"] for chunk in similar_job_chunks[:max_chunks]])
        cv_combined = " ".join([chunk["content"] for chunk in similar_cv_chunks[:max_chunks]])

        # Step 2: Summarize the job (if summarize_job implemented)
        generator = CoverLetterGenerator()
        try:
            job_summary = generator.summarize_job(jd_combined)
        except AttributeError:
            job_summary = jd_combined

        # Step 3: Generate cover letter
        cover_letter = generator.generate_cover_letter(cv_combined, job_summary)
        logger.info(f"Generated cover letter (words: {len(cover_letter.split())})")
        log_metrics({"cover_letter_length": len(cover_letter.split())})

        # Step 4: Compile results
        results = {
            "retrieved_jobs": similar_job_chunks[:max_chunks],
            "retrieved_cvs": similar_cv_chunks[:max_chunks],
            "job_summary": job_summary,
            "cover_letter": cover_letter
        }

        return results


def main():
    """Quick local testing."""
    cv_text = "I’m an NLP engineer with 3 years of experience in Python, HuggingFace, and LLM fine-tuning."
    jd_text = "We’re hiring an AI engineer with strong experience in NLP, Transformers, and model deployment."

    results = run_rag_pipeline(cv_text, jd_text)
    logger.info("---- Job Summary ----")
    logger.info(results["job_summary"])
    logger.info("---- Cover Letter ----")
    logger.info(results["cover_letter"])


if __name__ == "__main__":
    main()
