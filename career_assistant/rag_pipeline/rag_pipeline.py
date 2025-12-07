# career_assistant/rag_pipeline/rag_pipeline.py
"""
RAG-based cover letter generation pipeline.

Supports two generator approaches:
1. LLM-based (generator.py) - Uses transformer models
2. Template-based (generator_template.py) - Uses intelligent templates (ideal result)

Both use the same RAG retrieval and semantic matching logic.
Difference is only in the cover letter generation step.
"""

from typing import Literal
from career_assistant.rag_pipeline.retriever import Retriever
from career_assistant.rag_pipeline.generator import CoverLetterGenerator
from career_assistant.rag_pipeline.generator_template import CoverLetterGeneratorTemplate
from career_assistant.mlflow_logger import start_run, log_params, log_metrics
from career_assistant.utils.logger import get_logger
from career_assistant.utils.config import Config

import mlflow

config = Config(env="dev").load_yaml_config()
generator_cfg = config.get("generator", {})

mlflow_cfg = config.get("mlflow", {})

mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
mlflow.set_experiment(mlflow_cfg["experiment_name"])

logger = get_logger(__name__)


def run_rag_pipeline(cv_text: str, jd_text: str, top_k: int = 5, max_chunks: int = 2, 
                     generator_type: Literal["llm", "template"] = "llm"):
    """
    RAG pipeline for cover letter generation.
    
    Args:
        cv_text: Candidate CV text
        jd_text: Job description text
        top_k: Number of chunks to retrieve
        max_chunks: Maximum chunks to combine
        generator_type: "llm" for transformer-based, "template" for template-based
        
    Returns:
        Dictionary with match assessment and cover letter
    """
    if not cv_text.strip() or not jd_text.strip():
        logger.warning("Empty CV or job description text provided to RAG pipeline")
        raise ValueError("CV text and job description text cannot be empty.")

    logger.info(f"Running RAG pipeline (approach: {generator_type}, CV length: {len(cv_text)}, JD length: {len(jd_text)}, top_k: {top_k}, max_chunks: {max_chunks})")
    with start_run(run_name=f"rag_pipeline_run_{generator_type}") as run_id:
        log_params({
            "cv_text_length": len(cv_text),
            "jd_text_length": len(jd_text),
            "top_k": top_k,
            "max_chunks": max_chunks,
            "generator_type": generator_type
        })

        # Step 1: Retrieve chunks from vector DB
        retriever = Retriever(collection_name="career_assistant_qdrant")
        similar_job_chunks = retriever.retrieve_similar_jobs(jd_text, top_k=top_k)
        similar_cv_chunks = retriever.retrieve_similar_cvs(cv_text, top_k=top_k)
        logger.info(f"Retrieved {len(similar_job_chunks)} job chunks and {len(similar_cv_chunks)} CV chunks")

        # Fallback: Use input text if retrieval returns no results
        if not similar_job_chunks:
            logger.warning("No job chunks retrieved; using input JD text as fallback")
            similar_job_chunks = [{"content": jd_text}]
        
        if not similar_cv_chunks:
            logger.warning("No CV chunks retrieved; using input CV text as fallback")
            similar_cv_chunks = [{"content": cv_text}]

        # Only take top max_chunks chunks
        jd_combined = " ".join([chunk["content"] for chunk in similar_job_chunks[:max_chunks]])
        cv_combined = " ".join([chunk["content"] for chunk in similar_cv_chunks[:max_chunks]])

        # Step 2: Generate cover letter using specified approach
        if generator_type == "template":
            logger.info("Using TEMPLATE-BASED generator (ideal result)")
            generator = CoverLetterGeneratorTemplate(log_mlflow=True)
        else:
            logger.info("Using LLM-BASED generator (transformer model)")
            generator = CoverLetterGenerator(log_mlflow=True)
            
        generation_result = generator.generate_cover_letter(cv_combined, jd_combined)
        
        logger.info(f"Generation result: level={generation_result['match_level']}, score={generation_result['match_score']}")
        log_metrics({
            "match_score": generation_result["match_score"],
            "cover_letter_generated": 1 if generation_result["cover_letter"] else 0
        })

        # Step 3: Compile results
        results = {
            "approach": generator_type,
            "match_score": generation_result["match_score"],
            "match_level": generation_result["match_level"],
            "assessment_message": generation_result["message"],
            "matched_skills": generation_result["matched_skills"],
            "missing_skills": generation_result["missing_skills"],
            "cover_letter": generation_result["cover_letter"],
            "retrieved_jobs": similar_job_chunks[:max_chunks],
            "retrieved_cvs": similar_cv_chunks[:max_chunks]
        }

        return results


def main():
    """Quick local testing with both approaches."""
    cv_text = "I'm an NLP engineer with 3 years of experience in Python, HuggingFace, and LLM fine-tuning."
    jd_text = "We're hiring an AI engineer with strong experience in NLP, Transformers, and model deployment."

    logger.info("\n" + "="*80)
    logger.info("COMPARISON: LLM vs TEMPLATE APPROACHES")
    logger.info("="*80)
    
    # Test LLM approach
    logger.info("\n[1] LLM-BASED APPROACH (Transformer Model)")
    logger.info("-"*80)
    results_llm = run_rag_pipeline(cv_text, jd_text, generator_type="llm")
    logger.info(f"Match Level: {results_llm['match_level']} ({results_llm['match_score']})")
    if results_llm["cover_letter"]:
        logger.info("Cover Letter: Generated")
        logger.info(results_llm["cover_letter"][:200] + "...")
    else:
        logger.info("Cover Letter: Not generated (poor match)")
    
    # Test Template approach
    logger.info("\n[2] TEMPLATE-BASED APPROACH (Ideal Result)")
    logger.info("-"*80)
    results_template = run_rag_pipeline(cv_text, jd_text, generator_type="template")
    logger.info(f"Match Level: {results_template['match_level']} ({results_template['match_score']})")
    if results_template["cover_letter"]:
        logger.info("Cover Letter: Generated")
        logger.info(results_template["cover_letter"][:200] + "...")
    else:
        logger.info("Cover Letter: Not generated (poor match)")
    
    logger.info("\n" + "="*80)
    logger.info("USAGE EXAMPLES:")
    logger.info("  # LLM-based (transformer model)")
    logger.info("  result = run_rag_pipeline(cv, jd, generator_type='llm')")
    logger.info("  # Template-based (ideal result)")
    logger.info("  result = run_rag_pipeline(cv, jd, generator_type='template')")
    logger.info("="*80)



if __name__ == "__main__":
    main()
