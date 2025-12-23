"""Comparison endpoint: LLM vs Template generators side-by-side."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from career_assistant.rag_pipeline.generator import CoverLetterGenerator
from career_assistant.rag_pipeline.generator_template import CoverLetterGeneratorTemplate
from career_assistant.utils.logger import get_logger
from career_assistant.utils.config import Config
import time
import os

env = os.getenv("ENVIRONMENT", "dev")
config = Config(env=env)
logger = get_logger(__name__, level=config.get("logging.level"))

router = APIRouter(prefix="/compare", tags=["compare"])

class CompareRequest(BaseModel):
    cv_text: str
    jd_text: str

class GeneratorOutput(BaseModel):
    match_score: float
    match_level: str
    assessment_message: str
    matched_skills: list[str]
    missing_skills: list[str]
    cover_letter: str | None
    inference_time_ms: float

class CompareResponse(BaseModel):
    llm_output: GeneratorOutput
    template_output: GeneratorOutput
    comparison: dict

@router.post("/", response_model=CompareResponse)
def compare_generators(payload: CompareRequest):
    """
    Compare LLM-based and template-based generators side-by-side.
    
    Returns outputs from both approaches with inference timing metrics.
    Useful for demonstrating production (template) vs. cutting-edge (LLM) approaches.
    """
    try:
        # Initialize generators
        llm_gen = CoverLetterGenerator(log_mlflow=False)
        template_gen = CoverLetterGeneratorTemplate(log_mlflow=False)
        
        # LLM-based generation with timing
        start_time = time.time()
        llm_result = llm_gen.generate_cover_letter(payload.cv_text, payload.jd_text)
        llm_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Template-based generation with timing
        start_time = time.time()
        template_result = template_gen.generate_cover_letter(payload.cv_text, payload.jd_text)
        template_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Package LLM output
        llm_output = GeneratorOutput(
            match_score=llm_result.get("match_score", 0.0),
            match_level=llm_result.get("match_level", "unknown"),
            assessment_message=llm_result.get("message", ""),
            matched_skills=llm_result.get("matched_skills", []),
            missing_skills=llm_result.get("missing_skills", []),
            cover_letter=llm_result.get("cover_letter"),
            inference_time_ms=llm_time
        )
        
        # Package template output
        template_output = GeneratorOutput(
            match_score=template_result.get("match_score", 0.0),
            match_level=template_result.get("match_level", "unknown"),
            assessment_message=template_result.get("message", ""),
            matched_skills=template_result.get("matched_skills", []),
            missing_skills=template_result.get("missing_skills", []),
            cover_letter=template_result.get("cover_letter"),
            inference_time_ms=template_time
        )
        
        # Comparison metrics
        comparison = {
            "speed_advantage": f"Template is {llm_time / template_time:.1f}x faster",
            "llm_inference_ms": round(llm_time, 2),
            "template_inference_ms": round(template_time, 2),
            "match_level_agreement": llm_output.match_level == template_output.match_level,
            "match_score_diff": abs(llm_output.match_score - template_output.match_score),
            "skill_coverage_agreement": len(set(llm_output.matched_skills) & set(template_output.matched_skills)) / max(len(set(llm_output.matched_skills) | set(template_output.matched_skills)), 1)
        }
        
        logger.info(f"Comparison complete - LLM: {llm_time:.2f}ms, Template: {template_time:.2f}ms")
        
        return CompareResponse(
            llm_output=llm_output,
            template_output=template_output,
            comparison=comparison
        )
        
    except Exception as e:
        logger.error(f"Error in /compare endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
