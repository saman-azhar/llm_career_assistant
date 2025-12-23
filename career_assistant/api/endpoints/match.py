from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from career_assistant.api.utils import evaluate_cv_job
from career_assistant.rag_pipeline.generator_template import CoverLetterGeneratorTemplate
from career_assistant.utils.logger import get_logger
from career_assistant.utils.config import Config
import os

env = os.getenv("ENVIRONMENT", "dev")
config = Config(env=env)
logger = get_logger(__name__, level=config.get("logging.level"))

router = APIRouter(prefix="/match", tags=["match"])

cover_letter_generator = CoverLetterGeneratorTemplate()

COVER_LETTER_THRESHOLD = config.get("semantic_matching.similarity_threshold", 0.6)

class MatchRequest(BaseModel):
    cv_text: str
    jd_text: str

class MatchResponse(BaseModel):
    match_score: float
    verdict: str
    matched_skills: list[str]
    missing_skills: list[str]
    assessment_message: str
    cover_letter: str | None = None

@router.post("/", response_model=MatchResponse)
def match_cv_jd(payload: MatchRequest):
    try:
        # Use template-based generator for ideal output
        result = cover_letter_generator.generate_cover_letter(payload.cv_text, payload.jd_text)
        match_score = result.get("match_score", 0.0)
        match_level = result.get("match_level", "unknown")
        verdict = "good match" if match_level == "good" else ("moderate match" if match_level == "moderate" else "poor match")
        matched_skills = result.get("matched_skills", [])
        missing_skills = result.get("missing_skills", [])
        assessment_message = result.get("message", "")
        cover_letter = result.get("cover_letter")

        return MatchResponse(
            match_score=match_score,
            verdict=verdict,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            assessment_message=assessment_message,
            cover_letter=cover_letter
        )

    except Exception as e:
        logger.error(f"Error in /match endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
