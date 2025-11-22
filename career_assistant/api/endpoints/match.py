from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from career_assistant.api.utils import evaluate_cv_job
from career_assistant.rag_pipeline.generator import CoverLetterGenerator
from career_assistant.utils.logger import get_logger
from career_assistant.config import Config
import os

env = os.getenv("ENVIRONMENT", "dev")
config = Config(env=env)
logger = get_logger(__name__, level=config.get("logging.level"))

router = APIRouter(prefix="/match", tags=["match"])

generator_cfg = config.get("generator")
cover_letter_generator = CoverLetterGenerator(model_name=generator_cfg.get("model_name"))

COVER_LETTER_THRESHOLD = config.get("semantic_matching.similarity_threshold", 0.6)

class MatchRequest(BaseModel):
    cv_text: str
    jd_text: str

class MatchResponse(BaseModel):
    match_score: float
    verdict: str
    missing_skills: list[str]
    cover_letter: str | None = None

@router.post("/", response_model=MatchResponse)
def match_cv_jd(payload: MatchRequest):
    try:
        result = evaluate_cv_job(payload.cv_text, payload.jd_text)
        match_score = result.get("match_score", 0.0)
        verdict = "good match" if match_score >= COVER_LETTER_THRESHOLD else "poor match"
        missing_skills = result.get("missing_skills", [])

        cover_letter = None
        if match_score >= COVER_LETTER_THRESHOLD:
            cover_letter = cover_letter_generator.generate_cover_letter(
                cv_text=payload.cv_text,
                jd_text=payload.jd_text
            )

        return MatchResponse(
            match_score=match_score,
            verdict=verdict,
            missing_skills=missing_skills,
            cover_letter=cover_letter
        )

    except Exception as e:
        logger.error(f"Error in /match endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
