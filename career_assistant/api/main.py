from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from career_assistant.api.utils import evaluate_cv_job, generate_cover_letter
from qdrant_client import QdrantClient
import os

app = FastAPI(title="Career Assistant API")

# Qdrant setup
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Pydantic schemas for input/output
class MatchRequest(BaseModel):
    cv_text: str
    jd_text: str

class MatchResponse(BaseModel):
    match_score: float
    verdict: str
    missing_skills: list[str]
    cover_letter: str | None = None

# Threshold to decide if cover letter should be generated
COVER_LETTER_THRESHOLD = 0.6

@app.post("/match", response_model=MatchResponse)
def match_cv_jd(payload: MatchRequest):
    """
    Input your CV and a Job Description (JD). Returns:
    - semantic match score
    - verdict (good/poor match)
    - missing skills (if any)
    - cover letter (if match is good)
    """
    try:
        # Evaluate CV-JD alignment
        result = evaluate_cv_job(payload.cv_text, payload.jd_text)
        match_score = result.get("match_score", 0.0)
        verdict = "good match" if match_score >= COVER_LETTER_THRESHOLD else "poor match"
        missing_skills = result.get("missing_skills", [])

        # Generate cover letter if match is good
        cover_letter = None
        if match_score >= COVER_LETTER_THRESHOLD:
            cover_letter = generate_cover_letter(
                cv_text=payload.cv_text,
                jd_text=payload.jd_text,
                match_score=match_score,
                missing_skills=missing_skills
            )

        return MatchResponse(
            match_score=match_score,
            verdict=verdict,
            missing_skills=missing_skills,
            cover_letter=cover_letter
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Career Assistant API running!"}


@app.get("/collections")
def list_collections():
    return client.get_collections()
