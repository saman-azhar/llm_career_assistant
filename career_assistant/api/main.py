from fastapi import FastAPI
from career_assistant.api.schemas import MatchRequest, MatchResponse
from career_assistant.api.utils import evaluate_cv_job

app = FastAPI(title="CV-JD Match Agent", version="1.0")

@app.post("/match", response_model=MatchResponse)
def match_cv_jd(payload: MatchRequest):
    """
    Input your CV and a Job Description (JD). Returns:
    - semantic match score
    - verdict (good/poor match)
    - missing skills (if any)
    - cover letter (if match is good)
    """
    result = evaluate_cv_job(payload.cv_text, payload.jd_text)
    return result

@app.get("/")
def root():
    return {"message": "CV-JD Match Agent running. POST /match with CV and JD."}
