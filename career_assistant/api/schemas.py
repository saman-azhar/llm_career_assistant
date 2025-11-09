from pydantic import BaseModel
from typing import List, Optional

class MatchRequest(BaseModel):
    cv_text: str
    jd_text: str

class MatchResponse(BaseModel):
    match_score: float
    verdict: str
    missing_skills: Optional[List[str]]
    matched_skills: Optional[List[str]]
    cover_letter: Optional[str]