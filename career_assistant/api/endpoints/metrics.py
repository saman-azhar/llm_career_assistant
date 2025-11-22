from fastapi import APIRouter
from career_assistant.api.dependencies import logger

router = APIRouter()

@router.get("/metrics")
async def metrics():
    # Ideally integrate Prometheus/FastAPI metrics here
    logger.info("Metrics endpoint called")
    return {"active_requests": 5, "uptime_seconds": 3600}  # placeholders
