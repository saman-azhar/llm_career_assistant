from fastapi import APIRouter
from career_assistant.api.dependencies import logger

router = APIRouter()

@router.get("/health")
async def health():
    logger.info("Health check called")
    return {"status": "ok"}
