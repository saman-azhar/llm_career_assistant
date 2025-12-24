from fastapi import APIRouter
from career_assistant.api.dependencies import client, logger

router = APIRouter()

@router.get("/collections")
async def list_collections():
    logger.info("Listing Qdrant collections")
    return client.get_collections()
