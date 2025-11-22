import os
from functools import lru_cache
from qdrant_client import QdrantClient
from career_assistant.utils.logger import get_logger

# Qdrant client
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Logger
logger = get_logger("api")

# Optional: cache decorator for expensive calls
@lru_cache(maxsize=128)
def cached_call(key: str):
    """Simple in-memory cache placeholder"""
    return None
