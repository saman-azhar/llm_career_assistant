# career_assistant/utils/chunking.py
from typing import List
from itertools import islice
from career_assistant.utils.logger import get_logger

logger = get_logger(__name__)

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 70) -> List[str]:
    """
    Split text into overlapping chunks for RAG ingestion.
    
    Args:
        text: Full text to split
        chunk_size: number of tokens/words per chunk
        overlap: number of tokens/words to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move with overlap

    logger.info(f"Text split into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})")
    return chunks
