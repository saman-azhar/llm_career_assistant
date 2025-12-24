# career_assistant/rag_pipeline/embedder.py
from langchain_huggingface import HuggingFaceEmbeddings
from career_assistant.mlflow_logger import start_run, log_params, log_metrics
from career_assistant.utils.logger import get_logger
from career_assistant.utils.chunking import chunk_text

logger = get_logger(__name__)

class Embedder:
    def __init__(self, model_name: str = "intfloat/e5-base-v2", log_mlflow: bool = True):
        """
        Wrapper around Hugging Face embedding model.
        Can embed single text, batch texts, or chunked texts.
        """
        self.model_name = model_name
        self.log_mlflow = log_mlflow
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model {self.model_name}: {e}")
            raise

    def embed_query(self, text: str):
        """Return vector for single text input."""
        if not text or not text.strip():
            logger.warning("Received empty text for embedding")
            raise ValueError("Input text for embedding cannot be empty.")
        logger.info(f"Embedding query of length {len(text)}")
        vector = None
        try:
            if self.log_mlflow:
                with start_run(run_name="embed_query") as run:
                    log_params({"model_name": self.model_name, "text_length": len(text)})
                    vector = self.embedding_model.embed_query(text)
                    log_metrics({"embedding_dim": len(vector)})
            else:
                vector = self.embedding_model.embed_query(text)
        except Exception as e:
            logger.error(f"Error during query embedding: {e}")
            raise
        logger.info(f"Query embedding completed, vector dim: {len(vector)}")
        return vector

    def embed_documents(self, texts: list[str]):
        """Batch embed multiple documents."""
        if not texts:
            logger.warning("Received empty list for document embedding")
            raise ValueError("Input text list cannot be empty.")
        logger.info(f"Embedding batch of {len(texts)} documents")
        vectors = None
        try:
            if self.log_mlflow:
                with start_run(run_name="embed_documents") as run:
                    log_params({"model_name": self.model_name, "num_documents": len(texts)})
                    vectors = self.embedding_model.embed_documents(texts)
                    log_metrics({
                        "num_documents": len(texts),
                        "embedding_dim": len(vectors[0]) if vectors else 0
                    })
            else:
                vectors = self.embedding_model.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error during batch document embedding: {e}")
            raise
        logger.info(f"Document embedding completed, first vector dim: {len(vectors[0]) if vectors else 0}")
        return vectors

    def embed_chunked_text(self, text: str, chunk_size: int = 300, overlap: int = 70):
        """
        Split a long text into chunks and embed each chunk.
        Returns a list of embeddings and chunk texts.
        """
        if not text or not text.strip():
            return [], []

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        logger.info(f"Embedding {len(chunks)} chunks for text length {len(text)}")
        embeddings = self.embed_documents(chunks)
        return embeddings, chunks


def main():
    """Standalone test."""
    logger.info("Embedder module executed directly. No operations performed.")

if __name__ == "__main__":
    main()
