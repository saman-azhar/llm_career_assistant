# career_assistant/rag_pipeline/embedder.py
from langchain_huggingface import HuggingFaceEmbeddings


class Embedder:
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        """
        Wrapper around Hugging Face embedding model.
        Uses the same model as VectorStore for consistency.
        """
        self.model_name = model_name
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)

    def embed_query(self, text: str):
        """Return a vector representation of a single text input."""
        if not text or not text.strip():
            raise ValueError("Input text for embedding cannot be empty.")
        return self.embedding_model.embed_query(text)

    def embed_documents(self, texts: list[str]):
        """Batch embed multiple documents."""
        if not texts:
            raise ValueError("Input text list cannot be empty.")
        return self.embedding_model.embed_documents(texts)


def main():
    """Reserved for future standalone execution or testing."""
    pass


if __name__ == "__main__":
    main()