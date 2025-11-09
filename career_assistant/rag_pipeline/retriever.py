from career_assistant.rag_pipeline.vector_store import VectorStore


class Retriever:
    def __init__(self, collection_name="career_assistant"):
        """Initialize retriever that wraps around Qdrant via LangChain."""
        self.vs = VectorStore(collection_name=collection_name)

    def retrieve_similar_jobs(self, query_text: str, top_k: int = 5):
        """Return the most similar job descriptions for a given query text."""
        results = self.vs.search(query_text, top_k=top_k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]

    def retrieve_similar_cvs(self, query_text: str, top_k: int = 5):
        """Return the most similar CVs for a given query text."""
        results = self.vs.search(query_text, top_k=top_k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]


def main():
    pass


if __name__ == "__main__":
    main()
