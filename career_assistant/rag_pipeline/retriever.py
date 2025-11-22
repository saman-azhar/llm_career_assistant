# career_assistant/rag_pipeline/retriever.py
from career_assistant.rag_pipeline.vector_store import VectorStore
from career_assistant.mlflow_logger import start_run, log_params, log_metrics
from career_assistant.utils.logger import get_logger

logger = get_logger(__name__)

class Retriever:
    def __init__(self, collection_name="career_assistant"):
        """Initialize retriever that wraps around Qdrant via LangChain."""
        self.vs = VectorStore(collection_name=collection_name)

    def _aggregate_chunks(self, results):
        """Aggregate multiple chunks of the same document by highest score."""
        aggregated = {}
        for doc in results:
            doc_id = doc.metadata.get("doc_id")
            if doc_id is None:
                logger.warning(f"Doc without 'doc_id' found in results: {doc.metadata}")
                continue
            # Keep the chunk with highest similarity score
            if doc_id not in aggregated or doc.score > aggregated[doc_id]["score"]:
                aggregated[doc_id] = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.score
                }
        return list(aggregated.values())

    def retrieve_similar_jobs(self, query_text: str, top_k: int = 5):
        """Return the most similar job descriptions (aggregated from chunks) for a given query text."""
        with start_run(run_name="retrieve_similar_jobs") as run_id:
            log_params({"query_length": len(query_text), "top_k": top_k})
            results = self.vs.search(query_text, top_k=top_k)
            log_metrics({"num_results": len(results)})
            aggregated_results = self._aggregate_chunks(results)
            logger.info(f"Aggregated {len(aggregated_results)} unique job documents from {len(results)} chunks")
            return aggregated_results

    def retrieve_similar_cvs(self, query_text: str, top_k: int = 5):
        """Return the most similar CVs (aggregated from chunks) for a given query text."""
        with start_run(run_name="retrieve_similar_cvs") as run_id:
            log_params({"query_length": len(query_text), "top_k": top_k})
            results = self.vs.search(query_text, top_k=top_k)
            log_metrics({"num_results": len(results)})
            aggregated_results = self._aggregate_chunks(results)
            logger.info(f"Aggregated {len(aggregated_results)} unique CV documents from {len(results)} chunks")
            return aggregated_results


def main():
    retriever = Retriever()
    sample_query = "Looking for a data scientist with Python and ML experience."
    jobs = retriever.retrieve_similar_jobs(sample_query)
    cvs = retriever.retrieve_similar_cvs(sample_query)
    logger.info(f"Retrieved {len(jobs)} jobs and {len(cvs)} CVs.")


if __name__ == "__main__":
    main()
