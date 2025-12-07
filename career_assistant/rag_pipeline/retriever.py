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
            # Get score from metadata (stored by VectorStore.search)
            score = doc.metadata.get("_score", 0)
            # Keep the chunk with highest similarity score
            if doc_id not in aggregated or score > aggregated[doc_id]["score"]:
                aggregated[doc_id] = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
        return list(aggregated.values())

    def retrieve_similar_jobs(self, query_text: str, top_k: int = 5):
        """Return the most similar job descriptions (aggregated from chunks) for a given query text."""
        with start_run(run_name="retrieve_similar_jobs") as run_id:
            log_params({"query_length": len(query_text), "top_k": top_k})
            # Get more results to ensure we have enough of each source
            results = self.vs.search(query_text, top_k=top_k*5)
            
            # Filter for JD (job description) source
            jd_results = [r for r in results if r.metadata.get("source") == "JD"][:top_k]
            
            log_metrics({"num_results": len(jd_results)})
            aggregated_results = self._aggregate_chunks(jd_results)
            logger.info(f"Aggregated {len(aggregated_results)} unique job documents from {len(jd_results)} chunks")
            return aggregated_results

    def retrieve_similar_cvs(self, query_text: str, top_k: int = 5):
        """Return the most similar CVs (aggregated from chunks) for a given query text."""
        with start_run(run_name="retrieve_similar_cvs") as run_id:
            log_params({"query_length": len(query_text), "top_k": top_k})
            # Get more results to ensure we have enough of each source
            results = self.vs.search(query_text, top_k=top_k*5)
            
            # Filter for CV source
            cv_results = [r for r in results if r.metadata.get("source") == "CV"][:top_k]
            
            log_metrics({"num_results": len(cv_results)})
            aggregated_results = self._aggregate_chunks(cv_results)
            logger.info(f"Aggregated {len(aggregated_results)} unique CV documents from {len(cv_results)} chunks")
            return aggregated_results


def main():
    retriever = Retriever()
    sample_query = "Looking for a data scientist with Python and ML experience."
    jobs = retriever.retrieve_similar_jobs(sample_query)
    cvs = retriever.retrieve_similar_cvs(sample_query)
    logger.info(f"Retrieved {len(jobs)} jobs and {len(cvs)} CVs.")


if __name__ == "__main__":
    main()
