# career_assistant/rag_pipeline/vector_store.py
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_core.documents import Document
from career_assistant.mlflow_logger import start_run, log_params, log_metrics
from career_assistant.utils.chunking import chunk_text

class VectorStore:
    def __init__(self, 
                 host="localhost", 
                 port=6333, 
                 collection_name="career_assistant_qdrant", 
                 embedding_model="intfloat/e5-base-v2"):

        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # LangChain embedding model wrapper
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        # Qdrant client
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            with start_run(run_name="create_qdrant_collection") as run_id:
                log_params({"collection_name": self.collection_name, "vector_size": 768, "distance": "COSINE"})
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                log_metrics({"collection_created": 1})

    def insert_embeddings(self, embeddings, payloads):
        """Manual insert (non-LangChain route)"""
        with start_run(run_name="insert_embeddings") as run_id:
            log_params({"num_embeddings": len(embeddings)})
            points = [
                PointStruct(id=i, vector=embeddings[i].tolist(), payload=payloads[i])
                for i in range(len(embeddings))
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            log_metrics({"num_inserted": len(points)})

    def add_documents(self, documents, chunk_size=300, overlap=70):
        """
        Add documents with automatic chunking.
        Each chunk becomes a separate point; original doc_id preserved in payload.
        """
        all_points = []
        for doc_id, doc in enumerate(documents):
            chunks = chunk_text(doc.page_content, chunk_size=chunk_size, overlap=overlap)
            for i, chunk in enumerate(chunks):
                payload = {
                    **doc.metadata,
                    "doc_id": doc_id,  # original document id for aggregation later
                    "chunk_idx": i,
                    "text": chunk
                }
                # compute embedding per chunk
                vector = self.embeddings.embed_query(chunk)
                all_points.append(PointStruct(id=len(all_points), vector=vector, payload=payload))

        with start_run(run_name="add_documents_chunked") as run_id:
            log_params({"num_documents": len(documents), "num_points": len(all_points)})
            self.client.upsert(collection_name=self.collection_name, points=all_points)
            log_metrics({"num_inserted": len(all_points)})

    def search(self, query_text, top_k=5):
        """Search using Qdrant's direct query_points to preserve all metadata"""
        with start_run(run_name="vector_store_search") as run_id:
            log_params({"query_length": len(query_text), "top_k": top_k})
            
            # Embed the query
            query_vector = self.embeddings.embed_query(query_text)
            
            # Search Qdrant directly to preserve metadata
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert Qdrant results to LangChain Document format for compatibility
            results = []
            for scored_point in search_results.points:
                # Add score to metadata for retriever to use
                payload = dict(scored_point.payload)
                payload["_score"] = scored_point.score if hasattr(scored_point, 'score') else 0
                doc = Document(
                    page_content=payload.get("text", ""),
                    metadata=payload
                )
                results.append(doc)
            
            log_metrics({"num_results": len(results)})
            return results
