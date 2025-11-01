from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


class VectorStore:
    def __init__(self, 
                 host="localhost", 
                 port=6333, 
                 collection_name="career_assistant", 
                 embedding_model="intfloat/e5-base-v2"):
        
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # LangChain embedding model wrapper
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        # Qdrant client
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()

        # LangChain Qdrant wrapper (acts as retriever interface)
        self.store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

    def insert_embeddings(self, embeddings, payloads):
        """Manual insert (non-LangChain route)"""
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(id=i, vector=embeddings[i].tolist(), payload=payloads[i])
            for i in range(len(embeddings))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def add_documents(self, documents):
        """LangChain route for direct document ingestion"""
        self.store.add_documents(documents)

    def search(self, query_text, top_k=5):
        """Search using LangChain’s retriever abstraction"""
        """Perform a similarity search in Qdrant using LangChain abstraction.

        Parameters:
            query_text (str): Raw natural language query. 
                NOTE: This function expects **plain text**, not precomputed embeddings.
                The embedding conversion is handled automatically via the
                HuggingFaceEmbeddings model configured in this VectorStore.
            top_k (int): Number of nearest results to retrieve.

        Returns:
            List[langchain_core.documents.Document]: Top-matching documents.
        """
        return self.store.similarity_search(query_text, k=top_k)
