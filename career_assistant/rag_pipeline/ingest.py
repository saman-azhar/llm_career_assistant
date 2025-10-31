import numpy as np
import pandas as pd
from langchain_core.documents import Document
from .vector_store import VectorStore


def ingest_data():
    vs = VectorStore()

    # Load precomputed embeddings
    job_embeddings_path = "career_assistant/data/vector_store/default/embeddings/job_embeddings.npy"
    cv_embeddings_path = "career_assistant/data/vector_store/default/embeddings/cv_embeddings.npy"

    try:
        job_embeddings = np.load(job_embeddings_path)
        cv_embeddings = np.load(cv_embeddings_path)
    except FileNotFoundError:
        print(f"Embeddings not found at {job_embeddings_path} or {cv_embeddings_path}")
        return

    # Load preprocessed data
    df_jobs = pd.read_csv("career_assistant/data/processed/cleaned_job_data_final.csv")
    df_cvs = pd.read_csv("career_assistant/data/processed/cleaned_resume_data_final.csv")

    # Column names
    job_col = "cleaned_job_description"
    cv_col = "cleaned_resume"

    # Convert to LangChain Documents
    job_docs = [
        Document(page_content=row[job_col], metadata={
            "role": row.get("simplified_job_title") or row.get("Job Title", ""), 
            "source": "JD"
        })
        for _, row in df_jobs.iterrows()
    ]

    cv_docs = [
        Document(page_content=row[cv_col], metadata={
            "name": row.get("Name", ""), 
            "source": "CV"
        })
        for _, row in df_cvs.iterrows()
    ]

    # Directly inject embeddings to avoid recomputation
    vs.store.add_documents(documents=job_docs, embeddings=job_embeddings.tolist())
    vs.store.add_documents(documents=cv_docs, embeddings=cv_embeddings.tolist())

    print("Ingestion completed successfully with precomputed embeddings.")

if __name__ == "__main__":
    ingest_data()

