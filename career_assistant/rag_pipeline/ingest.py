import numpy as np
import pandas as pd
from qdrant_client.models import PointStruct
from langchain_core.documents import Document
from .vector_store import VectorStore


def ingest_data():
    vs = VectorStore()
    print("Initializing Qdrant ingestion with precomputed embeddings...")

    # Paths
    job_embeddings_path = "career_assistant/data/vector_store/default/embeddings/job_embeddings.npy"
    cv_embeddings_path = "career_assistant/data/vector_store/default/embeddings/cv_embeddings.npy"

    # Load precomputed embeddings
    try:
        job_embeddings = np.load(job_embeddings_path)
        cv_embeddings = np.load(cv_embeddings_path)
        print(f"Loaded {len(job_embeddings)} job and {len(cv_embeddings)} CV embeddings.")
    except FileNotFoundError:
        print("Embedding files not found. Run embedding generation first.")
        return

    # Load CSVs
    df_jobs = pd.read_csv("career_assistant/data/processed/cleaned_job_data_final.csv")
    df_cvs = pd.read_csv("career_assistant/data/processed/cleaned_resume_data_final.csv")

    # Use preprocessed column names
    job_col = "cleaned_job_description"
    cv_col = "cleaned_resume"

    # Sanity checks
    if job_col not in df_jobs.columns or cv_col not in df_cvs.columns:
        print("Required columns not found in CSVs.")
        return

    # Prepare points for Qdrant
    job_points = [
        PointStruct(
            id=i,
            vector=job_embeddings[i].tolist(),
            payload={
                "role": str(row.simplified_job_title),
                "source": "JD",
                "text": str(row.cleaned_job_description)
            }
        )
        for i, row in enumerate(df_jobs.itertuples())
    ]

    cv_points = [
        PointStruct(
            id=len(job_points) + i,
            vector=cv_embeddings[i].tolist(),
            payload={
                "name": str(row.category),
                "source": "CV",
                "text": str(row.cleaned_resume)
            }
        )
        for i, row in enumerate(df_cvs.itertuples())
    ]

    # Combine and insert
    all_points = job_points + cv_points
    vs.client.upsert(collection_name=vs.collection_name, points=all_points)

    print(f"Successfully ingested {len(all_points)} total documents into Qdrant.")
    print("Ingestion complete (used precomputed embeddings, no recomputation).")


if __name__ == "__main__":
    ingest_data()
