# career_assistant/rag_pipeline/ingest.py
import numpy as np
from qdrant_client.models import PointStruct
from .vector_store import VectorStore
from .embedder import Embedder
from career_assistant.mlflow_logger import start_run, log_params, log_metrics
from career_assistant.utils.file_utils import read_csv
from career_assistant.utils.chunking import chunk_text
from career_assistant.utils.logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 300
CHUNK_OVERLAP = 70

def ingest_data(chunking=True):
    vs = VectorStore()
    embedder = Embedder()
    logger.info("Initializing Qdrant ingestion with embeddings and optional chunking...")

    job_csv_path = "career_assistant/data/processed/cleaned_job_data_final.csv"
    cv_csv_path = "career_assistant/data/processed/cleaned_resume_data_final.csv"

    with start_run(run_name="qdrant_ingestion") as run_id:
        log_params({
            "job_csv_path": job_csv_path,
            "cv_csv_path": cv_csv_path,
            "collection_name": vs.collection_name,
            "chunking": chunking,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        })

        try:
            df_jobs = read_csv(job_csv_path)
            df_cvs = read_csv(cv_csv_path)
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        job_col = "cleaned_job_description"
        cv_col = "cleaned_resume"
        if job_col not in df_jobs.columns or cv_col not in df_cvs.columns:
            logger.error(f"Required columns '{job_col}' or '{cv_col}' not found in CSVs.")
            return

        all_points = []
        job_chunks_total = 0
        cv_chunks_total = 0

        # --- Ingest jobs ---
        logger.info(f"Ingesting {len(df_jobs)} job descriptions")
        for i, row in enumerate(df_jobs.itertuples()):
            text = str(row.cleaned_job_description)
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP) if chunking else [text]
            embeddings = embedder.embed_documents(chunks)
            job_chunks_total += len(chunks)
            for j, (chunk_text_val, vec) in enumerate(zip(chunks, embeddings)):
                all_points.append(PointStruct(
                    id=f"job_{i}_{j}",
                    vector=vec.tolist(),
                    payload={
                        "doc_id": i,  # original job index
                        "role": str(row.simplified_job_title),
                        "source": "JD",
                        "chunk_idx": j,
                        "text": chunk_text_val
                    }
                ))

        # --- Ingest CVs ---
        logger.info(f"Ingesting {len(df_cvs)} CVs")
        for i, row in enumerate(df_cvs.itertuples()):
            text = str(row.cleaned_resume)
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP) if chunking else [text]
            embeddings = embedder.embed_documents(chunks)
            cv_chunks_total += len(chunks)
            for j, (chunk_text_val, vec) in enumerate(zip(chunks, embeddings)):
                all_points.append(PointStruct(
                    id=f"cv_{i}_{j}",
                    vector=vec.tolist(),
                    payload={
                        "doc_id": i,  # original CV index
                        "name": str(row.category),
                        "source": "CV",
                        "chunk_idx": j,
                        "text": chunk_text_val
                    }
                ))

        vs.client.upsert(collection_name=vs.collection_name, points=all_points)
        logger.info(f"Successfully ingested {len(all_points)} total chunks into Qdrant.")

        log_metrics({
            "num_job_chunks": job_chunks_total,
            "num_cv_chunks": cv_chunks_total,
            "num_total_points": len(all_points)
        })


if __name__ == "__main__":
    ingest_data()
