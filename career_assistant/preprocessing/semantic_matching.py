import os
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import argparse

from career_assistant.utils.file_utils import read_csv, write_csv
from career_assistant.utils.logger import get_logger
from career_assistant.mlflow_logger import start_run, log_params, log_metrics, log_artifacts_from_path

logger = get_logger(__name__)

# --- Known skills list ---
KNOWN_SKILLS = [
    "python", "r", "java", "c++", "c#", "scala", "sql", "bash", "javascript",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "statsmodels",
    "excel", "tableau", "power bi", "lookerstudio", "data analysis", 
    "data visualization", "data wrangling", "etl", "data cleaning",
    "machine learning", "deep learning", "reinforcement learning",
    "scikit-learn", "xgboost", "lightgbm", "tensorflow", "pytorch", "keras",
    "model training", "model evaluation", "feature engineering", "mlops",
    "nlp", "text mining", "text classification", "word embeddings",
    "transformers", "bert", "hugging face", "spacy", "nltk",
    "hadoop", "spark", "kafka", "airflow", "dbt", "snowflake", 
    "redshift", "bigquery", "data pipeline", "data warehouse", 
    "data lake", "sql server", "postgresql", "mongodb",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "jenkins",
    "ci cd", "api development", "microservices", "linux", "shell scripting",
    "object oriented programming", "data structures", "algorithms", 
    "rest api", "unit testing", "version control", "design patterns",
    "agile", "scrum", "system design",
    "statistics", "probability", "regression", "classification", "clustering",
    "hypothesis testing", "time series", "optimization", "a b testing"
]


def extract_skills(text: str):
    found = []
    # Extract words as separate tokens
    words = re.findall(r'\b\w+\b', text.lower())
    for skill in KNOWN_SKILLS:
        if skill in words:
            found.append(skill)
    return found



def compute_similarity_runtime(cv_text: str, jd_text: str, model_name: str = "intfloat/e5-base-v2"):
    """Compute similarity and missing skills for a single CV-JD pair."""
    model = SentenceTransformer(model_name)
    cv_emb = model.encode([cv_text], normalize_embeddings=True)
    jd_emb = model.encode([jd_text], normalize_embeddings=True)
    score = float(cosine_similarity(cv_emb, jd_emb)[0][0])

    cv_skills = set(extract_skills(cv_text))
    jd_skills = set(extract_skills(jd_text))
    missing_skills = sorted(list(jd_skills - cv_skills))
    matched_skills = sorted(list(jd_skills & cv_skills))

    logger.info(f"Similarity: {score:.3f}, Matched skills: {matched_skills}, Missing skills: {missing_skills}")
    return {"similarity_score": score, "matched_skills": matched_skills, "missing_skills": missing_skills}


def semantic_matching(cv_csv: str, job_csv: str, output_dir: str, model_name: str = "intfloat/e5-base-v2",
                      top_n: int = 5, cached_embeddings_dir: str = None):
    """Compute semantic similarity between CVs and Job Descriptions, output top matches and missing skills."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting semantic matching. CVs: {cv_csv}, Jobs: {job_csv}, Output: {output_dir}")

    with start_run(run_name="semantic_matching") as run:
        log_params({
            "cv_csv": cv_csv,
            "job_csv": job_csv,
            "output_dir": output_dir,
            "model_name": model_name,
            "top_n": top_n,
            "cached_embeddings_dir": cached_embeddings_dir
        })

        # --- Load CSVs using utils ---
        df_cv = read_csv(cv_csv)
        df_job = read_csv(job_csv)
        log_metrics({"num_cv_raw": len(df_cv), "num_jobs_raw": len(df_job)})
        logger.info(f"Loaded {len(df_cv)} CVs and {len(df_job)} job descriptions")

        # --- Load or generate embeddings ---
        if cached_embeddings_dir:
            cv_embed_path = os.path.join(cached_embeddings_dir, "cv_embeddings.npy")
            job_embed_path = os.path.join(cached_embeddings_dir, "job_embeddings.npy")
            if os.path.exists(cv_embed_path) and os.path.exists(job_embed_path):
                logger.info(f"Using cached embeddings from {cached_embeddings_dir}")
                cv_embeddings = np.load(cv_embed_path)
                job_embeddings = np.load(job_embed_path)
            else:
                logger.info("Cached embeddings not found. Generating new embeddings...")
                model = SentenceTransformer(model_name)
                cv_embeddings = model.encode(df_cv['cleaned_resume'].tolist(), show_progress_bar=True)
                job_embeddings = model.encode(df_job['cleaned_job_description'].tolist(), show_progress_bar=True)
                np.save(cv_embed_path, cv_embeddings)
                np.save(job_embed_path, job_embeddings)
        else:
            logger.info("Generating new embeddings...")
            model = SentenceTransformer(model_name)
            cv_embeddings = model.encode(df_cv['cleaned_resume'].tolist(), show_progress_bar=True)
            job_embeddings = model.encode(df_job['cleaned_job_description'].tolist(), show_progress_bar=True)
            np.save(os.path.join(output_dir, "cv_embeddings.npy"), cv_embeddings)
            np.save(os.path.join(output_dir, "job_embeddings.npy"), job_embeddings)

        # --- Compute cosine similarity ---
        cv_embeddings = normalize(cv_embeddings)
        job_embeddings = normalize(job_embeddings)
        similarity = cosine_similarity(cv_embeddings, job_embeddings)
        logger.info("Cosine similarity matrix computed")

        # --- Build DataFrame of similarities ---
        job_keys = df_job.apply(lambda r: f"{r['simplified_job_title']}___{r.name}", axis=1)
        cv_keys = [f"{cat}_{i}" for i, cat in enumerate(df_cv['category'])]
        similarity_df = pd.DataFrame(similarity, index=cv_keys, columns=job_keys)
        similarity_csv = os.path.join(output_dir, "similarity_scores.csv")
        write_csv(similarity_df, similarity_csv)
        log_artifacts_from_path(similarity_csv)
        logger.info(f"Similarity scores saved to {similarity_csv}")

        # --- Extract top N matches and missing skills ---
        top_matches = similarity_df.apply(lambda row: row.nlargest(top_n).index.tolist(), axis=1)
        top_scores = similarity_df.apply(lambda row: row.nlargest(top_n).values.tolist(), axis=1)

        expanded_rows = []
        cv_skill_cache = {}
        job_skill_cache = {}

        for cv_idx, cv_row in df_cv.iterrows():
            cv_cat = cv_row['category']
            cv_text = cv_row['cleaned_resume']
            cv_skills = cv_skill_cache.setdefault(cv_idx, set(extract_skills(cv_text)))

            jd_keys = top_matches.iloc[cv_idx]
            jd_scores = top_scores.iloc[cv_idx]

            for jd_key, jd_score in zip(jd_keys, jd_scores):
                _, job_idx = jd_key.rsplit('___', 1)
                job_idx = int(job_idx)
                jd_row = df_job.loc[job_idx]
                jd_skills = job_skill_cache.setdefault(job_idx, set(extract_skills(jd_row['cleaned_job_description'])))
                missing_skills = sorted(list(jd_skills - cv_skills))

                expanded_rows.append({
                    "CV_ID": f"{cv_cat}_{cv_idx}",
                    "CV_Category": cv_cat,
                    "Top_JD_Categories": jd_row['simplified_job_title'],
                    "Top_JD_Key": jd_key,
                    "Match_Score": jd_score,
                    "cv_skills": sorted(list(cv_skills)),
                    "jd_skills": sorted(list(jd_skills)),
                    "Missing_Skills": missing_skills
                })

        matches_df = pd.DataFrame(expanded_rows)
        matches_df["Top_Score"] = matches_df.groupby("CV_ID")["Match_Score"].transform("max")
        top_matches_csv = os.path.join(output_dir, "top_matches_with_missing_skills.csv")
        write_csv(matches_df, top_matches_csv)
        log_artifacts_from_path(top_matches_csv)
        logger.info(f"Top matches with missing skills saved to {top_matches_csv}")

        log_metrics({
            "num_cv": len(df_cv),
            "num_jobs": len(df_job),
            "num_matches": len(matches_df),
            "max_score": matches_df["Match_Score"].max(),
            "avg_score": matches_df["Match_Score"].mean()
        })

        logger.info("Semantic matching pipeline completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Semantic matching between CVs and Job Descriptions")
    parser.add_argument("cv_csv", type=str, help="Path to cleaned CV CSV")
    parser.add_argument("job_csv", type=str, help="Path to cleaned Job CSV")
    parser.add_argument("output_dir", type=str, help="Directory to save outputs")
    parser.add_argument("--model_name", type=str, default="intfloat/e5-base-v2", help="SentenceTransformer model name")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top matches to extract")
    parser.add_argument("--cached_embeddings_dir", type=str, default=None, help="Optional cached embeddings directory")
    args = parser.parse_args()

    semantic_matching(
        cv_csv=args.cv_csv,
        job_csv=args.job_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        top_n=args.top_n,
        cached_embeddings_dir=args.cached_embeddings_dir
    )


if __name__ == "__main__":
    main()
