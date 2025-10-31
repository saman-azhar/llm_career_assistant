import pandas as pd
import numpy as np
import os
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

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

def extract_skills(text):
    """Extract known skills from text."""
    text = str(text).lower()
    return [skill for skill in KNOWN_SKILLS if skill in text]


def semantic_matching(cv_csv, job_csv, output_dir, model_name='intfloat/e5-base-v2',
                      top_n=5, cached_embeddings_dir=None):
    """Compute semantic similarity and missing skills analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Load data ---
    df_cv = pd.read_csv(cv_csv)
    df_job = pd.read_csv(job_csv)

    # --- Load or generate embeddings ---
    if cached_embeddings_dir:
        cv_embed_path = os.path.join(cached_embeddings_dir, "cv_embeddings.npy")
        job_embed_path = os.path.join(cached_embeddings_dir, "job_embeddings.npy")

        if os.path.exists(cv_embed_path) and os.path.exists(job_embed_path):
            print(f"Using cached embeddings from {cached_embeddings_dir}")
            cv_embeddings = np.load(cv_embed_path)
            job_embeddings = np.load(job_embed_path)
        else:
            print("Cached embeddings not found, generating new ones...")
            model = SentenceTransformer(model_name)
            cv_embeddings = model.encode(df_cv['cleaned_resume'].tolist(), show_progress_bar=True)
            job_embeddings = model.encode(df_job['cleaned_job_description'].tolist(), show_progress_bar=True)
            np.save(cv_embed_path, cv_embeddings)
            np.save(job_embed_path, job_embeddings)
    else:
        print("Generating new embeddings...")
        model = SentenceTransformer(model_name)
        cv_embeddings = model.encode(df_cv['cleaned_resume'].tolist(), show_progress_bar=True)
        job_embeddings = model.encode(df_job['cleaned_job_description'].tolist(), show_progress_bar=True)
        np.save(os.path.join(output_dir, "cv_embeddings.npy"), cv_embeddings)
        np.save(os.path.join(output_dir, "job_embeddings.npy"), job_embeddings)

    # --- Compute similarity ---
    cv_embeddings = normalize(cv_embeddings)
    job_embeddings = normalize(job_embeddings)
    similarity = cosine_similarity(cv_embeddings, job_embeddings)

    # --- Create unique job column names so lookups are unambiguous ---
    # Use row index to guarantee uniqueness. If you have a job_id column, use that instead.
    job_unique_keys = df_job.apply(lambda r: f"{r['simplified_job_title']}___{r.name}", axis=1).tolist()
    cv_index = [f"{cat}_{i}" for i, cat in enumerate(df_cv['category'])]
    similarity_df = pd.DataFrame(similarity, index=cv_index, columns=job_unique_keys)
    similarity_df.to_csv(os.path.join(output_dir, "similarity_scores.csv"))

    # --- Top N matches (now columns are unique keys) ---
    similarity_df = similarity_df.apply(pd.to_numeric, errors='coerce')
    top_matches = similarity_df.apply(lambda row: row.nlargest(top_n).index.tolist(), axis=1)
    top_scores = similarity_df.apply(lambda row: row.nlargest(top_n).values.tolist(), axis=1)

    # --- Expand rows and compute missing skills ---
    expanded_rows = []

    # Optional: cache skill extraction for speed
    cv_skill_cache = {}
    job_skill_cache = {}

    for cv_idx, cv_row in df_cv.iterrows():
        cv_cat = cv_row['category']
        cv_text = cv_row['cleaned_resume']

        # Extract CV skills once
        if cv_idx not in cv_skill_cache:
            cv_skill_cache[cv_idx] = set(extract_skills(cv_text))
        cv_skills = cv_skill_cache[cv_idx]

        # Top matched job unique keys and their scores
        jd_unique_keys = top_matches.iloc[cv_idx]
        jd_scores = top_scores.iloc[cv_idx]

        for jd_key, jd_score in zip(jd_unique_keys, jd_scores):
            # jd_key looks like "<title>___<row_index>"
            try:
                title_part, idx_part = jd_key.rsplit('___', 1)
                job_row_idx = int(idx_part)
            except Exception:
                # fallback (defensive): if the format isn't as expected, try to find exact match
                # but this should not happen if we created keys above
                job_row_idx = None

            if job_row_idx is not None:
                jd_row = df_job.loc[job_row_idx]
            else:
                # defensive fallback: pick the first matching title
                title_part = jd_key
                jd_row = df_job.loc[df_job['simplified_job_title'] == title_part].iloc[0]

            # Extract or compute jd_skills (cache)
            if job_row_idx not in job_skill_cache:
                jd_text = jd_row.get('cleaned_job_description', "")
                job_skill_cache[job_row_idx] = set(extract_skills(jd_text))
            jd_skills = job_skill_cache[job_row_idx]

            missing_skills = sorted(list(jd_skills - cv_skills))

            expanded_rows.append({
                "CV_ID": f"{cv_cat}_{cv_idx}",
                "CV_Category": cv_cat,
                "Top_JD_Categories": jd_row['simplified_job_title'],  # human-friendly title
                "Top_JD_Key": jd_key,  # unique key so you can trace back if needed
                "Match_Score": jd_score,
                "cv_skills": sorted(list(cv_skills)),
                "jd_skills": sorted(list(jd_skills)),
                "Missing_Skills": missing_skills
            })

    matches_df = pd.DataFrame(expanded_rows)
    matches_df["Top_Score"] = matches_df.groupby("CV_ID")["Match_Score"].transform("max")
    matches_df.to_csv(os.path.join(output_dir, "top_matches_with_missing_skills.csv"), index=False)

    print(f"Semantic matching and missing skills analysis saved to {output_dir}")


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

# usage
# python semantic_matching.py cleaned_resume.csv cleaned_jobs.csv outputs

# optional filters:
# python semantic_matching.py cleaned_resume.csv cleaned_jobs.csv outputs --model_name all-MiniLM-L6-v2 --top_n 10

# testing
# python semantic_matching.py ../data/processed/cleaned_resume_data_final.csv ../data/processed/cleaned_job_data_final.csv ../data/processed/semantic_matching_output/test2
# python semantic_matching.py ../data/processed/cleaned_resume_data_final.csv ../data/processed/cleaned_job_data_final.csv ../data/processed/semantic_matching_output --cached_embeddings_dir ../data/vector_store/default/embeddings/

