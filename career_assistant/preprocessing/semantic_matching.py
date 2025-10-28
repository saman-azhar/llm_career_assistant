# semantic_matching.py
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
    text = str(text).lower()
    return [skill for skill in KNOWN_SKILLS if skill in text]

def semantic_matching(cv_csv, job_csv, output_dir, model_name='all-MiniLM-L6-v2', top_n=5):
    """Compute embeddings, similarity, top matches, and missing skills"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df_cv = pd.read_csv(cv_csv)
    df_job = pd.read_csv(job_csv)

    # Reset indices
    df_cv = df_cv.reset_index(drop=True)
    df_job = df_job.reset_index(drop=True)

    # Load model
    model = SentenceTransformer(model_name)

    # Generate embeddings
    cv_embeddings = model.encode(df_cv['cleaned_resume'].tolist(), show_progress_bar=True)
    job_embeddings = model.encode(df_job['cleaned_job_description'].tolist(), show_progress_bar=True)

    # Similarity matrix
    similarity = cosine_similarity(cv_embeddings, job_embeddings)

    # Save embeddings
    np.save(os.path.join(output_dir, "cv_embeddings.npy"), cv_embeddings)
    np.save(os.path.join(output_dir, "job_embeddings.npy"), job_embeddings)

    # Build similarity DataFrame
    similarity_df = pd.DataFrame(
        similarity,
        index=df_cv['category'],
        columns=df_job['simplified_job_title']
    )
    similarity_df.to_csv(os.path.join(output_dir, "similarity_scores.csv"), index=True)

    # Top N matches
    similarity_df = similarity_df.apply(pd.to_numeric, errors='coerce')
    top_matches = similarity_df.apply(lambda row: row.nlargest(top_n).index.tolist(), axis=1)
    top_scores = similarity_df.apply(lambda row: row.nlargest(top_n).values.tolist(), axis=1)

    matches_df = pd.DataFrame({
        "CV_ID": range(len(df_cv)),
        "CV_Category": df_cv['category'].reset_index(drop=True),
        "Top_JD_Categories": top_matches.reset_index(drop=True),
        "Match_Scores": top_scores.reset_index(drop=True)
    })
    matches_df["Top_Score"] = matches_df["Match_Scores"].apply(lambda x: max(x))

    # --- Compute missing skills for only the top JD match ---
    missing_skills = []
    for i, row in matches_df.iterrows():
        top_jd = row["Top_JD_Categories"][0]  # take only the top JD
        jd_rows = df_job[df_job["simplified_job_title"] == top_jd]
        jd_text = jd_rows["cleaned_job_description"].str.cat(sep=" ")
        cv_text = df_cv.loc[i, "cleaned_resume"]

        cv_skills = extract_skills(cv_text)
        jd_skills = extract_skills(jd_text)
        missing = [s for s in jd_skills if s not in cv_skills]
        missing_skills.append(missing)

    matches_df["Missing_Skills"] = missing_skills

    # Save final file
    matches_df.to_csv(os.path.join(output_dir, "top_matches_with_missing_skills.csv"), index=False)
    print(f"Final file saved to: {os.path.join(output_dir, 'top_matches_with_missing_skills.csv')}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic matching between CVs and Job Descriptions")
    parser.add_argument("cv_csv", type=str, help="Path to preprocessed CV CSV")
    parser.add_argument("job_csv", type=str, help="Path to preprocessed Job CSV")
    parser.add_argument("output_dir", type=str, help="Folder to save outputs")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top matches to extract")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    semantic_matching(
        cv_csv=args.cv_csv,
        job_csv=args.job_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        top_n=args.top_n
    )

if __name__ == "__main__":
    main()


# usage
# python semantic_matching.py cleaned_resume.csv cleaned_jobs.csv outputs

# optional filters:
# python semantic_matching.py cleaned_resume.csv cleaned_jobs.csv outputs --model_name all-MiniLM-L6-v2 --top_n 10
