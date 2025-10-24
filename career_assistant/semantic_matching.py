# semantic_matching.py
import pandas as pd
import numpy as np
import re
import argparse
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

    # Load model
    model = SentenceTransformer(model_name)

    # Generate embeddings
    cv_embeddings = model.encode(df_cv['cleaned_resume'].tolist(), show_progress_bar=True)
    job_embeddings = model.encode(df_job['cleaned_job_description'].tolist(), show_progress_bar=True)

    # Similarity matrix
    similarity = cosine_similarity(cv_embeddings, job_embeddings)

    # Save embeddings if desired
    np.save(os.path.join(output_dir, "cv_embeddings.npy"), cv_embeddings)
    np.save(os.path.join(output_dir, "job_embeddings.npy"), job_embeddings)

    # Build similarity DataFrame
    similarity_df = pd.DataFrame(
        similarity,
        index=df_cv['category'],                  # or unique CV ID
        columns=df_job['simplified_job_title']    # job titles/categories
    )
    similarity_df.to_csv(os.path.join(output_dir, "similarity_scores.csv"), index=True)

    # Top N matches
    similarity_df = similarity_df.apply(pd.to_numeric, errors='coerce')
    top_matches = similarity_df.apply(lambda row: row.nlargest(top_n).index.tolist(), axis=1)
    top_scores = similarity_df.apply(lambda row: row.nlargest(top_n).values.tolist(), axis=1)

    matches_df = pd.DataFrame({
        "CV_ID": df_cv['category'],
        "CV_Category": df_cv['category'],
        "Top_JD_Categories": top_matches,
        "Match_Scores": top_scores
    })
    matches_df["Top_Score"] = matches_df["Match_Scores"].apply(lambda x: max(x))
    matches_df.to_csv(os.path.join(output_dir, "top_matches.csv"), index=False)

    # Extract CV and JD skills
    df_cv["extracted_skills"] = df_cv["cleaned_resume"].apply(extract_skills)
    df_job["extracted_skills"] = df_job["cleaned_job_description"].apply(extract_skills)

    # Merge skills with top matches
    top_matches_exploded = matches_df.explode('Top_JD_Categories')
    merged = top_matches_exploded.merge(
        df_cv[['category', 'extracted_skills']],
        left_on='CV_Category',
        right_on='category',
        how='left'
    ).rename(columns={'extracted_skills':'cv_skills'}).drop(columns=['category'])

    merged = merged.merge(
        df_job[['simplified_job_title', 'extracted_skills']],
        left_on='Top_JD_Categories',
        right_on='simplified_job_title',
        how='left'
    ).rename(columns={'extracted_skills':'jd_skills'}).drop(columns=['simplified_job_title'])

    # Compute missing skills
    merged['Missing_Skills'] = merged.apply(
        lambda row: [s for s in row['jd_skills'] if s not in row['cv_skills']]
        if isinstance(row['jd_skills'], list) and isinstance(row['cv_skills'], list)
        else [],
        axis=1
    )

    # Save final output
    final_columns = [
        'CV_ID', 'CV_Category', 'Top_JD_Categories', 'Match_Scores', 'Top_Score',
        'cv_skills', 'jd_skills', 'Missing_Skills'
    ]
    merged[final_columns].to_csv(os.path.join(output_dir, "missing_skills_analysis.csv"), index=False)
    print(f"Semantic matching and missing skills analysis saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Semantic matching between CVs and Job Descriptions")

    # Positional arguments for files
    parser.add_argument("cv_csv", type=str, help="Path to preprocessed CV CSV")
    parser.add_argument("job_csv", type=str, help="Path to preprocessed Job CSV")
    parser.add_argument("output_dir", type=str, help="Folder to save outputs")

    # Optional arguments
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top matches to extract")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run semantic matching
    semantic_matching(
        cv_file=args.cv_csv,
        job_file=args.job_csv,
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
