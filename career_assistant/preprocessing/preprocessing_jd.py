# job_preprocessing.py
import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import argparse
import os

# --- NLTK setup ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Custom stopwords
custom_stopwords = [
    'experience', 'team', 'work', 'ability', 'company', 
    'project', 'opportunity', 'year', 'new', 'knowledge',
    'support', 'strong', 'working', 'develop', 'environment',
    'customer', 'technical', 'including', 'using', 'business',
    'skill', 'development', 'solution', 'management', 'tool',
    'product', 'analytics', 'system', 'problem', 'process',
    'information', 'required', 'degree', 'position', 'requirement',
    'service', 'client', 'job', 'status', 'computer', 'provide',
    'etc', 'communication', 'related', 'need', 'preferred',
    'responsibility', 'field', 'help', 'employee', 'advanced',
    'role', 'must', 'across', 'employment', 'use', 'understanding'
]
stop_words = stop_words.union(custom_stopwords)

# --- Helper functions ---
def simplify_job_title(title: str) -> str:
    title = title.lower()
    if any(x in title for x in ["machine learning", "ml", "deep learning", "ai", "research engineer"]):
        return "Machine Learning Engineer"
    elif any(x in title for x in ["data scientist", "science", "scientist", "modeler", "r&d", "quantitative", "analytical"]):
        return "Data Scientist"
    elif any(x in title for x in ["data engineer", "platform", "etl", "pipeline", "architecture", "database"]):
        return "Data Engineer"
    elif any(x in title for x in ["data analyst", "analytics", "bi", "business analyst", "consultant", "manager"]):
        return "Data Analyst"
    elif any(x in title for x in ["software", "developer", "programmer"]):
        return "Software Engineer"
    else:
        return "Other"

def clean_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # remove non-alpha chars
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# --- Main preprocessing ---
def preprocess_job_data(input_csv: str, output_csv: str, min_desc_len: int = 50) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    # Keep relevant columns
    columns_to_keep = ['Job Title', 'Job Description', 'Company Name', 'Location']
    df = df[columns_to_keep].copy()

    # Drop empty descriptions & duplicates
    df = df.dropna(subset=['Job Description']).drop_duplicates(subset=['Job Description'])

    # Simplify titles
    df['simplified_job_title'] = df['Job Title'].apply(simplify_job_title)
    df = df[df['simplified_job_title'] != 'Other']

    # Clean text
    df['cleaned_job_description'] = df['Job Description'].apply(clean_text)

    # Filter by minimum description length
    df = df[df['cleaned_job_description'].str.split().apply(len) >= min_desc_len]

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed job data saved to {output_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess job descriptions CSV.")

    # Positional arguments
    parser.add_argument("input_csv", type=str, help="Path to input job CSV file")
    parser.add_argument("output_csv", type=str, help="Path to save cleaned CSV file")

    # Optional arguments
    parser.add_argument("--min_desc_len", type=int, default=50,
                        help="Minimum words in job description to keep")

    args = parser.parse_args()

    preprocess_job_data(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        min_desc_len=args.min_desc_len
    )

if __name__ == "__main__":
    main()

# usage
# python job_preprocessing.py data/glassdoor_jobs.csv data/cleaned_job_data_final.csv

# optional filters:
# python preprocessing_jd.py data/glassdoor_jobs.csv data/cleaned_job_data_final.csv --min_desc_len 100

