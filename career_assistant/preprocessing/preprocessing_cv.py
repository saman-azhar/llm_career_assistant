# resume_preprocessing.py
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import argparse
import os

# --- NLTK Setup ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Custom stopwords
custom_stopwords = [
    'experience', 'exprience', 'month', 'project', 'company', 'description', 'detail',
    'worked', 'responsible', 'responsibility', 'responsibilities',
    'team', 'task', 'year', 'based', 'work', 'developed', 
    'technology', 'skill', 'skills', 'requirement', 'requirements',
    'maharashtra', 'using', 'environment', 'less', 'role', 'roles', 'ltd'
]
stop_words = stop_words.union(custom_stopwords)

# --- Helper functions ---
def clean_resume(text: str) -> str:
    """Clean resume text: remove HTML, URLs, non-alpha chars (except +.#), lowercase, lemmatize, remove stopwords"""
    text = re.sub(r'<[^>]+>', ' ', str(text))  # remove HTML tags
    text = re.sub(r'http\S+', ' ', text)       # remove URLs
    text = re.sub(r'[^a-zA-Z#\+\.\s]', ' ', text)  # keep +, #, . for skills
    text = text.lower()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    cleaned = ' '.join(tokens)
    # replace isolated dots with space
    cleaned = re.sub(r'(?<!\w)\.(?!\w)', ' ', cleaned)
    return cleaned.strip()

def filter_categories(df: pd.DataFrame, keep_categories: list) -> pd.DataFrame:
    """Keep only resumes in relevant job categories"""
    return df[df['Category'].isin(keep_categories)].reset_index(drop=True)

# --- Main preprocessing function ---
def preprocess_resumes(input_csv: str, output_csv: str,
                       min_resume_len: int = 50,
                       min_word_count: int = 100,
                       categories: list = None) -> pd.DataFrame:
    """
    Load resume dataset, clean text, filter duplicates, remove short resumes, filter categories.
    Saves cleaned dataset to output_csv.
    """
    df = pd.read_csv(input_csv)

    # Drop duplicates and short/empty resumes
    df = df.drop_duplicates(subset='Resume')
    df = df.dropna(subset=['Resume'])
    df = df[df['Resume'].str.len() >= min_resume_len].reset_index(drop=True)

    # Clean resume text
    df['cleaned_resume'] = df['Resume'].apply(clean_resume)

    # Drop resumes with fewer than min_word_count words after cleaning
    df['text_length'] = df['cleaned_resume'].apply(lambda x: len(x.split()))
    df = df[df['text_length'] >= min_word_count].reset_index(drop=True)

    # Filter by categories if provided
    if categories:
        df = filter_categories(df, categories)

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed resumes saved to {output_csv}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Preprocess resume CSV file.")

    # Positional args
    parser.add_argument("input_csv", type=str, help="Path to input resume CSV file")
    parser.add_argument("output_csv", type=str, help="Path to save cleaned CSV file")

    # Optional args
    parser.add_argument("--min_resume_len", type=int, default=50,
                        help="Minimum characters in raw resume to keep")
    parser.add_argument("--min_word_count", type=int, default=100,
                        help="Minimum words in cleaned resume to keep")
    parser.add_argument("--categories", type=str, nargs='*',
                        help="List of categories to keep (space-separated, e.g., 'Data Science Data Analyst')")

    args = parser.parse_args()

    preprocess_resumes(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        min_resume_len=args.min_resume_len,
        min_word_count=args.min_word_count,
        categories=args.categories
    )

if __name__ == "__main__":
    main()

# usage
# python preprocessing_cv.py data/UpdatedResumeDataSet.csv data/cleaned_resume_data_final.csv

# optional filters:
# python preprocessing_cv.py data/UpdatedResumeDataSet.csv data/cleaned_resume_data_final.csv --min_word_count 120 --categories "Data Science" "Data Analyst"
