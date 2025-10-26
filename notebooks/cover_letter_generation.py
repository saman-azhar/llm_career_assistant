# cover_letter_generation.py
import argparse
import os
import ast
import pandas as pd
from typing import List
from transformers import pipeline


# -----------------------------
# Hugging Face Setup
# -----------------------------
def setup_hf_model(model_name: str):
    """Load a summarization pipeline from Hugging Face."""
    print(f"Loading model: {model_name}")
    return pipeline("summarization", model=model_name)


# -----------------------------
# Core Functions
# -----------------------------
def truncate_text(text: str, max_chars: int = 3000):
    """Prevent model overload by trimming overly long job descriptions."""
    return text[:max_chars]


def summarize_job_description(job_text: str, model, max_length: int, min_length: int) -> List[str]:
    """Summarize a job description into 3–4 concise bullet points."""
    job_text = truncate_text(job_text)
    try:
        summary = model(job_text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    except Exception as e:
        print(f"Summarization failed: {e}")
        return []
    bullets = [f"- {b.strip()}" for b in summary.split('. ') if b.strip()]
    return bullets[:4]


def draft_cover_letter(cv_text: str, jd_summary: List[str]) -> str:
    """Draft a simple cover letter using CV and JD summary."""
    jd_points = " ".join([b.lstrip("- ").strip() for b in jd_summary])
    first_sentence = f"Based on your requirements, {jd_points}."
    cv_tokens = [word for word in cv_text.split() if word[0].isupper()]
    top_cv_skills = ", ".join(cv_tokens[:5])
    return (
        f"Dear Hiring Manager,\n\n"
        f"{first_sentence}\n\n"
        f"I have relevant experience with {top_cv_skills}, and I’m confident my background aligns with your needs.\n\n"
        f"Sincerely,\nCandidate"
    )


# -----------------------------
# CLI Entry Point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate cover letters using CVs and Job Descriptions.")
    parser.add_argument("cv_file", type=str, help="Path to CV CSV file.")
    parser.add_argument("jd_file", type=str, help="Path to Job Description CSV file.")
    parser.add_argument("matches_file", type=str, help="Path to matches CSV file.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small", help="Model name.")
    parser.add_argument("--max_length", type=int, default=120, help="Max summary length.")
    parser.add_argument("--min_length", type=int, default=40, help="Min summary length.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    cv_df = pd.read_csv(args.cv_file)
    jd_df = pd.read_csv(args.jd_file)
    matches_df = pd.read_csv(args.matches_file)

    # Clean up column names
    cv_df.columns = cv_df.columns.str.strip().str.lower()
    jd_df.columns = jd_df.columns.str.strip().str.lower()
    matches_df.columns = matches_df.columns.str.strip()

    # Pick text columns
    cv_col = 'cleaned_resume' if 'cleaned_resume' in cv_df.columns else 'resume'
    jd_col = 'cleaned_job_description' if 'cleaned_job_description' in jd_df.columns else 'job_description'

    summarizer = setup_hf_model(args.model_name)

    # Loop through matches
    for i, row in matches_df.iterrows():
        cv_id = row['CV_ID']
        try:
            top_jds = ast.literal_eval(row['Top_JD_Categories'])
        except Exception:
            print(f"Skipping CV_ID {cv_id}: bad Top_JD_Categories format.")
            continue

        # Clean JD titles
        clean_titles = [t.split('.')[0].strip() for t in top_jds]

        jd_summaries = []
        for jd_title in clean_titles:
            jd_row = jd_df[jd_df['simplified_job_title'].str.lower() == jd_title.lower()]
            if jd_row.empty:
                continue
            jd_text = jd_row[jd_col].values[0]
            summary_points = summarize_job_description(jd_text, summarizer, args.max_length, args.min_length)
            jd_summaries.extend(summary_points)

        if not jd_summaries:
            print(f"No JD summaries for CV_ID {cv_id}, skipping...")
            continue

        # Match CV by category or fallback
        cv_text = cv_df[cv_col].iloc[0]
        if 'CV_Category' in row and 'category' in cv_df.columns:
            match_cv = cv_df[cv_df['category'].str.lower() == row['CV_Category'].lower()]
            if not match_cv.empty:
                cv_text = match_cv[cv_col].values[0]

        cover_letter = draft_cover_letter(cv_text, jd_summaries)
        out_path = os.path.join(args.output_dir, f"cover_letter_{cv_id}.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(cover_letter)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()



# usage
# python cover_letter_generation.py ../data/cleaned_resume_data_final.csv ../data/cleaned_job_data_final.csv ../outputs/top_matches_with_missing_skills.csv

# optional filters:
# python cover_letter_generation.py my_cv.csv my_jds.csv --output_dir my_outputs --model_name google/flan-t5-large --max_length 150 --min_length 50

