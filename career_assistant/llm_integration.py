# llm_integration.py
import argparse
import os
import pandas as pd
from typing import List
from transformers import pipeline

# -----------------------------
# Hugging Face Setup
# -----------------------------
def setup_hf_model(model_name: str):
    """Load a summarization pipeline from Hugging Face."""
    print(f"Loading model: {model_name}")
    summarizer = pipeline("summarization", model=model_name)
    return summarizer

# -----------------------------
# Core Functions
# -----------------------------
def summarize_job_description(job_text: str, model, max_length: int, min_length: int) -> List[str]:
    """Summarize a job description into 3–4 concise bullet points."""
    summary_text = model(job_text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    bullets = summary_text.split('. ')
    bullets = [f"- {b.strip()}" for b in bullets if b.strip()]
    return bullets[:4]

def draft_cover_letter(cv_text: str, jd_summary: List[str]) -> str:
    """Draft a cover letter from CV and JD summary."""
    summary_paragraph = " ".join([b.lstrip("- ").strip() for b in jd_summary])
    cv_skills = [word for word in cv_text.split() if word[0].isupper()]
    cv_paragraph = ", ".join(cv_skills[:5])
    return (
        f"Dear Hiring Manager,\n\n"
        f"Based on your requirements: {summary_paragraph}.\n\n"
        f"I align with your needs and have experience with {cv_paragraph}.\n\n"
        f"Sincerely,\nCandidate"
    )

# -----------------------------
# CLI Entry Point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate cover letters from CVs and Job Descriptions using Hugging Face LLM.")
    
    # Positional arguments for files
    parser.add_argument("cv_file", type=str, help="Path to CV file (CSV with 'Resume' column or TXT).")
    parser.add_argument("jd_file", type=str, help="Path to Job Description file (CSV with 'cleaned_job_description' column or TXT).")
    parser.add_argument("matches_file", type=str, help="Path to top_matches_with_missing_skills CSV.")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="outputs", help="Folder to save summaries and cover letters.")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small", help="Hugging Face model name.")
    parser.add_argument("--max_length", type=int, default=130, help="Max tokens for summarization.")
    parser.add_argument("--min_length", type=int, default=40, help="Min tokens for summarization.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load CVs
    if args.cv_file.endswith(".csv"):
        cv_df = pd.read_csv(args.cv_file)
        cv_df.columns = cv_df.columns.str.strip()
        if "cleaned_resume" in cv_df.columns:
            cv_texts = cv_df['cleaned_resume'].tolist()
        elif "Resume" in cv_df.columns:
            cv_texts = cv_df['Resume'].tolist()
        else:
            raise ValueError("CV CSV must contain 'Resume' or 'cleaned_resume' column")
    else:
        with open(args.cv_file, 'r', encoding='utf-8') as f:
            cv_texts = [f.read()]

    # Load JDs
    jd_df = pd.read_csv(args.jd_file)
    jd_df.columns = jd_df.columns.str.strip()
    required_cols = ['simplified_job_title', 'cleaned_job_description']
    if not all(col in jd_df.columns for col in required_cols):
        raise ValueError("Job CSV must contain 'simplified_job_title' and 'cleaned_job_description' columns")

    # Load top matches
    matches_df = pd.read_csv(args.matches_file)
    matches_df.columns = matches_df.columns.str.strip()

    # Load summarization model
    summarizer = setup_hf_model(args.model_name)

    # Process each CV
    for idx, row in matches_df.iterrows():
        cv_id = row['CV_ID']
        top_jds = row['Top_JD_Categories'].split(',')  # split comma-separated titles

        # Strip numeric suffixes
        cleaned_top_jds = [jd_title.split('.')[0].strip() for jd_title in top_jds]

        # Aggregate JD summaries
        jd_summaries = []
        for jd_title in cleaned_top_jds:
            jd_row = jd_df[jd_df['simplified_job_title'] == jd_title]
            if jd_row.empty:
                print(f"No JD found for title '{jd_title}', skipping...")
                continue
            jd_text = jd_row['cleaned_job_description'].values[0]
            bullets = summarize_job_description(jd_text, summarizer, args.max_length, args.min_length)
            jd_summaries.extend(bullets)

        if not jd_summaries:
            print(f"No valid JD summaries for CV_ID {cv_id}, skipping cover letter generation.")
            continue

        # Draft cover letter
        cv_text = cv_texts[idx] if idx < len(cv_texts) else cv_texts[0]
        cover_letter = draft_cover_letter(cv_text, jd_summaries)

        # Save output
        cl_file = os.path.join(args.output_dir, f"cover_letter_{cv_id}.txt")
        with open(cl_file, 'w', encoding='utf-8') as f:
            f.write(cover_letter)
        print(f"Saved cover letter to {cl_file}")

if __name__ == "__main__":
    main()



# usage
# python llm_integration.py ../data/cleaned_resume_data_final.csv ../data/cleaned_job_data_final.csv ../outputs/top_matches_with_missing_skills.csv

# optional filters:
# python llm_integration.py my_cv.csv my_jds.csv --output_dir my_outputs --model_name google/flan-t5-large --max_length 150 --min_length 50

