# cover_letter_generator.py
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
    print(f"Loading Hugging Face model: {model_name}")
    summarizer = pipeline("summarization", model=model_name)
    return summarizer

# -----------------------------
# Core Functions
# -----------------------------
def summarize_job_description(job_text: str, model, max_length: int, min_length: int) -> List[str]:
    """
    Summarize a job description into 3–4 concise bullet points.
    """
    summary_text = model(job_text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    bullets = summary_text.split('. ')
    bullets = [f"- {b.strip()}" for b in bullets if b.strip()]
    return bullets[:4]

def draft_cover_letter(cv_text: str, jd_summary: List[str]) -> str:
    """
    Draft a cover letter from CV and JD summary.
    """
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
    parser = argparse.ArgumentParser(description="Generate cover letters from CVs and Job Descriptions using a Hugging Face model.")
    
    # Positional arguments for files
    parser.add_argument("cv_file", type=str, help="Path to CV file (CSV with 'Resume' column or TXT).")
    parser.add_argument("jd_file", type=str, help="Path to Job Description file (CSV with 'Job Description' column or TXT).")
    
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
        if "Resume" not in cv_df.columns:
            raise ValueError("CSV must contain a 'Resume' column")
        cv_texts = cv_df['Resume'].tolist()
    else:
        with open(args.cv_file, 'r', encoding='utf-8') as f:
            cv_texts = [f.read()]

    # Load JDs
    if args.jd_file.endswith(".csv"):
        jd_df = pd.read_csv(args.jd_file)
        if "Job Description" not in jd_df.columns:
            raise ValueError("CSV must contain a 'Job Description' column")
        jd_texts = jd_df['Job Description'].tolist()
    else:
        with open(args.jd_file, 'r', encoding='utf-8') as f:
            jd_texts = [f.read()]

    # Load summarization model
    summarizer = setup_hf_model(args.model_name)

    # Process each JD-CV pair
    for i, jd_text in enumerate(jd_texts, start=1):
        jd_summary = summarize_job_description(jd_text, summarizer, args.max_length, args.min_length)
        summary_file = os.path.join(args.output_dir, f"jd_summary_{i}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(jd_summary))
        print(f"Saved JD summary to {summary_file}")

        for j, cv_text in enumerate(cv_texts, start=1):
            cover_letter = draft_cover_letter(cv_text, jd_summary)
            cl_file = os.path.join(args.output_dir, f"cover_letter_{i}_{j}.txt")
            with open(cl_file, 'w', encoding='utf-8') as f:
                f.write(cover_letter)
            print(f"Saved cover letter to {cl_file}")


if __name__ == "__main__":
    main()

# usage
# python llm_integration.py my_cv.csv my_jds.csv

# optional filters:
# python generate_cover_letters.py my_cv.csv my_jds.csv --output_dir my_outputs --model_name google/flan-t5-large --max_length 150 --min_length 50

