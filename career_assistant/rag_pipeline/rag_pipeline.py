from career_assistant.rag_pipeline.retriever import Retriever
from career_assistant.rag_pipeline.generator import CoverLetterGenerator


def run_rag_pipeline(cv_text: str, jd_text: str, top_k: int = 5):
    """
    Main runtime RAG pipeline:
    1. Retrieves similar CVs/jobs from Qdrant
    2. Summarizes the job description
    3. Generates a cover letter aligned to the job
    """
    # Step 1: Retrieve similar content from the vector DB
    retriever = Retriever(collection_name="career_assistant")

    similar_jobs = retriever.retrieve_similar_jobs(jd_text, top_k=top_k)
    similar_cvs = retriever.retrieve_similar_cvs(cv_text, top_k=top_k)

    # Step 2: Summarize the job
    generator = CoverLetterGenerator()
    job_summary = generator.summarize_job(jd_text)

    # Step 3: Generate a cover letter
    cover_letter = generator.generate_cover_letter(cv_text, job_summary)

    # Step 4: (Optional) Compile results for API response
    return {
        "retrieved_jobs": similar_jobs,
        "retrieved_cvs": similar_cvs,
        "job_summary": job_summary,
        "cover_letter": cover_letter
    }


def main():
    """For quick local testing."""
    cv_text = "I’m an NLP engineer with 3 years of experience in Python, HuggingFace, and LLM fine-tuning."
    jd_text = "We’re hiring an AI engineer with strong experience in NLP, Transformers, and model deployment."

    results = run_rag_pipeline(cv_text, jd_text)
    print("---- Job Summary ----")
    print(results["job_summary"])
    print("\n---- Cover Letter ----")
    print(results["cover_letter"])


if __name__ == "__main__":
    main()
