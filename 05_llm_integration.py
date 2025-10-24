from typing import List, Union
from transformers import pipeline
from openai import OpenAI

# --- LLM Setup Functions ---
def setup_openai_api(api_key: str = None):
    """
    Initialize OpenAI client (placeholder if API not available)
    """
    try:
        if api_key:
            client = OpenAI(api_key=api_key)
            print("OpenAI API client initialized")
            return client
        else:
            print("No OpenAI API key provided. Skipping setup.")
            return None
    except ImportError:
        print("OpenAI package not installed. Run 'pip install openai'.")
        return None


def setup_huggingface_model(model_name: str = "facebook/bart-large-cnn"):
    """
    Load Hugging Face summarization pipeline
    """
    print(f"Loading HuggingFace model: {model_name}")
    summarizer = pipeline("summarization", model=model_name)
    return summarizer

from typing import List, Optional


def summarize_job_description(
    job_text: str,
    model: Optional[object] = None,
    method: str = "huggingface",
    max_length: int = 130,
    min_length: int = 40
) -> List[str]:
    """
    Summarize a job description into 3–4 concise bullet points.
    
    Parameters:
        job_text: str - raw job description
        model: Hugging Face pipeline object OR OpenAI client
        method: "huggingface" or "openai"
        max_length, min_length: for Hugging Face summarizer
    Returns:
        List of bullet points
    """
    if method.lower() == "huggingface":
        from transformers import pipeline
        # If no pipeline passed, create one
        if model is None:
            print("Using default Hugging Face summarizer")
            model = pipeline("summarization", model="facebook/bart-large-cnn")
        summary_text = model(job_text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    
    elif method.lower() == "openai":
        if model is None:
            print("OpenAI model not provided. Skipping OpenAI summarization.")
            return ["[OpenAI summarization skipped – no API key provided]"]
        # Example OpenAI call using text-davinci-003
        response = model.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize this job description in 4 concise bullet points:\n{job_text}",
            max_tokens=150
        )
        summary_text = response.choices[0].text.strip()
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'huggingface' or 'openai'.")
    
    # Convert summary text to bullet points
    bullets = summary_text.split('. ')
    bullets = [f"- {b.strip()}" for b in bullets if b.strip()]
    return bullets[:4]


def draft_cover_letter(cv_text: str, jd_summary: list, model=None, method: str = "huggingface") -> str:
    print(f"\nDrafting cover letter using: {method}")

    if isinstance(jd_summary, list):
        summary_paragraph = " ".join([b.lstrip("- ").strip() for b in jd_summary])
    else:
        summary_paragraph = jd_summary

    # Extract "skills" from CV as all capitalized words or simple keywords
    cv_skills = [word for word in cv_text.split() if word[0].isupper()]
    cv_paragraph = ", ".join(cv_skills[:5])  # just top 5 for demo

    return (
        f"Dear Hiring Manager,\n\n"
        f"Based on your requirements: {summary_paragraph}.\n\n"
        f"Here's how I align with your needs: I have experience with {cv_paragraph}."
    )


# --- Main Execution for Testing ---
if __name__ == "__main__":
    jd_example = """
    We are seeking a Data Scientist with experience in Python, SQL, and machine learning.
    The ideal candidate will develop predictive models, perform statistical analysis,
    and collaborate with cross-functional teams to deploy AI-driven solutions.
    Experience with cloud platforms (AWS/Azure) and NLP is a plus.
    """
    cv_example = "Sample CV text here."

    # Setup LLMs
    openai_client = setup_openai_api()  # (add key later)
    hf_summarizer = setup_huggingface_model()

    # Run both summarizations for comparison
    jd_summary_hf = summarize_job_description(jd_example, hf_summarizer, method="huggingface")
    jd_summary_gpt = summarize_job_description(jd_example, openai_client, method="openai")

    print("\n--- HuggingFace Summary ---")
    print(jd_summary_hf)

    print("\n--- GPT Summary (placeholder) ---")
    print(jd_summary_gpt)

    # Generate cover letter
    cover_letter = draft_cover_letter(cv_example, jd_summary_hf)
    print("\n--- Draft Cover Letter ---")
    print(cover_letter)
