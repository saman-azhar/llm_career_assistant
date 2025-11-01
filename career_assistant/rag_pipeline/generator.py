
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


class CoverLetterGenerator:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """Initialize a local LLM for summarization and cover letter generation."""
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

    def summarize_job(self, job_text: str, max_tokens: int = 200) -> str:
        """Summarize the job description into 3–4 bullet points."""
        # Truncate overly long inputs to avoid token overflow
        if len(job_text) > 4000:
            job_text = job_text[:4000]

        prompt = (
            f"Summarize the following job description into 3–4 short bullet points:\n\n"
            f"{job_text}\n\nBullet points:"
        )
        response = self.generator(prompt, max_new_tokens=max_tokens, do_sample=False)[0]["generated_text"]
        return response.strip()


    def generate_cover_letter(self, cv_text: str, job_summary: str) -> str:
        """Generate a cover letter aligning CV to the job summary."""
        # Truncate overly long inputs
        if len(job_summary) > 4000:
            job_summary = job_summary[:4000]
        if len(cv_text) > 4000:
            cv_text = cv_text[:4000]

        prompt = (
            f"Write a professional cover letter (3 short paragraphs) based on the following:\n\n"
            f"Job Summary:\n{job_summary}\n\n"
            f"Candidate Background:\n{cv_text}\n\n"
            f"Tone: confident, concise, relevant to the role."
        )
        response = self.generator(prompt, max_new_tokens=400, do_sample=True, temperature=0.7)[0]["generated_text"]
        return response.strip()


def main():
    """Placeholder for manual testing."""
    pass


if __name__ == "__main__":
    main()
