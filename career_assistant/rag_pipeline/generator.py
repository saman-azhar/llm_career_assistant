
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


class CoverLetterGenerator:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
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
        """Generate a professional cover letter aligning CV and job summary."""

        # Safety truncation to avoid context overflow
        if len(job_summary) > 4000:
            job_summary = job_summary[:4000]
        if len(cv_text) > 4000:
            cv_text = cv_text[:4000]

        prompt = (
            f"You are an experienced career assistant helping a candidate apply for jobs.\n\n"
            f"Job Description:\n{job_summary}\n\n"
            f"Candidate Background:\n{cv_text}\n\n"
            f"Write a concise, professional cover letter in three short paragraphs.\n"
            f"Emphasize overlap between the candidate’s experience and the job requirements.\n"
            f"Keep the tone confident and natural. Only output the final cover letter text."
        )

        # Generate response
        response = self.generator(
            prompt,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        # --- Post-processing: clean up extra text ---
        # Sometimes TinyLlama echoes the prompt or adds unwanted preamble
        cleaned = response
        if "Dear" in response:
            cleaned = response.split("Dear", 1)[1]
            cleaned = "Dear " + cleaned.strip()

        # Remove trailing junk (e.g., duplicated instructions)
        cleaned = cleaned.split("Tone:")[0].strip()
        cleaned = cleaned.split("Write a")[0].strip()

        return cleaned



def main():
    """Placeholder for manual testing."""
    pass


if __name__ == "__main__":
    main()
