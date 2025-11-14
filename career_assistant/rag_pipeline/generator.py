from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class CoverLetterGenerator:
    def __init__(self, model_name="TheBloke/guanaco-7B-HF", device=None):
        """Initialize a stronger LLM on AWS (GPU recommended)."""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model {model_name} on device {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          device_map="auto" if self.device=="cuda" else None,
                                                          torch_dtype=torch.float16 if self.device=="cuda" else torch.float32)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if self.device=="cuda" else -1)


    def summarize_job(self, job_text: str, max_tokens: int = 200):
        if len(job_text) > 4000: job_text = job_text[:4000]
        prompt = f"Summarize the following job description into 3-4 concise bullet points:\n{job_text}\nBullet points:"
        return self.generator(prompt, max_new_tokens=max_tokens, do_sample=False)[0]["generated_text"].strip()


    def generate_cover_letter(self, cv_text: str, job_summary: str, max_tokens=400):
        if len(cv_text) > 4000: cv_text = cv_text[:4000]
        if len(job_summary) > 4000: job_summary = job_summary[:4000]

        prompt = (
            f"You are a professional career assistant that writes polished cover letters.\n\n"
            f"Job Description Summary:\n{job_summary}\n\n"
            f"Candidate Background:\n{cv_text}\n\n"
            f"Instructions:\n"
            f"- Write a concise cover letter in 3 short paragraphs.\n"
            f"- Use confident, professional tone suitable for technical jobs.\n"
            f"- Highlight exact skills that match the job requirements.\n"
            f"- Mention achievements where possible.\n"
            f"- Do NOT include generic phrases; focus on specifics.\n"
            f"- Output only the cover letter, ready to send."
        )

        # Generate response
        response = self.generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"]
        return response.strip()
        # --- Post-processing: clean up extra text ---
        # Sometimes TinyLlama echoes the prompt or adds unwanted preamble
        # cleaned = response
        # if "Dear" in response:
        #     cleaned = response.split("Dear", 1)[1]
        #     cleaned = "Dear " + cleaned.strip()

        # # Remove trailing junk (e.g., duplicated instructions)
        # cleaned = cleaned.split("Tone:")[0].strip()
        # cleaned = cleaned.split("Write a")[0].strip()

        # return cleaned



def main():
    """Placeholder for manual testing."""
    pass


if __name__ == "__main__":
    main()
