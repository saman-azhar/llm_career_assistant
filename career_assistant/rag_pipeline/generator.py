# career_assistant/rag_pipeline/generator.py
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from career_assistant.mlflow_logger import start_run, log_params, log_metrics
from career_assistant.utils.logger import get_logger
from career_assistant.utils.chunking import chunk_text

logger = get_logger(__name__)

class CoverLetterGenerator:
    def __init__(self, model_name="google/flan-t5-large", log_mlflow=True, chunk_size=400, overlap=50):
        self.model_name = model_name
        self.log_mlflow = log_mlflow
        self.device = "cpu"
        self.chunk_size = chunk_size
        self.overlap = overlap

        try:
            logger.info(f"Loading model {model_name} on CPU...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1  # CPU
            )
            logger.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def generate_cover_letter(self, cv_text: str, job_text: str, max_tokens=350):
        if not cv_text.strip() or not job_text.strip():
            logger.warning("Received empty CV or job text for cover letter generation")
            raise ValueError("CV text and job description text cannot be empty.")

        # Optional chunking for very long texts
        cv_chunks = chunk_text(cv_text, self.chunk_size, self.overlap)
        job_chunks = chunk_text(job_text, self.chunk_size, self.overlap)

        # Concatenate top chunks to fit max token limit
        cv_prompt = " ".join(cv_chunks[:3])
        job_prompt = " ".join(job_chunks[:3])
        prompt = f"Write a professional cover letter using CV: {cv_prompt} and JD: {job_prompt}. Highlight matching skills and experience."

        logger.info(f"Generating cover letter (CV chunks: {len(cv_chunks)}, JD chunks: {len(job_chunks)})")
        response = ""

        try:
            if self.log_mlflow:
                with start_run(run_name="generate_cover_letter") as run:
                    log_params({
                        "model_name": self.model.config.name_or_path,
                        "cv_text_length": len(cv_text),
                        "job_text_length": len(job_text),
                        "num_cv_chunks": len(cv_chunks),
                        "num_job_chunks": len(job_chunks),
                        "max_tokens": max_tokens
                    })
                    response = self.generator(
                        prompt,
                        max_new_tokens=max_tokens,
                        temperature=0.9,
                        top_p=0.9
                    )[0]["generated_text"].strip()
                    log_metrics({"generated_text_length": len(response.split())})
            else:
                response = self.generator(prompt, max_new_tokens=max_tokens, temperature=0.9, top_p=0.9)[0]["generated_text"].strip()
        except Exception as e:
            logger.error(f"Error during cover letter generation: {e}")
            raise

        logger.info(f"Cover letter generation completed (words: {len(response.split())})")
        return response
