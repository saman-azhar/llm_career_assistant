# career_assistant/rag_pipeline/generator.py
import re
from typing import Optional
from career_assistant.mlflow_logger import start_run, log_params, log_metrics
from career_assistant.utils.logger import get_logger
from career_assistant.preprocessing.semantic_matching import compute_similarity_runtime

logger = get_logger(__name__)

# CPU-friendly LLM setup
try:
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        LANGCHAIN_AVAILABLE = False
    except ImportError:
        logger.warning("Transformers not available. LLM generation will use fallback.")
        LANGCHAIN_AVAILABLE = False


class CoverLetterGenerator:
    """Generate cover letters with intelligent matching assessment using LLM."""
    
    # Matching thresholds
    POOR_MATCH_THRESHOLD = 0.60
    GOOD_MATCH_THRESHOLD = 0.80
    
    def __init__(self, log_mlflow=True, model_name: Optional[str] = None, use_cpu: bool = True):
        """
        Initialize the cover letter generator with a CPU-friendly LLM.
        
        Args:
            log_mlflow: Whether to log to MLflow
            model_name: HuggingFace model name. Defaults to CPU-friendly model.
            use_cpu: Force CPU usage (default True for CPU-friendly operation)
        """
        self.log_mlflow = log_mlflow
        self.use_cpu = use_cpu
        # Default to a small, CPU-friendly model
        # Alternatives: "google/flan-t5-small", "microsoft/phi-2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.model_name = model_name or "google/flan-t5-base"  # Very CPU-friendly, good for text generation
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM pipeline for CPU-friendly inference."""
        try:
            logger.info(f"Initializing CPU-friendly LLM: {self.model_name}")
            
            # Always load on CPU safely
            import torch
            torch_dtype = torch.float32  # CPU uses float32 for stability
            
            # Check if it's a T5 model (text2text-generation) or causal LM (text-generation)
            is_t5_model = "t5" in self.model_name.lower() or "flan" in self.model_name.lower()
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                if is_t5_model:
                    from transformers import T5ForConditionalGeneration
                    model = T5ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch_dtype,
                        device_map=None,  # Ensures model is loaded on CPU
                        trust_remote_code=True
                    )
                    hf_pipeline = pipeline(
                        "text2text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=-1,  # -1 for CPU
                        max_length=512,
                        do_sample=True,
                        temperature=0.7
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch_dtype,
                        device_map=None,  # Ensures model is loaded on CPU
                        trust_remote_code=True
                    )
                    hf_pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=-1,  # -1 for CPU
                        max_new_tokens=400,  # Limit for CPU efficiency
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
                    )
                
                if LANGCHAIN_AVAILABLE:
                    self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
                else:
                    self.llm = hf_pipeline
                    # Store pipeline type for later use
                    self.llm_type = "text2text-generation" if is_t5_model else "text-generation"
                    
                logger.info(f"LLM initialized successfully (type: {'T5' if is_t5_model else 'CausalLM'})")
                
            except Exception as e:
                logger.warning(f"Failed to load {self.model_name}, trying fallback model: {e}")
                # Fallback to a simpler, guaranteed-to-work model
                self.model_name = "google/flan-t5-small"
                from transformers import T5ForConditionalGeneration
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = T5ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype
                )
                hf_pipeline = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7
                )
                if LANGCHAIN_AVAILABLE:
                    self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
                else:
                    self.llm = hf_pipeline
                    self.llm_type = "text2text-generation"
                logger.info("Fallback LLM initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}. Will use template fallback.")
            self.llm = None

    def generate_cover_letter(self, cv_text: str, job_text: str):
        """
        Generate cover letter with matching score assessment.
        
        Returns: dict with keys:
            - match_score: float between 0-1 (based on skill overlap: matched / (matched + missing))
            - match_level: "poor" | "moderate" | "good"
            - message: str with assessment
            - matched_skills: list of matching skills
            - missing_skills: list of missing skills
            - cover_letter: str (only if match_score > POOR_MATCH_THRESHOLD)
        """
        if not cv_text.strip() or not job_text.strip():
            logger.warning("Received empty CV or job text for cover letter generation")
            raise ValueError("CV text and job description text cannot be empty.")

        logger.info("Computing semantic match and extracting skills...")
        
        # Compute similarity and extract skills using semantic matching
        matching_data = compute_similarity_runtime(cv_text, job_text)
        matched_skills = matching_data["matched_skills"]
        missing_skills = matching_data["missing_skills"]
        
        # Compute match score based on skill overlap: matched / (matched + missing)
        # If no skills found in JD, use semantic similarity as fallback
        total_jd_skills = len(matched_skills) + len(missing_skills)
        if total_jd_skills > 0:
            match_score = len(matched_skills) / total_jd_skills
        else:
            # Fallback to semantic similarity if no KNOWN_SKILLS found in JD
            match_score = matching_data["similarity_score"]
            logger.warning(f"No skills extracted from JD. Using semantic similarity as fallback: {match_score:.2f}")

        logger.info(f"Match score: {match_score:.2f}, Matched: {len(matched_skills)}, Missing: {len(missing_skills)}")

        # Determine match level and generate appropriate message
        if match_score < self.POOR_MATCH_THRESHOLD:
            match_level = "poor"
            message = self._generate_poor_match_message(matched_skills, missing_skills, match_score)
            cover_letter = None
        elif match_score < self.GOOD_MATCH_THRESHOLD:
            match_level = "moderate"
            message = self._generate_moderate_match_message(matched_skills, missing_skills, match_score)
            cover_letter = self._build_cover_letter_template(cv_text, job_text, matched_skills, missing_skills, address_gaps=True)
        else:
            match_level = "good"
            message = self._generate_good_match_message(matched_skills, missing_skills, match_score)
            cover_letter = self._build_cover_letter_template(cv_text, job_text, matched_skills, missing_skills, address_gaps=False)

        # Log metrics
        if self.log_mlflow:
            with start_run(run_name="generate_cover_letter") as run:
                log_params({
                    "cv_length": len(cv_text),
                    "jd_length": len(job_text),
                    "matched_skills_count": len(matched_skills),
                    "missing_skills_count": len(missing_skills),
                    "match_level": match_level
                })
                log_metrics({
                    "match_score": match_score,
                    "cover_letter_generated": 1 if cover_letter else 0
                })

        result = {
            "match_score": round(match_score, 2),
            "match_level": match_level,
            "message": message,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "cover_letter": cover_letter
        }

        logger.info(f"Result - Level: {match_level}, Score: {match_score:.2f}, CL Generated: {cover_letter is not None}")
        return result

    def _generate_poor_match_message(self, matched_skills, missing_skills, score):
        """Generate message for poor match (<60%)."""
        msg = (
            f"\n{'='*80}\n"
            f"MATCH ASSESSMENT: POOR FIT - {score*100:.1f}%\n"
            f"{'='*80}\n"
            f"This role may not be the best fit for your current profile.\n\n"
            f"Your Matched Skills ({len(matched_skills)}): {', '.join(matched_skills) if matched_skills else 'None'}\n"
            f"Critical Missing Skills ({len(missing_skills)}): {', '.join(missing_skills[:10]) if missing_skills else 'None'}\n\n"
            f"RECOMMENDATION:\n"
            f"Consider gaining hands-on experience in the following key areas before applying:\n"
        )
        for skill in missing_skills[:5]:
            msg += f"  - {skill}\n"
        msg += (
            f"\nOnce you have developed these skills, this role could be an excellent opportunity.\n"
            f"{'='*80}\n"
        )
        return msg

    def _generate_moderate_match_message(self, matched_skills, missing_skills, score):
        """Generate message for moderate match (60-80%)."""
        msg = (
            f"\n{'='*80}\n"
            f"MATCH ASSESSMENT: MODERATE FIT - {score*100:.1f}%\n"
            f"{'='*80}\n"
            f"You have solid relevant experience with some skill gaps to address.\n\n"
            f"Your Strongest Matches ({len(matched_skills)}): {', '.join(matched_skills) if matched_skills else 'None'}\n"
            f"Skills to Develop ({len(missing_skills)}): {', '.join(missing_skills[:8]) if missing_skills else 'None'}\n\n"
            f"Below is a cover letter that highlights your strengths and addresses your skill gaps.\n"
            f"We recommend emphasizing your willingness to learn and grow in the areas where you're less experienced.\n"
            f"{'='*80}\n\n"
        )
        return msg

    def _generate_good_match_message(self, matched_skills, missing_skills, score):
        """Generate message for good match (>80%)."""
        msg = (
            f"\n{'='*80}\n"
            f"MATCH ASSESSMENT: EXCELLENT FIT - {score*100:.1f}%\n"
            f"{'='*80}\n"
            f"You are a strong candidate for this role! Your profile aligns very well with the requirements.\n\n"
            f"Your Core Strengths ({len(matched_skills)}): {', '.join(matched_skills)}\n"
            f"Minor Skill Gaps ({len(missing_skills)}): {', '.join(missing_skills[:5]) if missing_skills else 'None - Perfect Match!'}\n\n"
            f"Use the cover letter below to highlight why you're an excellent fit for this position.\n"
            f"{'='*80}\n\n"
        )
        return msg

    def _build_cover_letter_template(self, cv_text, job_text, matched_skills, missing_skills, address_gaps=False):
        """Generate professional cover letter using LLM, highlighting skill alignment."""
        
        if self.llm is None:
            logger.warning("LLM not available, falling back to template-based generation")
            return self._build_cover_letter_fallback(cv_text, job_text, matched_skills, missing_skills, address_gaps)
        
        # Extract candidate information
        cv_info = self._extract_candidate_info(cv_text)
        
        # Build a comprehensive prompt for the LLM
        matched_str = ", ".join(matched_skills[:8]) if matched_skills else "relevant technical skills"
        missing_str = ", ".join(missing_skills[:5]) if missing_skills else "specialized tools"
        
        # Truncate inputs to fit within token limits (CPU-friendly)
        cv_summary = cv_text[:1500] if len(cv_text) > 1500 else cv_text
        jd_summary = job_text[:1500] if len(job_text) > 1500 else job_text
        
        prompt = f"""Write a professional cover letter for a job application. 

CANDIDATE BACKGROUND:
{cv_summary}

JOB REQUIREMENTS:
{jd_summary}

MATCHED SKILLS: {matched_str}
"""
        
        if address_gaps and missing_skills:
            prompt += f"""SKILLS TO DEVELOP: {missing_str}

Write a cover letter that:
1. Highlights the candidate's matched skills and experience
2. Addresses the skill gaps by showing willingness to learn and adapt
3. Demonstrates enthusiasm for the role
4. Is professional, concise (3-4 paragraphs), and compelling

Cover Letter:"""
        else:
            prompt += f"""
Write a cover letter that:
1. Highlights the candidate's matched skills and experience
2. Demonstrates strong alignment with the job requirements
3. Shows enthusiasm and value proposition
4. Is professional, concise (3-4 paragraphs), and compelling

Cover Letter:"""
        
        try:
            logger.info("Generating cover letter using LLM...")
            
            # Generate using LLM
            if LANGCHAIN_AVAILABLE and hasattr(self.llm, 'invoke'):
                # LangChain interface
                response = self.llm.invoke(prompt)
                cover_letter = response.strip()
            elif hasattr(self.llm, '__call__'):
                # Direct pipeline interface
                pipeline_type = getattr(self, 'llm_type', None)
                if pipeline_type == "text2text-generation" or "t5" in str(type(self.llm)).lower():
                    # For T5-style models (text2text-generation)
                    result = self.llm(
                        prompt,
                        max_length=512,
                        min_length=200,
                        do_sample=True,
                        temperature=0.7,
                        num_return_sequences=1
                    )
                    cover_letter = result[0]['generated_text'].strip()
                else:
                    # For causal LM models (text-generation)
                    result = self.llm(
                        prompt,
                        max_new_tokens=400,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        num_return_sequences=1
                    )
                    # Extract generated text (remove prompt)
                    generated = result[0]['generated_text']
                    # Remove the prompt from the beginning if it's there
                    if generated.startswith(prompt):
                        cover_letter = generated[len(prompt):].strip()
                    else:
                        cover_letter = generated.strip()
            else:
                raise ValueError("Unknown LLM interface")
            
            # Post-process: Ensure proper formatting
            cover_letter = self._format_cover_letter(cover_letter, cv_info)
            
            logger.info(f"Successfully generated cover letter (length: {len(cover_letter)})")
            return cover_letter
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}. Falling back to template.")
            return self._build_cover_letter_fallback(cv_text, job_text, matched_skills, missing_skills, address_gaps)
    
    def _format_cover_letter(self, text: str, cv_info: dict) -> str:
        """Format the generated cover letter to ensure proper structure."""
        # Ensure it starts with a greeting
        if not text.strip().startswith(('Dear', 'To', 'Hello')):
            text = f"Dear Hiring Manager,\n\n{text}"
        
        # Ensure it ends with a closing
        if not any(text.strip().endswith(phrase) for phrase in ['Sincerely', 'Best regards', 'Thank you', 'Regards']):
            text = f"{text}\n\nSincerely,\n[Your Name]"
        
        # Clean up any weird formatting
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = text.strip()
        
        return text
    
    def _build_cover_letter_fallback(self, cv_text, job_text, matched_skills, missing_skills, address_gaps=False):
        """Fallback template-based generation if LLM fails."""
        cv_info = self._extract_candidate_info(cv_text)
        matched_str = ", ".join(matched_skills[:5]) if matched_skills else "relevant technical skills"
        missing_str = ", ".join(missing_skills[:3]) if missing_skills else "specialized tools"
        
        cover_letter = f"""Dear Hiring Manager,

I am writing to express my strong interest in the position at your organization. With my {cv_info['years']}+ years of professional experience and proven expertise in {matched_str}, I am confident that I am an excellent fit for your team.

My background demonstrates a strong alignment with your requirements. I have successfully developed and demonstrated proficiency in {', '.join(matched_skills[:3]) if matched_skills else 'core technical competencies'}, which are essential for this role. These skills have enabled me to deliver measurable results and contribute meaningfully in previous positions.

Throughout my career, I have consistently leveraged my expertise in {matched_skills[0] if matched_skills else 'technical skills'} to solve complex problems and drive innovation. I am particularly drawn to this opportunity because it aligns perfectly with my professional strengths and career aspirations."""

        if address_gaps and missing_skills:
            cover_letter += f"""

While my primary strengths lie in {', '.join(matched_skills[:2]) if len(matched_skills) >= 2 else (matched_skills[0] if matched_skills else 'core skills')}, I recognize the value of expanding my expertise in {missing_str}. I am highly motivated to develop these skills and am confident that my strong foundation will enable me to quickly master any specialized knowledge required for the role."""

        cover_letter += """

I would welcome the opportunity to discuss how my experience, skills, and enthusiasm can contribute to your team's success. Thank you for considering my application, and I look forward to speaking with you soon.

Sincerely,
[Your Name]"""

        return cover_letter

    def _extract_candidate_info(self, cv_text: str) -> dict:
        """Extract candidate information from CV."""
        text_lower = cv_text.lower()
        
        # Extract experience level from years mentioned
        years_match = None
        year_patterns = [r'(\d+)\s*(?:\+)?\s*years?', r'(\d+)\s*(?:to\s*)?years?']
        for pattern in year_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                years_match = max(int(m) for m in matches if m.isdigit())
                break
        
        experience_level = "entry-level"
        if years_match:
            if years_match >= 5:
                experience_level = "senior"
            elif years_match >= 2:
                experience_level = "mid-level"
        
        # Extract education
        has_master = "master" in text_lower or "m.s." in text_lower or "m.tech" in text_lower
        has_phd = "phd" in text_lower or "doctorate" in text_lower
        education = "PhD" if has_phd else ("Master's" if has_master else "Bachelor's")
        
        return {
            "experience_level": experience_level,
            "years": years_match or 3,
            "education": education
        }
