# career_assistant/rag_pipeline/generator_template.py
"""
Template-based cover letter generator.

IDEAL RESULT approach: Uses skill matching and intelligent templates
to generate professional cover letters without LLM inference.

This demonstrates the target behavior for the system.
"""

import re
from career_assistant.mlflow_logger import start_run, log_params, log_metrics
from career_assistant.utils.logger import get_logger
from career_assistant.preprocessing.semantic_matching import compute_similarity_runtime

logger = get_logger(__name__)


class CoverLetterGeneratorTemplate:
    """Generate cover letters using intelligent templates with skill matching."""
    
    # Matching thresholds
    POOR_MATCH_THRESHOLD = 0.60
    GOOD_MATCH_THRESHOLD = 0.80
    
    def __init__(self, log_mlflow=True):
        self.log_mlflow = log_mlflow

    def generate_cover_letter(self, cv_text: str, job_text: str):
        """
        Generate cover letter with matching score assessment using templates.
        
        Returns: dict with keys:
            - match_score: float between 0-1 (based on skill overlap)
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
        total_jd_skills = len(matched_skills) + len(missing_skills)
        if total_jd_skills > 0:
            match_score = len(matched_skills) / total_jd_skills
        else:
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
            with start_run(run_name="generate_cover_letter_template") as run:
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
        """Build professional cover letter template highlighting skill alignment."""
        
        # Extract candidate information
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
