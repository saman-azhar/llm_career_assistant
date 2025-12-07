# LLM Career Assistant - Pipeline Approaches

## Overview

This project implements an intelligent RAG-based cover letter generation system with two complementary approaches:

### 1. **LLM-Based Approach** (`rag_pipeline.py`)

**Status:** Primary approach - Uses transformer models for intelligent cover letter generation

**Features:**
- Uses FLAN-T5 (or configurable CPU-friendly LLM)
- Semantic matching for skill extraction
- 3-tier matching assessment (Poor/Moderate/Good)
- LLM generates contextual cover letters

**Advantages:**
- Shows deep learning expertise
- Demonstrates LLM integration skills
- Can generate diverse, contextual content
- Impressive for recruiters (shows AI knowledge)

**Challenges:**
- Token limit constraints (512 tokens for FLAN-T5)
- CPU inference is slower
- Risk of hallucination/repetition with small models
- Requires careful prompt engineering

**Run:**
```bash
python -m career_assistant.rag_pipeline.rag_pipeline
```

---

### 2. **Template-Based Approach** (`rag_pipeline_template.py`)

**Status:** Ideal result - Shows target behavior using intelligent templates

**Features:**
- Skill-based matching algorithm
- Professional cover letter templates
- 3-tier assessment with conditional generation
- Fast, deterministic output

**Advantages:**
- **Zero hallucination** - exact skill data used
- **Fast inference** - no model loading
- **Reliable** - same input = same output
- **Scalable** - lightweight architecture
- Shows pragmatic engineering decisions

**Use Case:**
- Demonstrate ideal system behavior
- Show scalability without resource constraints
- Illustrate proper tier-based logic

**Run:**
```bash
python -m career_assistant.rag_pipeline.rag_pipeline_template
```

---

## Comparison

| Feature | LLM-Based | Template-Based |
|---------|-----------|----------------|
| **Generation Method** | Transformer model | Intelligent templates |
| **Skill Usage** | Input context | Direct data source |
| **Hallucination Risk** | Moderate | None |
| **Speed** | Slower (model inference) | Fast |
| **Reproducibility** | Non-deterministic | Deterministic |
| **Resource Usage** | GPU/CPU intensive | Minimal |
| **Scalability** | Requires GPU for prod | Built for scale |
| **Recruiter Appeal** | Shows LLM skills | Shows engineering maturity |

---

## Architecture

### Pipeline Flow

```
CV + JD Text
    ↓
[RAG Retrieval] → Retrieve similar chunks from vector DB
    ↓
[Semantic Matching] → Extract skills from CV and JD
    ↓
[Skill Scoring] → Calculate match_score = matched/(matched+missing)
    ↓
[Assessment Decision]
├─ score < 0.60  → POOR: No cover letter
├─ 0.60-0.80    → MODERATE: Cover letter addressing gaps
└─ score > 0.80  → GOOD: Strong cover letter
    ↓
[Generation]
├─ LLM-Based:    Use transformer model + skill context
└─ Template:     Use professional template + actual skills
    ↓
[Output] → Assessment message + Cover letter + Metadata
```

---

## Skill Extraction

Both approaches use the same semantic matching system:

```python
def extract_skills(text):
    # Finds skills from KNOWN_SKILLS list in text
    # Uses regex for multi-word skills (e.g., "machine learning")
    # Returns: list of matched skills
```

**KNOWN_SKILLS** includes 100+ technical skills across:
- Programming languages
- ML/AI frameworks
- Data tools
- Cloud platforms
- DevOps tools
- And more...

---

## 3-Tier Assessment Logic

### Tier 1: Poor Match (< 60%)
```
Message: "This role may not be the best fit"
Skills: Shows matched + missing skills
Action: Lists skills to develop
Letter: NOT generated
```

### Tier 2: Moderate Match (60-80%)
```
Message: "You have solid experience with skill gaps"
Skills: Lists strengths and gaps
Action: Generates letter addressing gaps
Letter: Generated with honest gap disclosure
```

### Tier 3: Good Match (> 80%)
```
Message: "Excellent fit for this role"
Skills: Highlights core strengths
Action: Professional cover letter
Letter: Generated celebrating strengths
```

---

## Generator Classes

### `CoverLetterGenerator` (LLM-based)
- Initializes and manages transformer model
- Generates contextual cover letters using LLM
- Falls back to template if LLM unavailable
- Configurable model selection

**Key Methods:**
- `generate_cover_letter()` - Main generation method
- `_initialize_llm()` - Set up transformer pipeline
- `_generate_poor/moderate/good_match_message()` - Assessment messages

### `CoverLetterGeneratorTemplate` (Template-based)
- Uses professional templates
- Fills templates with actual skill data
- Zero dependencies on LLM
- Fast and deterministic

**Key Methods:**
- `generate_cover_letter()` - Main generation method
- `_build_cover_letter_template()` - Create letter from template
- `_extract_candidate_info()` - Parse CV for experience level

---

## Usage Examples

### Using LLM Approach
```python
from career_assistant.rag_pipeline.rag_pipeline import run_rag_pipeline

result = run_rag_pipeline(cv_text, jd_text)
print(f"Match: {result['match_level']} ({result['match_score']})")
print(result['assessment_message'])
print(result['cover_letter'])
```

### Using Template Approach
```python
from career_assistant.rag_pipeline.rag_pipeline_template import run_rag_pipeline_template

result = run_rag_pipeline_template(cv_text, jd_text)
print(f"Match: {result['match_level']} ({result['match_score']})")
print(result['cover_letter'])  # Ideal result
```

---

## Recommended Usage

### For Recruiter Pitch
1. **Show LLM approach** - Demonstrates AI/ML skills
2. **Show template approach** - Demonstrates engineering maturity
3. **Explain decision** - "Chose templates for production due to resource constraints, but showcasing scalable architecture"

### For Production Deployment
- **Short-term:** Use template approach (fast, reliable, scalable)
- **Long-term:** GPU-based LLM approach for more sophisticated generation

### For Portfolio/GitHub
- Keep both implementations visible
- Add comparison documentation
- Show experimentation journey
- Explain pragmatic trade-offs

---

## Configuration

Both pipelines use the same config file (`config/config.yml`):

```yaml
generator:
  model_name: "google/flan-t5-base"
  max_tokens: 350
  temperature: 0.7
  top_p: 0.9
```

---

## Performance Metrics

### LLM-Based
- Inference time: ~2-5 seconds (CPU)
- Memory: ~2-4GB
- Token efficiency: 350 tokens avg

### Template-Based
- Generation time: ~100ms
- Memory: ~10MB
- Always consistent output

---

## Future Enhancements

1. **GPU Deployment** - Run LLM on cloud (AWS SageMaker, Replicate, etc.)
2. **Fine-tuning** - Train on recruitment data
3. **Hybrid Approach** - Use LLM for specific tiers only
4. **User Feedback** - Learn from acceptance rates
5. **Multi-language** - Support different languages
6. **A/B Testing** - Compare LLM vs template quality

---

## See Also

- `career_assistant/rag_pipeline/retriever.py` - Vector similarity search
- `career_assistant/preprocessing/semantic_matching.py` - Skill extraction
- `test_full_pipeline.py` - End-to-end testing
- `test_matching_levels.py` - Tier validation tests

