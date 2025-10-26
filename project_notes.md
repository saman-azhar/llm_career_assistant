# Project Notes

## 1. Job Description Dataset Overview

**Dataset:** Jobs Dataset from Glassdoor (Kaggle, by *thedevastator*)  
**Format:** CSV  
**Purpose:** Used to train and evaluate job title classification models.

### Initial Observations
- Contains job listings with attributes such as title, description, company, location, salary, and rating.
- Focus columns:
  - `Job Title`
  - `Job Description`
- Irrelevant columns dropped (e.g., company, salary, perks).

### Cleaning Strategy
1. Remove duplicates.
2. Drop empty or very short descriptions.
3. Strip HTML, symbols, and emojis.
4. Lowercase, tokenize, and remove stopwords.
5. Lemmatize text.

**Job Title Simplification:**
Created a mapping to reduce 300+ titles into 5 primary categories:
  - Data Scientist  
  - Data Engineer  
  - Data Analyst  
  - ML Engineer  
  - Software Engineer  

### Exploratory Analysis
- Generated frequency plots for top job titles.
- Visualized token counts and word frequencies.
- Stored cleaned dataset for reproducibility.

### Final Dataset Summary
File name: data/cleaned_job_data_final.csv
Columns to be used: cleaned_job_description, simplified_job_title

---

**2. Resume Dataset Overview:**

* Source: [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
* Total Records (raw): 962 resumes
* Text column: `Resume`
* Label column: `Category`

---

### Data Cleaning Steps

1. **Removed duplicates and empty rows** – 796 invalid/duplicate entries dropped.
2. **Standardized text**

   * Lowercased all text.
   * Removed HTML tags, symbols, emojis, and URLs.
3. **Tokenization and stopword removal** using NLTK.
4. **Lemmatization** to normalize word forms.
5. **Removed generic/filler terms**: “experience”, “project”, “company”, “description”, “detail”, “requirement”, “technology”, “responsibility”, “maharashtra”.
6. **Preserved domain-relevant tokens**: programming languages, tools, technical terms, punctuation for abbreviations (like “C++”, “.NET”).
7. **Filtered resumes by word length** – kept records with > 100 words to ensure quality and relevance.

---

### Exploratory Analysis

* **Category Distribution (before cleaning):** 25 unique job categories.
* **After filtering:** 12 categories retained for variation and relevance.

  * Examples: *Data Science, Business Analyst, Web Designing, Automation Testing, Blockchain, SAP Developer, DevOps Engineer,* etc.
* **Average resume length:** ~300 words.
* **Most frequent words:** `data`, `database`, `system`, `sql`, `java`, `developer`, `testing`, `management`, `software`, `user`.

---

### Final Dataset Summary

| Metric                 |                                       Count |
| :--------------------- | ------------------------------------------: |
| Records retained       |                                         166 |
| Unique categories      |                                          12 |
| Avg. tokens per resume |                                         300 |
| Saved file             | `data/cleaned_resume_data_final.csv` |

**File name:** data/cleaned_resume_final.csv
**Columns to be used:** cleaned_resume, category

---

## Week 1 – Data Preparation & Exploration

### Key Activities
- Loaded dataset from Kaggle.
- Inspected structure and contents (`df.info()`, `df.head()`).
- Cleaned and standardized job titles.
- Created a mapping to reduce 300+ titles into 5 primary categories

### Exploratory Analysis
- Generated frequency plots for top job titles.
- Visualized token counts and word frequencies.
- Stored cleaned dataset for reproducibility.

### Deliverable
`01_job_data_preprocessing.ipynb` containing:
- Cleaning pipeline
- Visualizations
- Exported clean dataset

---

## 3. Week 2 – Baseline Model (Logistic Regression & SVM)

### Objective
Build a baseline text classification model to categorize job descriptions into the defined job categories.

### Approach
1. Used **TF-IDF vectorization** for text features.  
2. Trained **Logistic Regression** as a baseline model.  
3. Evaluated using **accuracy** and **weighted F1 score**.  
4. Addressed class imbalance using `class_weight='balanced'`.  
5. Compared performance with an **SVM** model.

**Note:** File used is cleaned_job_data_v1.csv

### Results

| Metric | Before Balancing | After Balancing |
|---------|------------------|-----------------|
| Accuracy | 0.73 | 0.86 |
| Weighted F1 | 0.67 | 0.86 |

- Logistic Regression performed well after balancing.
- SVM achieved comparable results.
- Class imbalance awareness highlighted as key insight.

### Rationale for Logistic Regression
- Simple, interpretable baseline for text classification.
- Fast to train on TF-IDF matrices.
- Serves as a strong reference before moving to deep models.

### Deliverable
`02_ml_baseline_pipeline.ipynb` containing:
- Preprocessing- Model training and evaluationJ- Comparative analysis (LR vs SVM)

---

## Week 3 – Semantic Matching (CV & JD Datasets)

### Key Activities

* Loaded CV dataset from Kaggle.
* Performed text cleaning and preprocessing — lowercasing, tokenization, lemmatization, and stopword removal.
* Identified **high-frequency words** in both JD and CV datasets that could bias similarity scoring (e.g., “data”, “experience”, “team”, “work”).
* Created **custom stopword lists** for both datasets to ensure the model focuses on meaningful, domain-relevant terms.
* Filtered and retained relevant CV categories aligned with the JD dataset (e.g., Data Scientist, Data Engineer, Data Analyst, ML Engineer, Software Engineer).

---

### Exploratory Analysis of CV Data

* Analyzed word frequencies before and after applying custom stopwords.
* Visualized category distributions and token counts to confirm dataset balance and clarity.

---

### Semantic Similarity

* Installed and configured **SentenceTransformers**.
* Selected the `all-MiniLM-L6-v2` model — compact, fast, and reliable for semantic embedding on CPU systems.
* Encoded both **job descriptions** and **CVs** into dense vector embeddings to capture contextual meaning, not just keyword overlap.
* Computed **cosine similarity scores** to measure how closely each CV aligns with every JD.
* Structured results into a **similarity matrix (DataFrame)** with CV categories as rows and JD categories as columns and exported it as `similarity_scores.csv`.
* Built a **match scoring pipeline** to extract the **Top 5 most similar job roles** for each CV and saved results as `top_matches.csv`.

---

### Semantic Understanding

This step bridges the gap between **semantic understanding** and **practical job matching**.
Instead of raw similarity numbers, the model now provides interpretable insights:

> “Which job roles best align with this resume, and how strong is that match?”

This enables:

* Quantitative ranking of resume–job relevance.
* Skill gap analysis between matched roles.
* Generation of human-readable match summaries.

In short — the system now understands *what* the text means and *acts* on it intelligently.

---

### Use of spaCy for Skill Extraction (Didn't work)

* Integrated **spaCy** for deeper linguistic understanding and domain-specific skill extraction.
* Used the **`en_core_web_sm`** model for efficient tokenization, POS tagging, and noun phrase extraction.
* Extracted likely skill terms (nouns and noun chunks) while filtering generic terms or stopwords.
* This adds **linguistic precision** to skill matching, beyond simple word overlap.

---

### Keyword-based Skill Extraction using a Domain Vocabulary

* Expanded a **domain-specific skill vocabulary** to identify key competencies across job descriptions.
* Frequency distribution analysis confirmed strong coverage across analytical, engineering, and software domains.

---

### Skill Gap Analysis (Final Step of Week 3)

* Combined the **semantic matching output** (`top_matches.csv`) with **extracted skill data** to identify **missing or underrepresented skills** in each CV compared to its top-matched JD.
* For each CV:

  * Retrieved the **Top 1 matching job role** and its **match score**.
  * Compared skills extracted from CV vs JD to generate a **`Missing_Skills`** list.
  * Saved the summarized results as `top_matches_with_missing_skills.csv`.

---

### Insights

* High match scores (>0.7) were common within closely related domains (e.g., Data Scientist ↔ Data Science).
* Cross-domain matches (e.g., ML Engineer ↔ Data Engineer) appeared when overlapping skill sets were strong — mainly Python, SQL, and ML tools.
* The **Missing_Skills** column highlights specific skills that candidates may need to learn or emphasize for stronger alignment.
* Some resumes had **no missing skills**, indicating near-complete overlap with their top-matched JD.
* Overall, the pipeline now not only matches resumes semantically but also pinpoints *how* they fall short, forming the foundation for **targeted upskilling recommendations**.

---

### Deliverables

**`03_resume_preprocessing.ipynb`**

* Text cleaning and preprocessing pipeline for CV dataset
* Custom stopword lists
* Dataset visualizations
* Exported clean dataset ready for embedding

**`04_semantic_matching.ipynb`**

* SentenceTransformer setup & embedding generation
* Cosine similarity computation
* Match score extraction (Top 5 per resume)
* Exports: `similarity_scores.csv`, `top_matches.csv`

**`top_matches_with_missing_skills.csv`**

* Final combined file showing:

  * `CV_ID`
  * `CV_Category`
  * `Top_Match_Score`
  * `Missing_Skills`
* Used to interpret skill coverage and improvement areas for each CV.

---

## Week 4 – Production-Level Scripts & LLM-Based Automation

### Key Activities

* Refactored all preprocessing, semantic matching, and LLM-based cover letter scripts for **production readiness**.
* Converted hardcoded paths and parameters into **CLI-friendly arguments** for flexible execution.
* Introduced **default values** for optional parameters to allow smooth first-time execution without manual tweaking.
* Ensured scripts handle both **CSV and plain text inputs** with automatic column validation (`Resume` for CVs, `Job Description` for JDs).
* Encapsulated core logic in **modular functions**, enabling reuse across pipelines, batch processing, or APIs.
* Conducted first **end-to-end test run** of semantic matching + LLM cover letter generation using real CV–JD pairs.
* Observed limitations in current cover letter generation — repetitive and non-contextual language — prompting plan for **RAG-based enhancement** in upcoming iteration.

---

### CLI & Optional Arguments

* Users can now run scripts with a simple, positional interface or include optional arguments:

  ```bash
  python resume_preprocessing.py input.csv output.csv --min_resume_len 50 --min_word_count 100
  python job_preprocessing.py input.csv output.csv --min_desc_len 50
  python semantic_matching.py cv_cleaned.csv job_cleaned.csv outputs --top_n 5
  python llm_coverletter.py cv.txt jd.txt outputs --model google/flan-t5-small --max_length 130 --min_length 40
  ```

* Optional args allow fine-tuning: minimum resume length, top N matches, summarization length, or LLM model selection.

---

### Production-Level Preprocessing

* **Resume preprocessing:** cleaned text, removed duplicates, filtered short resumes, and kept only relevant categories.
* **Job preprocessing:** cleaned job descriptions, simplified titles, and removed irrelevant entries.
* Both scripts now save processed data as CSV for downstream tasks.

---

### Semantic Matching

* Scripts now accept **CLI arguments** for CV and JD CSV paths, output folder, SentenceTransformer model, and top N matches.
* Sentence embeddings and cosine similarity computation are fully modular.
* Top N matches and missing skills are generated automatically and exported as CSVs for analysis.
* Entire workflow is **pipeline-ready**, enabling integration with future automation steps.
* **New this week:** validated semantic matching results as input context for cover letter generation, ensuring CV–JD alignment.

---

### LLM-Based Job Description Summarization & Cover Letter Drafting

* Integrated an **open-source instruction-tuned LLM**: `google/flan-t5-small` via Hugging Face `transformers`.
* Each job description is summarized into **3–4 bullet points**.
* Cover letters are drafted automatically using **CV skills**, **JD summaries**, and **semantic match insights**.
* **Observation:** while structure and coherence were consistent, output lacked contextual fluency — confirming the need to move beyond vanilla LLM prompting.
* Next step: enhance pipeline with a **vector database (e.g., Qdrant or FAISS)** to enable **RAG-style retrieval augmentation** for more accurate, context-aware generation.

---

### Workflow Integration

* CLI-friendly preprocessing feeds into semantic matching.
* Semantic matching outputs (Top N matches, missing skills) feed into LLM-based cover letter generation.
* Entire Week 4 workflow forms a **complete end-to-end pipeline** from raw CVs and job descriptions to actionable insights and draft cover letters.
* **Next Milestone:** migrate to modular architecture (`/src/` structure), add Qdrant-based RAG pipeline, and begin containerization in Week 5.

---

### Deliverables

**`preprocessing_cv.py`**

* Clean, filter, and export CV datasets with optional parameters for minimum length and categories.

**`preprocessing_jd.py`**

* Clean and simplify job descriptions with optional parameters for minimum description length.

**`semantic_matching.py`**

* Generate embeddings, compute cosine similarity, extract Top N matches, and calculate missing skills.
* Outputs: `similarity_scores.csv`, `top_matches.csv`, `missing_skills_analysis.csv`.

**`llm_integration.py`**

* Summarize JDs and generate draft cover letters automatically using LLMs.
* Outputs: JD summaries (`jd_summary_*.txt`) and cover letters (`cover_letter_*_*.txt`).

**`cover_letter_generation.py`**

* Summarize JDs and generate draft cover letters automatically using LLMs and semantic matching output from week 3.
* Outputs: JD summaries (`jd_summary_*.txt`) and cover letters (`cover_letter_*_*.txt`).
* Planned improvement: **upgrade to RAG pipeline with vector database** for enhanced contextual understanding and generation.

---

## Summary of Progress
**Week 1:** Data cleaning, title standardization, and exploration complete.  
**Week 2:** Baseline ML models (LR, SVM) implemented and evaluated.  
**Week 3:** Implemented semantic matching using transformer embeddings and extracted missing skills to identify CV–JD alignment gaps.

---

**Repository Status:**

* All preprocessing, semantic matching, and LLM cover letter scripts refactored for **production and CLI use**.
* Scripts accept **positional and optional arguments**, support CSV/TXT inputs, and handle batch processing.
* Semantic embeddings, top N match extraction, and missing skill analysis fully automated.
* Open-source LLM (`google/flan-t5-small`) integrated for **JD summarization and cover letter drafting**.
* Outputs consistently saved to structured directories (`outputs/`), ready for downstream evaluation.
* Environment captured in `requirements.txt`; repository is **pipeline-ready** for end-to-end automation.

---