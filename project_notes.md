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
  * Saved the summarized results as `week3_missing_skills_summary.csv`.

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

**`week3_missing_skills_summary.csv`**

* Final combined file showing:

  * `CV_ID`
  * `CV_Category`
  * `Top_Match_Score`
  * `Missing_Skills`
* Used to interpret skill coverage and improvement areas for each CV.

---

## Summary of Progress
**Week 1:** Data cleaning, title standardization, and exploration complete.  
**Week 2:** Baseline ML models (LR, SVM) implemented and evaluated.  
**Week 3:** Implemented semantic matching using transformer embeddings and extracted missing skills to identify CV–JD alignment gaps.

---

**Repository Status:**  
- Environment configured and versioned with `requirements.txt`.  
- Data and model experiments pushed to GitHub.  
- Ready for LLM/NLP pipeline integration in upcoming phases.

---