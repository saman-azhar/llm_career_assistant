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
Columns to be used: clean_text, Simplified Job Title

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
**Columns to be used:** cleaned_resume, Category

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
* Performed text cleaning and preprocessing: lowercasing, tokenization, lemmatization, and stopword removal.
* **Identified high-frequency words in both JD and CV datasets that could bias similarity scoring** (e.g., “data”, “experience”, “team”, “work”, etc.).
* Created **custom stopword lists** for both datasets and removed them to ensure semantic matching focuses on meaningful content.
* Analyzed CV categories and filtered to retain relevant roles aligned with JD dataset (e.g., Data Scientist, Data Engineer, Data Analyst, ML Engineer, Software Engineer).

### Exploratory Analysis of CV Data

* Examined word frequencies before and after removing custom stopwords.
* Visualized category distributions and token counts to confirm clean datasets.

### Deliverable

`03_resume_preprocessing.ipynb` containing:

* Cleaning and preprocessing pipelines for CV and JD datasets
* Custom stopword lists
* Visualizations of cleaned datasets
* Exported ready-to-use datasets for semantic matching

---


## Summary of Progress
**Week 1:** Data cleaning, title standardization, and exploration complete.  
**Week 2:** Baseline ML models (LR, SVM) implemented and evaluated.  
**Next Step:** Move toward semantic matching using SentenceTransformers (Week 3).

---

**Repository Status:**  
- Environment configured and versioned with `requirements.txt`.  
- Data and model experiments pushed to GitHub.  
- Ready for LLM/NLP pipeline integration in upcoming phases.

---