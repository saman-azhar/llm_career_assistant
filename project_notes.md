# Project Notes

## 1. Dataset Overview

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

**Expected Output:**  
`data/job_descriptions_clean.csv` – ready for EDA and modeling.

---

## 2. Week 1 – Data Preparation & Exploration

### Key Activities
- Loaded dataset from Kaggle.
- Inspected structure and contents (`df.info()`, `df.head()`).
- Cleaned and standardized job titles.
- Created a mapping to reduce 300+ titles into 6 primary categories:
  - Data Scientist  
  - Data Engineer  
  - Data Analyst  
  - ML Engineer  
  - Software Engineer  
  - Other

### Exploratory Analysis
- Generated frequency plots for top job titles.
- Visualized token counts and word frequencies.
- Stored cleaned dataset for reproducibility.

### Deliverable
`notebooks/week1_data_preprocessing.ipynb` containing:
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
`notebooks/week2_classification_baseline.ipynb` containing:
- Preprocessing
- Model training and evaluation
- Comparative analysis (LR vs SVM)

---

## 4. Summary of Progress
**Week 1:** Data cleaning, title standardization, and exploration complete.  
**Week 2:** Baseline ML models (LR, SVM) implemented and evaluated.  
**Next Step:** Move toward semantic matching using SentenceTransformers (Week 3).

---

**Repository Status:**  
- Environment configured and versioned with `requirements.txt`.  
- Data and model experiments pushed to GitHub.  
- Ready for LLM/NLP pipeline integration in upcoming phases.

---

## 3. Week 3 – Semantic Matching

### Resume Dataset – Analysis & Preprocessing Summary

**Dataset Overview:**

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
| Saved file             | `data/processed/cleaned_resume_dataset.csv` |

---