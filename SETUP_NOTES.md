# Day 1: WSL + Python Setup

## Installed Ubuntu LTS
```powershell
wsl --install -d Ubuntu



sudo apt update
sudo apt install python3-venv -y

python3 --version : Python 3.12.3

cd ../..
cd ~
mkdir projects
cd projects
mkdir llm_career_assistant

python3 -m venv venv
source venv/bin/activate

cd llm_career_assistant

# Install Week 1 Libraries
pip install pandas numpy scikit-learn matplotlib seaborn nltk spacy

pip freeze > requirements.txt

# Deploy to Git
# Initialize git
init git

touch .gitignore (tells Git which files or folders to ignore when you run git add . or push your code to GitHub.)
nano .gitignore 

git config --global user.name "samwise"
git config --global user.email "saman.azharr@gmail.com"

for verification:
git config --global --list

git add .
git commit -m "Initial setup: venv, libraries, requirements.txt"

git remote add origin https://github.com/saman-azhar/llm_career_assistant.git
git branch -M main
git push -u origin main


# Day 2
Dataset: https://www.kaggle.com/datasets/thedevastator/jobs-dataset-from-glassdoor

# 🗂️ Data Notes: Glassdoor Jobs Dataset

## 1. Dataset Overview
- **Name/Source:** Jobs Dataset from Glassdoor (Kaggle, by *thedevastator*)
- **Format:** CSV
- **Rows:** [to fill after inspection]
- **Columns:** [list them here after `df.columns`]

---

## 2. Content Description
- **Job Title Column:** [column name here]
- **Job Description Column:** [column name here]
- **Other Columns:** [company, location, salary, rating, etc.]
- **Notes:** [any immediate impressions]

---

## 3. Quality & Noise Observations
- **Duplicates:** [yes/no, how many]
- **Empty/Short Descriptions:** [count % of rows]
- **Formatting issues:** (HTML tags, bullet symbols, emojis, etc.)
- **Example noisy snippet:**
  > “We offer FREE lunch 🍕 and gym memberships. Requirements: Python, SQL, ML pipelines.”
- **Decision:** Remove perks/culture mentions, keep requirement sections.

---

## 4. Cleaning Decisions
- Keep:
  - Job Title
  - Job Description
- Drop:
  - [columns you won’t use]
- Cleaning pipeline steps:
  1. Deduplicate rows  
  2. Drop very short/empty descriptions  
  3. Strip HTML, symbols, emojis  
  4. Lowercasing, tokenization  
  5. Remove stopwords  
  6. Lemmatization  

---

## 5. Final Dataset Plan
- **Expected rows after cleaning:** [e.g. ~10,000 JDs]
- **Output file:** `data/job_descriptions_clean.csv`
- **Next step:** exploratory analysis (word frequencies, skills, bigrams, etc.)

---


# kaggle
Created API Token

create private dir:
mkdir -p ~/.kaggle
move from local to linux:
mv '/mnt/c/Users/Lenovo -E16/Downloads/kaggle.json' ~/.kaggle/

pip install kaggle

set permission:
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets list -s glassdoor

mkdir data
cd data

kaggle datasets download -d thedevastator/jobs-dataset-from-glassdoor

unzip jobs-dataset-from-glassdoor.zip

DATA ANALYSIS:
pip install jupyterlab
pip install plotly

pip install jupyter

Week 2
Progress Summary — Thursday

Explored job title distribution — found 327 unique titles and analyzed top occurrences.

Created a job title standardization logic — merged similar titles (e.g., “Senior Data Scientist” → “Data Scientist”).

Refined category mapping — condensed hundreds of roles down to six clean, usable labels:
→ Data Scientist, Data Engineer, Data Analyst, ML Engineer, Software Engineer, Other.

Saved the processed dataset — preserving your progress before moving to modeling.

Committed and pushed changes to GitHub — project now synced and versioned properly.


Why we’re using Logistic Regression (LR)

Even though it has “regression” in the name, Logistic Regression is actually one of the most reliable classification algorithms — and that’s exactly what we’re doing here: classifying job descriptions into categories.

Reasons it’s a solid baseline:

Simple but effective: It’s often the go-to starting point for text classification before trying more complex models like SVM or deep learning.

Interpretable: You can inspect the learned weights to see which words contribute most to each class (great for explainability).

Fast to train: Works efficiently even on large TF-IDF matrices.

Strong baseline: If tuned well, it can perform surprisingly close to more complex models — great for benchmarking.

By default, scikit-learn uses max_iter=100.

For text data (like TF-IDF vectors), the feature space is often huge → it takes longer for the optimizer to converge.

So we increase it to 300 to give the algorithm more room to find the optimal solution and avoid warnings like:

Model achieves 73% accuracy and 0.67 weighted F1 — performs well on dominant “Data Scientist” class but struggles to recall minority job titles due to class imbalance and linear model constraints. Precision remains high, indicating correct predictions when confident.

Setting class_weight='balanced' told Logistic Regression to penalize mistakes on small classes more heavily.
Before, the model could ignore “ML Engineer” (only 7 samples) and still score well.
Now, each class matters equally in training. From 73% → 86% accuracy and 0.67 → 0.86 F1 — that’s a massive improvement just by adding class_weight='balanced'

Week 2 – Classic ML Baseline

Goal: Build a baseline text classification model to categorize job descriptions into roles (Data Scientist, Data Engineer, etc.) using classic ML methods.

Work Done:

Preprocessed cleaned text using TF-IDF vectorization.

Trained a Logistic Regression model (baseline) and evaluated with accuracy, F1 score, and classification report.

Addressed class imbalance using class_weight='balanced', improving performance significantly.

Results:

Metric	Before Balancing	After Balancing
Accuracy	0.73	0.86
F1 Score (weighted)	0.67	0.86