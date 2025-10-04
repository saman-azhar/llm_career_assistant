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
