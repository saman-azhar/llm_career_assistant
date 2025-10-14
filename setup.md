# Setup Notes

## 1. Environment Setup (WSL + Python)

### Install Ubuntu and Python Environment

```bash
wsl --install -d Ubuntu

sudo apt update
sudo apt install python3-venv -y

python3 --version
```

### Project Directory Setup

```bash
cd ~
mkdir projects
cd projects
mkdir llm_career_assistant
cd llm_career_assistant
```

### Create Virtual Environment and Activate

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Core Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk spacy
pip freeze > requirements.txt
```

---

## 2. Git Configuration

### Initialize Git and Set Up `.gitignore`

```bash
git init
touch .gitignore
nano .gitignore
```

**Example `.gitignore` content:**

```
venv/
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
data/
```

### Configure Git User

```bash
git config --global user.name "samwise"
git config --global user.email "abc@gmail.com"
git config --global --list
```

### First Commit and Push to GitHub

```bash
git add .
git commit -m "Initial setup: venv, libraries, requirements.txt"
git remote add origin https://github.com/saman-azhar/llm_career_assistant.git
git branch -M main
git push -u origin main
```

---

## 3. Dataset Setup (Kaggle)

### Create Kaggle API Token

1. Go to your Kaggle account → **Settings → Create New API Token**
2. Download the `kaggle.json` file.

### Move API Token to WSL and Set Permissions

```bash
mkdir -p ~/.kaggle
mv '/mnt/c/Users/Lenovo -E16/Downloads/kaggle.json' ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Verify Kaggle CLI

```bash
pip install kaggle
kaggle datasets list -s glassdoor
```

### Download Dataset

```bash
mkdir data
cd data
kaggle datasets download -d thedevastator/jobs-dataset-from-glassdoor
unzip jobs-dataset-from-glassdoor.zip
```

---

## 4. Data Analysis Environment

### Install Jupyter and Visualization Libraries

```bash
pip install jupyter jupyterlab plotly
```
