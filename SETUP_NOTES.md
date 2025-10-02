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

