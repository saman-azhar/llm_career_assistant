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

---

## 5. Docker & Qdrant Setup (Vector Database Integration)

### Install Docker Desktop on Windows

1. Download from [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
2. During installation, **enable WSL 2 integration** and select your Ubuntu distro.

After installation, verify Docker is running correctly:

```bash
docker --version
docker run hello-world
```

If it runs successfully, Docker is now accessible from WSL.

---

### Fix Docker Permission Issues (If Any)

If you see an error like
`permission denied while trying to connect to the Docker daemon socket`,
run the following inside your Ubuntu WSL terminal:

```bash
sudo usermod -aG docker $USER
newgrp docker
sudo service docker start
```

Then test again:

```bash
docker ps
```

---

### Run Qdrant Vector Database (Docker)

Pull and start the latest **Qdrant** container:

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

This:

* Exposes Qdrant on port `6333`
* Persists vector data under `qdrant_storage` in your project directory

Verify it’s running:

```bash
docker ps
```

You should see a container named `qdrant/qdrant` listed as **Up**.

---

### Python Client Installation and Connection Test

Install the official Qdrant client in your virtual environment:

```bash
pip install qdrant-client
```

Then test connectivity:

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
print(client.get_collections())
```

Expected output (after ingestion):

```
collections=[CollectionDescription(name='career_vectors')]
```

---

### Create and Verify Qdrant Collection (Optional Manual Step)

If you need to create the collection manually:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name="career_vectors",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
```

Check it exists:

```python
res = client.get_collection("career_vectors")
print(res.vectors_config)
```

Expected output:

```
size=768 distance=<Distance.COSINE: 'Cosine'>
```

---

### Confirm Ingestion Pipeline

Run your ingestion script (this will push embeddings into Qdrant):

```bash
python -m career_assistant.rag_pipeline.ingest
```

Verify the vector count:

```python
res = client.get_collection("career_vectors")
print(res.vectors_count)
```

If it prints a numeric value (e.g., `125`), ingestion succeeded.

---

### Optional Cleanup Commands

Stop or remove the Qdrant container if needed:

```bash
docker stop $(docker ps -q --filter ancestor=qdrant/qdrant)
docker rm $(docker ps -a -q --filter ancestor=qdrant/qdrant)
```


