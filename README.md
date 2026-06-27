# 🎫 IT Support Ticket Data — 3-Stage DVC Pipeline & Prod Labeling

> **Topic:** Data Management in MLOps
>
> **Goal:** Demonstrate a complete MLOps lifecycle (Immutable Data Lineage) using DVC Pipelines and automated API ingestion into a production-grade infrastructure.

---

## 🛠️ Project Overview

### What tool was used for the markup?
We use **[Label Studio](https://labelstud.io/)** as our primary data annotation and markup tool. It provides a highly customizable UI that allows annotators to read the text of an IT support ticket and classify it into specific departments (e.g., "Hardware Support", "Network Operations").

### How to run/open the markup tool?
Label Studio is hosted locally via Docker Compose alongside a robust PostgreSQL database. To run it:
1. Open your terminal in the project directory.
2. Run `cd docker && docker compose up -d` to spin up the infrastructure.
3. Open a web browser and navigate to `http://localhost:8080`.
4. Log in using `admin@example.com` and the password `admin123`.

### How does dataset versioning work?
Data versioning is handled entirely by **[DVC (Data Version Control)](https://dvc.org/)**. 
- Instead of manually tracking files like `dataset_v1.csv` or `dataset_final.json`, DVC cryptographically hashes every dataset file and stores it in a local **MinIO (S3-compatible) Vault**.
- The `dvc.yaml` file acts as a pipeline: it tracks the transformation from the raw CSV source into the JSON format needed for Label Studio.
- Finally, when markup is finished, we export the annotations back into the workspace and run `dvc add` and `dvc push` to safely lock the "Gold Standard" dataset into the cold-storage vault.

### What tasks is this data planned for in the future?
The labeled dataset (our "Gold Standard") is being prepared to train an **NLP (Natural Language Processing) Machine Learning model**. 
In the future, this model will be deployed to automatically read incoming IT support tickets and predict which department they should be routed to, entirely eliminating the need for human triage agents.

---

## 🚀 Quick Start Guide

### 1. Launch Prod Infrastructure
Start PostgreSQL, MinIO, and Label Studio:
```bash
cd docker && docker compose up -d && cd ..
python -m venv .venv
source .venv/bin/activate
pip install kaggle dvc "dvc[s3]" requests python-dotenv
```

### 2. Stage 1: The Raw Source
Download the raw data from Kaggle and freeze it with DVC.
```bash
bash scripts/download_raw.sh
dvc add dataset/raw_tickets.csv
```

### 3. Stage 2: The DVC Pipeline (Transformation)
Generate a sample payload for Label Studio using our reproducible pipeline:
```bash
dvc repro
```
*DVC will automatically run `src/tokenize_and_sample.py` and generate `dataset/sample_tickets.json`.*

### 4. Stage 3: API Ingestion & Annotation
Open http://localhost:8080 (admin@example.com / admin123) and create a project named **"IT Support Ticket Classification"** (ID: 1).

Paste the following XML code into the project settings (Labeling Interface -> Code):
```xml
<View>
  <Header value="IT Support Ticket" />
  <Text name="text" value="$text" />
  <Header value="Department Classification" />
  <Choices name="department" toName="text" choice="single" showInline="true">
    <Choice value="Technical Support" />
    <Choice value="Network Operations" />
    <Choice value="Software Development" />
    <Choice value="Hardware Support" />
    <Choice value="User Accounts" />
    <Choice value="Billing &amp; Licensing" />
  </Choices>
</View>
```

**(Optional) Fetch your API Key:**
Go to the top right corner of Label Studio -> **Account & Settings**. Copy your **Access Token** and paste it into your `docker/.env` file as `LABEL_STUDIO_AUTH_TOKEN`. *(Note: Our automated Python scripts use session login and actually bypass the need for this key, but it is good practice to configure it!)*

Automatically push your tasks into the PostgreSQL database via our API script:
```bash
python scripts/api_sync_tasks.py --project-id 1 --input dataset/sample_tickets.json
```
*Complete some basic labeling on the tickets in the UI.*

### 5. Exporting the "Gold Standard"
Once labeling is complete, automatically pull the results from the database and push them into cold S3 storage:
```bash
# Pull labeled data via API
python scripts/export_annotations.py --project-id 1

# Lock the final dataset into DVC
dvc add dataset/labeled_tickets.json
git add dataset/labeled_tickets.json.dvc
git commit -m "Update labeled dataset"
dvc push
```

You are all set! You have successfully built a perfect, automated MLOps data management platform. 🎯
