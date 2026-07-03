# 🎫 IT Support Ticket Classification MLOps Platform

> **Goal:** A complete, production-grade MLOps lifecycle demonstrating immutable data lineage (DVC), containerized data labeling (Label Studio), distributed deep learning fine-tuning (Ray Train + DistilBERT), tracking (MLflow), and automated version promotion in a Model Registry.

---

## 🛠️ Project Overview

This platform covers the full MLOps lifecycle:

```
[Kaggle Source] ──► raw_tickets.csv (Stage 1)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    prepare (Stage 2)              train (Stage 4)
             │                           │
    sample_tickets.json                  ▼
             │                   [Ray Torch Workers]
             ▼                           │
    [Label Studio API] (Stage 3)         ├── Fine-tune DistilBERT
             │                           ├── Log metrics/params to MLflow
    labeled_tickets.json (Gold)          └── Export best checkpoint
                                                 │
                                                 ▼
                                       register_model (Stage 5)
                                                 │
                                                 ▼
                                       [MLflow Model Registry]
```

### Infrastructure Services (Docker Compose)
All infrastructure runs locally and self-contained via Docker:
- **Label Studio (`:8080`)**: Annotation interface.
- **MinIO (`:9000/:9001`)**: Local S3 vault storing DVC datasets and MLflow model artifacts.
- **PostgreSQL (`:5432`)**: Database backend for Label Studio and MLflow.
- **MLflow Tracking Server (`:5001`)**: Centralized experiment tracker and Model Registry.

---

## 🚀 Quick Start Guide

### 1. Launch MLOps Stack
Start PostgreSQL, MinIO, MLflow, and Label Studio:
```bash
cd docker && docker compose up -d && cd ..
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Stage 1: The Raw Source
Download the raw data from Kaggle and track it with DVC:
```bash
bash scripts/download_raw.sh
dvc add dataset/raw_tickets.csv
```

### 3. Stage 2: Data Preparation & Labeling (Optional)
Generate a sample payload and ingest it into Label Studio:
```bash
# Run DVC prepare stage
dvc repro prepare

# Sync tasks into Label Studio (ensure Project ID 1 is created)
python scripts/api_sync_tasks.py --project-id 1 --input dataset/sample_tickets.json
```
Once labeling is complete, export the annotations and lock them into S3 via DVC:
```bash
python scripts/export_annotations.py --project-id 1
dvc add dataset/labeled_tickets.json
dvc push
```

### 4. Stage 4: Model Training (Ray Train & DistilBERT)
We fine-tune a pre-trained **DistilBERT** transformer model using **Ray Train** for distributed execution. All hyperparameters, validation curves, confusion matrices, and model weights are tracked automatically in MLflow.

To configure parameters, edit [params.yaml](file:///Users/ihedz/it-support-mlops/params.yaml):
```yaml
train:
  epochs: 1
  batch_size: 16
  learning_rate: 2e-5
  max_length: 256
  quick_test: true  # Set to false to train on the full 29K dataset!
```

Run the training pipeline stage using DVC:
```bash
dvc repro train
```
Or run the training script directly:
```bash
python src/train.py --input dataset/raw_tickets.csv --epochs 1 --batch-size 16 --quick-test false
```

### 5. Stage 5: Experiment Tracking & Model Registry
Open the MLflow Tracking Server UI at **`http://localhost:5001`** to compare runs, inspect parameters, download evaluation plots, and examine saved model artifacts.

#### Promoting Models
To automatically fetch the best run from MLflow (using `test_f1_weighted`), register it, and promote it through lifecycle stages, use the registry helper script:

```bash
# Register the best model as 'Staging'
python scripts/register_model.py

# Register and promote directly to 'Production'
python scripts/register_model.py --promote
```

---

## 🎯 Verification & Local Testing

Check training metrics logged by DVC:
```bash
dvc metrics show
```

Inspect generated plots:
- Check out the confusion matrix plot at `plots/confusion_matrix.png`.
- View the classification report at `plots/classification_report.txt`.
