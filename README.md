# 🎫 IT Support Ticket Classification MLOps Platform

> **Goal:** A complete, production-grade MLOps lifecycle demonstrating immutable data lineage (DVC), containerized data labeling (Label Studio), distributed deep learning fine-tuning (Ray Train + DistilBERT), tracking (MLflow), automated version promotion in a Model Registry, and CI/CD pipelines (GitHub Actions).

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

### 6. Stage 6: Model Serving & Inference API (FastAPI + Docker)
Deploy the REST API serving the registered model:
```bash
# Build and run the serving container
cd docker && docker compose up -d --build inference-api && cd ..
```

### 7. Stage 7: Monitoring & Observability (Prometheus + Grafana + Evidently)
The serving service collects Prometheus metrics and supports drift detection.

#### Start Monitoring Stack
Launch all services including Prometheus, Grafana, and the instrumented Inference API:
```bash
cd docker && docker compose up -d --build && cd ..
```

This launches:
- **FastAPI Inference Service (`:8000`)**
- **Prometheus (`:9090`)** — Scrapes API metrics every 5 seconds.
- **Grafana (`:3000`)** — Auto-configured with Prometheus datasource and dashboard. Credentials: `admin` / `admin`.

#### View Dashboard
Open **`http://localhost:3000`** in your browser, log in, and view the **ML Model Monitoring** dashboard for:
- **Predictions per Minute (RPS)**
- **Average Processing Time (Latency)**
- **Model Loaded Status**

#### Request Drift Report
Trigger Evidently's drift report comparing prediction text inputs against the reference dataset (`dataset/raw_tickets.csv`):
```bash
curl http://localhost:8000/drift/report
```

### 8. Stage 8: CI/CD Pipeline (GitHub Actions)
The project includes a fully automated CI/CD pipeline that orchestrates model training and deployment.

#### Pipeline Architecture
```
┌─────────────┐     ┌──────────────────────┐     ┌──────────────────────────┐
│   Trigger    │────►│   Job 1: Train        │────►│   Job 2: Deploy          │
│              │     │                      │     │   (skipped on PRs)       │
│ • push main  │     │ 1. Setup Python      │     │                          │
│ • PR to main │     │ 2. Launch infra      │     │ 1. Launch full stack     │
│ • manual     │     │ 3. Download dataset  │     │ 2. Build inference API   │
│ • schedule   │     │ 4. Train model       │     │ 3. Smoke test /health    │
│              │     │ 5. Register in MLflow│     │ 4. Smoke test /predict   │
└─────────────┘     │ 6. Upload artifacts  │     │ 5. Report status         │
                    └──────────────────────┘     └──────────────────────────┘
```

#### Triggers

| Trigger | Event | Behavior |
|---------|-------|----------|
| **Push** | Merge to `main` | Full train + deploy pipeline |
| **Pull Request** | PR targeting `main` | Quick-test training only (validation) |
| **Manual** | `workflow_dispatch` | Configurable epochs, quick_test, promote |
| **Schedule** | `cron: 0 6 * * 1` | Weekly retrain every Monday at 06:00 UTC |

#### Required GitHub Secrets
Configure these in **Settings → Secrets and variables → Actions**:

| Secret | Purpose |
|--------|---------|
| `KAGGLE_TOKEN` | Kaggle API token (`KGAT_...`) for dataset download |

#### Manual Trigger
1. Go to **Actions** tab → **ML Pipeline — Train & Deploy**
2. Click **Run workflow**
3. Configure inputs (epochs, quick_test, promote_to_production)
4. Click **Run workflow** to start

#### Viewing Results
- **Run logs**: Click on a completed workflow run to see step-by-step logs
- **Artifacts**: Training metrics and plots are uploaded as run artifacts (downloadable)
- **Workflow file**: [`.github/workflows/ml-pipeline.yml`](.github/workflows/ml-pipeline.yml)

---


## 🎯 Verification & Local Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My laptop screen is flickering and I cannot work"}'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Cannot connect to VPN", "I need a refund for my order"]}'
```

### Visualizing Metrics & Plots
Check training metrics logged by DVC:
```bash
dvc metrics show
```

Inspect generated plots:
- Check out the confusion matrix plot at `plots/confusion_matrix.png`.
- View the classification report at `plots/classification_report.txt`.

