import os
import json
import time
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from mlflow import MlflowClient
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from src.drift import DriftDetector

# Environment defaults for MLflow
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")
os.environ.setdefault("MLFLOW_S3_IGNORE_TLS", "true")

MODEL_NAME = "it-support-classifier"
model = None
label_mapping = {}

# Prometheus instrumentation
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["endpoint", "status"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["endpoint"]
)
MODEL_LOADED = Gauge(
    "model_loaded",
    "Model loaded status (1 = loaded, 0 = not loaded)"
)

# Drift detection initialization
drift_detector = DriftDetector(reference_path="dataset/raw_tickets.csv")

def get_department_name(pred_label: str) -> str:
    # ponytail: decode pipeline LABEL_X to human name using label_mapping.json
    if pred_label.startswith("LABEL_"):
        idx = pred_label.split("_")[-1]
        return label_mapping.get(idx, pred_label)
    if str(pred_label).isdigit():
        return label_mapping.get(str(pred_label), pred_label)
    return pred_label

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, label_mapping
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    
    # ponytail: try Production, then Staging, then alias 'best'
    run_id = None
    for stage in ["Production", "Staging"]:
        try:
            model_uri = f"models:/{MODEL_NAME}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)
            mv = client.get_latest_versions(MODEL_NAME, stages=[stage])[0]
            run_id = mv.run_id
            break
        except Exception:
            continue
            
    if model is None:
        try:
            model_uri = f"models:/{MODEL_NAME}@best"
            model = mlflow.pyfunc.load_model(model_uri)
            mv = client.get_model_version_by_alias(MODEL_NAME, "best")
            run_id = mv.run_id
        except Exception as e:
            print(f"Warning: could not load model from registry: {e}")
            
    if run_id:
        try:
            local_path = client.download_artifacts(run_id, "label_mapping.json")
            with open(local_path) as f:
                label_mapping = json.load(f)
        except Exception as e:
            print(f"Warning: could not download label_mapping: {e}")
            
    # Set model loaded gauge status
    MODEL_LOADED.set(1 if model is not None else 0)
    yield

app = FastAPI(title="IT Support Ticket Classifier API", lifespan=lifespan)

# Mount Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app())

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    department: str
    confidence: float

class BatchPredictionRequest(BaseModel):
    texts: list[str]

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/drift/report")
def drift_report():
    return drift_detector.calculate_drift()

def predict_internal(text: str) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame({"text": [text]})
    res = model.predict(df)
    pred_label = res.iloc[0]["label"]
    confidence = float(res.iloc[0]["score"])
    department = get_department_name(pred_label)
    return PredictionResponse(department=department, confidence=confidence)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = time.perf_counter()
    status = "success"
    try:
        response = predict_internal(request.text)
        drift_detector.add_prediction(request.text)
        return response
    except Exception as e:
        status = "error"
        raise e
    finally:
        latency = time.perf_counter() - start_time
        PREDICTION_REQUESTS.labels(endpoint="predict", status=status).inc()
        PREDICTION_LATENCY.labels(endpoint="predict").observe(latency)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    start_time = time.perf_counter()
    status = "success"
    try:
        predictions = []
        for text in request.texts:
            predictions.append(predict_internal(text))
            drift_detector.add_prediction(text)
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        status = "error"
        raise e
    finally:
        latency = time.perf_counter() - start_time
        PREDICTION_REQUESTS.labels(endpoint="predict_batch", status=status).inc()
        PREDICTION_LATENCY.labels(endpoint="predict_batch").observe(latency)

if __name__ == "__main__":
    # ponytail: inline runnable check for decoding logic
    label_mapping = {"5": "Product Support"}
    assert get_department_name("LABEL_5") == "Product Support"
    assert get_department_name("LABEL_99") == "LABEL_99"
    print("Self-test passed!")

