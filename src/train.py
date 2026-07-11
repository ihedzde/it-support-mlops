#!/usr/bin/env python3
"""
train.py — Train DistilBERT model for IT Support Ticket Classification using Ray Train and MLflow.
"""

import argparse
import json
import os
import sys
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

# Ray and MLflow imports
import ray
from ray import train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

# MLflow
import mlflow
import mlflow.transformers

# Hugging Face
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

# Configure environment defaults for MLflow local running
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

VALID_DEPARTMENTS = {
    "Technical Support", "Product Support", "Customer Service", "IT Support",
    "Billing and Payments", "Returns and Exchanges", "Service Outages and Maintenance",
    "Sales and Pre-Sales", "Human Resources", "General Inquiry"
}


def clean_and_prepare_data(input_csv, quick_test=False):
    """Load data, clean out invalid categories and labels, and return split datasets and mapping."""
    print(f"🔄 Loading data from {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input file {input_csv} not found.")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    
    # Clean column names (sometimes there is leading/trailing whitespace or index columns)
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    
    # Drop rows with null body or department
    df = df.dropna(subset=["Body", "Department"])
    
    # Clean strings
    df["Body"] = df["Body"].astype(str).str.strip()
    df["Department"] = df["Department"].astype(str).str.strip()
    
    # Filter only valid departments to clean noise
    df = df[df["Department"].isin(VALID_DEPARTMENTS)]
    
    if quick_test:
        print("⚡ Quick test mode enabled: sampling 2000 rows to speed up execution.")
        # Ensure we have representation of all classes in quick-test if possible, or just sample
        df = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)
    
    print(f"📊 Cleaned dataset size: {len(df)} rows across {df['Department'].nunique()} classes.")
    
    # Encode targets
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Department"])
    
    # Save the label mapping
    mapping = {int(idx): str(name) for idx, name in enumerate(label_encoder.classes_)}
    
    # Split: 70% Train, 15% Val, 15% Test
    # Only stratify if all classes have at least 2 members
    train_class_counts = df["label"].value_counts()
    stratify_train = df["label"] if train_class_counts.min() >= 2 else None
    
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=stratify_train)
    
    temp_class_counts = temp_df["label"].value_counts()
    stratify_val = temp_df["label"] if temp_class_counts.min() >= 2 else None
    
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=stratify_val)
    
    print(f"📝 Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df, mapping


def compute_metrics_fn(eval_pred):
    """Evaluation metrics computation for Hugging Face Trainer."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    
    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "precision_weighted": precision,
        "recall_weighted": recall
    }


def train_func_per_worker(config):
    """Ray Train worker function."""
    # Read training config parameters
    epochs = config.get("epochs", 3)
    batch_size = config.get("batch_size", 16)
    lr = config.get("learning_rate", 2e-5)
    max_length = config.get("max_length", 256)
    
    # Load tokenized datasets passed via local temp storage or direct python objects
    # In Ray Train, we can read datasets directly or load them
    train_data = config["train_data"]
    val_data = config["val_data"]
    num_labels = config["num_labels"]
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Helper to tokenize datasets
    def tokenize_fn(batch):
        return tokenizer(batch["Body"], padding="max_length", truncation=True, max_length=max_length)
    
    # Convert pandas to HF datasets
    train_ds = Dataset.from_pandas(train_data[["Body", "label"]])
    val_ds = Dataset.from_pandas(val_data[["Body", "label"]])
    
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    use_gpu = config.get("use_gpu", False)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        disable_tqdm=True,
        use_cpu=not use_gpu,
        report_to="none"  # Disable standard logger integrations in Ray workers
    )
    
    # Ray Train Hugging Face callback to report metrics and checkpoints
    from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_fn,
        callbacks=[RayTrainReportCallback()]
    )
    
    trainer = prepare_trainer(trainer)
    trainer.train()


def evaluate_and_plot(model, tokenizer, test_df, mapping):
    """Evaluate the trained model on test data and return metrics & plots."""
    print("🧪 Running final evaluation on test set...")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    texts = test_df["Body"].tolist()
    true_labels = test_df["label"].tolist()
    preds = []
    
    # Predict in batches
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds.extend(batch_preds)
            
    # Calculate metrics
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="weighted")
    precision = precision_score(true_labels, preds, average="weighted", zero_division=0)
    recall = recall_score(true_labels, preds, average="weighted", zero_division=0)
    
    print(f"📊 Test Results: Accuracy={acc:.4f}, F1-Score={f1:.4f}")
    
    # Classification report
    label_names = [mapping[i] for i in sorted(mapping.keys())]
    class_report = classification_report(
        true_labels, 
        preds, 
        target_names=[mapping[i] for i in sorted(mapping.keys()) if i in np.unique(true_labels + preds)],
        zero_division=0
    )
    
    # Confusion matrix plot
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=label_names, 
        yticklabels=label_names
    )
    plt.title("Confusion Matrix — IT Support Ticket Classifier")
    plt.ylabel("True Category")
    plt.xlabel("Predicted Category")
    plt.tight_layout()
    
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    cm_path = "plots/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    metrics = {
        "test_accuracy": acc,
        "test_f1_weighted": f1,
        "test_precision_weighted": precision,
        "test_recall_weighted": recall
    }
    
    return metrics, class_report, cm_path


def main():
    parser = argparse.ArgumentParser(description="Distributed DistilBERT Training with Ray and MLflow")
    parser.add_argument("-i", "--input", default="dataset/raw_tickets.csv", help="Input raw CSV path")
    parser.add_argument("-e", "--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Batch size per worker")
    parser.add_argument("-lr", "--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max sequence length")
    parser.add_argument("-w", "--num-workers", type=int, default=1, help="Number of Ray Torch workers")
    parser.add_argument("--use-gpu", action="store_true", help="Train on GPU if available")
    parser.add_argument("--quick-test", type=str, default="false", help="Run a quick test with sampled data ('true' or 'false')")
    
    args = parser.parse_args()
    
    quick_test = args.quick_test.lower() == "true"
    if quick_test:
        args.epochs = 1
        args.batch_size = 8
    
    # Clean and split dataset
    train_df, val_df, test_df, label_mapping = clean_and_prepare_data(args.input, quick_test)
    num_labels = len(label_mapping)
    
    # Configure Ray Train loop config
    train_loop_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "train_data": train_df,
        "val_data": val_df,
        "num_labels": num_labels,
        "use_gpu": args.use_gpu
    }
    
    # ── MLflow Tracking Setup ─────────────────────────────────────
    print(f"📡 Connecting to MLflow at {os.environ['MLFLOW_TRACKING_URI']}...")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("it-support-classification")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(log_to_driver=False)
        
    print(f"🚀 Initializing Ray Train (workers: {args.num_workers}, gpu: {args.use_gpu})...")
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=args.use_gpu
        ),
        run_config=RunConfig(
            storage_path=os.path.abspath("./ray_results"),
            name="distilbert_ticket_classification"
        )
    )
    
    # Run distributed training
    try:
        with mlflow.start_run() as run:
            print(f"📝 Logging experiment parameters to MLflow run: {run.info.run_id}")
            
            # Log params
            mlflow.log_params({
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "num_workers": args.num_workers,
                "use_gpu": args.use_gpu,
                "model_name": "distilbert-base-uncased",
                "num_classes": num_labels,
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df)
            })
            
            # Run the trainer
            results = trainer.fit()
            print("🎉 Distributed training completed successfully!")
            
            # Log validation metrics from best checkpoint
            best_metrics = results.metrics
            # Filter standard ray training metrics for logging
            mlflow_metrics = {
                k: v for k, v in best_metrics.items() 
                if isinstance(v, (int, float)) and not k.startswith("config/")
            }
            mlflow.log_metrics(mlflow_metrics)
            
            # Retrieve checkpoint and load the best model
            checkpoint = results.checkpoint
            with checkpoint.as_directory() as checkpoint_dir:
                model_dir = os.path.join(checkpoint_dir, "checkpoint")
                if not os.path.exists(os.path.join(model_dir, "config.json")):
                    model_dir = checkpoint_dir
                print(f"📥 Loading best model from checkpoint: {model_dir}")
                
                best_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                
                # Final evaluation on held-out test data
                test_metrics, report, cm_path = evaluate_and_plot(best_model, tokenizer, test_df, label_mapping)
                
                # Log test metrics
                mlflow.log_metrics(test_metrics)
                
                # Log classification report
                report_path = "plots/classification_report.txt"
                os.makedirs("plots", exist_ok=True)
                with open(report_path, "w") as f:
                    f.write(report)
                mlflow.log_artifact(report_path)
                print(report)
                
                # Log confusion matrix plot
                mlflow.log_artifact(cm_path)
                
                # Save label mapping to a file and log it as artifact
                mapping_path = "plots/label_mapping.json"
                with open(mapping_path, "w") as f:
                    json.dump(label_mapping, f, indent=2)
                mlflow.log_artifact(mapping_path)
                
                # Log model & tokenizer to MLflow model tracking
                print("💾 Logging Transformers model and tokenizer to MLflow Artifact store...")
                mlflow.transformers.log_model(
                    transformers_model={"model": best_model, "tokenizer": tokenizer},
                    artifact_path="model",
                    task="text-classification"
                )
                
            # Write metrics out to local file for DVC integration
            with open("metrics.json", "w") as f:
                json.dump(test_metrics, f, indent=2)
                
            print("✅ All metrics, plots, and models successfully stored in MLflow!")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
