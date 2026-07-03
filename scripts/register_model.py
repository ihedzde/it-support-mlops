#!/usr/bin/env python3
"""
register_model.py — Register the best run's model in MLflow Model Registry and promote it.
"""

import argparse
import os
import sys
import mlflow
from mlflow import MlflowClient

# Configure environment defaults for MLflow local running
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"


def register_best_model(experiment_name, model_name, metric_name, promote_to_prod=False):
    print(f"📡 Connecting to MLflow at {os.environ['MLFLOW_TRACKING_URI']}...")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    
    # 1. Find the experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"❌ Error: Experiment '{experiment_name}' not found. Did you run training?")
        sys.exit(1)
        
    print(f"🔍 Searching for the best run in experiment '{experiment_name}' sorted by '{metric_name}'...")
    
    # 2. Search runs, sorted by metric descending
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1
    )
    
    if not runs:
        print(f"❌ Error: No runs found in experiment '{experiment_name}'.")
        sys.exit(1)
        
    best_run = runs[0]
    run_id = best_run.info.run_id
    metric_val = best_run.data.metrics.get(metric_name, "N/A")
    
    print(f"🏆 Best run found:")
    print(f"   Run ID: {run_id}")
    print(f"   {metric_name}: {metric_val}")
    
    # 3. Register the model
    model_uri = f"runs:/{run_id}/model"
    print(f"📦 Registering model from {model_uri} as '{model_name}'...")
    
    try:
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        sys.exit(1)
        
    print(f"✅ Model registered successfully! Version: {model_version.version}")
    
    # 4. Promote model to target stage
    target_stage = "Production" if promote_to_prod else "Staging"
    print(f"🚀 Transitioning '{model_name}' version {model_version.version} to stage '{target_stage}'...")
    
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=target_stage,
            archive_existing_versions=True
        )
        print(f"🎉 Promotion complete! Model '{model_name}' v{model_version.version} is now in '{target_stage}'.")
    except Exception as e:
        print(f"⚠️  Warning: Stage transition failed: {e}")
        print("Note: In newer MLflow environments, you might need to use aliases or tags, but standard stages should work on our server.")
        
    # Also assign an alias 'best' for easier downstream referencing
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias="best" if not promote_to_prod else "production",
            version=model_version.version
        )
        print(f"🏷️  Assigned alias '{'best' if not promote_to_prod else 'production'}' to version {model_version.version}.")
    except Exception as e:
        print(f"ℹ️  Could not set alias: {e}")


def main():
    parser = argparse.ArgumentParser(description="Register the best model from MLflow tracking to the Model Registry")
    parser.add_argument("-e", "--experiment", default="it-support-classification", help="MLflow experiment name")
    parser.add_argument("-m", "--model-name", default="it-support-classifier", help="Registered model name")
    parser.add_argument("--metric", default="test_f1_weighted", help="Metric to rank runs (default: test_f1_weighted)")
    parser.add_argument("--promote", action="store_true", help="Promote directly to Production instead of Staging")
    
    args = parser.parse_args()
    
    register_best_model(
        experiment_name=args.experiment,
        model_name=args.model_name,
        metric_name=args.metric,
        promote_to_prod=args.promote
    )


if __name__ == "__main__":
    main()
