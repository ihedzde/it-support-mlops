#!/bin/bash
# ==============================================================================
# download_raw.sh - Download the raw dataset for Stage 1 of the pipeline
# 
# What it does:
#   1. Uses Kaggle CLI to download the dataset
#   2. Extracts it directly into the dataset/ folder as raw_tickets.csv
# ==============================================================================

set -e

DATASET_NAME="parthpatil256/it-support-ticket-data"
DEST_DIR="dataset"
FINAL_FILE="$DEST_DIR/raw_tickets.csv"

echo "🚀 Starting Raw Data Download (Stage 1)..."
mkdir -p "$DEST_DIR"

if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found. Please run this script from your virtual environment (pip install kaggle)."
    exit 1
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "📥 Downloading dataset..."
kaggle datasets download -d "$DATASET_NAME" -p "$TMP_DIR" --unzip

# Move the CSV to the final destination
CSV_FILE=$(find "$TMP_DIR" -name "*.csv" | head -n 1)
mv "$CSV_FILE" "$FINAL_FILE"

echo "✅ Success! Raw dataset saved to $FINAL_FILE"
echo "👉 Next step: Run 'dvc add $FINAL_FILE' to track it."
