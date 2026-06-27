#!/usr/bin/env python3
"""
tokenize_and_sample.py — Transform raw CSV into a Label Studio JSON payload.

Usage:
    python src/tokenize_and_sample.py -i dataset/raw_tickets.csv -o dataset/sample_tickets.json

This script represents Stage 2 of the DVC pipeline.
"""

import argparse
import csv
import json
import os
import sys

def transform_data(input_csv, output_json, sample_size=100):
    print(f"🔄 Reading raw data from: {input_csv}")
    
    if not os.path.exists(input_csv):
        print(f"❌ Error: {input_csv} does not exist. Did you run Stage 1?")
        sys.exit(1)

    tasks = []
    with open(input_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= sample_size:
                break
                
            # Create a task structured for Label Studio
            task = {
                "data": {
                    "text": row.get("Body", ""),
                },
                "meta": {
                    "original_id": row.get(""), # First column has no name in this dataset
                    "department": row.get("Department", ""),
                    "priority": row.get("Priority", ""),
                    "tags": row.get("Tags", "")
                }
            }
            tasks.append(task)
            
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    print(f"💾 Saving {len(tasks)} tasks to: {output_json}")
    with open(output_json, mode="w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)
        
    print("✅ Transformation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform CSV to Label Studio JSON")
    parser.add_argument("-i", "--input", required=True, help="Input raw CSV file")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    parser.add_argument("-s", "--sample", type=int, default=50, help="Number of rows to sample")
    
    args = parser.parse_args()
    transform_data(args.input, args.output, args.sample)
