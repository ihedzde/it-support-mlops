#!/usr/bin/env python3
"""
api_sync_tasks.py — Idempotent API pipeline to sync tasks to Label Studio

Usage:
    python scripts/api_sync_tasks.py --project-id 1 --input dataset/sample_tickets.json

What it does:
    1. Reads the Label Studio API token from docker/.env
    2. Fetches all existing tasks from the project
    3. Compares the original_id of the JSON against the existing tasks
    4. ONLY uploads tasks that do not already exist (protecting your annotations)
"""

import argparse
import json
import os
import sys
import requests
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# Try to load python-dotenv if available to read docker/.env
try:
    # pyrefly: ignore [missing-import]
   
    # Load from docker/.env
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker", ".env")
    load_dotenv(env_path)
except ImportError:
    pass

LABEL_STUDIO_URL = "http://localhost:8080"

def sync_tasks(project_id, input_json):
    # Get credentials from .env
    username = os.environ.get("LABEL_STUDIO_USERNAME", "admin@example.com")
    password = os.environ.get("LABEL_STUDIO_PASSWORD", "admin123")

    print(f"🔐 Logging into Label Studio automatically as {username}...")
    session = requests.Session()
    
    # 1. Get CSRF Token
    try:
        login_page = session.get(f"{LABEL_STUDIO_URL}/user/login/")
        login_page.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to reach Label Studio at {LABEL_STUDIO_URL}: {e}")
        sys.exit(1)
        
    csrf_token = session.cookies.get("csrftoken")
    
    # 2. Authenticate
    login_data = {
        "email": username,
        "password": password,
        "csrfmiddlewaretoken": csrf_token
    }
    
    res = session.post(
        f"{LABEL_STUDIO_URL}/user/login/", 
        data=login_data, 
        headers={"Referer": f"{LABEL_STUDIO_URL}/user/login/"}
    )
    
    if res.status_code != 200:
        print(f"❌ Login failed! Status {res.status_code}")
        sys.exit(1)
        
    print("✅ Login successful!")

    # 3. Read the input tasks
    print(f"🔄 Reading input tasks from {input_json}...")
    if not os.path.exists(input_json):
        print(f"❌ Error: {input_json} not found. Did you run 'dvc repro'?")
        sys.exit(1)
        
    with open(input_json, "r") as f:
        incoming_tasks = json.load(f)

    # 4. Fetch existing tasks from Label Studio to prevent duplicates
    print(f"📡 Fetching existing tasks from Label Studio Project {project_id}...")
    res = session.get(f"{LABEL_STUDIO_URL}/api/tasks?project={project_id}")
    
    if res.status_code == 404:
        print(f"\n❌ Error: Project ID {project_id} does not exist!")
        print("   Please go to http://localhost:8080 and click 'Create' to make your project first.")
        sys.exit(1)
    elif res.status_code != 200:
        print(f"❌ Failed to fetch tasks! Status: {res.status_code}")
        sys.exit(1)
        
    existing_data = res.json()
    existing_tasks = existing_data.get('tasks', existing_data.get('items', [])) if isinstance(existing_data, dict) else existing_data

    existing_ids = set()
    for t in existing_tasks:
        meta = t.get("meta", {})
        if "original_id" in meta:
            existing_ids.add(str(meta["original_id"]))

    # 5. Filter out tasks that already exist
    tasks_to_upload = [t for t in incoming_tasks if str(t.get("meta", {}).get("original_id", "")) not in existing_ids]

    if not tasks_to_upload:
        print("✅ All tasks are already in Label Studio! Nothing to sync.")
        return

    print(f"📤 Uploading {len(tasks_to_upload)} NEW tasks to Label Studio...")
    res = session.post(
        f"{LABEL_STUDIO_URL}/api/projects/{project_id}/import",
        json=tasks_to_upload,
        headers={"X-CSRFToken": session.cookies.get("csrftoken"), "Referer": LABEL_STUDIO_URL}
    )
    
    if res.status_code in [200, 201]:
        print("✅ Sync complete! New tasks successfully added to the database.")
    else:
        print(f"❌ Failed to upload tasks: {res.status_code} - {res.text}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Idempotent task sync for Label Studio")
    parser.add_argument("--project-id", required=True, type=int, help="Label Studio Project ID")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    args = parser.parse_args()
    
    sync_tasks(args.project_id, args.input)
