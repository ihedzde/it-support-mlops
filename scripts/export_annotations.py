#!/usr/bin/env python3
"""
export_annotations.py — Export annotations from Label Studio and save for DVC versioning.

Usage:
    python scripts/export_annotations.py --project-id 1

What it does:
    1. Connects to Label Studio API
    2. Exports all completed annotations from the given project
    3. Saves them to dataset/annotated/annotations_v{N}.json
    4. Prints DVC commands to version the new annotations
"""

import argparse
import json
import os
import sys
from datetime import datetime

try:
    import requests
except ImportError:
    print("❌ requests is required. Install it with: pip install requests")
    sys.exit(1)


# ── Label Studio Configuration ──────────────────────────────────
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")

def get_session():
    """Programmatically log into Label Studio to bypass broken API tokens."""
    # Try to load python-dotenv if available to read docker/.env
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker", ".env")
        load_dotenv(env_path)
    except ImportError:
        pass

    username = os.environ.get("LABEL_STUDIO_USERNAME", "admin@example.com")
    password = os.environ.get("LABEL_STUDIO_PASSWORD", "admin123")

    print(f"🔐 Logging into Label Studio automatically as {username}...")
    session = requests.Session()
    
    try:
        login_page = session.get(f"{LABEL_STUDIO_URL}/user/login/")
        login_page.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to reach Label Studio at {LABEL_STUDIO_URL}: {e}")
        sys.exit(1)
        
    csrf_token = session.cookies.get("csrftoken")
    
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
    return session


def export_annotations(project_id: int, session: requests.Session) -> list:
    """Export all annotations from a Label Studio project using the authenticated session."""
    url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}/export"
    params = {"exportType": "JSON"}

    print(f"📥 Exporting annotations from project {project_id}...")
    response = session.get(url, params=params)

    if response.status_code == 404:
        print(f"❌ Project {project_id} not found.")
        sys.exit(1)
    elif response.status_code != 200:
        print(f"❌ API error {response.status_code}: {response.text}")
        sys.exit(1)

    annotations = response.json()
    print(f"   Found {len(annotations)} exported items")
    return annotations


def save_annotations(annotations: list, output_file: str) -> str:
    """Save annotations to a JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Add export metadata
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "total_annotations": len(annotations),
        "annotations": annotations,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(annotations)} annotations to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Export Label Studio annotations for DVC versioning"
    )
    parser.add_argument(
        "--project-id",
        type=int,
        required=True,
        help="Label Studio project ID",
    )
    parser.add_argument(
        "--output-file",
        default="dataset/labeled_tickets.json",
        help="File to save annotations (default: dataset/labeled_tickets.json)",
    )
    args = parser.parse_args()

    # 1. Get Session
    session = get_session()

    # 2. Export annotations
    annotations = export_annotations(args.project_id, session)

    if not annotations:
        print("⚠️  No completed annotations found. Label some data first!")
        sys.exit(0)

    # 3. Save directly
    filepath = save_annotations(annotations, args.output_file)

    # 4. Print DVC commands
    print(f"\n🔄 To version this dataset with DVC, run:")
    print(f"   dvc add {filepath}")
    print(f"   git add {filepath}.dvc .gitignore")
    print(f'   git commit -m "Update labeled dataset"')
    print(f"   dvc push")


if __name__ == "__main__":
    main()
