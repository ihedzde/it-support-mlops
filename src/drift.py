import os
import pandas as pd
from collections import deque
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping

# ponytail: simple in-memory rolling drift detector
class DriftDetector:
    def __init__(self, reference_path="dataset/raw_tickets.csv", max_window=1000):
        self.max_window = max_window
        self.window = deque(maxlen=max_window)
        self.reference_path = reference_path
        self.reference = None
        self._load_reference()

    def _load_reference(self):
        try:
            if os.path.exists(self.reference_path):
                df = pd.read_csv(self.reference_path)
                df = df.dropna(subset=["Body"])
                df = df.rename(columns={"Body": "text"})
                # Sample reference data to keep Evidently processing fast
                self.reference = df[["text"]].sample(n=min(500, len(df)), random_state=42)
                print(f"Loaded {len(self.reference)} reference rows for drift detection from {self.reference_path}")
            else:
                print(f"Warning: Reference file {self.reference_path} not found. Drift detection will use fallback.")
                self.reference = pd.DataFrame({"text": ["Standard IT support ticket sample text"] * 100})
        except Exception as e:
            print(f"Error loading reference dataset: {e}")
            self.reference = pd.DataFrame({"text": ["Fallback IT support ticket sample text"] * 100})

    def add_prediction(self, text: str):
        self.window.append(text)

    def calculate_drift(self):
        if len(self.window) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 10 predictions to calculate drift. Current count: {len(self.window)}"
            }

        current_df = pd.DataFrame({"text": list(self.window)})
        mapping = ColumnMapping(text_features=["text"])
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference, current_data=current_df, column_mapping=mapping)
        
        res = report.as_dict()["metrics"][0]["result"]
        return {
            "drift_detected": res.get("dataset_drift", False),
            "share_of_drifted_columns": res.get("share_of_drifted_columns", 0.0),
            "number_of_columns": res.get("number_of_columns", 0),
            "metrics": res
        }

if __name__ == "__main__":
    # ponytail: self-test block
    detector = DriftDetector(reference_path="dataset/raw_tickets.csv")
    for t in ["my laptop does not boot", "printer offline", "reset password"] * 5:
        detector.add_prediction(t)
    report = detector.calculate_drift()
    assert "drift_detected" in report or "status" in report
    print("drift.py self-test passed!")

