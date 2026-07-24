import csv
import os
from datetime import datetime

LOG_FILES= "logs/predictions_log.csv"
LOG_HEADERS = ["timestamp", "review_text", "predicted_label", "confidence_score"]

def log_prediction(text: str, label: str, confidence: float):
    os.makedirs(os.path.dirname(LOG_FILES), exist_ok=True)
    file_exists = os.path.isfile(LOG_FILES)

    with open(LOG_FILES, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(LOG_HEADERS)
        writer.writerow([datetime.utcnow().isoformat(), text, label, confidence])