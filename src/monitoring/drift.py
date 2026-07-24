import os
import pandas as pd
from loguru import logger
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

REFERENCE_PATH = "data/reference/reference_data.csv"
CURRENT_PATH = "logs/predictions_log.csv"
REPORT_DIR = "reports"
MIN_ROWS_REQUIRED = 30


def check_sufficient_data(current_path, min_rows):
    if not os.path.isfile(current_path):
        return False, "No predictions logged yet."

    current_df = pd.read_csv(current_path)
    if len(current_df) < min_rows:
        return False, f"Only {len(current_df)} predictions logged; need at least {min_rows}."

    return True, current_df


def run_drift_report():
    is_ready, result = check_sufficient_data(CURRENT_PATH, MIN_ROWS_REQUIRED)
    if not is_ready:
        logger.warning(result)
        return None, result

    current_df = result[["review_text"]]
    reference_df = pd.read_csv(REFERENCE_PATH)[["review_text"]]

    definition = DataDefinition(text_columns=["review_text"])
    reference_dataset = Dataset.from_pandas(reference_df, data_definition=definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=definition)

    logger.info("Running Evidently drift report...")
    report = Report([DataDriftPreset()])
    snapshot = report.run(current_dataset, reference_dataset)

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = f"{REPORT_DIR}/drift_report.html"
    snapshot.save_html(report_path)

    logger.success(f"Drift report saved to {report_path}")
    return report_path, "Report generated successfully."