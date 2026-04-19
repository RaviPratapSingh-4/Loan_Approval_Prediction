import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "prediction_logs.csv"

os.makedirs(LOG_PATH.parent, exist_ok=True)