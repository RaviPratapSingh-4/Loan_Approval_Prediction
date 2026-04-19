from pathlib import Path
import joblib

MODEL_PATH = Path("models/best_model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run training/train.py first."
        )
    model = joblib.load(MODEL_PATH)
    return model