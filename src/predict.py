"""
Inference: load the trained model, predict on new SMS.

Usage:
    python src/predict.py "Mwapambana K500000! Tumizani PIN yanu"
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "scam_detector.pkl"

_model = None  # lazy-loaded singleton


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_sms(text: str, threshold: float = 0.5) -> dict:
    """
    Classify a single SMS.

    Args:
        text: raw SMS message
        threshold: decision threshold for 'scam'.
                   0.5 = balanced (default)
                   < 0.5 = more aggressive (fewer missed scams, more false alarms)
                   > 0.5 = more conservative (fewer false alarms, more missed scams)

    Returns:
        {
          "label": "scam" | "legitimate",
          "confidence": float,     # probability of being scam
          "threshold": float,
        }
    """
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from preprocess import clean_text

    model = _load_model()

    # Model expects DataFrame with 'clean' and 'raw' columns
    X = pd.DataFrame({
        "clean": [clean_text(text)],
        "raw":   [str(text)],
    })

    # LinearSVC has no predict_proba — use decision_function instead.
    # Decision function is a signed distance from the hyperplane.
    # We squash it with a sigmoid-like transform for interpretability.
    clf = model.named_steps["clf"]

    if hasattr(clf, "predict_proba"):
        proba = float(model.predict_proba(X)[0, 1])
    elif hasattr(clf, "decision_function"):
        score = float(model.decision_function(X)[0])
        # Sigmoid squash: turns score → 0..1
        import math
        proba = 1.0 / (1.0 + math.exp(-score))
    else:
        proba = float(model.predict(X)[0])

    label = "scam" if proba >= threshold else "legitimate"

    return {
        "label": label,
        "confidence": round(proba, 4),
        "threshold": threshold,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify an SMS as scam or legitimate."
    )
    parser.add_argument("text", nargs="+", help="SMS text to classify")
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.4,
        help="Decision threshold (default 0.4, biased toward catching scams)"
    )
    args = parser.parse_args()

    text = " ".join(args.text)
    result = predict_sms(text, threshold=args.threshold)

    print(f"Text: {text}")
    print(f"Verdict: {result['label'].upper()}")
    print(f"Scam probability: {result['confidence']:.2%}")
    print(f"Threshold: {result['threshold']}")