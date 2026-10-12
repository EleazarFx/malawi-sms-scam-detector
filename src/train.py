"""
Train and evaluate SMS scam classifiers.
Combines TF-IDF (on cleaned text) with rule features (on raw text).
"""

import sys
from pathlib import Path

import math
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import math

# ---- Path setup ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocess import clean_text
from features import build_vectorizer, rule_vectorizer, build_feature_union, TextRouter
from rules import RULE_FEATURE_NAMES



# ---- Data loading ----

def load_data():
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "interim" / "sms_deduped.csv",
        encoding="utf-8",
    )
    df["clean"] = df["message"].apply(clean_text)
    df["raw"] = df["message"].astype(str)
    df["label_num"] = (df["label"] == "scam").astype(int)
    return df


def build_pipelines():
    """Each model is a Pipeline: TextRouter (features) -> classifier."""
    return {
        "Naive Bayes": Pipeline([
            ("feat", TextRouter()),
            ("clf", MultinomialNB()),
        ]),
        "Logistic Regression": Pipeline([
            ("feat", TextRouter()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42,
            )),
        ]),
       "Linear SVM": Pipeline([
            ("feat", TextRouter()),
            ("clf", LinearSVC(
                class_weight="balanced",
                random_state=42,
                max_iter=5000,       # was default (1000)
                dual="auto",         # auto-picks primal/dual based on n_samples
                tol=1e-3,
            )),
        ]),
    }


def main():
    print("Loading and cleaning data...")
    df = load_data()
    print(f"Rows: {len(df)}  |  scam: {df.label_num.sum()}  |  legit: {(df.label_num==0).sum()}")

    # X is a DataFrame with 'clean' and 'raw' columns
    X = df[["clean", "raw"]]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # ---- Cross-validate each model ----
    print("=" * 60)
    print("Cross-validation (5-fold, F1 on scam class):")
    print("=" * 60)

    pipelines = build_pipelines()
    results = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")
        results[name] = scores.mean()
        print(f"{name:25s}  F1 = {scores.mean():.3f}  (+/- {scores.std():.3f})")

    best_name = max(results, key=results.get)
    best_pipe = pipelines[best_name]
    print(f"\nBest model: {best_name}")

    # ---- Train final model ----
    print("\nTraining final model on full training set...")
    best_pipe.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = best_pipe.predict(X_test)

    print("\n" + "=" * 60)
    print("Test set evaluation")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion matrix:")
    print(f"                 Pred Ham   Pred Scam")
    print(f"Actual Ham       {cm[0,0]:>6}     {cm[0,1]:>6}")
    print(f"Actual Scam      {cm[1,0]:>6}     {cm[1,1]:>6}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "scam"]))


    # Add to train.py after evaluation
    import numpy as np
    from sklearn.metrics import f1_score, precision_recall_curve

    # Only for models with decision_function
    router = best_pipe.named_steps["feat"]
    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "decision_function"):
        scores = best_pipe.decision_function(X_test)
        probs = np.array([1 / (1 + math.exp(-s)) for s in scores])

    print("\n" + "="*60)
    print("Threshold tuning (tradeoff: catch more scams vs fewer false alarms)")
    print("="*60)
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        print(f"  threshold={t:.1f}  FP={fp:2d}  FN={fn:2d}  F1={f1_score(y_test, preds):.3f}")

    # ---- Misclassified examples ----
    print("=" * 60)
    print("Misclassified test examples:")
    print("=" * 60)
    X_test_raw = X_test["raw"].reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    # Get per-sample scores
    router = best_pipe.named_steps["feat"]
    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "decision_function"):
        raw_scores = best_pipe.decision_function(X_test)
        probs = np.array([1 / (1 + math.exp(-s)) for s in raw_scores])
    else:
        probs = best_pipe.predict_proba(X_test)[:, 1]

    for i, (true, pred) in enumerate(zip(y_test_reset, y_pred)):
        if true != pred:
            kind = "MISSED SCAM (FN)" if true == 1 else "FALSE ALARM (FP)"
            print(f"\n[{kind}]  score={probs[i]:.3f}")
            print(f"   {X_test_raw[i][:140]}")

    # ---- Top features (only for linear models) ----
    if best_name in ("Logistic Regression", "Linear SVM"):
        print("\n" + "=" * 60)
        print("Top features")
        print("=" * 60)
        router = best_pipe.named_steps["feat"]
        clf = best_pipe.named_steps["clf"]
        tfidf_names = list(router.tfidf.get_feature_names_out())
        all_names = tfidf_names + list(RULE_FEATURE_NAMES)
        coefs = clf.coef_[0]

        pairs = list(zip(all_names, coefs))
        print("\nTop 20 scam-indicative:")
        for name, c in sorted(pairs, key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {name:30s} {c:+.3f}")
        print("\nTop 20 ham-indicative:")
        for name, c in sorted(pairs, key=lambda x: x[1])[:20]:
            print(f"  {name:30s} {c:+.3f}")


    # At the end of train.py, before saving
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[y_test_reset == 0], bins=20, alpha=0.6, label="Actual Ham", color="green")
    ax.hist(probs[y_test_reset == 1], bins=20, alpha=0.6, label="Actual Scam", color="red")
    ax.axvline(0.4, color="blue", linestyle="--", label="Prod threshold (0.4)")
    ax.axvline(0.5, color="black", linestyle=":", label="Default (0.5)")
    ax.set_xlabel("Predicted scam probability")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution by true class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "models" / "score_distribution.png", dpi=100)
    print(f"\nPlot saved → models/score_distribution.png")

    # ---- Save ----
    model_path = PROJECT_ROOT / "models" / "scam_detector.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, model_path)
    print(f"\nModel saved → {model_path}")


if __name__ == "__main__":
    main()