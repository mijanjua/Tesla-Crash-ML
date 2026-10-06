"""Model building, evaluation and persistence.

Run from the project root to train both models and save the best one:
    python -m src.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
    make_scorer, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, PROJECT_ROOT, TARGET, load_dataset

MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42

# Only ~13% of crashes have Autopilot claimed, so accuracy is misleading
# (predicting "no" every time scores ~87%). These metrics focus on the minority class.
CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    # zero_division=0: a fold where the model predicts no positives scores 0, not a warning
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
}


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def build_models() -> dict[str, Pipeline]:
    # class_weight="balanced" up-weights the rare positive class during training
    return {
        "Baseline (most frequent)": Pipeline([
            ("prep", make_preprocessor(scale_numeric=False)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression": Pipeline([
            ("prep", make_preprocessor(scale_numeric=True)),
            ("clf", LogisticRegression(class_weight="balanced", C=0.5, max_iter=1000)),
        ]),
        "Random Forest": Pipeline([
            ("prep", make_preprocessor(scale_numeric=False)),
            ("clf", RandomForestClassifier(
                n_estimators=500, max_depth=5, min_samples_leaf=3,
                class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }


def split(df: pd.DataFrame, test_size=0.25):
    return train_test_split(
        df[FEATURES], df[TARGET], test_size=test_size,
        stratify=df[TARGET], random_state=RANDOM_STATE,
    )


def cross_validate_models(models, X, y, n_splits=5, n_repeats=5) -> pd.DataFrame:
    """Repeated stratified k-fold CV.

    With ~35 positives, one split (or even one 5-fold run) is too noisy to trust:
    which rows land in which fold can move ROC-AUC by 0.1. Repeating with
    different shuffles and averaging gives a much steadier estimate.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    rows = {}
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=CV_SCORING)
        rows[name] = {
            f"{metric} ({stat})": fn(scores[f"test_{metric}"])
            for metric in CV_SCORING
            for stat, fn in (("mean", np.mean), ("std", np.std))
        }
    return pd.DataFrame(rows).T


def evaluate_on_test(model, X_test, y_test, threshold=0.5) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "recall": recall_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]),
    }


def out_of_fold_proba(model, X, y, n_splits=5, n_repeats=5) -> np.ndarray:
    """Average out-of-fold probabilities over several shuffles: every row is scored by a model that never saw it."""
    probas = [
        cross_val_predict(model, X, y, method="predict_proba",
                          cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed))[:, 1]
        for seed in range(n_repeats)
    ]
    return np.mean(probas, axis=0)


def tune_threshold(model, X, y, thresholds=np.arange(0.05, 0.951, 0.025)) -> tuple[float, pd.DataFrame]:
    """Pick the probability cut-off that maximises F1 on out-of-fold predictions.

    0.5 is arbitrary for a 13% minority class. Tuning on training-set
    out-of-fold predictions keeps the test set untouched for the final check.
    """
    proba = out_of_fold_proba(model, X, y)
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 3),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "predicted_yes": int(pred.sum()),
        })
    table = pd.DataFrame(rows)
    best = float(table.loc[table["f1"].idxmax(), "threshold"])
    return best, table


def feature_names(model: Pipeline) -> list[str]:
    return [n.split("__", 1)[1] for n in model.named_steps["prep"].get_feature_names_out()]


def logistic_coefficients(model: Pipeline) -> pd.Series:
    """Positive = pushes toward 'Autopilot claimed'. Comparable because numerics are scaled."""
    coefs = model.named_steps["clf"].coef_[0]
    return pd.Series(coefs, index=feature_names(model)).sort_values(key=abs, ascending=False)


def forest_importances(model: Pipeline) -> pd.Series:
    imp = model.named_steps["clf"].feature_importances_
    return pd.Series(imp, index=feature_names(model)).sort_values(ascending=False)


def permutation_importances(model, X, y, n_repeats=30) -> pd.DataFrame:
    """Model-agnostic importance on raw features: drop in PR-AUC when a feature is shuffled."""
    result = permutation_importance(
        model, X, y, scoring="average_precision", n_repeats=n_repeats,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return pd.DataFrame(
        {"mean": result.importances_mean, "std": result.importances_std}, index=X.columns,
    ).sort_values("mean", ascending=False)


def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split(df)
    models = build_models()

    print(f"Rows: {len(df)}  |  positive rate: {df[TARGET].mean():.1%}")
    print("\n5x5 repeated CV on training set:")
    cv_results = cross_validate_models(models, X_train, y_train)
    print(cv_results.filter(like="mean").round(3).to_string())

    candidates = [n for n in models if not n.startswith("Baseline")]
    best_name = max(candidates, key=lambda n: cv_results.loc[n, "pr_auc (mean)"])

    threshold, _ = tune_threshold(build_models()[best_name], X_train, y_train)
    print(f"\nTuned threshold for {best_name} (max out-of-fold F1): {threshold:.3f}")

    print("\nHeld-out test set:")
    test_results = {}
    for name in candidates:
        models[name].fit(X_train, y_train)
        test_results[name] = evaluate_on_test(models[name], X_test, y_test)
        r = test_results[name]
        print(f"  {name} @0.5: ROC-AUC={r['roc_auc']:.3f} PR-AUC={r['pr_auc']:.3f} "
              f"recall={r['recall']:.2f} precision={r['precision']:.2f}")
        print("   confusion matrix [[TN FP] [FN TP]]:", r["confusion_matrix"].tolist())
    tuned = evaluate_on_test(models[best_name], X_test, y_test, threshold=threshold)
    print(f"  {best_name} @{threshold:.3f}: recall={tuned['recall']:.2f} precision={tuned['precision']:.2f} "
          f"f1={tuned['f1']:.2f}")
    print("   confusion matrix [[TN FP] [FN TP]]:", tuned["confusion_matrix"].tolist())

    # Refit the chosen model on all labelled data before saving it for the app
    final = build_models()[best_name].fit(df[FEATURES], df[TARGET])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_DIR / "autopilot_model.joblib")
    metadata = {
        "model_name": best_name,
        "n_rows": len(df),
        "positive_rate": float(df[TARGET].mean()),
        "threshold": threshold,
        "cv_metrics": cv_results.loc[best_name].round(4).to_dict(),
        # Metrics that depend on a cut-off use the tuned threshold
        "test_metrics": {k: (v.tolist() if hasattr(v, "tolist") else round(v, 4))
                         for k, v in tuned.items()},
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved {best_name} (best CV PR-AUC) to {MODEL_DIR / 'autopilot_model.joblib'}")


if __name__ == "__main__":
    main()
