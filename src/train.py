"""Train and evaluate next-day AQI class models against a persistence baseline.

Design:
  * Time-based split: train on target dates before TEST_START, test on the rest.
    Never shuffle time series data.
  * Hyperparameters are tuned with TimeSeriesSplit CV on the training period only.
  * The final model is picked by CV macro-F1 (never by test score), then
    evaluated once on the held-out test year.
  * Two feature sets (A: today only, B: + next-day weather) measure how much a
    weather forecast helps.

Usage (from the project root, venv active):
    python -m src.train
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import CLASS_NAMES, feature_groups
from src.ingest import ROOT

log = logging.getLogger("train")

TEST_START = "2024-01-01"
LABELS = list(range(len(CLASS_NAMES)))
MODELS_DIR = ROOT / "models"


def candidates() -> dict:
    """Model -> (estimator, hyperparameter grid). class_weight counters the imbalance."""
    return {
        "logreg": (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=5000),
            ),
            {"logisticregression__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
        ),
        "hgb": (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=0),
            {
                "learning_rate": [0.03, 0.1],
                "max_depth": [3, 6],
                "min_samples_leaf": [20, 50],
                "l2_regularization": [0.0, 1.0],
            },
        ),
    }


def scores(y_true, y_pred) -> dict:
    recall = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
        "recall": {c: float(r) for c, r in zip(CLASS_NAMES, recall)},
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def _json(o):
    return o.item() if isinstance(o, np.generic) else str(o)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--input", type=Path, default=Path("data/processed/texas_dallas_features.parquet")
    )
    args = p.parse_args(argv)
    path = args.input if args.input.is_absolute() else ROOT / args.input

    df = pd.read_parquet(path)
    groups = feature_groups(list(df.columns))

    is_test = df["target_date"] >= TEST_START
    train, test = df[~is_test], df[is_test]
    y_tr, y_te = train["target"].to_numpy(), test["target"].to_numpy()
    log.info("train: %d rows (..%s)   test: %d rows (%s..)",
             len(train), train["target_date"].max().date(), len(test), TEST_START)
    log.info("test class counts: %s",
             dict(zip(CLASS_NAMES, np.bincount(y_te, minlength=len(LABELS)).tolist())))

    # ---- baseline: tomorrow = today ----
    results: dict[str, dict] = {
        "baseline_persistence": {
            "features": "-",
            "cv_macro_f1": float(f1_score(y_tr, train["class_today"], average="macro")),
            "test": scores(y_te, test["class_today"]),
        }
    }

    # ---- tuned models on both feature sets ----
    cv = TimeSeriesSplit(n_splits=4)
    fitted: dict[str, tuple] = {}
    for set_name, cols in groups.items():
        for model_name, (estimator, grid) in candidates().items():
            key = f"{model_name}__{set_name}"
            log.info("tuning  %s (%d features)", key, len(cols))
            search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
            search.fit(train[cols], y_tr)
            pred = search.best_estimator_.predict(test[cols])
            results[key] = {
                "features": set_name,
                "best_params": search.best_params_,
                "cv_macro_f1": float(search.best_score_),
                "test": scores(y_te, pred),
            }
            fitted[key] = (search.best_estimator_, cols)

    # ---- results table ----
    table = pd.DataFrame(
        [
            {
                "model": k,
                "cv_macroF1": r["cv_macro_f1"],
                "test_acc": r["test"]["accuracy"],
                "test_macroF1": r["test"]["macro_f1"],
                "unhealthy_recall": r["test"]["recall"]["Unhealthy+"],
            }
            for k, r in results.items()
        ]
    ).sort_values("cv_macroF1", ascending=False)
    log.info("results (sorted by CV macro-F1):\n%s", table.round(3).to_string(index=False))

    # ---- pick by CV, report on test ----
    best_key = max(fitted, key=lambda k: results[k]["cv_macro_f1"])
    best, base = results[best_key]["test"], results["baseline_persistence"]["test"]
    gain = best["macro_f1"] - base["macro_f1"]
    log.info("best by CV: %s", best_key)
    log.info("test macro-F1: %.3f vs baseline %.3f  (+%.1f points, %+.0f%% relative)",
             best["macro_f1"], base["macro_f1"], 100 * gain, 100 * gain / base["macro_f1"])
    log.info("confusion matrix (rows=true, cols=pred; %s):\n%s",
             "/".join(CLASS_NAMES), np.array(best["confusion"]))

    # ---- what does the best model rely on? ----
    model, cols = fitted[best_key]
    pi = permutation_importance(model, test[cols], y_te, scoring="f1_macro",
                                n_repeats=20, random_state=0, n_jobs=-1)
    importance = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    log.info("top 10 features (permutation importance, drop in macro-F1):\n%s",
             importance.head(10).round(4).to_string())

    # ---- save artifacts ----
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "features": cols, "class_names": CLASS_NAMES,
         "key": best_key, "test_start": TEST_START},
        MODELS_DIR / "best_model.joblib",
    )
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(
            {"best": best_key, "test_start": TEST_START,
             "n_train": len(train), "n_test": len(test),
             "results": results,
             "permutation_importance": importance.round(4).to_dict()},
            indent=2, default=_json,
        )
    )
    log.info("wrote   models/best_model.joblib, models/metrics.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    main()
