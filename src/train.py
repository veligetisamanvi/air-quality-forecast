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
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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


# ---------------------------------------------------------------------------
# Threshold tuning for the Unhealthy+ warning
# ---------------------------------------------------------------------------
UNHEALTHY = CLASS_NAMES.index("Unhealthy+")
MIN_UNHEALTHY_RECALL = 0.70  # policy: catch at least 70% of bad-air days
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def full_proba(model, X: pd.DataFrame) -> np.ndarray:
    """predict_proba with one column per class in LABELS, even if a fold missed a class."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(LABELS)))
    out[:, model.classes_] = p
    return out


def predict_with_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Warn (Unhealthy+) when P(Unhealthy+) >= threshold; otherwise pick Good vs Moderate."""
    safe = [i for i in LABELS if i != UNHEALTHY]
    pred = np.array(safe)[proba[:, safe].argmax(axis=1)]
    pred[proba[:, UNHEALTHY] >= threshold] = UNHEALTHY
    return pred


def tune_threshold(estimator, X: pd.DataFrame, y: np.ndarray, cv) -> tuple[float, pd.DataFrame]:
    """Choose the threshold on out-of-fold predictions from the TRAINING years only.

    Rule: among thresholds whose OOF Unhealthy+ recall >= MIN_UNHEALTHY_RECALL,
    take the one with the best macro-F1 (fewest needless false alarms).
    Falls back to best macro-F1 overall if no threshold reaches the recall target.
    """
    probas, ys = [], []
    for tr_idx, val_idx in cv.split(X):
        fold_model = clone(estimator).fit(X.iloc[tr_idx], y[tr_idx])
        probas.append(full_proba(fold_model, X.iloc[val_idx]))
        ys.append(y[val_idx])
    proba, y_oof = np.vstack(probas), np.concatenate(ys)

    rows = []
    for t in THRESHOLDS:
        pred = predict_with_threshold(proba, t)
        rows.append({
            "threshold": float(t),
            "macro_f1": f1_score(y_oof, pred, labels=LABELS, average="macro", zero_division=0),
            "unhealthy_recall": recall_score(y_oof, pred, labels=[UNHEALTHY], average=None, zero_division=0)[0],
            "unhealthy_precision": precision_score(y_oof, pred, labels=[UNHEALTHY], average=None, zero_division=0)[0],
        })
    curve = pd.DataFrame(rows)

    ok = curve[curve["unhealthy_recall"] >= MIN_UNHEALTHY_RECALL]
    if ok.empty:
        log.warning("no threshold reaches %.0f%% OOF recall; using best macro-F1", 100 * MIN_UNHEALTHY_RECALL)
        ok = curve
    best = ok.sort_values("macro_f1", ascending=False).iloc[0]
    return float(best["threshold"]), curve


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

    cv = TimeSeriesSplit(n_splits=4)

    # ---- baseline: tomorrow = today ----
    # Scored on the SAME validation folds as the models, so CV numbers are comparable.
    today_tr = train["class_today"].to_numpy()
    base_cv = np.mean([
        f1_score(y_tr[val], today_tr[val], labels=LABELS, average="macro", zero_division=0)
        for _, val in cv.split(train)
    ])
    results: dict[str, dict] = {
        "baseline_persistence": {
            "features": "-",
            "cv_macro_f1": float(base_cv),
            "test": scores(y_te, test["class_today"]),
        }
    }

    # ---- tuned models on both feature sets ----
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

    # ---- tune the Unhealthy+ warning threshold (training years only) ----
    model, cols = fitted[best_key]
    threshold, curve = tune_threshold(model, train[cols], y_tr, cv)
    chosen = curve.loc[curve["threshold"] == threshold].iloc[0]
    log.info("threshold tuning (OOF on training years, target recall >= %.0f%%):",
             100 * MIN_UNHEALTHY_RECALL)
    log.info("  chosen P(Unhealthy+) >= %.2f -> OOF macro-F1 %.3f, recall %.2f, precision %.2f",
             threshold, chosen["macro_f1"], chosen["unhealthy_recall"], chosen["unhealthy_precision"])

    tuned = scores(y_te, predict_with_threshold(full_proba(model, test[cols]), threshold))
    tuned_key = f"{best_key}__tuned"
    results[tuned_key] = {
        "features": results[best_key]["features"],
        "threshold": threshold,
        "cv_macro_f1": float(chosen["macro_f1"]),
        "test": tuned,
    }
    compare = pd.DataFrame(
        [
            {"rule": name, "test_acc": r["accuracy"], "test_macroF1": r["macro_f1"],
             "unhealthy_recall": r["recall"]["Unhealthy+"],
             "false_alarms": sum(row[UNHEALTHY] for i, row in enumerate(r["confusion"]) if i != UNHEALTHY)}
            for name, r in [("baseline", base), ("argmax", best), (f"threshold {threshold:.2f}", tuned)]
        ]
    )
    log.info("2024 test, decision rules compared:\n%s", compare.round(3).to_string(index=False))
    log.info("tuned confusion matrix (rows=true, cols=pred; %s):\n%s",
             "/".join(CLASS_NAMES), np.array(tuned["confusion"]))

    # ---- what does the best model rely on? ----
    pi = permutation_importance(model, test[cols], y_te, scoring="f1_macro",
                                n_repeats=20, random_state=0, n_jobs=-1)
    importance = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    log.info("top 10 features (permutation importance, drop in macro-F1):\n%s",
             importance.head(10).round(4).to_string())

    # ---- save artifacts ----
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "features": cols, "class_names": CLASS_NAMES,
         "key": best_key, "test_start": TEST_START, "threshold": threshold},
        MODELS_DIR / "best_model.joblib",
    )
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(
            {"best": best_key, "test_start": TEST_START,
             "n_train": len(train), "n_test": len(test),
             "threshold": threshold, "min_unhealthy_recall": MIN_UNHEALTHY_RECALL,
             "results": results,
             "threshold_curve": curve.round(4).to_dict(orient="records"),
             "permutation_importance": importance.round(4).to_dict()},
            indent=2, default=_json,
        )
    )
    log.info("wrote   models/best_model.joblib, models/metrics.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    main()
