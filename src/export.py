"""Export small, deploy-friendly artifacts for the dashboard.

The dashboard reads only app_data/ (a CSV + a JSON), so the deployed app needs
no scikit-learn, no raw data, and no pickled model. That avoids version
mismatches between your laptop and the hosting server.

Usage (from the project root, venv active, after src.train):
    python -m src.export
"""

from __future__ import annotations

import logging
import shutil

import joblib
import pandas as pd

from src.ingest import ROOT
from src.train import MODELS_DIR, TEST_START, full_proba, predict_with_threshold

log = logging.getLogger("export")

APP_DATA = ROOT / "app_data"
DAILY = ROOT / "data" / "processed" / "texas_dallas.parquet"
FEATURES = ROOT / "data" / "processed" / "texas_dallas_features.parquet"


def main() -> None:
    bundle = joblib.load(MODELS_DIR / "best_model.joblib")
    feats = pd.read_parquet(FEATURES)
    daily = pd.read_parquet(DAILY).set_index("date")

    test = feats[feats["target_date"] >= TEST_START].reset_index(drop=True)
    proba = full_proba(bundle["model"], test[bundle["features"]])

    out = pd.DataFrame(
        {
            "date": test["date"],
            "target_date": test["target_date"],
            "aqi_today": test["aqi_lag0"],
            "p_good": proba[:, 0],
            "p_moderate": proba[:, 1],
            "p_unhealthy": proba[:, 2],
            "pred": predict_with_threshold(proba, bundle["threshold"]),
            "actual": test["target"],
            "aqi_actual": test["target_date"].map(daily["aqi"]),
            "pollutant_actual": test["target_date"].map(daily["defining_param"]),
            "next_temp_max_c": test["next_wx_temperature_2m_max"],
        }
    )

    APP_DATA.mkdir(exist_ok=True)
    out.round(4).to_csv(APP_DATA / "predictions_test.csv", index=False)
    shutil.copy(MODELS_DIR / "metrics.json", APP_DATA / "metrics.json")
    log.info("wrote   app_data/predictions_test.csv (%d days), app_data/metrics.json", len(out))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    main()
