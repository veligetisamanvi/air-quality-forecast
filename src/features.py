"""Build model-ready features + next-day target from the ingested daily table.

Each row t describes "end of day t" and its target is the AQI class on day t+1.

Leakage rule: every feature uses only information available at the end of day t,
EXCEPT columns prefixed `next_wx_` (tomorrow's actual weather, standing in for a
forecast). They are kept in a separate group so train.py can compare models
with and without them.

Usage (from the project root, venv active):
    python -m src.features --input data/processed/texas_dallas.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest import ROOT, WEATHER_VARS

log = logging.getLogger("features")

CLASS_NAMES = ["Good", "Moderate", "Unhealthy+"]
CATEGORY_TO_CLASS = {
    "Good": 0,
    "Moderate": 1,
    # too rare to learn separately (~5 "Unhealthy" days in 5 yrs) -> merged
    "Unhealthy for Sensitive Groups": 2,
    "Unhealthy": 2,
    "Very Unhealthy": 2,
    "Hazardous": 2,
}
AQI_LAGS = [0, 1, 2, 6]  # today, yesterday, 2 days ago, one week before target
ROLL_WINDOWS = [3, 7]
POLLUTANT_FLAGS = {"PM2.5": "param_pm25", "Ozone": "param_ozone"}
META_COLS = ["date", "target_date", "target"]


def _weather_block(df: pd.DataFrame, prefix: str, shift: int) -> pd.DataFrame:
    """Weather columns shifted by `shift` rows; wind direction made circular."""
    out = pd.DataFrame(index=df.index)
    for col in WEATHER_VARS:
        s = df[col].shift(shift)
        if col == "wind_direction_10m_dominant":
            # 359 deg and 1 deg are neighbours; raw degrees would put them far apart
            rad = np.deg2rad(s)
            out[f"{prefix}wind_dir_sin"] = np.sin(rad)
            out[f"{prefix}wind_dir_cos"] = np.cos(rad)
        else:
            out[f"{prefix}{col}"] = s
    return out


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values("date").reset_index(drop=True)

    # shift(k) means "k days" only if the calendar has no gaps
    gaps = df["date"].diff().dropna().ne(pd.Timedelta(days=1))
    if gaps.any():
        raise ValueError("Calendar has gaps; re-run ingest (it builds a full calendar).")

    unknown = set(df["aqi_category"].dropna()) - set(CATEGORY_TO_CLASS)
    if unknown:
        raise ValueError(f"Unmapped AQI categories: {unknown}")
    cls = df["aqi_category"].map(CATEGORY_TO_CLASS)
    aqi = df["aqi"]

    feats = pd.DataFrame({"date": df["date"]})

    # --- AQI history (known at end of day t) ---
    for k in AQI_LAGS:
        feats[f"aqi_lag{k}"] = aqi.shift(k)
    for w in ROLL_WINDOWS:
        roll = aqi.rolling(w, min_periods=w)
        feats[f"aqi_roll{w}_mean"] = roll.mean()
        feats[f"aqi_roll{w}_max"] = roll.max()
    feats["aqi_delta1"] = aqi - aqi.shift(1)
    feats["class_today"] = cls

    # --- which pollutant drove today's AQI ---
    for value, name in POLLUTANT_FLAGS.items():
        feats[name] = (df["defining_param"] == value).astype(int)

    # --- calendar features describe the TARGET day (known in advance, no leakage) ---
    target_date = df["date"] + pd.Timedelta(days=1)
    doy = target_date.dt.dayofyear
    feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    feats["is_weekend"] = (target_date.dt.dayofweek >= 5).astype(int)

    # --- weather: today's (safe) and tomorrow's (forecast stand-in) ---
    feats = feats.join(_weather_block(df, "wx_", 0))
    feats = feats.join(_weather_block(df, "next_wx_", -1))

    # --- target: tomorrow's class ---
    feats["target_date"] = target_date
    feats["target"] = cls.shift(-1)

    before = len(feats)
    feats = feats.dropna().reset_index(drop=True)
    log.info("dropped %d edge rows (lag warm-up + last day)", before - len(feats))

    feats["target"] = feats["target"].astype(int)
    feats["class_today"] = feats["class_today"].astype(int)
    return feats


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Feature sets for the ablation in train.py."""
    next_wx = [c for c in columns if c.startswith("next_wx_")]
    base = [c for c in columns if c not in META_COLS and c not in next_wx]
    return {"A_today_only": base, "B_with_next_weather": base + next_wx}


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, default=Path("data/processed/texas_dallas.parquet"))
    args = p.parse_args(argv)

    src = args.input if args.input.is_absolute() else ROOT / args.input
    feats = build_features(pd.read_parquet(src))

    groups = feature_groups(list(feats.columns))
    log.info("rows: %d  (%s .. %s)", len(feats), feats["date"].min().date(), feats["date"].max().date())
    for name, cols in groups.items():
        log.info("feature set %-20s %d features", name, len(cols))
    balance = feats["target"].value_counts(normalize=True).sort_index()
    balance.index = [CLASS_NAMES[i] for i in balance.index]
    log.info("target balance:\n%s", balance.round(3).to_string())

    out = src.with_name(src.stem + "_features.parquet")
    feats.to_parquet(out, index=False)
    log.info("wrote   %s", out.relative_to(ROOT))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    main()
