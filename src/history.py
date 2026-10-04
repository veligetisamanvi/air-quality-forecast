"""Yearly air quality history for the dashboard (separate from model training).

Downloads EPA daily county AQI for a long period (default 1980-2025) and writes
app_data/history.csv with one row per year. Weather is not needed here.

This does NOT touch data/processed/, so your training data stays 2010-2024.

Usage (from the project root, venv active):
    python -m src.history --state "Texas" --county "Dallas"
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import requests

from src.features import CATEGORY_TO_CLASS
from src.ingest import RAW_DIR, ROOT, load_epa

log = logging.getLogger("history")

APP_DATA = ROOT / "app_data"
UNHEALTHY_CATEGORIES = [c for c, k in CATEGORY_TO_CLASS.items() if k == 2]


def yearly_summary(epa: pd.DataFrame) -> pd.DataFrame:
    epa = epa.assign(year=epa["date"].dt.year)
    g = epa.groupby("year")
    return pd.DataFrame(
        {
            "days_reported": g.size(),
            "mean_aqi": g["aqi"].mean().round(1),
            "unhealthy_days": g["aqi_category"].apply(lambda s: int(s.isin(UNHEALTHY_CATEGORIES).sum())),
            "good_share": g["aqi_category"].apply(lambda s: round(float((s == "Good").mean()), 3)),
            "top_pollutant": g["defining_param"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else ""),
        }
    ).reset_index()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--state", required=True)
    p.add_argument("--county", required=True)
    p.add_argument("--start-year", type=int, default=1980)
    p.add_argument("--end-year", type=int, default=2025)
    args = p.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        epa = load_epa(args.state, args.county, range(args.start_year, args.end_year + 1))
    except requests.HTTPError as e:
        sys.exit(f"Download failed ({e}). If the newest year isn't published yet, "
                 f"re-run with --end-year {args.end_year - 1}.")

    hist = yearly_summary(epa)
    thin = hist[hist["days_reported"] < 300]
    if not thin.empty:
        log.warning("years with < 300 reported days (less reliable): %s", thin["year"].tolist())

    APP_DATA.mkdir(exist_ok=True)
    hist.to_csv(APP_DATA / "history.csv", index=False)
    log.info("years: %d-%d", hist["year"].min(), hist["year"].max())
    log.info("\n%s", hist.to_string(index=False))
    log.info("wrote   app_data/history.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    main()
