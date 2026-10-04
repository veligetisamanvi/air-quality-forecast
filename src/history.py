"""Yearly air quality history for the dashboard (separate from model training).

Downloads EPA daily county AQI for a long period (default 1980-2025) and writes
app_data/history.csv with one row per year. If --lat/--lon are given, it also
fetches daily weather and writes app_data/daily_lookup.csv (one row per day)
for the dashboard's "look up any day" feature.

This does NOT touch data/processed/, so your training data stays 2010-2024.

Usage (from the project root, venv active):
    python -m src.history --state "Texas" --county "Dallas" --lat 32.77 --lon -96.78
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import requests

from src.features import CATEGORY_TO_CLASS
from src.ingest import RAW_DIR, ROOT, load_epa, load_weather

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


def daily_lookup(epa: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day: AQI + weather in US units, for the date lookup."""
    cal = pd.DataFrame({"date": pd.date_range(wx["date"].min(), wx["date"].max(), freq="D")})
    df = cal.merge(wx, on="date", how="left").merge(
        epa[["date", "aqi", "aqi_category", "defining_param"]], on="date", how="left"
    )
    return pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d"),
            "aqi": df["aqi"],
            "category": df["aqi_category"],
            "pollutant": df["defining_param"],
            "temp_max_f": (df["temperature_2m_max"] * 9 / 5 + 32).round(0),
            "temp_min_f": (df["temperature_2m_min"] * 9 / 5 + 32).round(0),
            "rain_in": (df["precipitation_sum"] / 25.4).round(2),
            "wind_max_mph": (df["wind_speed_10m_max"] * 0.621371).round(0),
            "humidity_pct": df["relative_humidity_2m_mean"].round(0),
        }
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--state", required=True)
    p.add_argument("--county", required=True)
    p.add_argument("--start-year", type=int, default=1980)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--lat", type=float, help="latitude; with --lon, also builds daily_lookup.csv")
    p.add_argument("--lon", type=float, help="longitude; with --lat, also builds daily_lookup.csv")
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

    if args.lat is not None and args.lon is not None:
        wx = load_weather(args.lat, args.lon, f"{args.start_year}-01-01", f"{args.end_year}-12-31")
        lookup = daily_lookup(epa, wx)
        lookup.to_csv(APP_DATA / "daily_lookup.csv", index=False)
        log.info("wrote   app_data/daily_lookup.csv (%d days, %d without AQI)",
                 len(lookup), lookup["aqi"].isna().sum())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    main()
