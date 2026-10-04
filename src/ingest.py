"""Ingest EPA daily county AQI + Open-Meteo historical weather into one daily table.

Output: data/processed/<state>_<county>.parquet with one row per calendar day.
Days with no EPA report keep NaN AQI (do NOT drop them: lag features must be
computed on a complete calendar, or "t-1" silently becomes "t-3" across gaps).

Usage (from the project root, venv active):
    python -m src.ingest --state "Texas" --county "Dallas" \
        --lat 32.78 --lon -96.80 --start-year 2020 --end-year 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

EPA_URL = "https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_county_{year}.zip"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "relative_humidity_2m_mean",
]
TIMEOUT = 60

log = logging.getLogger("ingest")


def _snake(cols: pd.Index) -> list[str]:
    return [c.strip().lower().replace(" ", "_") for c in cols]


def download(url: str, dest: Path) -> Path:
    """Stream a file to disk once; reuse the cached copy afterwards."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info("cached  %s", dest.name)
        return dest
    log.info("fetch   %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(dest)  # atomic: a crash never leaves a half-written "cached" file
    return dest


def load_epa(state: str, county: str, years: range) -> pd.DataFrame:
    frames = []
    for year in years:
        path = download(
            EPA_URL.format(year=year), RAW_DIR / f"daily_aqi_by_county_{year}.zip"
        )
        df = pd.read_csv(path, compression="zip")
        df.columns = _snake(df.columns)
        mask = (df["state_name"].str.casefold() == state.casefold()) & (
            df["county_name"].str.casefold() == county.casefold()
        )
        if not mask.any():
            log.warning("%d: no rows for %s County, %s", year, county, state)
        frames.append(df.loc[mask])

    epa = pd.concat(frames, ignore_index=True)
    if epa.empty:
        sys.exit(
            f"No EPA rows for '{county}' County, '{state}'. "
            "Use the EPA spelling without the word 'County' (e.g. --county Dallas)."
        )

    epa["date"] = pd.to_datetime(epa["date"])
    epa = epa.rename(
        columns={
            "category": "aqi_category",
            "defining_parameter": "defining_param",
            "number_of_sites_reporting": "n_sites",
        }
    )[["date", "aqi", "aqi_category", "defining_param", "n_sites"]]

    dupes = epa["date"].duplicated().sum()
    if dupes:
        log.warning("dropping %d duplicate EPA dates", dupes)
    return epa.drop_duplicates("date").sort_values("date")


def load_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    cache = RAW_DIR / f"weather_{lat:.3f}_{lon:.3f}_{start}_{end}.parquet"
    if cache.exists():
        log.info("cached  %s", cache.name)
        return pd.read_parquet(cache)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(WEATHER_VARS),
        "timezone": "auto",  # daily aggregates in local time, matching EPA local dates
    }
    log.info("fetch   Open-Meteo archive %s..%s", start, end)
    r = requests.get(WEATHER_URL, params=params, timeout=TIMEOUT)
    if r.status_code != 200:
        sys.exit(f"Open-Meteo error {r.status_code}: {r.text[:500]}")

    wx = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
    wx["date"] = pd.to_datetime(wx["date"])
    wx.to_parquet(cache, index=False)
    return wx


def build(args: argparse.Namespace) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    start, end = f"{args.start_year}-01-01", f"{args.end_year}-12-31"
    epa = load_epa(args.state, args.county, range(args.start_year, args.end_year + 1))
    wx = load_weather(args.lat, args.lon, start, end)

    calendar = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    df = calendar.merge(wx, on="date", how="left").merge(epa, on="date", how="left")

    # ---- data quality report ----
    n = len(df)
    log.info("rows: %d days (%s .. %s)", n, start, end)
    log.info("missing AQI:     %d days (%.1f%%)", df["aqi"].isna().sum(), 100 * df["aqi"].isna().mean())
    log.info("missing weather: %d days", df[WEATHER_VARS].isna().any(axis=1).sum())
    log.info(
        "category balance:\n%s",
        df["aqi_category"].value_counts(normalize=True).round(3).to_string(),
    )
    log.info(
        "defining pollutant:\n%s",
        df["defining_param"].value_counts(normalize=True).round(3).to_string(),
    )

    slug = f"{args.state}_{args.county}".lower().replace(" ", "-")
    out = PROCESSED_DIR / f"{slug}.parquet"
    df.to_parquet(out, index=False)
    log.info("wrote   %s", out.relative_to(ROOT))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--state", required=True, help='EPA state name, e.g. "Texas"')
    p.add_argument("--county", required=True, help='EPA county name, e.g. "Dallas"')
    p.add_argument("--lat", type=float, required=True, help="latitude for weather")
    p.add_argument("--lon", type=float, required=True, help="longitude for weather")
    p.add_argument("--start-year", type=int, default=2020)
    p.add_argument("--end-year", type=int, default=2024)
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    build(parse_args())
