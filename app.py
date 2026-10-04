"""Streamlit dashboard: next-day air quality warnings for Dallas County, backtested on 2024.

Run locally:   streamlit run app.py
Reads only app_data/ (produced by `python -m src.export`).
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "app_data"
CLASSES = ["Good", "Moderate", "Unhealthy+"]
UNHEALTHY = 2

MODEL_NAMES = {
    "baseline_persistence": "Baseline: tomorrow = today",
    "logreg__A_today_only": "Logistic regression · today's data",
    "hgb__A_today_only": "Gradient boosting · today's data",
    "logreg__B_with_next_weather": "Logistic regression · + tomorrow's weather",
    "hgb__B_with_next_weather": "Gradient boosting · + tomorrow's weather",
}
FEATURE_NAMES = {
    "next_wx_temperature_2m_max": "Tomorrow's high temperature",
    "next_wx_temperature_2m_min": "Tomorrow's low temperature",
    "next_wx_relative_humidity_2m_mean": "Tomorrow's humidity",
    "next_wx_shortwave_radiation_sum": "Tomorrow's sunlight",
    "next_wx_wind_speed_10m_max": "Tomorrow's wind speed",
    "next_wx_wind_dir_sin": "Tomorrow's wind direction (E–W)",
    "next_wx_wind_dir_cos": "Tomorrow's wind direction (N–S)",
    "next_wx_precipitation_sum": "Tomorrow's rain",
    "wx_temperature_2m_max": "Today's high temperature",
    "wx_temperature_2m_min": "Today's low temperature",
    "wx_relative_humidity_2m_mean": "Today's humidity",
    "wx_shortwave_radiation_sum": "Today's sunlight",
    "wx_wind_speed_10m_max": "Today's wind speed",
    "aqi_lag0": "Today's AQI",
    "aqi_lag1": "Yesterday's AQI",
    "aqi_lag2": "AQI 2 days ago",
    "aqi_lag6": "AQI 6 days ago",
    "aqi_roll3_mean": "3-day average AQI",
    "aqi_roll7_mean": "7-day average AQI",
    "aqi_roll3_max": "3-day max AQI",
    "aqi_roll7_max": "7-day max AQI",
    "aqi_delta1": "AQI change since yesterday",
    "class_today": "Today's AQI category",
    "param_ozone": "Today driven by ozone",
    "param_pm25": "Today driven by PM2.5",
    "doy_sin": "Season (sin)",
    "doy_cos": "Season (cos)",
    "is_weekend": "Weekend",
}
OUTCOME_COLORS = {
    "Warned – correct": "#d62728",
    "False alarm": "#f2a541",
    "Missed bad day": "#6a3d9a",
}
VERDICT = {
    0: ("🟢 Good", "Air quality expected to be satisfactory."),
    1: ("🟡 Moderate", "Acceptable; unusually sensitive people may be affected."),
    2: ("🟠 Warning: Unhealthy+", "Sensitive groups should limit prolonged outdoor exertion."),
}


# --------------------------------------------------------------------------- data
@st.cache_data
def load() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA / "predictions_test.csv", parse_dates=["date", "target_date"])
    metrics = json.loads((DATA / "metrics.json").read_text())
    return df, metrics


def apply_rule(df: pd.DataFrame, threshold: float) -> np.ndarray:
    """Same decision rule as training: warn if P(Unhealthy+) >= threshold, else Good vs Moderate."""
    safe = np.where(df["p_good"] >= df["p_moderate"], 0, 1)
    return np.where(df["p_unhealthy"] >= threshold, UNHEALTHY, safe)


def macro_f1(actual: np.ndarray, pred: np.ndarray) -> float:
    f1s = []
    for c in range(len(CLASSES)):
        tp = np.sum((actual == c) & (pred == c))
        fp = np.sum((actual != c) & (pred == c))
        fn = np.sum((actual == c) & (pred != c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def false_alarms_from_confusion(confusion: list[list[int]]) -> int:
    return int(sum(row[UNHEALTHY] for i, row in enumerate(confusion) if i != UNHEALTHY))


# --------------------------------------------------------------------------- page
st.set_page_config(page_title="Dallas Air Quality Forecaster", page_icon="🌫️", layout="wide")
df, metrics = load()
actual = df["actual"].to_numpy()
n_bad = int(np.sum(actual == UNHEALTHY))

st.title("🌫️ Next-Day Air Quality Forecaster · Dallas County, TX")
st.caption(
    "Predicts tomorrow's EPA air quality category from today's air quality and the weather. "
    f"Trained on {metrics['n_train']:,} days (2010–2023); every result below is from "
    "**2024, a year the model never saw during training.**"
)

# ---- sidebar: warning sensitivity ----
with st.sidebar:
    st.header("Warning sensitivity")
    tuned = float(metrics["threshold"])
    threshold = st.slider(
        "Warn when the model's risk score is at least",
        min_value=0.02, max_value=0.60, value=tuned, step=0.01,
    )
    st.caption(
        f"Default **{tuned:.2f}** was chosen on the training years to catch at least "
        f"{metrics['min_unhealthy_recall']:.0%} of bad-air days. Lower = more warnings "
        "(fewer missed days, more false alarms). The score is a ranking, not a calibrated probability."
    )

pred = apply_rule(df, threshold)
caught = int(np.sum((actual == UNHEALTHY) & (pred == UNHEALTHY)))
false_alarms = int(np.sum((actual != UNHEALTHY) & (pred == UNHEALTHY)))
base = metrics["results"]["baseline_persistence"]["test"]
base_caught = round(base["recall"]["Unhealthy+"] * n_bad)
base_fa = false_alarms_from_confusion(base["confusion"])

# ---- headline metrics ----
st.subheader("How well does it work in 2024?")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Bad-air days caught", f"{caught} of {n_bad}", f"{caught - base_caught:+d} vs baseline")
c2.metric("False alarms", false_alarms, f"{false_alarms - base_fa:+d} vs baseline", delta_color="inverse")
c3.metric("Accuracy", f"{np.mean(pred == actual):.0%}", f"{np.mean(pred == actual) - base['accuracy']:+.0%} vs baseline")
c4.metric("Macro-F1", f"{macro_f1(actual, pred):.3f}", f"{macro_f1(actual, pred) - base['macro_f1']:+.3f} vs baseline")
st.caption("Baseline = \"tomorrow will be the same as today.\" Bad-air day = Unhealthy for Sensitive Groups or worse.")

# ---- timeline ----
plot = df.assign(
    outcome=np.select(
        [
            (pred == UNHEALTHY) & (actual == UNHEALTHY),
            (pred == UNHEALTHY) & (actual != UNHEALTHY),
            (pred != UNHEALTHY) & (actual == UNHEALTHY),
        ],
        list(OUTCOME_COLORS),
        default="",
    ),
    actual_label=[CLASSES[a] for a in actual],
)
line = alt.Chart(plot).mark_line(color="#9aa0a6", strokeWidth=1).encode(
    x=alt.X("target_date:T", title=None),
    y=alt.Y("aqi_actual:Q", title="Actual AQI"),
)
limit = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(strokeDash=[4, 4], color="#d62728").encode(y="y:Q")
points = alt.Chart(plot[plot["outcome"] != ""]).mark_point(filled=True, size=70).encode(
    x="target_date:T",
    y="aqi_actual:Q",
    color=alt.Color(
        "outcome:N",
        scale=alt.Scale(domain=list(OUTCOME_COLORS), range=list(OUTCOME_COLORS.values())),
        legend=alt.Legend(title=None, orient="top"),
    ),
    tooltip=[
        alt.Tooltip("target_date:T", title="Date"),
        alt.Tooltip("aqi_actual:Q", title="Actual AQI"),
        alt.Tooltip("pollutant_actual:N", title="Main pollutant"),
        alt.Tooltip("outcome:N", title="Outcome"),
    ],
)
st.altair_chart((line + limit + points).properties(height=320))
st.caption("Dashed line: AQI 100. Above it, air is unhealthy for sensitive groups.")

# ---- single-day explorer ----
st.subheader("Pick a day")
day = st.date_input(
    "Forecast for",
    value=df.loc[df["aqi_actual"].idxmax(), "target_date"].date(),
    min_value=df["target_date"].min().date(),
    max_value=df["target_date"].max().date(),
)
row_idx = df.index[df["target_date"].dt.date == day]
if len(row_idx):
    i = row_idx[0]
    r = df.loc[i]
    label, advice = VERDICT[int(pred[i])]
    a, b, c = st.columns(3)
    a.metric("Forecast (made the evening before)", label)
    b.metric("What actually happened", f"{CLASSES[int(r['actual'])]} · AQI {r['aqi_actual']:.0f}")
    c.metric("Forecast high temperature", f"{r['next_temp_max_c'] * 9 / 5 + 32:.0f} °F")
    st.caption(f"{advice}  Inputs: AQI on {r['date']:%b %d} was {r['aqi_today']:.0f}.")
else:
    st.info("No forecast for that date.")

# ---- model comparison + drivers ----
left, right = st.columns(2)
with left:
    st.subheader("Models compared")
    rows = [
        {
            "Model": MODEL_NAMES.get(k, k),
            "Training CV macro-F1": r["cv_macro_f1"],
            "2024 macro-F1": r["test"]["macro_f1"],
            "2024 bad days caught": f"{round(r['test']['recall']['Unhealthy+'] * n_bad)}/{n_bad}",
        }
        for k, r in metrics["results"].items()
        if k in MODEL_NAMES
    ]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
    st.caption(f"Deployed model: **{MODEL_NAMES.get(metrics['best'], metrics['best'])}**, "
               "chosen by training CV score (not by its 2024 score).")
with right:
    st.subheader("What drives the forecast")
    imp = (
        pd.Series(metrics["permutation_importance"]).sort_values(ascending=False).head(10)
        .rename(index=lambda k: FEATURE_NAMES.get(k, k)).rename("importance")
        .reset_index().rename(columns={"index": "feature"})
    )
    st.altair_chart(
        alt.Chart(imp).mark_bar(color="#4c78a8").encode(
            x=alt.X("importance:Q", title="Drop in macro-F1 when shuffled"),
            y=alt.Y("feature:N", sort="-x", title=None),
        ).properties(height=320)
    )

# ---- long-term history (optional: needs app_data/history.csv from src.history) ----
history_path = DATA / "history.csv"
if history_path.exists():
    hist = pd.read_csv(history_path)
    first, last = hist.iloc[0], hist.iloc[-1]
    st.subheader(f"Dallas County air over {int(last['year'] - first['year']) + 1} years")
    h1, h2 = st.columns(2)
    h1.metric(
        f"Bad-air days in {int(last['year'])}",
        int(last["unhealthy_days"]),
        f"{int(last['unhealthy_days'] - first['unhealthy_days']):+d} vs {int(first['year'])}",
        delta_color="inverse",
    )
    h2.metric(
        f"Average AQI in {int(last['year'])}",
        f"{last['mean_aqi']:.0f}",
        f"{last['mean_aqi'] - first['mean_aqi']:+.0f} vs {int(first['year'])}",
        delta_color="inverse",
    )
    tooltip = [
        alt.Tooltip("year:O", title="Year"),
        alt.Tooltip("unhealthy_days:Q", title="Bad-air days"),
        alt.Tooltip("mean_aqi:Q", title="Average AQI"),
        alt.Tooltip("top_pollutant:N", title="Most common main pollutant"),
        alt.Tooltip("days_reported:Q", title="Days with data"),
    ]
    bars = alt.Chart(hist).mark_bar(color="#d62728", opacity=0.75).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=-45, values=list(range(1980, 2031, 5)))),
        y=alt.Y("unhealthy_days:Q", title="Bad-air days per year"),
        tooltip=tooltip,
    )
    st.altair_chart(bars.properties(height=280))
    st.caption(
        "Bad-air day = Unhealthy for Sensitive Groups or worse. Compare decades loosely: "
        "the pollutants measured (PM2.5 only from ~1999), the number of monitors, and EPA's "
        "AQI rules have all changed over time. The model is trained on 2010–2023 only, "
        "because older years reflect a different pollution mix."
    )

# ---- method ----
with st.expander("Method and limitations"):
    st.markdown(
        """
- **Data:** EPA daily county AQI (AirData) and Open-Meteo historical weather, 2010–2024.
- **Target:** tomorrow's category. Good, Moderate, or Unhealthy+ (USG and worse merged; too rare to learn separately).
- **Validation:** trained on 2010–2023 with time-ordered cross-validation; tested once on 2024.
- **Imbalance:** bad-air days are ~4% of days, so models use balanced class weights and are judged by macro-F1, not accuracy.
- **Warning rule:** the cutoff was tuned on out-of-fold training predictions to catch ≥70% of bad days.
- **Limitation:** "tomorrow's weather" uses the *actual* next-day weather as a stand-in for a forecast,
  so real-time performance would likely be somewhat lower.
- **Limitation:** 2024 had only 19 bad-air days, so these test numbers have wide uncertainty.
"""
    )
