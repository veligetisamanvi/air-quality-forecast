# Next-Day Air Quality Forecaster · Dallas County, TX

**Live app:** https://dallas-air-quality.streamlit.app

A machine learning model that predicts whether tomorrow's air quality in Dallas County will be
**Good**, **Moderate**, or **Unhealthy+** (Unhealthy for Sensitive Groups or worse), using
15 years of EPA air quality data and historical weather.

On 2024, a year the model never saw during training, it **caught 14 of 19 unhealthy-air days (74%)**,
compared with 4 of 19 (21%) for a simple "tomorrow will be the same as today" baseline.

---

## What the app shows

- **2024 results** with a slider to adjust how sensitive the warnings are
- **Timeline** of correct warnings, false alarms, and missed bad-air days
- **Pick a day:** what the model predicted the evening before vs. what actually happened
- **Model comparison** and **what drives the forecast** (permutation importance)
- **46 years of history:** bad-air days per year in Dallas County, 1980–2025
- **Look up any day:** weather and air quality for any date from 1980 to 2025

## Data

| Source | What | Years |
|---|---|---|
| [EPA AirData](https://aqs.epa.gov/aqsweb/airdata/download_files.html) | Daily county AQI, category, main pollutant | 2010–2024 (model), 1980–2025 (history) |
| [Open-Meteo Historical Weather API](https://open-meteo.com/) | Daily temperature, rain, wind, sunlight, humidity | Same |

Dallas County had **no missing days** in either source for 2010–2024.

## Method

**Target.** Tomorrow's AQI category. "Unhealthy for Sensitive Groups" and worse are merged into one
class, **Unhealthy+**, because the most severe categories were too rare to learn separately
(about 5 "Unhealthy" days in 5 years). Unhealthy+ is still only **~4% of days**.

**Features** (each row = the end of one day):
- AQI lags (today, yesterday, 2 and 6 days back) and 3/7-day rolling mean and max
- Which pollutant drove today's AQI (PM2.5 or ozone)
- Season (day of year as sin/cos) and weekend flag
- Today's weather, plus tomorrow's weather as a stand-in for a forecast (see limitations)
- Wind direction encoded as sin/cos, so 359° and 1° are treated as neighbors

**Validation.** Trained on 2010–2023 (5,106 days) and tested once on **2024** (366 days).
Hyperparameters were tuned with time-ordered cross-validation (`TimeSeriesSplit`) on the training years only.
The final model was chosen by its cross-validation score, never by its test score.

**Class imbalance.** Models use balanced class weights and are judged by **macro-F1**
(which weights all three classes equally), since accuracy alone would reward ignoring the rare unhealthy days.

**Warning threshold.** Instead of only warning when Unhealthy+ is the model's top guess, the app warns
whenever the model's Unhealthy+ score passes a cutoff. The cutoff was tuned on out-of-fold predictions
from the training years to catch at least 70% of bad-air days.

## Results (2024 test year)

| Decision rule | Bad-air days caught | False alarms | Accuracy | Macro-F1 |
|---|---|---|---|---|
| Baseline: tomorrow = today | 4 / 19 | 15 | 62% | 0.489 |
| Gradient boosting, top guess | 8 / 19 | 9 | 72% | **0.631** |
| **Gradient boosting, tuned warning cutoff** | **14 / 19** | 34 | 68% | 0.603 |

**Model comparison** (macro-F1):

| Model | Training CV | 2024 test |
|---|---|---|
| Baseline: tomorrow = today | 0.573 | 0.489 |
| Logistic regression, today's data only | 0.505 | 0.514 |
| Gradient boosting, today's data only | 0.551 | 0.576 |
| Logistic regression, + tomorrow's weather | 0.566 | 0.556 |
| **Gradient boosting, + tomorrow's weather** | **0.593** | **0.631** |

**Takeaways**
- Adding tomorrow's weather improved the gradient boosting model by about **5.5 macro-F1 points** on 2024.
- Training on 15 years instead of 4 made the results more trustworthy. In an earlier run trained only on
  2020–2023, the models beat the baseline on 2024 but not on the training years.
- The most important features were **tomorrow's high temperature**, **today's AQI**, tomorrow's humidity,
  and whether today was an **ozone** day. This matches the chemistry: ozone forms on hot, sunny days.
- The tuned cutoff caught 70% of bad days on the training years and **74%** on 2024, so the rule held up on unseen data.

## Limitations

- **Tomorrow's weather is the actual weather, not a forecast.** In real use, forecasts have errors,
  so live performance would likely be somewhat lower than these results.
- **Small test set.** 2024 had only 19 bad-air days, so the test numbers have wide uncertainty.
  The improvement over the baseline in cross-validation (0.593 vs. 0.573) is real but modest.
- **The risk score is not a calibrated probability.** Balanced class weights shift the scores,
  which is why the tuned cutoff is low (0.07). The app shows risk levels, not percentages.
- **More warnings means more false alarms.** The tuned rule issues about one unnecessary warning
  every 11 days. That trade-off was chosen deliberately, since missing a bad-air day is worse for health.
- **Historical comparisons are approximate.** PM2.5 was only measured from about 1999, and monitors
  and EPA's AQI rules have changed over time. That's why the model trains on 2010 onward only.
- **One county.** The model was built and tested only for Dallas County.

## Project structure

```
src/ingest.py     # download EPA AQI + Open-Meteo weather, join into one daily table
src/features.py   # lags, rolling stats, seasonality, weather, next-day target
src/train.py      # baseline vs. models, time-split CV, threshold tuning, permutation importance
src/export.py     # small CSV/JSON for the app (no scikit-learn needed to run the app)
src/history.py    # 1980–2025 yearly summary + daily weather/AQI lookup
app.py            # Streamlit dashboard
app_data/         # files the app reads
```

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib streamlit requests pyarrow

python -m src.ingest --state "Texas" --county "Dallas" --lat 32.77 --lon -96.78 --start-year 2010 --end-year 2024
python -m src.features
python -m src.train
python -m src.export
python -m src.history --state "Texas" --county "Dallas" --lat 32.77 --lon -96.78

streamlit run app.py
```

## Author

Built by [@veligetisamanvi](https://github.com/veligetisamanvi).
