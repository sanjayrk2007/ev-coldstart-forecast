# EV Cold-Start Demand Forecasting

> Predicting hourly EV charging demand for newly installed stations with **zero usage history** — plus pre-installation site scoring for operators.


---

## The Problem

Demand forecasting requires history. New stations have none. Without forecasts, operators can't optimise pricing, staffing, or grid load in the critical weeks after installation.

This system solves it with four layers:

| Layer | What it does |
|-------|-------------|
| **Transfer learning** | Borrows temporal patterns from 107 data-rich stations |
| **Synthetic augmentation** | Generates plausible session history via Gaussian Copula |
| **Conformal prediction** | Calibrated uncertainty intervals, not just point estimates |
| **Site selection** | Scores candidate locations using OSM POI features — before installation |

---

## Results

| Metric | Value |
|--------|-------|
| MAE improvement (transfer vs baseline) | ~1.1–1.4× on held-out site |
| Coverage @ 80% intervals | **0.92** (target: 0.80) |
| Coverage @ 90% intervals | **0.96** (target: 0.90) |
| Training stations | 107 (Caltech + JPL) |
| Test site | office001 — 8 stations, never seen during training |
| Forecast horizon | 168 hours (one week) |

**Notable findings**

- Synthetic augmentation hurt performance in all configurations — the transfer model saturates the available signal at one week of fine-tuning. Augmentation damage decreases monotonically with real data volume. Documented as a finding, not a failure.
- Zero-inflation (86–96% of hours are zero demand) breaks single-quantile conformal coverage. Conditional quantiles per regime (zero vs non-zero) reduced worst-station gap from −0.057 to −0.014.

---

## Quick Start

```bash
git clone https://github.com/your-username/ev-coldstart-forecast
cd ev-coldstart-forecast
cp .env.example .env
python -m venv venv && venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Start API
uvicorn src.api.main:app --reload --port 8000

# Run tests
pytest tests/test_api.py -v   # 10 tests, all passing

# MLflow UI
mlflow ui --port 5000
```


## How It Works

**Transfer learning** — One LightGBM model trained on all 107 stations. `site_encoded` as a coarse integer feature (not one-hot) outperforms under zero-inflation + L1 loss. Fine-tuning adds station-specific correction trees via `lgb.train(init_model=global_booster)`.

**Conformal prediction** — Distribution-free coverage using MAPIE, calibrated on the held-out office001 site. Conditional quantiles: `q80_nonzero=0.295`, `q90_nonzero=0.703`, both zero-regime quantiles near 0.

**Site selection** — 14-dimension OSM POI feature vectors per station (parking, offices, transit, highway proximity, etc.). Cosine similarity against 107 training profiles → top-3 similar stations → hour-matched weighted synthetic profile → global model inference.

---

## Project Status

| Phase | | Status |
|-------|-|--------|
| 0–2 | Setup, data pipeline, baselines | ✅ |
| 3 | Transfer learning | ✅ |
| 4 | Synthetic augmentation | ✅ |
| 5 | Conformal prediction | ✅ |
| 6 | Site selection scoring | ✅ |
| 7 | FastAPI backend | ✅ |
| 8 | React frontend | 🔄 In progress |
| 9–10 | Docker, deployment, docs | ⏳ |

---

## Repo Structure

```
src/
├── data/         fetch, loader, preprocessor, features, poi_features
├── models/       global_model, transfer, uncertainty, baseline
├── augmentation/ synthesizer (GaussianCopula + TimeGAN stub)
├── evaluation/   metrics
└── api/          schemas, predictor, main

models/           global_model.pkl, calibration.json
data/cache/       station_profiles.json (107 stations), poi_cache.json
tests/            test_api.py (10 tests)
notebooks/        eda.ipynb
reports/          calibration_curve.png, site_selection_weekly_profile.png
```

---

## Limitations

- **Augmentation** consistently hurt — transfer model already saturates at 1 week of data
- **Overpass API** blocked on dev network — Caltech/JPL POI features hardcoded; live path preserved
- **Geography** — trained on US campus/office data; wider uncertainty for other contexts
- **TimeGAN** — stub only; full training requires GPU
- **ARIMA** — skipped due to statsforecast/Windows multiprocessing conflict

---

## Data

ACN-Data (`ev.caltech.edu`) · 3 sites · Caltech (55 stations) · JPL (52 stations) · office001 (8 stations, held out)

Features: `hour, day_of_week, month, is_weekend, is_holiday, lag_1h, lag_24h, lag_168h, rolling_24h_mean, rolling_7d_mean, site_encoded`


<!-- PR-Agent Review Trigger -->
