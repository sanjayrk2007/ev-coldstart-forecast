# EV Cold-Start Forecasting
> Predicting hourly demand for newly installed EV charging stations with zero usage history.

**Status:** Phase 3 complete — global model + transfer learning. Synthetic augmentation next.

---

## Problem Statement
Newly installed EV charging stations have zero usage history. Standard forecasting
models fail without data. This system solves the cold-start problem by borrowing
knowledge from 107 existing stations, adapting it to a new station with as little
as one week of local data, and wrapping predictions in calibrated uncertainty intervals.

---

## Architecture (coming soon in Phase 9)

---

## Results So Far

### Phase 3 — Transfer Learning Ablation (evaluated on held-out office001 site)

| Weeks of Data | Transfer MAE | Vanilla LightGBM | Seasonal Naive |
|---------------|-------------|-----------------|----------------|
| 1 week        | **0.0279**  | 0.0300          | 0.3488         |
| 2 weeks       | **0.0290**  | 0.0402          | 0.3438         |
| 3 weeks       | **0.0284**  | 0.0353          | 0.3481         |

Key finding: transfer learning is stable at ~0.028 MAE regardless of fine-tuning
data volume. One week of new-station data is sufficient for near-optimal predictions.
Vanilla LightGBM degrades with more weeks, suggesting overfitting on sparse data.

---

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/ev-coldstart-forecast.git
cd ev-coldstart-forecast
cp .env.example .env
# fill in .env values
docker-compose up
```

---

## Phases

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Environment setup | ✅ Done |
| 1 | Data ingestion + EDA | ✅ Done |
| 2 | Baseline models | ✅ Done |
| 3 | Transfer learning | ✅ Done |
| 4 | Synthetic augmentation | 🔄 In Progress |
| 5 | Uncertainty quantification | — |
| 6 | Site selection | — |
| 7 | FastAPI backend | — |
| 8 | React frontend | — |
| 9 | Docker + AWS | — |
| 10 | README + docs | — |

---

## Key Decisions

**Why LightGBM over a neural network?**
Limited data per station makes deep models unstable. LightGBM is more robust
under scarce data conditions and trains in seconds, enabling rapid ablation.

**Why office001 as the held-out test site?**
office001 was never seen during global model training or validation. It represents
a genuinely unseen operational context — the closest simulation of a real cold-start
deployment.

**Why not ARIMA?**
statsforecast has a known multiprocessing conflict on Windows + Jupyter. pmdarima
is a compatible alternative but was excluded to maintain phase momentum. ARIMA
results would be expected to fall between SeasonalNaive and LightGBM baselines.

---

## Experiment Tracking
All runs logged to MLflow. Start the UI with:
```bash
mlflow ui
# opens at http://localhost:5000
```


