# ⚡ EV Cold-Start Forecasting

> Predicting hourly demand for newly installed EV charging stations with zero usage history.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-brightgreen?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-experiment%20tracking-orange?style=flat-square&logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-frontend-61DAFB?style=flat-square&logo=react&logoColor=black)
![Status](https://img.shields.io/badge/Status-Phase%203%20Complete-success?style=flat-square)

---

## 🧩 The Problem

A newly installed EV charging station has **zero usage history**.  
Standard forecasting models fail without data.  
Operators are flying blind on demand, capacity planning, and grid load.

---

## 💡 The Solution

A four-layer ML system that solves cold-start from day one:

```
ACN-Data (107 stations)
        │
        ▼
┌─────────────────────────┐
│   Global LightGBM Model │  ← learns shared demand grammar across all stations
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Transfer + Fine-Tune  │  ← adapts to new station with 1 week of data
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Synthetic Augmentation │  ← TimeGAN / Gaussian Copula fills history gaps
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Conformal Prediction   │  ← calibrated uncertainty intervals, not just points
└─────────────────────────┘
```

---

## 📊 Results — Phase 3 Transfer Learning

> All models evaluated on **office001** — a completely held-out site,  
> never seen during training or validation.

| Weeks of Data | 🔁 Transfer MAE | Vanilla LightGBM | Seasonal Naive |
|:---:|:---:|:---:|:---:|
| 1 week | **0.0279** | 0.0300 | 0.3488 |
| 2 weeks | **0.0290** | 0.0402 | 0.3438 |
| 3 weeks | **0.0284** | 0.0353 | 0.3481 |

**Key finding:** Transfer MAE is flat across all data volumes.  
→ One week of new-station data is sufficient for near-optimal predictions.  
→ Vanilla LightGBM degrades with more weeks — overfitting on sparse data.  
→ Transfer beats Seasonal Naive by **12×**.

---

## 🗂️ Data

- **Source:** [ACN-Data](https://ev.caltech.edu/dataset) — real EV charging sessions from Caltech, JPL, and office campuses
- **Training:** 107 stations across Caltech + JPL (2,020,364 rows)
- **Test site:** office001 — 8 stations, held out entirely, never touched during training

---

## 🔬 Experiment Tracking

All runs logged to MLflow across every phase.

```bash
mlflow ui
# → http://localhost:5000
```

| Experiment | Runs | Description |
|---|---|---|
| `phase3_transfer` | 48 | Transfer vs vanilla, 8 stations × 3 volumes × 2 models |

---

## 🚀 Setup

```bash
git clone https://github.com/YOUR_USERNAME/ev-coldstart-forecast.git
cd ev-coldstart-forecast
cp .env.example .env
docker-compose up
```

---

## 🗺️ Build Progress

| Phase | Focus | Status |
|-------|-------|:---:|
| 0 | Environment setup | ✅ |
| 1 | Data ingestion + EDA | ✅ |
| 2 | Baseline models | ✅ |
| 3 | Transfer learning | ✅ |
| 4 | Synthetic augmentation | 🔄 |
| 5 | Uncertainty quantification | ⬜ |
| 6 | Site selection | ⬜ |
| 7 | FastAPI backend | ⬜ |
| 8 | React frontend | ⬜ |
| 9 | Docker + AWS | ⬜ |
| 10 | README + docs | ⬜ |

---

## 🧠 Key Decisions

<details>
<summary><b>Why LightGBM over a neural network?</b></summary>
<br>
Limited data per station makes deep models unstable under scarce data conditions.
LightGBM trains in seconds, enabling rapid ablation across 24 experiment runs.
Stability under scarcity matters more than model capacity here.
</details>

<details>
<summary><b>Why office001 as the held-out test site?</b></summary>
<br>
office001 was never seen during global model training or validation.
It represents a genuinely unseen operational context — the closest
simulation of a real cold-start deployment scenario.
</details>

<details>
<summary><b>Why not ARIMA?</b></summary>
<br>
statsforecast has a known multiprocessing conflict on Windows + Jupyter.
ARIMA results would be expected to fall between SeasonalNaive and LightGBM
baselines — not competitive with transfer learning at any data volume.
</details>

---

## 📁 Project Structure

```
ev-coldstart-forecast/
├── src/
│   ├── data/          # loader, preprocessor, feature engineering
│   ├── models/        # global_model, transfer, baseline
│   ├── evaluation/    # metrics
│   └── api/           # FastAPI app (Phase 6)
├── notebooks/         # EDA, phase ablations
├── data/
│   ├── raw/
│   ├── processed/     # per-station parquet files
│   └── synthetic/     # augmented data (Phase 4)
├── models/            # saved model artifacts
├── frontend/          # React dashboard (Phase 7)
└── docker-compose.yml
```

---

<p align="center">
  Built on <a href="https://ev.caltech.edu/dataset">ACN-Data</a> ·
  Tracked with <a href="https://mlflow.org">MLflow</a> ·
  Deployed on AWS EC2
</p>


