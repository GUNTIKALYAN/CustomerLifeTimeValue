# Customer Lifetime Value (CLTV) Prediction — End-to-End ML

Predict how much revenue each insurance customer will generate over their lifetime, and serve that prediction through a production-grade API.

## Overview

This project takes insurance customer records (demographics, policy history, claims) and predicts **CLTV** (Customer Lifetime Value) — the total money a customer brings the company over their relationship. The trained pipeline is served via a **FastAPI** endpoint, containerized with **Docker**, and released through an automated **CI/CD** pipeline.

```
RawData/ → feature pipeline → LightGBM → model.pkl → FastAPI → Docker → CI/CD → Power BI
```

## Why this matters

The predicted CLTV tells the business **how much it is worth spending** to acquire/retain a customer:

| Decision | Rule |
|---|---|
| Acquisition | Spend up to the predicted CLTV to acquire a customer |
| Retention | Prioritize high-CLTV customers at renewal |
| Segmentation | Premium vs low-value customers get different treatment |
| Valuation | Sum of all CLTV = value of the customer book |

## Project Structure

```
.
├── app.py                  # FastAPI inference endpoint
├── Dockerfile              # Container image (python:3.10-slim + libgomp1)
├── requirements.txt        # Python dependencies
├── model.pkl               # Trained pipeline artifact (joblib)
├── lightgbm_model.pkl      # Raw LightGBM model from notebook
├── notebook.ipynb          # Full EDA + model experimentation
├── submission.csv          # Generated test predictions
├── powerbi_cltv_data.csv   # Power BI-ready predictions dataset
├── src/
│   ├── data_loader.py      # Loads train/test CSVs
│   ├── pipeline.py         # Custom sklearn transformers (FE + one-hot)
│   ├── model.py            # LightGBM model config (huber objective)
│   ├── train.py            # Training: outlier clip → log1p → pipeline → save
│   ├── predict.py          # Batch inference → submission.csv
│   └── export_powerbi.py   # Predictions + segments for Power BI
├── powerbi/
│   ├── power_query_cltv.m  # Power Query (M) import/clean code
│   ├── dax_measures.txt    # DAX measures for the dashboard
│   └── README.md           # Power BI build guide
├── RawData/                # train.csv / test.csv
└── .github/workflows/ci.yml # CI/CD: test container + deploy to Docker Hub
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the model
```bash
python -m src.train
```
- Clips `cltv` at the 99th percentile, applies `log1p` target transform
- Full sklearn Pipeline: `Preprocessor → FeatureEngineer → OneHotEncoder → LightGBM`
- Early stopping (100 rounds), saves `model.pkl`

### 2. Generate test predictions
```bash
python -m src.predict
```
Writes `submission.csv` (id + cltv).

### 3. Serve the API
```bash
uvicorn app:app --reload
```
- `GET /` → health check
- `POST /predict` → prediction from JSON input
- Interactive docs at `http://localhost:8000/docs`

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "gender": "Male",
    "area": "Urban",
    "qualification": "Bachelor",
    "income": "5L-10L",
    "policy": "A",
    "type_of_policy": "Platinum",
    "num_policies": "More than 1",
    "vintage": 5,
    "claim_amount": 5000,
    "marital_status": 1
  }'
```

### 4. Docker
```bash
docker build -t cltv-api .
docker run -d -p 8000:8000 cltv-api
```

### 5. Power BI dashboard
```bash
python -m src.export_powerbi
```
See `powerbi/README.md` for import + DAX + visuals.

## Modeling Approach

| Step | Technique | Why |
|---|---|---|
| Outlier handling | Clip target at 99th percentile | Extreme CLTV values would dominate training |
| Target transform | `log1p` (inverted with `expm1`) | Fixes right-skew, makes target near-normal |
| Feature engineering | Ratios/interactions (`claim_to_income`, `avg_policy_age`, `claim_per_vintage`, `high_value_customer`, ...) | Gives the model structure it can split on |
| Encoding | One-hot inside pipeline, `reindex(..., fill_value=0)` | Aligns train/test columns, unseen categories → 0 |
| Model | LightGBM (huber objective) vs XGBoost, 5-fold CV | Selected the better generalizer on validation |
| Validation | Holdout + early stopping | Prevents overfitting, picks optimal tree count |
| Metric | MAE (rupees) | Business-interpretable; RMSE over-penalizes rare huge errors |
| Serving | Save whole pipeline via joblib | Inference needs no training code or data |

## CI/CD

`.github/workflows/ci.yml` runs on every push to `main` / PR:
1. Install dependencies
2. Verify `model.pkl` exists
3. Build Docker image
4. Run container, `curl` `/` and `/predict` endpoints
5. On `main`: tag + push to Docker Hub (latest + commit SHA)

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, FastAPI, Pydantic, Uvicorn, Docker, GitHub Actions, Power BI