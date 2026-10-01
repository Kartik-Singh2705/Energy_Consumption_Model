# Energy Forecasting Studio — User Forecast Dashboard

This project implements a browser-based energy forecasting workflow using the supplied synthetic datasets and the FRD.

## Main user workflow

1. Select Region A, Building B, or Home C.
2. Select a forecast horizon: 1 hour, 24 hours, or 7 days.
3. Click **Generate Forecast**.
4. The dashboard shows the next forecast point in kWh.
5. Demand is classified as **LOW / MEDIUM / HIGH** using dataset-relative thresholds (40th and 80th percentiles).
6. The dashboard shows the forecast across the selected horizon.
7. A driver panel explains which model features are elevated/reduced relative to recent history.

## Important synthetic-data limitation

The supplied CSVs end in 2023. Therefore this demo does not claim to forecast real-world 2026 consumption. The user-facing forecast starts immediately after the latest timestamp in the selected CSV. Future exogenous values are approximated from the same hour one week earlier. For production use, connect a current hourly consumption feed and prediction-time weather/holiday data.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

## Technical workflow

The existing pipeline performs data validation/cleaning, lag and rolling feature creation, XGBoost feature scoring, test metrics, forecasts, and technical explainability. LSTM, GRU and CNN-LSTM training modules are included for development experiments.

The supplied FRD states that this is a batch data and machine-learning pipeline, not a live grid-control system, and that synthetic data is for pipeline development rather than final reported results.
