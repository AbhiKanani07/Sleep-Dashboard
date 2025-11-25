# Sleep Insights Dashboard
Unified Fitbit + Apple Health sleep analytics with ML and explainability.

## Quickstart
1) python -m venv .venv && .\.venv\Scripts\Activate.ps1 or cmd .\.venv\Scripts\Activate.ps1
2) pip install -r requirements.txt
3) Optional: python train_model.py  # creates models/sleep_rf_model.pkl
4) streamlit run dashboard.py
Env vars (for hosted data): FITBIT_CSV_URL, APPLE_CSV_URL

## Features
- Multi-page Streamlit app: Insights Summary, Predictor (what-if), Fitbit/Apple insights, No-overlap comparison, Weekly patterns, Regularity & chronotype, Recommendations, Correlations, SHAP explainability.
- Filters: date range, weekdays, daily/weekly/monthly grouping.
- Exports: CSV download buttons.
- Plotly interactivity: lines, histograms, heatmaps; SHAP bar/beeswarm if installed.

## Deployment
- Local: see Quickstart.
- Hugging Face Spaces: streamlit SDK, app file `dashboard.py`, push `requirements.txt`, data/model; set FITBIT_CSV_URL/APPLE_CSV_URL. (Details in DEPLOY_HF.md.)

## Data & Model
- Expects cleaned CSVs under `data/clean/` or provided via URLs.
- Predictor loads `models/sleep_rf_model.pkl` if present.



