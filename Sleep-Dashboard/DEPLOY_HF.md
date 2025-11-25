# Deploy to Hugging Face Spaces (Streamlit)

1) Create a new public Space at https://huggingface.co/spaces and choose **SDK: Streamlit**.
2) In the Space settings, set **App file** to `dashboard.py` and **Python version** to 3.12 (or compatible with your local venv).
3) Push these files to the Space repository:
   - `dashboard.py`
   - `requirements.txt`
   - `train_model.py` (optional, for retraining)
   - `data/clean/fitbit_clean.csv` and `data/clean/apple_sleep_nightly_summary.csv` (or point to your own storage)
   - `models/sleep_rf_model.pkl` (optional; without it, the predictor will show a warning)
4) The Space will build automatically and launch the Streamlit app.

If you prefer the CLI route:
```bash
pip install huggingface_hub
huggingface-cli login  # paste your token
huggingface-cli repo create Sleep-Dashboard --type=space --sdk=streamlit --public
git clone https://huggingface.co/spaces/<your-username>/Sleep-Dashboard
cd Sleep-Dashboard
# copy your project files here, then:
git add .
git commit -m "Add sleep dashboard"
git push
```

Notes:
- Keep CSV/model sizes within the free Space storage limits (or host data externally and load via URL).
- The app file name must match what you set in Space settings (`dashboard.py`).
