import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def main() -> None:
    data_path = Path("data/clean/fitbit_clean.csv")
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "sleep_rf_model.pkl"

    df = pd.read_csv(data_path)
    features = [
        "HOURS_OF_SLEEP_HOURS",
        "REM_SLEEP",
        "DEEP_SLEEP",
        "HEART_RATE_UNDER_RESTING",
    ]
    target = "SLEEP_SCORE"

    df = df.dropna(subset=features + [target])
    if df.empty:
        raise RuntimeError("No rows left after dropping NaNs for features/target.")

    X = df[features]
    y = df[target]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    joblib.dump(model, model_path)
    print(f"Saved model to {os.fspath(model_path)}")


if __name__ == "__main__":
    main()
