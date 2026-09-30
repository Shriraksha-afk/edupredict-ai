
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data" / "student_performance.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "student_model.joblib"

# Features used to predict final score
FEATURES = [
    "attendance_percent",
    "study_hours_per_day",
    "assignment_score",
    "internal_exam1",
    "internal_exam2",
    "class_participation",
    "sleep_hours"
]

TARGET = "final_score"


def evaluate_model(name, y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    rmse = mean_squared_error(y_true, predictions) ** 0.5
    r2 = r2_score(y_true, predictions)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def main():
    # Check that the dataset exists
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    df = pd.read_csv(DATA)

    # Check required columns
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required CSV columns: {missing}"
        )

    # Convert required fields to numeric values
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rows without a target cannot be used for training/testing
    df = df.dropna(subset=[TARGET]).copy()

    if len(df) < 10:
        raise ValueError(
            "Not enough valid rows to train and evaluate the model."
        )

    X = df[FEATURES]
    y = df[TARGET]

    # Keep a separate test set for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Baseline: always predict the training-set mean
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)
    baseline_metrics = evaluate_model(
        "Baseline (training mean)",
        y_test,
        baseline_predictions
    )

    # Random Forest regression pipeline
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    # Train only on training data
    model.fit(X_train, y_train)

    # Evaluate on unseen test data
    predictions = model.predict(X_test)
    model_metrics = evaluate_model(
        "Random Forest",
        y_test,
        predictions
    )

    # Compare model with baseline
    print("\nComparison")
    print("----------")
    if model_metrics["mae"] < baseline_metrics["mae"]:
        print("Random Forest has lower test MAE than baseline.")
    else:
        print("Random Forest did not improve test MAE over baseline.")

    # Save trained pipeline
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()