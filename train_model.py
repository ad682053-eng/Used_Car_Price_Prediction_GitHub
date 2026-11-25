"""Train and evaluate the Used Car Price Prediction model.

Dataset expected at data/used_cars.csv. The source dataset uses USD prices and
mileage in miles. The source column is misspelled as `milage`; this script
intentionally supports that exact source name.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "used_cars.csv"
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "used_car_price_model.cbm"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

TARGET = "price"
NUMERIC_FEATURES = ["model_year", "milage"]
CATEGORICAL_FEATURES = [
    "brand", "model", "fuel_type", "engine", "transmission",
    "ext_col", "int_col", "accident", "clean_title",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


def parse_number(series: pd.Series) -> pd.Series:
    """Convert formatted prices/mileage (for example '$10,300', '51,000 mi.') to numbers."""
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def load_and_prepare_data(path: Path = DATA_PATH) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load CSV, validate its schema, clean units/missing values, and return X, y, raw."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nPlace the supplied CSV at data/used_cars.csv."
        )

    raw = pd.read_csv(path)
    # Accept the common spelling if a user later replaces the supplied file.
    if "milage" not in raw.columns and "mileage" in raw.columns:
        raw = raw.rename(columns={"mileage": "milage"})

    required = set(FEATURES + [TARGET])
    missing_columns = sorted(required.difference(raw.columns))
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing_columns)
            + "\nKeep the original column names or update the schema in train_model.py."
        )

    data = raw[FEATURES + [TARGET]].copy()
    data[TARGET] = parse_number(data[TARGET])
    data["milage"] = parse_number(data["milage"])
    data["model_year"] = pd.to_numeric(data["model_year"], errors="coerce")

    # A target row cannot be used for supervised learning when its price is absent/invalid.
    data = data.dropna(subset=[TARGET]).copy()
    data = data[data[TARGET] > 0].copy()

    # Preserve missing categorical values as an explicit category instead of dropping rows.
    for column in CATEGORICAL_FEATURES:
        data[column] = (
            data[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        )

    # Numeric missing values are filled with the training median in this prepared dataset.
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        median_value = data[column].median()
        data[column] = data[column].fillna(median_value)

    X = data[FEATURES].copy()
    y = data[TARGET].astype(float).copy()
    return X, y, raw


def make_price_bins(y: pd.Series) -> pd.Series:
    """Create log-price deciles so rare expensive records are represented in each split."""
    bins = pd.qcut(np.log1p(y), q=10, labels=False, duplicates="drop")
    if bins.nunique() < 2:
        raise ValueError("Not enough variation in price values to create stratified splits.")
    return bins


def evaluate_predictions(actual: pd.Series, predicted: np.ndarray) -> dict:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    absolute_percentage_error = (
        np.abs(actual_array - predicted_array) / np.maximum(np.abs(actual_array), 1.0)
    ) * 100.0
    return {
        "mae_usd": float(mean_absolute_error(actual_array, predicted_array)),
        "rmse_usd": float(np.sqrt(mean_squared_error(actual_array, predicted_array))),
        "r2": float(r2_score(actual_array, predicted_array)),
        "median_absolute_error_usd": float(median_absolute_error(actual_array, predicted_array)),
        "median_absolute_percentage_error": float(np.median(absolute_percentage_error)),
        "sample_count": int(len(actual_array)),
    }


def main() -> None:
    print("=" * 68)
    print("USED CAR PRICE PREDICTION - MODEL TRAINING")
    print("=" * 68)

    X, y, raw = load_and_prepare_data()
    bins = make_price_bins(y)

    # 70% training, 10% validation, 20% untouched final test set.
    X_train_val, X_test, y_train_val, y_test, bins_train_val, _ = train_test_split(
        X, y, bins, test_size=0.20, random_state=RANDOM_STATE, stratify=bins
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.125,
        random_state=43,
        stratify=bins_train_val,
    )

    print(f"Rows loaded: {len(raw):,}")
    print(f"Rows used for modelling: {len(X):,}")
    print(f"Train / validation / test: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
    print("Training CatBoostRegressor on log-transformed prices...")

    validation_model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.04,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=8,
        random_seed=RANDOM_STATE,
        random_strength=1,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    validation_model.fit(
        X_train,
        np.log1p(y_train),
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_val, np.log1p(y_val)),
        early_stopping_rounds=100,
        verbose=False,
    )

    test_prediction = np.expm1(validation_model.predict(X_test))
    test_prediction = np.maximum(test_prediction, 0.0)
    overall_metrics = evaluate_predictions(y_test, test_prediction)

    typical_mask = y_test <= 200_000
    typical_metrics = evaluate_predictions(
        y_test.loc[typical_mask], test_prediction[typical_mask.to_numpy()]
    ) if int(typical_mask.sum()) >= 2 else None

    # Retrain the deployable model on all supplied records using the selected iteration count.
    selected_iterations = max(int(validation_model.best_iteration_) + 1, 100)
    final_model = CatBoostRegressor(
        iterations=selected_iterations,
        depth=6,
        learning_rate=0.04,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=8,
        random_seed=RANDOM_STATE,
        random_strength=1,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    final_model.fit(X, np.log1p(y), cat_features=CATEGORICAL_FEATURES, verbose=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(MODEL_PATH), format="cbm")

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_percent": final_model.get_feature_importance(),
        }
    ).sort_values("importance_percent", ascending=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    prediction_frame = pd.DataFrame(
        {
            "actual_price_usd": y_test.to_numpy(),
            "predicted_price_usd": test_prediction,
            "absolute_error_usd": np.abs(y_test.to_numpy() - test_prediction),
        }
    )
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)

    missing_values = raw[FEATURES + [TARGET]].isna().sum().to_dict()
    mileage_numeric = parse_number(raw["milage"])
    target_numeric = parse_number(raw[TARGET])
    metrics = {
        "project_title": "Used Car Price Prediction",
        "dataset_file": "data/used_cars.csv",
        "dataset_currency": "USD",
        "mileage_unit": "miles",
        "source_rows": int(len(raw)),
        "rows_used": int(len(X)),
        "columns_in_source": int(len(raw.columns)),
        "duplicate_rows": int(raw.duplicated().sum()),
        "missing_values_by_column": {k: int(v) for k, v in missing_values.items()},
        "unique_brands": int(raw["brand"].nunique()),
        "unique_models": int(raw["model"].nunique()),
        "model_year_min": int(X["model_year"].min()),
        "model_year_max": int(X["model_year"].max()),
        "mileage_min": float(mileage_numeric.min()),
        "mileage_median": float(mileage_numeric.median()),
        "mileage_max": float(mileage_numeric.max()),
        "price_min_usd": float(target_numeric.min()),
        "price_median_usd": float(target_numeric.median()),
        "price_mean_usd": float(target_numeric.mean()),
        "price_max_usd": float(target_numeric.max()),
        "split": {"train": int(len(X_train)), "validation": int(len(X_val)), "test": int(len(X_test))},
        "model": "CatBoostRegressor",
        "target_transform": "log1p(price); inverse transform with expm1 for predictions",
        "best_validation_iteration": int(validation_model.best_iteration_),
        "final_model_iterations": int(selected_iterations),
        "overall_test_metrics": overall_metrics,
        "typical_price_segment_up_to_200k_usd": typical_metrics,
        "notes": [
            "R2 is a regression goodness-of-fit metric, not classification accuracy.",
            "The full test set includes a small number of very high-priced rare listings that strongly affect RMSE and R2.",
            "The typical-price segment is a separately reported subset where actual test price is <= USD 200,000.",
            "Predictions are estimates from this supplied US-market dataset, not guaranteed selling prices.",
        ],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nHELD-OUT TEST RESULTS (20% test set)")
    for key, value in overall_metrics.items():
        print(f"{key}: {value:,.4f}" if isinstance(value, (float, int)) else f"{key}: {value}")
    if typical_metrics:
        print("\nTEST SUBSET: actual prices <= $200,000")
        for key, value in typical_metrics.items():
            print(f"{key}: {value:,.4f}" if isinstance(value, (float, int)) else f"{key}: {value}")
    print(f"\nSaved trained model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved metrics: {METRICS_PATH.relative_to(ROOT)}")
    print("Training completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
