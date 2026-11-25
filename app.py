"""Streamlit interface for the Used Car Price Prediction project."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "used_cars.csv"
MODEL_PATH = ROOT / "model" / "used_car_price_model.cbm"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
IMPORTANCE_PATH = ROOT / "outputs" / "feature_importance.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "test_predictions.csv"

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


def read_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "milage" not in df.columns and "mileage" in df.columns:
        df = df.rename(columns={"mileage": "milage"})
    for col in ["brand", "model", "fuel_type", "engine", "transmission", "ext_col", "int_col", "accident", "clean_title"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = "Unknown"
    return df


@st.cache_resource
def load_model() -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH), format="cbm")
    return model


@st.cache_data
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def money(value: float) -> str:
    return f"${value:,.0f}"


st.title("🚘 Used Car Price Prediction")
st.caption("Machine Learning Project | CatBoost Regression | Dataset prices in USD and mileage in miles")

if not DATA_PATH.exists():
    st.error("Dataset not found. Keep the supplied CSV at data/used_cars.csv.")
    st.stop()
if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train_model.py` once, then refresh this page.")
    st.stop()

try:
    dataset = read_dataset()
    model = load_model()
    metrics = load_metrics()
except Exception as exc:
    st.error(f"Could not load project files: {exc}")
    st.stop()

predict_tab, data_tab, performance_tab, about_tab = st.tabs(
    ["Price Predictor", "Dataset Overview", "Model Performance", "About Project"]
)

with predict_tab:
    st.subheader("Enter vehicle details")
    st.info("This project was trained on a US-market dataset. Enter mileage in miles; the predicted price is shown in US dollars (USD).")

    all_brands = sorted(dataset["brand"].dropna().unique().tolist())
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_brand = st.selectbox("Brand", all_brands, index=all_brands.index("Toyota") if "Toyota" in all_brands else 0)
        models_for_brand = sorted(dataset.loc[dataset["brand"] == selected_brand, "model"].dropna().unique().tolist())
        if not models_for_brand:
            models_for_brand = ["Unknown"]
        selected_model = st.selectbox("Model", models_for_brand)
        year_min = int(pd.to_numeric(dataset["model_year"], errors="coerce").min())
        year_max = int(pd.to_numeric(dataset["model_year"], errors="coerce").max())
        model_year = st.number_input("Model year", min_value=year_min, max_value=year_max, value=min(2020, year_max), step=1)
    with c2:
        mileage = st.number_input("Mileage (miles)", min_value=0, max_value=1_000_000, value=50_000, step=1_000)
        fuel_values = sorted(set(dataset["fuel_type"].tolist()) | {"Unknown"})
        fuel_type = st.selectbox("Fuel type", fuel_values, index=fuel_values.index("Gasoline") if "Gasoline" in fuel_values else 0)
        transmission_values = sorted(set(dataset["transmission"].tolist()) | {"Unknown"})
        transmission = st.selectbox("Transmission", transmission_values)
    with c3:
        engine_defaults = dataset["engine"].value_counts().index.tolist()
        engine_default = engine_defaults[0] if engine_defaults else "Unknown"
        engine = st.text_input("Engine description", value=engine_default, help="Use the format shown in the dataset when possible.")
        exterior_values = sorted(set(dataset["ext_col"].tolist()) | {"Unknown"})
        exterior_default = "Black" if "Black" in exterior_values else exterior_values[0]
        ext_col = st.selectbox("Exterior colour", exterior_values, index=exterior_values.index(exterior_default))
        interior_values = sorted(set(dataset["int_col"].tolist()) | {"Unknown"})
        interior_default = "Black" if "Black" in interior_values else interior_values[0]
        int_col = st.selectbox("Interior colour", interior_values, index=interior_values.index(interior_default))

    c4, c5 = st.columns(2)
    with c4:
        accident_values = ["None reported", "At least 1 accident or damage reported", "Unknown"]
        accident = st.selectbox("Accident / damage history", accident_values)
    with c5:
        clean_title = st.selectbox("Clean title", ["Yes", "Unknown"])

    if st.button("Predict Used Car Price", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "model_year": float(model_year),
            "milage": float(mileage),
            "brand": str(selected_brand),
            "model": str(selected_model),
            "fuel_type": str(fuel_type),
            "engine": str(engine).strip() or "Unknown",
            "transmission": str(transmission),
            "ext_col": str(ext_col),
            "int_col": str(int_col),
            "accident": str(accident),
            "clean_title": str(clean_title),
        }])
        log_prediction = float(model.predict(row)[0])
        predicted_price = max(float(np.expm1(log_prediction)), 0.0)
        st.success("Prediction generated successfully")
        st.metric("Estimated used-car price", money(predicted_price))
        st.caption("This is a model estimate, not a dealer quote, inspection report, or guaranteed market price.")

with data_tab:
    st.subheader("Dataset summary")
    price_num = pd.to_numeric(dataset["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    mileage_num = pd.to_numeric(dataset["milage"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    cols = st.columns(4)
    cols[0].metric("Records", f"{len(dataset):,}")
    cols[1].metric("Brands", f"{dataset['brand'].nunique():,}")
    cols[2].metric("Car models", f"{dataset['model'].nunique():,}")
    cols[3].metric("Median listed price", money(float(price_num.median())))

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Price distribution (USD)**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(price_num.dropna(), bins=45, log=True)
        ax.set_xlabel("Price (USD)")
        ax.set_ylabel("Number of records (log scale)")
        ax.set_title("Used-car listing prices")
        st.pyplot(fig, clear_figure=True)
    with c2:
        st.write("**Missing values in source data**")
        missing_source = metrics.get("missing_values_by_column", {})
        missing = pd.Series(missing_source, dtype="int64")
        missing = missing[missing > 0].sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        if len(missing):
            missing.plot(kind="barh", ax=ax)
        else:
            ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.set_xlabel("Missing row count")
        ax.set_title("Missing values before cleaning")
        st.pyplot(fig, clear_figure=True)

    st.write("**Sample records**")
    st.dataframe(dataset.head(20), use_container_width=True, hide_index=True)
    st.caption(f"Mileage range in source file: {mileage_num.min():,.0f} to {mileage_num.max():,.0f} miles.")

with performance_tab:
    st.subheader("Held-out model evaluation")
    overall = metrics.get("overall_test_metrics", {})
    typical = metrics.get("typical_price_segment_up_to_200k_usd", {})
    if overall:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall test MAE", money(overall.get("mae_usd", 0)))
        m2.metric("Overall test RMSE", money(overall.get("rmse_usd", 0)))
        m3.metric("Overall test R²", f"{overall.get('r2', 0):.3f}")
        m4.metric("Overall MdAPE", f"{overall.get('median_absolute_percentage_error', 0):.1f}%")
        st.caption("The test set is held out from model fitting. R² is a regression metric, not classification accuracy.")
        st.markdown("**Typical-price subset (actual test price ≤ $200,000)**")
        if typical:
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Subset records", f"{typical.get('sample_count', 0):,}")
            t2.metric("Subset MAE", money(typical.get("mae_usd", 0)))
            t3.metric("Subset RMSE", money(typical.get("rmse_usd", 0)))
            t4.metric("Subset R²", f"{typical.get('r2', 0):.3f}")
        st.warning("A few very high-priced, rare cars have a large effect on overall RMSE and R². Read the overall and typical-price results together.")
    else:
        st.warning("Metrics are not available yet. Run train_model.py.")

    if IMPORTANCE_PATH.exists():
        importance = pd.read_csv(IMPORTANCE_PATH).sort_values("importance_percent", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance["feature"], importance["importance_percent"])
        ax.set_xlabel("Feature importance (%)")
        ax.set_title("CatBoost feature importance")
        st.pyplot(fig, clear_figure=True)
        st.dataframe(importance.sort_values("importance_percent", ascending=False), use_container_width=True, hide_index=True)

    if PREDICTIONS_PATH.exists():
        predictions = pd.read_csv(PREDICTIONS_PATH)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(predictions["actual_price_usd"], predictions["predicted_price_usd"], alpha=0.55)
        max_value = max(predictions["actual_price_usd"].max(), predictions["predicted_price_usd"].max())
        ax.plot([0, max_value], [0, max_value], linestyle="--")
        ax.set_xscale("symlog")
        ax.set_yscale("symlog")
        ax.set_xlabel("Actual price (USD)")
        ax.set_ylabel("Predicted price (USD)")
        ax.set_title("Actual vs. predicted prices (held-out test set)")
        st.pyplot(fig, clear_figure=True)

with about_tab:
    st.subheader("Project information")
    st.markdown(
        """
        **Objective:** estimate the listing price of a used vehicle from its attributes.

        **Algorithm:** CatBoostRegressor, with categorical inputs handled natively and the target transformed using `log1p` during training.

        **Input features:** brand, model, model year, mileage, fuel type, engine, transmission, exterior colour, interior colour, accident history, and clean-title status.

        **Evaluation:** a stratified 70% training / 10% validation / 20% test split. The final saved model is retrained on all supplied rows after the best iteration is selected on validation data.

        **Important limitation:** the supplied dataset is a US-market snapshot with USD prices and miles. It should not be interpreted as a direct Indian used-car price estimator. More geographically relevant Indian listings are needed for INR/km predictions.
        """
    )
    st.write("Dataset source file: `data/used_cars.csv`")
    st.write("Training script: `train_model.py`")
