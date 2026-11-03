
# ==========================================================
# REAL ESTATE INVESTMENT ADVISOR
# REGRESSION MODEL TRAINING
# ==========================================================

import pandas as pd
import numpy as np
import joblib
import json

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "india_housing_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

RESULT_DIR = (
    PROJECT_ROOT / "reports" / "model_results"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

ANNUAL_APPRECIATION_RATE = 0.08
FORECAST_YEARS = 5


# ==========================================================
# 2. DEFINE INPUT FEATURES AND TARGET
# ==========================================================

NUMERIC_FEATURES = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Property_Age_2026",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Amenities_Count",
    "Has_Gym",
    "Has_Pool",
    "Has_Garden",
    "Has_Clubhouse",
    "Has_Playground",
    "Has_Parking",
    "Has_Security",
    "Is_Ready_to_Move",
    "Transport_Accessibility_Score",
    "Floor_Info_Valid",
    "Floor_Ratio"
]

CATEGORICAL_FEATURES = [
    "State",
    "City",
    "Property_Type",
    "Furnished_Status",
    "Facing",
    "Owner_Type"
]

FEATURES = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)

TARGET = "Future_Price_5Y"


# ==========================================================
# 3. LOAD DATASET
# ==========================================================

def load_data():

    print("\nLoading feature-engineered dataset...", flush=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!", flush=True)

    print("Dataset Shape:", df.shape, flush=True)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Validate numerical target
    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    df = df.dropna(
        subset=[TARGET]
    ).copy()

    print("Final dataset:", df.shape, flush=True)

    return df


# ==========================================================
# 4. BUILD PREPROCESSING PIPELINE
# ==========================================================

def build_preprocessor():

    # Numerical preprocessing
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


# ==========================================================
# 5. DEFINE REGRESSION MODELS
# ==========================================================

def get_models():

    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=40,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.7,
            n_jobs=4,
            random_state=RANDOM_STATE
        ),

        "XGBoost Regressor": XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=4,
            random_state=RANDOM_STATE,
            verbosity=0
        )
    }

    return models


# ==========================================================
# 6. TRAINING SAMPLE CONFIGURATION
# ==========================================================

def get_training_data(
    name,
    X_train,
    y_train
):

    # Linear Regression: use full training data
    if name == "Linear Regression":

        return X_train, y_train

    # Random Forest: use 40,000 records
    elif name == "Random Forest Regressor":

        sample_size = min(
            40000,
            len(X_train)
        )

    # XGBoost: use 80,000 records
    elif name == "XGBoost Regressor":

        sample_size = min(
            80000,
            len(X_train)
        )

    else:

        return X_train, y_train

    # Deterministic random sampling
    sampled_X = X_train.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    sampled_y = y_train.loc[
        sampled_X.index
    ]

    return sampled_X, sampled_y


# ==========================================================
# 7. EVALUATE MODEL
# ==========================================================

def evaluate_model(
    y_test,
    y_pred
):

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    return mae, rmse, r2


# ==========================================================
# 8. SAVE ACTUAL VS PREDICTED GRAPH
# ==========================================================

def save_prediction_plot(
    name,
    y_test,
    y_pred,
    safe_name
):

    print(
        f"Generating prediction graph for {name}...",
        flush=True
    )

    # Sample points for visualization only
    plot_df = pd.DataFrame({
        "Actual": np.asarray(y_test),
        "Predicted": np.asarray(y_pred)
    })

    plot_df = plot_df.sample(
        n=min(10000, len(plot_df)),
        random_state=RANDOM_STATE
    )

    plt.figure(figsize=(9, 6))

    plt.scatter(
        plot_df["Actual"],
        plot_df["Predicted"],
        alpha=0.35,
        s=12
    )

    min_value = min(
        plot_df["Actual"].min(),
        plot_df["Predicted"].min()
    )

    max_value = max(
        plot_df["Actual"].max(),
        plot_df["Predicted"].max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Future Price (Lakhs)")
    plt.ylabel("Predicted Future Price (Lakhs)")

    plt.title(
        f"{name} - Actual vs Predicted"
    )

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / f"{safe_name}_prediction.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Graph saved successfully!", flush=True)


# ==========================================================
# 9. TRAIN AND COMPARE MODELS
# ==========================================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    models = get_models()

    results = []

    best_rmse = float("inf")

    best_model_name = None

    best_pipeline = None

    for name, regressor in models.items():

        print("\n====================================", flush=True)

        print(f"Starting training: {name}", flush=True)

        print("====================================", flush=True)

        # Select training sample
        model_X_train, model_y_train = get_training_data(
            name,
            X_train,
            y_train
        )

        print(
            f"Training records: {len(model_X_train)}",
            flush=True
        )

        # Build independent preprocessing pipeline
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor()
                ),
                (
                    "regressor",
                    regressor
                )
            ]
        )

        # Train model
        print(
            f"Fitting {name}...",
            flush=True
        )

        pipeline.fit(
            model_X_train,
            model_y_train
        )

        print(
            f"{name} training completed!",
            flush=True
        )

        # Generate predictions
        print(
            "Generating predictions...",
            flush=True
        )

        y_pred = pipeline.predict(X_test)

        # Evaluate model
        mae, rmse, r2 = evaluate_model(
            y_test,
            y_pred
        )

        print("\nModel:", name)

        print(f"MAE: {mae:.6f} Lakhs")

        print(f"RMSE: {rmse:.6f} Lakhs")

        print(f"R2 Score: {r2:.6f}")

        # Safe filename
        safe_name = (
            name.lower()
            .replace(" ", "_")
        )

        # Save graph
        save_prediction_plot(
            name,
            y_test,
            y_pred,
            safe_name
        )

        # Save individual model
        model_path = (
            MODEL_DIR / f"{safe_name}.joblib"
        )

        joblib.dump(
            pipeline,
            model_path
        )

        print(
            f"Model saved: {model_path.name}",
            flush=True
        )

        # Store metrics
        results.append({

            "Model": name,

            "Training_Records": len(model_X_train),

            "MAE": mae,

            "RMSE": rmse,

            "R2_Score": r2

        })

        # Select best model using RMSE
        if rmse < best_rmse:

            best_rmse = rmse

            best_model_name = name

            best_pipeline = pipeline

    return (
        results,
        best_model_name,
        best_pipeline
    )


# ==========================================================
# 10. SAVE MODEL COMPARISON AND BEST MODEL
# ==========================================================

def save_results(
    results,
    best_model_name,
    best_pipeline
):

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="RMSE",
        ascending=True
    )

    # Save metrics
    results_path = (
        RESULT_DIR / "regression_metrics.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # Save best model
    best_model_path = (
        MODEL_DIR / "regression_best_model.joblib"
    )

    joblib.dump(
        best_pipeline,
        best_model_path
    )

    # Save metadata
    metadata = {

        "best_model": best_model_name,

        "selection_metric": "RMSE",

        "best_rmse": float(
            results_df.iloc[0]["RMSE"]
        ),

        "target": TARGET,

        "features": FEATURES,

        "forecast_years": FORECAST_YEARS,

        "annual_appreciation_rate": (
            ANNUAL_APPRECIATION_RATE
        ),

        "target_type": "Assumption-based projection",

        "training_strategy": {
            "Linear Regression": "Full training data",
            "Random Forest Regressor": "Maximum 40000 sampled records",
            "XGBoost Regressor": "Maximum 80000 sampled records"
        }

    }

    with open(
        MODEL_DIR / "regression_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\n====================================")

    print("REGRESSION MODEL COMPARISON")

    print("====================================")

    print(
        results_df.to_string(index=False)
    )

    print("\nBest Model:", best_model_name)

    print(
        "Best RMSE:",
        round(
            float(results_df.iloc[0]["RMSE"]),
            6
        )
    )

    print("\nAll regression models saved successfully!")

    print("Metrics saved:", results_path)

    print("Best model saved:", best_model_path)


# ==========================================================
# 11. MAIN EXECUTION
# ==========================================================

def main():

    print("\n====================================")

    print("REAL ESTATE REGRESSION TRAINING")

    print("====================================")

    # Load data
    df = load_data()

    # Split dataset
    print("\nSplitting dataset...", flush=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    print(
        "Training records:",
        len(train_df),
        flush=True
    )

    print(
        "Testing records:",
        len(test_df),
        flush=True
    )

    # Select features
    X_train = train_df[FEATURES].copy()

    X_test = test_df[FEATURES].copy()

    # Select target
    y_train = train_df[TARGET].copy()

    y_test = test_df[TARGET].copy()

    print("\nTarget:", TARGET)

    print(
        "Training target average:",
        round(y_train.mean(), 4)
    )

    print(
        "Testing target average:",
        round(y_test.mean(), 4)
    )

    # Train models
    results, best_model_name, best_pipeline = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Save results
    save_results(
        results,
        best_model_name,
        best_pipeline
    )

    print("\n====================================")

    print("REGRESSION TRAINING COMPLETED!")

    print("====================================")

    print("\nModels directory:", MODEL_DIR)

    print("\nResults directory:", RESULT_DIR)


# ==========================================================
# 12. EXECUTION ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()
