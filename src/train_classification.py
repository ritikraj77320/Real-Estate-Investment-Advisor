
# ==========================================================
# REAL ESTATE INVESTMENT ADVISOR
# CLASSIFICATION MODEL TRAINING
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from xgboost import XGBClassifier


# ==========================================================
# 1. PROJECT PATHS
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

BENCHMARK_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "investment_benchmarks_train.csv"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ==========================================================
# 2. DEFINE FEATURES
# ==========================================================


NUMERIC_FEATURES = [
    "BHK",
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
    "Floor_Ratio",
]


CATEGORICAL_FEATURES = [
    "State",
    "City",
    "Property_Type",
    "Furnished_Status",
    "Facing",
    "Owner_Type"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PEER_GROUP = [
    "City",
    "Property_Type",
    "BHK"
]


# ==========================================================
# 3. LOAD DATASET
# ==========================================================

def load_data():

    print("\nLoading feature-engineered dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    print("Dataset Shape:", df.shape)

    missing_features = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    return df


# ==========================================================
# 4. CREATE TRAINING-ONLY INVESTMENT BENCHMARKS
# ==========================================================

def create_benchmarks(train_df):

    print("\nCreating training-only investment benchmarks...")

    benchmarks = {

        "city_type_bhk": (
            train_df.groupby(PEER_GROUP)
            ["Price_per_SqFt"]
            .median()
        ),

        "city_type": (
            train_df.groupby(
                ["City", "Property_Type"]
            )["Price_per_SqFt"]
            .median()
        ),

        "city": (
            train_df.groupby("City")
            ["Price_per_SqFt"]
            .median()
        ),

        "global": train_df["Price_per_SqFt"].median()
    }

    # Save the most detailed training benchmarks
    benchmark_df = (
        benchmarks["city_type_bhk"]
        .reset_index()
    )

    benchmark_df = benchmark_df.rename(
        columns={
            "Price_per_SqFt":
            "Training_Peer_Median_Price_per_SqFt"
        }
    )

    benchmark_df.to_csv(
        BENCHMARK_PATH,
        index=False
    )

    print("Training benchmarks saved!")

    return benchmarks


# ==========================================================
# 5. GENERATE CLASSIFICATION TARGETS
# ==========================================================

def generate_target(data, benchmarks):

    # Start with detailed benchmark
    index_3 = pd.MultiIndex.from_frame(
        data[PEER_GROUP]
    )

    threshold = pd.Series(
        benchmarks["city_type_bhk"].reindex(
            index_3
        ).to_numpy(),
        index=data.index,
        dtype=float
    )

    # Fallback 1: City + Property Type
    index_2 = pd.MultiIndex.from_frame(
        data[["City", "Property_Type"]]
    )

    fallback_2 = pd.Series(
        benchmarks["city_type"].reindex(
            index_2
        ).to_numpy(),
        index=data.index,
        dtype=float
    )

    threshold = threshold.fillna(fallback_2)

    # Fallback 2: City
    fallback_1 = data["City"].map(
        benchmarks["city"]
    )

    threshold = threshold.fillna(fallback_1)

    # Fallback 3: Global training median
    threshold = threshold.fillna(
        benchmarks["global"]
    )

    if threshold.isna().any():
        raise ValueError(
            "Unable to calculate investment benchmark."
        )

    # 1 = Good Investment
    # 0 = Not Good Investment

    target = (
        data["Price_per_SqFt"] <= threshold
    ).astype(int)

    return target


# ==========================================================
# 6. PREPROCESSING PIPELINE
# ==========================================================

def build_preprocessor():

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

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

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
# 7. DEFINE CLASSIFICATION MODELS
# ==========================================================

def get_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=500,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    }

    return models


# ==========================================================
# 8. TRAIN AND EVALUATE MODELS
# ==========================================================

def train_models(X_train, X_test, y_train, y_test):

    models = get_models()

    results = []

    trained_models = {}

    best_f1 = -1

    best_model_name = None
    best_pipeline = None

    for name, classifier in models.items():

        print("\n====================================")
        print("Training:", name)
        print("====================================")

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor()
                ),
                (
                    "classifier",
                    classifier
                )
            ]
        )

        # Train model
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        y_proba = pipeline.predict_proba(
            X_test
        )[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(
            y_test, y_pred
        )

        precision = precision_score(
            y_test, y_pred, zero_division=0
        )

        recall = recall_score(
            y_test, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test, y_proba
        )

        # Display results
        print("\nModel:", name)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        # Classification report
        report = classification_report(
            y_test,
            y_pred,
            target_names=[
                "Not Good Investment",
                "Good Investment"
            ],
            zero_division=0
        )

        print("\nClassification Report:")
        print(report)

        safe_name = (
            name.lower()
            .replace(" ", "_")
        )

        # Save confusion matrix
        cm = confusion_matrix(
            y_test, y_pred
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Not Good",
                "Good"
            ]
        )

        display.plot(
            cmap="Blues",
            values_format="d"
        )

        plt.title(f"{name} - Confusion Matrix")

        plt.savefig(
            RESULT_DIR / f"{safe_name}_confusion_matrix.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        # Save classification report
        report_path = (
            RESULT_DIR / f"{safe_name}_report.txt"
        )

        report_path.write_text(
            report,
            encoding="utf-8"
        )

        # Save individual model
        model_path = (
            MODEL_DIR / f"{safe_name}.joblib"
        )

        joblib.dump(
            pipeline,
            model_path
        )

        trained_models[name] = pipeline

        results.append({

            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "ROC_AUC": roc_auc

        })

        # Select best model based on F1
        if f1 > best_f1:

            best_f1 = f1

            best_model_name = name

            best_pipeline = pipeline

    # Save all model results
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1_Score",
        ascending=False
    )

    results_df.to_csv(
        RESULT_DIR / "classification_metrics.csv",
        index=False
    )

    # Save best model
    joblib.dump(
        best_pipeline,
        MODEL_DIR / "classification_best_model.joblib"
    )

    metadata = {
        "best_model": best_model_name,
        "selection_metric": "F1_Score",
        "best_f1_score": float(best_f1),
        "features": FEATURES,
        "reference_target": "Good_Investment"
    }

    with open(
        MODEL_DIR / "classification_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\n====================================")
    print("CLASSIFICATION MODEL COMPARISON")
    print("====================================")

    print(
        results_df.to_string(index=False)
    )

    print("\nBest Model:", best_model_name)

    print("Best F1 Score:", round(best_f1, 4))

    print("\nBest model saved successfully!")

    return results_df


# ==========================================================
# 9. MAIN EXECUTION
# ==========================================================

def main():

    df = load_data()

    # Split before generating labels.
    # This prevents the test set from influencing
    # the investment benchmark.

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    print("\nTraining records:", len(train_df))
    print("Testing records:", len(test_df))

    # Calculate benchmarks using training data only
    benchmarks = create_benchmarks(train_df)

    # Generate separate training and test targets
    y_train = generate_target(
        train_df,
        benchmarks
    )

    y_test = generate_target(
        test_df,
        benchmarks
    )

    print("\nTraining Target Distribution:")
    print(y_train.value_counts())

    print("\nTesting Target Distribution:")
    print(y_test.value_counts())

    # Select model features
    X_train = train_df[FEATURES].copy()

    X_test = test_df[FEATURES].copy()

    # Train and compare classification algorithms
    results = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n====================================")
    print("CLASSIFICATION TRAINING COMPLETED!")
    print("====================================")

    print("\nResults directory:", RESULT_DIR)

    print("\nModels directory:", MODEL_DIR)


if __name__ == "__main__":

    main()
