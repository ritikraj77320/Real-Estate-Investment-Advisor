# ==========================================================
# REAL ESTATE INVESTMENT ADVISOR
# CLASSIFICATION MODEL EVALUATION
# ==========================================================

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# ==========================================================
# 1. PROJECT PATHS AND SETTINGS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "india_housing_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "classification_best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "classification_metadata.json"
EVALUATION_DIR = PROJECT_ROOT / "reports" / "evaluation"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# IMPORTANT: These must remain identical to train_classification.py.
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
    "Owner_Type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
PEER_GROUP = ["City", "Property_Type", "BHK"]
TARGET_COLUMN = "Good_Investment"


# ==========================================================
# 2. LOAD DATA AND CLASSIFICATION MODEL
# ==========================================================

def load_resources():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Classification model not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    required_columns = set(FEATURES + PEER_GROUP + ["Price_per_SqFt"])
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Guard against training/evaluation feature mismatch.
    model_features = list(model.named_steps["preprocessor"].feature_names_in_)
    if model_features != FEATURES:
        raise ValueError(
            "Feature mismatch between the saved model and evaluation script.\n"
            f"Model features: {model_features}\n"
            f"Evaluation features: {FEATURES}\n"
            "Make the feature lists identical before evaluating."
        )

    print(f"Dataset loaded: {df.shape}")
    print(f"Classification model loaded: {type(model.named_steps['classifier']).__name__}")
    print(f"Number of model input features: {len(FEATURES)}")
    return df, model


# ==========================================================
# 3. RECREATE TRAINING-ONLY PEER BENCHMARKS
# ==========================================================

def create_benchmarks(train_df):
    return {
        "city_type_bhk": train_df.groupby(PEER_GROUP)["Price_per_SqFt"].median(),
        "city_type": train_df.groupby(["City", "Property_Type"])["Price_per_SqFt"].median(),
        "city": train_df.groupby("City")["Price_per_SqFt"].median(),
        "global": train_df["Price_per_SqFt"].median(),
    }


def generate_classification_target(data, benchmarks):
    index_3 = pd.MultiIndex.from_frame(data[PEER_GROUP])
    threshold = pd.Series(
        benchmarks["city_type_bhk"].reindex(index_3).to_numpy(),
        index=data.index,
        dtype=float,
    )

    index_2 = pd.MultiIndex.from_frame(data[["City", "Property_Type"]])
    fallback_2 = pd.Series(
        benchmarks["city_type"].reindex(index_2).to_numpy(),
        index=data.index,
        dtype=float,
    )

    threshold = threshold.fillna(fallback_2)
    threshold = threshold.fillna(data["City"].map(benchmarks["city"]))
    threshold = threshold.fillna(benchmarks["global"])

    if threshold.isna().any():
        raise ValueError("Unable to calculate one or more peer-price benchmarks.")

    return (data["Price_per_SqFt"] <= threshold).astype(int)


# ==========================================================
# 4. EVALUATE CLASSIFICATION MODEL
# ==========================================================

def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Find the probability column for class 1 explicitly.
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"The trained model does not contain class 1. Classes: {classes}")
    positive_class_index = classes.index(1)
    y_proba = model.predict_proba(X_test)[:, positive_class_index]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_test.nunique() == 2 else np.nan

    metrics = pd.DataFrame([{
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
    }])
    metrics.to_csv(EVALUATION_DIR / "classification_evaluation_metrics.csv", index=False)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not Good Investment", "Good Investment"],
        zero_division=0,
    )
    (EVALUATION_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Good", "Good"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / "classification_confusion_matrix.png", dpi=150)
    plt.close()

    if y_test.nunique() == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Classification ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(EVALUATION_DIR / "classification_roc_curve.png", dpi=150)
        plt.close()

    print("\nCLASSIFICATION EVALUATION RESULTS")
    print(metrics.to_string(index=False))
    print("\nClassification report:\n")
    print(report)
    return metrics


# ==========================================================
# 5. SAVE CLASSIFICATION FEATURE IMPORTANCE
# ==========================================================

def save_classification_feature_importance(model):
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.asarray(estimator.coef_).ravel())
    else:
        print("This classifier does not expose feature importance or coefficients.")
        return

    if len(feature_names) != len(importances):
        raise ValueError("Feature-importance length does not match transformed feature names.")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(
        EVALUATION_DIR / "classification_feature_importance.csv",
        index=False,
    )

    top_features = importance_df.head(20).sort_values("Importance")
    plt.figure(figsize=(11, 8))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.xlabel("Importance")
    plt.title("Top 20 Classification Features")
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / "classification_feature_importance.png", dpi=150)
    plt.close()


# ==========================================================
# 6. MAIN
# ==========================================================

def main():
    print("\nREAL ESTATE INVESTMENT ADVISOR - CLASSIFICATION EVALUATION")

    df, model = load_resources()

    # This reproduces the exact split used in train_classification.py.
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print(f"Training records: {len(train_df)}")
    print(f"Testing records: {len(test_df)}")

    benchmarks = create_benchmarks(train_df)
    y_test = generate_classification_target(test_df, benchmarks)
    X_test = test_df[FEATURES].copy()

    print("\nTest target distribution:")
    print(y_test.value_counts().sort_index())

    metrics = evaluate_classification(model, X_test, y_test)
    save_classification_feature_importance(model)

    print("\nClassification evaluation files saved to:")
    print(EVALUATION_DIR)
    print("\nNOTE: Regression evaluation is intentionally excluded.")
    print("The current Future_Price_5Y target is an assumption-based 8% projection,")
    print("not an independently observed future market price.")
    return metrics


if __name__ == "__main__":
    main()
