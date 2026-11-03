
# ============================================
# REAL ESTATE INVESTMENT ADVISOR
# DATA PREPROCESSING
# ============================================

import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------
# 1. DEFINE PROJECT PATHS
# --------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "india_housing_prices.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_PATH = (
    PROCESSED_DIR / "india_housing_cleaned.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_PATH = REPORT_DIR / "data_quality_report.txt"


# --------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------

def load_data():

    print("\nLoading dataset...")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print("Dataset loaded successfully!")

    return df


# --------------------------------------------
# 3. INITIAL DATA INSPECTION
# --------------------------------------------

def inspect_data(df):

    print("\n========== DATASET INFORMATION ==========")

    print("Dataset Shape:", df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nStatistical Summary:")
    print(df.describe())


# --------------------------------------------
# 4. DATA CLEANING
# --------------------------------------------

def clean_data(df):

    print("\n========== DATA CLEANING ==========")

    original_rows = len(df)

    # Remove unwanted spaces from column names
    df.columns = df.columns.str.strip()

    # Clean categorical text values
    for col in df.select_dtypes(include="str").columns:

        df[col] = (
            df[col]
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        df[col] = df[col].replace("", np.nan)

    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()

    df = df.drop_duplicates().copy()

    print("Duplicates removed:", duplicate_count)

    # Remove records with invalid essential values
    invalid_mask = (
        (df["Size_in_SqFt"] <= 0)
        | (df["Price_in_Lakhs"] <= 0)
        | (df["BHK"] <= 0)
    )

    invalid_count = invalid_mask.sum()

    df = df.loc[~invalid_mask].copy()

    print("Invalid essential records removed:", invalid_count)

    # ----------------------------------------
    # 5. CHECK FLOOR INFORMATION
    # ----------------------------------------

    floor_valid = (
        (df["Floor_No"] >= 0)
        & (df["Total_Floors"] >= 1)
        & (df["Floor_No"] <= df["Total_Floors"])
    )

    floor_invalid_count = (~floor_valid).sum()

    # Preserve original values and add a
    # quality indicator instead of guessing
    # the correct floor numbers.

    df["Floor_Info_Valid"] = floor_valid.astype(int)

    print(
        "Inconsistent floor records:",
        floor_invalid_count
    )

    # ----------------------------------------
    # 6. VALIDATE PRICE PER SQUARE FOOT
    # ----------------------------------------

    # Dataset convention:
    # Price_per_SqFt = Price_in_Lakhs / Size_in_SqFt

    df["Price_per_SqFt"] = (
        df["Price_in_Lakhs"] / df["Size_in_SqFt"]
    ).round(2)

    print("Price per SqFt validated.")

    # ----------------------------------------
    # 7. HANDLE MISSING VALUES
    # ----------------------------------------

    missing_before = df.isnull().sum().sum()

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = df.select_dtypes(
        include="str"
    ).columns

    # Numeric columns: Median imputation
    for col in numeric_columns:

        if df[col].isnull().any():

            df[col] = df[col].fillna(df[col].median())

    # Categorical columns: Mode imputation
    for col in categorical_columns:

        if df[col].isnull().any():

            mode_values = df[col].mode()

            if not mode_values.empty:
                df[col] = df[col].fillna(mode_values.iloc[0])

    missing_after = df.isnull().sum().sum()

    print("Missing values before cleaning:", missing_before)
    print("Missing values after cleaning:", missing_after)

    print("Original rows:", original_rows)
    print("Final rows:", len(df))

    return df, floor_invalid_count


# --------------------------------------------
# 8. SAVE PROCESSED DATASET
# --------------------------------------------

def save_data(df):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print("\nCleaned dataset saved successfully!")
    print("Location:", PROCESSED_DATA_PATH)


# --------------------------------------------
# 9. GENERATE DATA QUALITY REPORT
# --------------------------------------------

def generate_report(df, floor_invalid_count):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = f"""
REAL ESTATE INVESTMENT ADVISOR
DATA QUALITY REPORT
========================================

Total Records: {len(df)}

Total Features: {len(df.columns)}

Missing Values: {df.isnull().sum().sum()}

Duplicate Records: {df.duplicated().sum()}

Unique Properties: {df["ID"].nunique()}

Inconsistent Floor Records: {floor_invalid_count}

Unique States: {df["State"].nunique()}

Unique Cities: {df["City"].nunique()}

Unique Localities: {df["Locality"].nunique()}

Property Types:
{df["Property_Type"].value_counts().to_string()}

Furnishing Distribution:
{df["Furnished_Status"].value_counts().to_string()}

Average Property Price:
{df["Price_in_Lakhs"].mean():.2f} Lakhs

Average Property Size:
{df["Size_in_SqFt"].mean():.2f} SqFt

========================================
PREPROCESSING COMPLETED
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\nData quality report saved:", REPORT_PATH)


# --------------------------------------------
# 10. MAIN EXECUTION
# --------------------------------------------

def main():

    df = load_data()

    inspect_data(df)

    df, floor_invalid_count = clean_data(df)

    save_data(df)

    generate_report(df, floor_invalid_count)

    print("\n===================================")
    print("DATA PREPROCESSING COMPLETED!")
    print("===================================")


if __name__ == "__main__":

    main()
