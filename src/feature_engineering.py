
# ==========================================================
# REAL ESTATE INVESTMENT ADVISOR
# FEATURE ENGINEERING AND TARGET CREATION
# ==========================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "india_housing_cleaned.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "india_housing_features.csv"
)

BENCHMARK_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "investment_benchmarks.csv"
)

REPORT_PATH = (
    PROJECT_ROOT / "reports"
    / "feature_engineering_report.txt"
)

REFERENCE_YEAR = 2026

ANNUAL_APPRECIATION_RATE = 0.08

FORECAST_YEARS = 5

PEER_GROUP = [
    "City",
    "Property_Type",
    "BHK"
]


# ==========================================================
# 2. LOAD CLEANED DATASET
# ==========================================================

def load_data():

    print("\nLoading cleaned dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("Dataset loaded successfully!")
    print("Dataset Shape:", df.shape)

    return df


# ==========================================================
# 3. CREATE PROPERTY-RELATED FEATURES
# ==========================================================

def create_property_features(df):

    print("\nCreating property-related features...")

    # Recalculate price per square foot
    df["Price_per_SqFt"] = (
        df["Price_in_Lakhs"] /
        df["Size_in_SqFt"]
    )

    # Calculate updated property age
    df["Property_Age_2026"] = (
        REFERENCE_YEAR - df["Year_Built"]
    ).clip(lower=0)

    # Create floor information validity
    valid_floor = df["Floor_Info_Valid"].eq(1)

    # Do not use inconsistent floor numbers
    df.loc[
        ~valid_floor,
        ["Floor_No", "Total_Floors"]
    ] = np.nan

    # Calculate floor ratio
    df["Floor_Ratio"] = (
        df["Floor_No"] / df["Total_Floors"]
    )

    print("Property features created.")

    return df


# ==========================================================
# 4. CREATE AMENITY FEATURES
# ==========================================================

def create_amenity_features(df):

    print("\nCreating amenity features...")

    # Convert amenity strings into sets
    amenity_sets = (
        df["Amenities"]
        .fillna("")
        .apply(
            lambda x: {
                item.strip()
                for item in x.split(",")
                if item.strip()
            }
        )
    )

    # Total number of amenities
    df["Amenities_Count"] = amenity_sets.map(len)

    # Individual amenity indicators
    amenity_list = [
        "Gym",
        "Pool",
        "Garden",
        "Clubhouse",
        "Playground"
    ]

    for amenity in amenity_list:

        df[f"Has_{amenity}"] = (
            amenity_sets
            .apply(
                lambda items: int(amenity in items)
            )
        )

    print("Amenity features created.")

    return df


# ==========================================================
# 5. CREATE LOCATION AND FACILITY FEATURES
# ==========================================================

def create_facility_features(df):

    print("\nCreating facility features...")

    # Parking availability
    df["Has_Parking"] = (
        df["Parking_Space"]
        .str.lower()
        .eq("yes")
        .astype(int)
    )

    # Security availability
    df["Has_Security"] = (
        df["Security"]
        .str.lower()
        .eq("yes")
        .astype(int)
    )

    # Ready-to-move indicator
    df["Is_Ready_to_Move"] = (
        df["Availability_Status"]
        .eq("Ready_to_Move")
        .astype(int)
    )

    # Transport accessibility score
    transport_mapping = {
        "Low": 1,
        "Medium": 2,
        "High": 3
    }

    df["Transport_Accessibility_Score"] = (
        df["Public_Transport_Accessibility"]
        .map(transport_mapping)
    )

    print("Facility features created.")

    return df


# ==========================================================
# 6. CREATE GOOD INVESTMENT TARGET
# ==========================================================

def create_investment_target(df):

    print("\nCreating Good Investment target...")

    # Calculate comparable property median
    benchmarks = (
        df.groupby(PEER_GROUP)["Price_per_SqFt"]
        .median()
        .reset_index()
    )

    benchmarks = benchmarks.rename(
        columns={
            "Price_per_SqFt":
            "Peer_Median_Price_per_SqFt"
        }
    )

    # Save benchmark for future application use
    benchmarks.to_csv(
        BENCHMARK_PATH,
        index=False
    )

    # Attach peer median to every property
    df = df.merge(
        benchmarks,
        on=PEER_GROUP,
        how="left",
        validate="many_to_one"
    )

    # Classification target
    df["Good_Investment"] = (
        df["Price_per_SqFt"]
        <= df["Peer_Median_Price_per_SqFt"]
    ).astype(int)

    print("Good Investment target created.")

    print("\nClassification Distribution:")

    print(
        df["Good_Investment"]
        .value_counts()
        .rename(
            index={
                0: "Not Good Investment",
                1: "Good Investment"
            }
        )
    )

    return df, benchmarks


# ==========================================================
# 7. CREATE FUTURE PRICE TARGET
# ==========================================================

def create_future_price_target(df):

    print("\nCreating future price target...")

    appreciation_multiplier = (
        1 + ANNUAL_APPRECIATION_RATE
    ) ** FORECAST_YEARS

    df["Future_Price_5Y"] = (
        df["Price_in_Lakhs"]
        * appreciation_multiplier
    ).round(2)

    print(
        "Annual Appreciation Rate:",
        ANNUAL_APPRECIATION_RATE * 100,
        "%"
    )

    print(
        "Forecast Duration:",
        FORECAST_YEARS,
        "Years"
    )

    print(
        "Appreciation Multiplier:",
        round(appreciation_multiplier, 4)
    )

    print("Future price target created.")

    return df


# ==========================================================
# 8. SAVE FEATURE ENGINEERED DATASET
# ==========================================================

def save_data(df):

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFeature engineered dataset saved!")
    print("Location:", OUTPUT_PATH)


# ==========================================================
# 9. GENERATE FEATURE ENGINEERING REPORT
# ==========================================================

def generate_report(df, benchmarks):

    classification_counts = (
        df["Good_Investment"].value_counts()
    )

    report = f"""
REAL ESTATE INVESTMENT ADVISOR
FEATURE ENGINEERING REPORT
==========================================

Total Records: {len(df)}

Total Features: {len(df.columns)}

REFERENCE YEAR: {REFERENCE_YEAR}

NEW ENGINEERED FEATURES
------------------------------------------

Property_Age_2026
Floor_Ratio
Amenities_Count
Has_Gym
Has_Pool
Has_Garden
Has_Clubhouse
Has_Playground
Has_Parking
Has_Security
Is_Ready_to_Move
Transport_Accessibility_Score

CLASSIFICATION TARGET
------------------------------------------

Target: Good_Investment

Comparison: City + Property_Type + BHK

Comparison Groups: {len(benchmarks)}

Not Good Investment: {classification_counts.get(0, 0)}

Good Investment: {classification_counts.get(1, 0)}

REGRESSION TARGET
------------------------------------------

Target: Future_Price_5Y

Annual Appreciation Rate: 8%

Forecast Duration: 5 Years

Appreciation Multiplier:
{(1 + ANNUAL_APPRECIATION_RATE) ** FORECAST_YEARS:.4f}

Average Current Price:
{df["Price_in_Lakhs"].mean():.2f} Lakhs

Average Projected Price:
{df["Future_Price_5Y"].mean():.2f} Lakhs

IMPORTANT:
Future price is an assumption-based
projection, not an observed market price.

==========================================
FEATURE ENGINEERING COMPLETED
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8"
    )

    print("\nReport saved:", REPORT_PATH)


# ==========================================================
# 10. MAIN EXECUTION
# ==========================================================

def main():

    df = load_data()

    df = create_property_features(df)

    df = create_amenity_features(df)

    df = create_facility_features(df)

    df, benchmarks = create_investment_target(df)

    df = create_future_price_target(df)

    save_data(df)

    generate_report(df, benchmarks)

    print("\n==========================================")
    print("FEATURE ENGINEERING COMPLETED!")
    print("==========================================")


if __name__ == "__main__":

    main()
