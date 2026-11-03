# ==========================================================
# REAL ESTATE INVESTMENT ADVISOR
# PROPERTY PREDICTION SYSTEM
# ==========================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "india_housing_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

CLASSIFICATION_MODEL_PATH = (
    MODEL_DIR / "classification_best_model.joblib"
)

# The regression model is retained for the separate model-evaluation
# page, but it is NOT used to claim an independent future-price forecast.

RANDOM_STATE = 42
REFERENCE_YEAR = 2026
ANNUAL_APPRECIATION_RATE = 0.08
FORECAST_YEARS = 5

PEER_GROUP = ["City", "Property_Type", "BHK"]


# ==========================================================
# 2. MODEL FEATURES
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
    "Owner_Type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ==========================================================
# 3. CACHED RESOURCES
# ==========================================================

_CLASSIFICATION_MODEL = None
_BENCHMARKS = None


# ==========================================================
# 4. LOAD CLASSIFICATION MODEL
# ==========================================================

def load_classification_model():
    """Load and cache the trained investment-classification model."""
    global _CLASSIFICATION_MODEL

    if _CLASSIFICATION_MODEL is not None:
        return _CLASSIFICATION_MODEL

    if not CLASSIFICATION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classification model not found: "
            f"{CLASSIFICATION_MODEL_PATH}"
        )

    print("Loading investment classification model...")
    _CLASSIFICATION_MODEL = joblib.load(CLASSIFICATION_MODEL_PATH)
    print(
        "Classification model loaded:",
        type(_CLASSIFICATION_MODEL).__name__,
    )

    return _CLASSIFICATION_MODEL


# Backward-compatible helper for any older code that imports load_models.
# The second return value is None because regression is not used for the
# user-facing five-year scenario calculation.
def load_models():
    return load_classification_model(), None


# ==========================================================
# 5. CREATE TRAINING-ONLY BENCHMARKS
# ==========================================================

def load_benchmarks():
    """Build peer-price benchmarks from the same training split."""
    global _BENCHMARKS

    if _BENCHMARKS is not None:
        return _BENCHMARKS

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading investment benchmarks...")
    df = pd.read_csv(DATA_PATH)

    required = {"City", "Property_Type", "BHK", "Price_per_SqFt"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing benchmark columns: {sorted(missing)}"
        )

    train_df, _ = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    benchmarks = {
        "city_type_bhk": (
            train_df.groupby(PEER_GROUP)["Price_per_SqFt"]
            .median()
            .to_dict()
        ),
        "city_type": (
            train_df.groupby(["City", "Property_Type"])["Price_per_SqFt"]
            .median()
            .to_dict()
        ),
        "city": (
            train_df.groupby("City")["Price_per_SqFt"]
            .median()
            .to_dict()
        ),
        "global": float(train_df["Price_per_SqFt"].median()),
    }

    _BENCHMARKS = benchmarks
    print("Investment benchmarks loaded!")
    return _BENCHMARKS


# ==========================================================
# 6. CALCULATE INVESTMENT BENCHMARK
# ==========================================================

def get_investment_benchmark(city, property_type, bhk):
    benchmarks = load_benchmarks()

    # Level 1: City + Property Type + BHK
    key_3 = (city, property_type, bhk)
    threshold = benchmarks["city_type_bhk"].get(key_3)

    if threshold is not None:
        return float(threshold)

    # Level 2: City + Property Type
    key_2 = (city, property_type)
    threshold = benchmarks["city_type"].get(key_2)

    if threshold is not None:
        return float(threshold)

    # Level 3: City
    threshold = benchmarks["city"].get(city)

    if threshold is not None:
        return float(threshold)

    # Level 4: Global median
    return float(benchmarks["global"])


# ==========================================================
# 7. VALIDATE PROPERTY INPUT
# ==========================================================

def validate_property_data(data):
    """Validate user inputs without changing the caller's dictionary."""
    data = data.copy()

    required_fields = [
        "State",
        "City",
        "Locality",
        "Property_Type",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Year_Built",
        "Furnished_Status",
        "Floor_No",
        "Total_Floors",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "Public_Transport_Accessibility",
        "Parking_Space",
        "Security",
        "Amenities",
        "Facing",
        "Owner_Type",
        "Availability_Status",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in data
        or data[field] is None
        or (isinstance(data[field], str) and not data[field].strip())
    ]

    if missing_fields:
        raise ValueError(f"Missing property details: {missing_fields}")

    numeric_fields = [
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Year_Built",
        "Floor_No",
        "Total_Floors",
        "Nearby_Schools",
        "Nearby_Hospitals",
    ]

    for field in numeric_fields:
        try:
            value = float(data[field])
        except (ValueError, TypeError):
            raise ValueError(f"{field} must be numeric.")

        if not np.isfinite(value):
            raise ValueError(f"{field} must be a finite number.")

        data[field] = value

    for field in ("BHK", "Size_in_SqFt", "Price_in_Lakhs", "Total_Floors"):
        if data[field] <= 0:
            raise ValueError(f"{field} must be greater than zero.")

    if data["Year_Built"] > REFERENCE_YEAR:
        raise ValueError("Year built cannot be in the future.")

    if data["Year_Built"] < 1900:
        raise ValueError("Please enter a valid construction year.")

    if data["Floor_No"] < 0:
        raise ValueError("Floor number cannot be negative.")

    if data["Nearby_Schools"] < 0 or data["Nearby_Hospitals"] < 0:
        raise ValueError("Nearby facilities cannot be negative.")

    if data["Floor_No"] > data["Total_Floors"]:
        raise ValueError("Floor number cannot exceed total floors.")

    return data


# ==========================================================
# 8. CREATE MACHINE LEARNING FEATURES
# ==========================================================

def create_property_features(data):
    print("Preparing property features...")

    data = data.copy()

    data["Price_per_SqFt"] = (
        data["Price_in_Lakhs"] / data["Size_in_SqFt"]
    )

    data["Property_Age_2026"] = (
        REFERENCE_YEAR - data["Year_Built"]
    )

    floor_no = data["Floor_No"]
    total_floors = data["Total_Floors"]

    valid_floor = (
        floor_no >= 0
        and total_floors > 0
        and floor_no <= total_floors
    )

    data["Floor_Info_Valid"] = int(valid_floor)

    if valid_floor:
        data["Floor_Ratio"] = floor_no / total_floors
    else:
        data["Floor_No"] = np.nan
        data["Total_Floors"] = np.nan
        data["Floor_Ratio"] = np.nan

    if isinstance(data["Amenities"], (list, tuple, set)):
        amenity_set = {
            str(item).strip().lower()
            for item in data["Amenities"]
        }
    else:
        amenity_set = {
            item.strip().lower()
            for item in str(data["Amenities"]).split(",")
            if item.strip()
        }

    data["Amenities_Count"] = len(amenity_set)

    amenity_names = [
        "Gym",
        "Pool",
        "Garden",
        "Clubhouse",
        "Playground",
    ]

    for amenity in amenity_names:
        data[f"Has_{amenity}"] = int(
            amenity.lower() in amenity_set
        )

    data["Has_Parking"] = int(
        str(data["Parking_Space"]).strip().lower() == "yes"
    )

    data["Has_Security"] = int(
        str(data["Security"]).strip().lower() == "yes"
    )

    data["Is_Ready_to_Move"] = int(
        str(data["Availability_Status"]).strip().lower()
        == "ready_to_move"
    )

    transport_mapping = {
        "low": 1,
        "medium": 2,
        "high": 3,
    }

    transport_score = transport_mapping.get(
        str(data["Public_Transport_Accessibility"]).strip().lower()
    )

    if transport_score is None:
        raise ValueError(
            "Transport accessibility must be Low, Medium or High."
        )

    data["Transport_Accessibility_Score"] = transport_score

    feature_df = pd.DataFrame([data])

    missing_features = [
        feature for feature in FEATURES
        if feature not in feature_df.columns
    ]
    if missing_features:
        raise ValueError(
            f"Could not create model features: {missing_features}"
        )

    return feature_df[FEATURES]


# ==========================================================
# 9. CLASSIFICATION PREDICTION
# ==========================================================

def predict_investment(classification_model, property_features, property_data):
    print("Predicting investment classification...")

    prediction = int(classification_model.predict(property_features)[0])

    if not hasattr(classification_model, "predict_proba"):
        raise TypeError(
            "The classification model does not support predict_proba()."
        )

    probabilities = classification_model.predict_proba(property_features)[0]
    classes = list(classification_model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The classification model does not contain class 1 "
            "for Good Investment."
        )

    good_probability = float(probabilities[classes.index(1)])

    benchmark = get_investment_benchmark(
        property_data["City"],
        property_data["Property_Type"],
        property_data["BHK"],
    )

    actual_price_per_sqft = (
        property_data["Price_in_Lakhs"]
        / property_data["Size_in_SqFt"]
    )

    benchmark_rule_result = actual_price_per_sqft <= benchmark

    return {
        "prediction": prediction,
        "investment_status": (
            "Good Investment" if prediction == 1
            else "Not Good Investment"
        ),
        "good_investment_probability": good_probability * 100,
        "peer_median_price_per_sqft": benchmark,
        "property_price_per_sqft": actual_price_per_sqft,
        "benchmark_rule_result": (
            "Good Investment" if benchmark_rule_result
            else "Not Good Investment"
        ),
    }


# ==========================================================
# 10. FIVE-YEAR SCENARIO ESTIMATION
# ==========================================================

def estimate_future_price_scenario(current_price):
    """
    Calculate an illustrative compound-growth scenario.

    This is a formula-based scenario, not an ML forecast or
    a guarantee of actual property appreciation.
    """
    current_price = float(current_price)

    if not np.isfinite(current_price) or current_price <= 0:
        raise ValueError("Current price must be a positive finite number.")

    scenario_multiplier = (
        1 + ANNUAL_APPRECIATION_RATE
    ) ** FORECAST_YEARS

    scenario_price = current_price * scenario_multiplier
    appreciation = scenario_price - current_price
    appreciation_percentage = (
        appreciation / current_price
    ) * 100

    return {
        "current_price_lakhs": current_price,
        "scenario_future_price_lakhs": scenario_price,
        "expected_appreciation_lakhs": appreciation,
        "expected_appreciation_percentage": appreciation_percentage,
        "annual_appreciation_rate": ANNUAL_APPRECIATION_RATE * 100,
        "forecast_years": FORECAST_YEARS,
        "forecast_method": "Assumption-based compound-growth scenario",
    }


# ==========================================================
# 11. MAIN PREDICTION FUNCTION
# ==========================================================

def predict_property(property_data):
    print("\n====================================")
    print("REAL ESTATE PROPERTY PREDICTION")
    print("====================================")

    property_data = validate_property_data(property_data)

    classification_model = load_classification_model()
    property_features = create_property_features(property_data)

    classification_result = predict_investment(
        classification_model,
        property_features,
        property_data,
    )

    scenario_result = estimate_future_price_scenario(
        property_data["Price_in_Lakhs"]
    )

    results = {
        "property_location": (
            f"{property_data['Locality']}, "
            f"{property_data['City']}, "
            f"{property_data['State']}"
        ),
        "property_type": property_data["Property_Type"],
        "bhk": property_data["BHK"],
        "property_size_sqft": property_data["Size_in_SqFt"],
        **classification_result,
        **scenario_result,
    }

    return results


# ==========================================================
# 12. FORMAT PREDICTION RESULTS
# ==========================================================

def display_prediction(results):
    print("\n====================================")
    print("PROPERTY PREDICTION RESULTS")
    print("====================================")

    print("\nProperty Information")
    print("--------------------")
    print("Location:", results["property_location"])
    print("Property Type:", results["property_type"])
    print("BHK:", results["bhk"])
    print("Size:", results["property_size_sqft"], "SqFt")

    print("\nInvestment Analysis")
    print("--------------------")
    print("ML Prediction:", results["investment_status"])
    print(
        "Good Investment Probability:",
        f"{results['good_investment_probability']:.2f}%",
    )
    print(
        "Peer Median Price per SqFt:",
        f"{results['peer_median_price_per_sqft']:.4f} Lakhs",
    )
    print(
        "Property Price per SqFt:",
        f"{results['property_price_per_sqft']:.4f} Lakhs",
    )
    print("Benchmark Rule Result:", results["benchmark_rule_result"])

    print("\nFive-Year Price Scenario")
    print("------------------------")
    print(
        "Current Price:",
        f"Rs. {results['current_price_lakhs']:.2f} Lakhs",
    )
    print(
        "Scenario Future Price:",
        f"Rs. {results['scenario_future_price_lakhs']:.2f} Lakhs",
    )
    print(
        "Expected Scenario Appreciation:",
        f"Rs. {results['expected_appreciation_lakhs']:.2f} Lakhs",
    )
    print(
        "Appreciation Percentage:",
        f"{results['expected_appreciation_percentage']:.2f}%",
    )
    print(
        f"Assumption: {results['annual_appreciation_rate']:.0f}% annually "
        f"for {results['forecast_years']} years."
    )
    print("This is an illustrative scenario, not a guaranteed return.")
    print("====================================")


# ==========================================================
# 13. MODULE TEST
# ==========================================================

def main():
    print("\n====================================")
    print("PREDICTION MODULE TEST")
    print("====================================")

    model = load_classification_model()
    load_benchmarks()

    print("\nPrediction module initialized successfully!")
    print("Classification model:", type(model).__name__)
    print("Five-year estimate: formula-based 8% scenario")
    print("\nReady to receive property details.")

    sample_property = {
        "State": "West Bengal",
        "City": "Durgapur",
        "Locality": "Bidhannagar",
        "Property_Type": "Apartment",
        "BHK": 3,
        "Size_in_SqFt": 1500,
        "Price_in_Lakhs": 65.0,
        "Year_Built": 2020,
        "Furnished_Status": "Semi-Furnished",
        "Floor_No": 3,
        "Total_Floors": 5,
        "Nearby_Schools": 4,
        "Nearby_Hospitals": 3,
        "Public_Transport_Accessibility": "High",
        "Parking_Space": "Yes",
        "Security": "Yes",
        "Amenities": ["Gym", "Garden", "Clubhouse"],
        "Facing": "East",
        "Owner_Type": "Builder",
        "Availability_Status": "Ready_to_Move",
    }

    results = predict_property(sample_property)
    display_prediction(results)


if __name__ == "__main__":
    main()
