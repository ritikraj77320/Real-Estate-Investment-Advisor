
# ==========================================================
# REAL ESTATE INVESTMENT ADVISOR
# EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "india_housing_cleaned.csv"
)

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

EDA_REPORT_DIR = PROJECT_ROOT / "reports" / "eda_results"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
EDA_REPORT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ==========================================================
# 2. LOAD DATA
# ==========================================================

def load_data():

    print("\nLoading cleaned dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    print("Shape:", df.shape)

    return df


# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================

def save_plot(filename):

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def save_table(df, filename):

    df.to_csv(
        EDA_REPORT_DIR / filename,
        index=True
    )


# ==========================================================
# GROUP 1: PRICE AND SIZE ANALYSIS
# QUESTIONS 1-5
# ==========================================================

def price_and_size_analysis(df):

    print("\n========== PRICE AND SIZE ANALYSIS ==========")

    # ------------------------------------------------------
    # Q1. Distribution of property prices
    # ------------------------------------------------------

    print("Q1: Property price distribution")

    plt.figure(figsize=(10, 5))

    sns.histplot(
        df["Price_in_Lakhs"],
        bins=50,
        kde=True
    )

    plt.title("Distribution of Property Prices")
    plt.xlabel("Property Price (Lakhs)")
    plt.ylabel("Frequency")

    save_plot("01_price_distribution.png")

    # ------------------------------------------------------
    # Q2. Distribution of property sizes
    # ------------------------------------------------------

    print("Q2: Property size distribution")

    plt.figure(figsize=(10, 5))

    sns.histplot(
        df["Size_in_SqFt"],
        bins=50,
        kde=True
    )

    plt.title("Distribution of Property Sizes")
    plt.xlabel("Size (Square Feet)")
    plt.ylabel("Frequency")

    save_plot("02_size_distribution.png")

    # ------------------------------------------------------
    # Q3. Price per sqft by property type
    # ------------------------------------------------------

    print("Q3: Price per sqft by property type")

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Property_Type",
        y="Price_per_SqFt"
    )

    plt.title("Price per SqFt by Property Type")
    plt.xlabel("Property Type")
    plt.ylabel("Price per SqFt (Lakhs)")

    save_plot("03_price_by_property_type.png")

    # ------------------------------------------------------
    # Q4. Relationship between size and price
    # ------------------------------------------------------

    print("Q4: Size versus property price")

    sample = df.sample(
        n=min(15000, len(df)),
        random_state=42
    )

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=sample,
        x="Size_in_SqFt",
        y="Price_in_Lakhs",
        alpha=0.4,
        s=15
    )

    plt.title("Property Size vs Property Price")
    plt.xlabel("Size (Square Feet)")
    plt.ylabel("Price (Lakhs)")

    save_plot("04_size_vs_price.png")

    # ------------------------------------------------------
    # Q5. Outlier analysis
    # ------------------------------------------------------

    print("Q5: Outlier analysis")

    columns = [
        "Price_in_Lakhs",
        "Size_in_SqFt",
        "Price_per_SqFt"
    ]

    outlier_results = []

    for col in columns:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = (
            (df[col] < lower)
            | (df[col] > upper)
        )

        outlier_results.append({
            "Feature": col,
            "Lower_Bound": lower,
            "Upper_Bound": upper,
            "Outlier_Count": int(outliers.sum()),
            "Outlier_Percentage": round(
                outliers.mean() * 100, 2
            )
        })

    outlier_df = pd.DataFrame(outlier_results)

    save_table(
        outlier_df,
        "outlier_analysis.csv"
    )

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df[columns]
    )

    plt.title("Outlier Detection in Price and Size Features")

    save_plot("05_outlier_analysis.png")

    print(outlier_df.to_string(index=False))


# ==========================================================
# GROUP 2: LOCATION-BASED ANALYSIS
# QUESTIONS 6-10
# ==========================================================

def location_analysis(df):

    print("\n========== LOCATION ANALYSIS ==========")

    # ------------------------------------------------------
    # Q6. Average price per sqft by state
    # ------------------------------------------------------

    print("Q6: Average price per sqft by state")

    state_prices = (
        df.groupby("State")["Price_per_SqFt"]
        .mean()
        .sort_values(ascending=False)
    )

    save_table(
        state_prices,
        "average_price_per_sqft_by_state.csv"
    )

    plt.figure(figsize=(12, 8))

    sns.barplot(
        x=state_prices.values,
        y=state_prices.index
    )

    plt.title("Average Price per SqFt by State")
    plt.xlabel("Average Price per SqFt (Lakhs)")

    save_plot("06_state_price_analysis.png")

    # ------------------------------------------------------
    # Q7. Average property price by city
    # ------------------------------------------------------

    print("Q7: Average property price by city")

    city_prices = (
        df.groupby("City")["Price_in_Lakhs"]
        .mean()
        .sort_values(ascending=False)
    )

    save_table(
        city_prices,
        "average_property_price_by_city.csv"
    )

    plt.figure(figsize=(12, 12))

    sns.barplot(
        x=city_prices.values,
        y=city_prices.index
    )

    plt.title("Average Property Price by City")
    plt.xlabel("Average Price (Lakhs)")

    save_plot("07_city_price_analysis.png")

    # ------------------------------------------------------
    # Q8. Median age by locality
    # ------------------------------------------------------

    print("Q8: Median property age by locality")

    locality_age = (
        df.groupby(["State", "City", "Locality"])
        ["Age_of_Property"]
        .median()
        .sort_values(ascending=False)
    )

    save_table(
        locality_age,
        "median_age_by_locality.csv"
    )

    # Display 20 oldest locality groups
    top_oldest = locality_age.head(20)

    labels = [
        f"{city} - {locality}"
        for state, city, locality in top_oldest.index
    ]

    plt.figure(figsize=(12, 8))

    sns.barplot(
        x=top_oldest.values,
        y=labels
    )

    plt.title("20 Locality Groups with Oldest Properties")
    plt.xlabel("Median Age of Property (Years)")

    save_plot("08_locality_age_analysis.png")

    # ------------------------------------------------------
    # Q9. BHK distribution across cities
    # ------------------------------------------------------

    print("Q9: BHK distribution across cities")

    bhk_city = pd.crosstab(
        df["City"],
        df["BHK"]
    )

    save_table(
        bhk_city,
        "bhk_distribution_by_city.csv"
    )

    plt.figure(figsize=(12, 14))

    sns.heatmap(
        bhk_city,
        annot=True,
        fmt="d",
        cmap="YlGnBu"
    )

    plt.title("BHK Distribution Across Cities")
    plt.xlabel("BHK")
    plt.ylabel("City")

    save_plot("09_bhk_city_distribution.png")

    # ------------------------------------------------------
    # Q10. Top 5 most expensive localities
    # ------------------------------------------------------

    print("Q10: Top 5 expensive localities")

    locality_prices = (
        df.groupby(["State", "City", "Locality"])
        .agg(
            Median_Price_Per_SqFt=(
                "Price_per_SqFt", "median"
            ),
            Average_Property_Price=(
                "Price_in_Lakhs", "mean"
            ),
            Property_Count=(
                "ID", "count"
            )
        )
    )

    # Avoid rankings based on very small groups
    locality_prices = locality_prices[
        locality_prices["Property_Count"] >= 20
    ]

    top5 = locality_prices.sort_values(
        "Median_Price_Per_SqFt",
        ascending=False
    ).head(5)

    save_table(
        top5,
        "top5_expensive_localities.csv"
    )

    labels = [
        f"{city} - {locality}"
        for state, city, locality in top5.index
    ]

    plt.figure(figsize=(11, 6))

    sns.barplot(
        x=top5["Median_Price_Per_SqFt"].values,
        y=labels
    )

    plt.title("Top 5 Most Expensive Localities")
    plt.xlabel("Median Price per SqFt (Lakhs)")

    save_plot("10_top5_expensive_localities.png")


# ==========================================================
# GROUP 3: FEATURE RELATIONSHIP AND CORRELATION
# QUESTIONS 11-15
# ==========================================================

def feature_relationship_analysis(df):

    print("\n========== FEATURE RELATIONSHIP ANALYSIS ==========")

    # ------------------------------------------------------
    # Q11. Numeric feature correlation
    # ------------------------------------------------------

    print("Q11: Correlation analysis")

    numeric_df = df.select_dtypes(
        include=np.number
    ).drop(
        columns=["ID"],
        errors="ignore"
    )

    correlation = numeric_df.corr()

    save_table(
        correlation,
        "correlation_matrix.csv"
    )

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        center=0
    )

    plt.title("Numerical Feature Correlation Heatmap")

    save_plot("11_correlation_heatmap.png")

    # ------------------------------------------------------
    # Q12. Nearby schools vs price per sqft
    # ------------------------------------------------------

    print("Q12: Nearby schools analysis")

    schools_price = (
        df.groupby("Nearby_Schools")
        ["Price_per_SqFt"]
        .mean()
    )

    save_table(
        schools_price,
        "schools_vs_price.csv"
    )

    plt.figure(figsize=(9, 5))

    sns.lineplot(
        x=schools_price.index,
        y=schools_price.values,
        marker="o"
    )

    plt.title("Nearby Schools vs Price per SqFt")
    plt.xlabel("Nearby Schools")
    plt.ylabel("Average Price per SqFt")

    save_plot("12_schools_vs_price.png")

    # ------------------------------------------------------
    # Q13. Nearby hospitals vs price per sqft
    # ------------------------------------------------------

    print("Q13: Nearby hospitals analysis")

    hospital_price = (
        df.groupby("Nearby_Hospitals")
        ["Price_per_SqFt"]
        .mean()
    )

    save_table(
        hospital_price,
        "hospitals_vs_price.csv"
    )

    plt.figure(figsize=(9, 5))

    sns.lineplot(
        x=hospital_price.index,
        y=hospital_price.values,
        marker="o"
    )

    plt.title("Nearby Hospitals vs Price per SqFt")
    plt.xlabel("Nearby Hospitals")
    plt.ylabel("Average Price per SqFt")

    save_plot("13_hospitals_vs_price.png")

    # ------------------------------------------------------
    # Q14. Property price by furnishing status
    # ------------------------------------------------------

    print("Q14: Furnishing status analysis")

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=df,
        x="Furnished_Status",
        y="Price_in_Lakhs"
    )

    plt.title("Property Price by Furnishing Status")
    plt.xlabel("Furnishing Status")
    plt.ylabel("Price (Lakhs)")

    save_plot("14_furnished_status_analysis.png")

    # ------------------------------------------------------
    # Q15. Price per sqft by facing direction
    # ------------------------------------------------------

    print("Q15: Facing direction analysis")

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=df,
        x="Facing",
        y="Price_per_SqFt"
    )

    plt.title("Price per SqFt by Facing Direction")
    plt.xlabel("Facing Direction")
    plt.ylabel("Price per SqFt")

    save_plot("15_facing_analysis.png")


# ==========================================================
# GROUP 4: INVESTMENT, AMENITIES AND OWNERSHIP
# QUESTIONS 16-20
# ==========================================================

def investment_analysis(df):

    print("\n========== INVESTMENT ANALYSIS ==========")

    # ------------------------------------------------------
    # Q16. Properties by owner type
    # ------------------------------------------------------

    print("Q16: Owner type analysis")

    owner_counts = df["Owner_Type"].value_counts()

    save_table(
        owner_counts,
        "owner_type_distribution.csv"
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=owner_counts.index,
        y=owner_counts.values
    )

    plt.title("Properties by Owner Type")
    plt.xlabel("Owner Type")
    plt.ylabel("Number of Properties")

    save_plot("16_owner_type_analysis.png")

    # ------------------------------------------------------
    # Q17. Availability status distribution
    # ------------------------------------------------------

    print("Q17: Availability status analysis")

    availability = (
        df["Availability_Status"].value_counts()
    )

    save_table(
        availability,
        "availability_distribution.csv"
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=availability.index,
        y=availability.values
    )

    plt.title("Property Availability Status")
    plt.xlabel("Availability Status")
    plt.ylabel("Number of Properties")

    save_plot("17_availability_analysis.png")

    # ------------------------------------------------------
    # Q18. Parking and property price
    # ------------------------------------------------------

    print("Q18: Parking space analysis")

    parking_price = (
        df.groupby("Parking_Space")
        ["Price_in_Lakhs"]
        .mean()
    )

    save_table(
        parking_price,
        "parking_vs_price.csv"
    )

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Parking_Space",
        y="Price_in_Lakhs"
    )

    plt.title("Parking Availability vs Property Price")
    plt.xlabel("Parking Space")
    plt.ylabel("Price (Lakhs)")

    save_plot("18_parking_analysis.png")

    # ------------------------------------------------------
    # Q19. Amenities vs price per sqft
    # ------------------------------------------------------

    print("Q19: Amenities analysis")

    amenities = df[
        ["Amenities", "Price_per_SqFt"]
    ].copy()

    amenities["Amenities"] = (
        amenities["Amenities"]
        .str.split(",")
    )

    amenities = amenities.explode("Amenities")

    amenities["Amenities"] = (
        amenities["Amenities"].str.strip()
    )

    amenity_price = (
        amenities.groupby("Amenities")
        .agg(
            Average_Price_Per_SqFt=(
                "Price_per_SqFt", "mean"
            ),
            Property_Count=(
                "Price_per_SqFt", "size"
            )
        )
        .sort_values(
            "Average_Price_Per_SqFt",
            ascending=False
        )
    )

    save_table(
        amenity_price,
        "amenities_vs_price.csv"
    )

    top_amenities = (
        amenity_price.head(15)
    )

    plt.figure(figsize=(11, 7))

    sns.barplot(
        x=top_amenities[
            "Average_Price_Per_SqFt"
        ].values,
        y=top_amenities.index
    )

    plt.title("Top 15 Amenities by Average Price per SqFt")
    plt.xlabel("Average Price per SqFt")

    save_plot("19_amenities_analysis.png")

    # ------------------------------------------------------
    # Q20. Transport accessibility vs price
    # ------------------------------------------------------

    print("Q20: Public transport analysis")

    transport_order = [
        "Low",
        "Medium",
        "High"
    ]

    transport_price = (
        df.groupby("Public_Transport_Accessibility")
        ["Price_per_SqFt"]
        .mean()
        .reindex(transport_order)
    )

    save_table(
        transport_price,
        "transport_vs_price.csv"
    )

    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=transport_price.index,
        y=transport_price.values
    )

    plt.title(
        "Transport Accessibility vs Price per SqFt"
    )

    plt.xlabel("Transport Accessibility")
    plt.ylabel("Average Price per SqFt")

    save_plot("20_transport_analysis.png")


# ==========================================================
# 5. MAIN EXECUTION
# ==========================================================

def main():

    df = load_data()

    price_and_size_analysis(df)

    location_analysis(df)

    feature_relationship_analysis(df)

    investment_analysis(df)

    print("\n========================================")
    print("ALL 20 EDA QUESTIONS COMPLETED!")
    print("========================================")

    print("\nCharts saved in:")
    print(FIGURE_DIR)

    print("\nAnalytical CSV files saved in:")
    print(EDA_REPORT_DIR)


if __name__ == "__main__":

    main()
