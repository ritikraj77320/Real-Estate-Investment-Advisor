
# ==========================================================
# REAL ESTATE INVESTMENT ADVISOR
# STREAMLIT WEB APPLICATION
# ==========================================================

import sys
import pandas as pd
import numpy as np
import streamlit as st

from pathlib import Path

# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "india_housing_features.csv"
)

REPORT_DIR = (
    PROJECT_ROOT / "reports" / "model_results"
)

EVALUATION_DIR = (
    PROJECT_ROOT / "reports" / "evaluation"
)

sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import predict_property


# ==========================================================
# 2. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 3. CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

.main-title {
    font-size: 32px;
    font-weight: 700;
}

.subtitle {
    color: #64748b;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 4. LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


try:

    df = load_dataset()

except Exception as e:

    st.error(f"Dataset loading error: {e}")

    st.stop()


# ==========================================================
# 5. HELPER FUNCTIONS
# ==========================================================

def get_options(column, default):

    if column not in df.columns:
        return [default]

    values = (
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    values = sorted(values)

    if not values:
        return [default]

    return values


def get_report(path):

    if path.exists():

        return pd.read_csv(path)

    return None


def format_lakhs(value):

    return f"₹{value:,.2f} Lakhs"


# ==========================================================
# 6. SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("🏠 Real Estate Advisor")

st.sidebar.caption(
    "AI-Powered Property Investment Analysis"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home Dashboard",
        "Property Prediction",
        "Market Analytics",
        "Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
    **Project Technologies**

    - Python
    - Pandas & NumPy
    - Scikit-learn
    - XGBoost
    - Streamlit
    - Matplotlib
    """
)

st.sidebar.caption(
    "Real Estate Investment Advisor | 2026"
)


# ==========================================================
# 7. HOME DASHBOARD
# ==========================================================

if page == "Home Dashboard":

    st.markdown(
        '<div class="main-title">'
        '🏠 Real Estate Investment Advisor'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered property investment analysis '
        'and five-year price estimation.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Dataset statistics

    total_properties = len(df)

    total_cities = (
        df["City"].nunique()
        if "City" in df.columns else 0
    )

    total_states = (
        df["State"].nunique()
        if "State" in df.columns else 0
    )

    average_price = (
        df["Price_in_Lakhs"].mean()
        if "Price_in_Lakhs" in df.columns else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Properties",
        f"{total_properties:,}"
    )

    col2.metric(
        "Cities Covered",
        total_cities
    )

    col3.metric(
        "States Covered",
        total_states
    )

    col4.metric(
        "Average Property Price",
        f"₹{average_price:.2f} L"
    )

    st.divider()

    st.subheader("📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Properties by Property Type")

        if "Property_Type" in df.columns:

            property_counts = (
                df["Property_Type"]
                .value_counts()
            )

            st.bar_chart(property_counts)

    with col2:

        st.markdown("#### Average Price by Property Type")

        if {
            "Property_Type",
            "Price_in_Lakhs"
        }.issubset(df.columns):

            avg_price_type = (
                df.groupby("Property_Type")
                ["Price_in_Lakhs"]
                .mean()
                .sort_values(ascending=False)
            )

            st.bar_chart(avg_price_type.rename("Average Price (₹ Lakhs)"))
            st.caption("Average property price is shown in lakh rupees.")

    st.divider()

    st.subheader("🌆 Top 10 Cities by Average Property Price")

    if {"City", "Price_in_Lakhs"}.issubset(df.columns):

        city_prices = (
            df.groupby("City")
            ["Price_in_Lakhs"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(city_prices.rename("Average Price (₹ Lakhs)"))
        st.caption("Average property price is shown in lakh rupees.")

    st.divider()

    st.subheader("📋 Sample Property Records")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.caption(
        "Dataset contains property information across "
        "Indian cities and states."
    )


# ==========================================================
# 8. PROPERTY PREDICTION
# ==========================================================

elif page == "Property Prediction":

    st.title("🔮 Property Investment Prediction")

    st.write(
        "Enter property information below to analyze "
        "its investment classification and estimated "
        "five-year price."
    )

    st.divider()

    # Dynamic dropdown values from dataset

    states = get_options("State", "West Bengal")

    cities = get_options("City", "Durgapur")

    localities = get_options("Locality", "Bidhannagar")

    property_types = get_options(
        "Property_Type", "Apartment"
    )

    furnishing_options = get_options(
        "Furnished_Status", "Semi-Furnished"
    )

    facing_options = get_options(
        "Facing", "East"
    )

    owner_options = get_options(
        "Owner_Type", "Builder"
    )

    availability_options = get_options(
        "Availability_Status", "Ready_to_Move"
    )

    transport_options = get_options(
        "Public_Transport_Accessibility", "High"
    )

    parking_options = get_options(
        "Parking_Space", "Yes"
    )

    security_options = get_options(
        "Security", "Yes"
    )

    # Extract available amenities

    amenities_options = []

    if "Amenities" in df.columns:

        for value in df["Amenities"].dropna().unique():

            if isinstance(value, str):

                amenities_options.extend(
                    [
                        item.strip()
                        for item in value.split(",")
                        if item.strip()
                    ]
                )

        amenities_options = sorted(
            set(amenities_options)
        )

    # Property form

    with st.form("property_prediction_form"):

        st.subheader("1. Property Location")

        col1, col2, col3 = st.columns(3)

        with col1:

            state = st.selectbox(
                "State",
                states
            )

        with col2:

            city = st.selectbox(
                "City",
                cities
            )

        with col3:

            locality = st.selectbox(
                "Locality",
                localities
            )

        st.divider()

        st.subheader("2. Property Details")

        col1, col2, col3 = st.columns(3)

        with col1:

            property_type = st.selectbox(
                "Property Type",
                property_types
            )

            bhk = st.number_input(
                "BHK",
                min_value=1,
                max_value=10,
                value=3
            )

        with col2:

            size = st.number_input(
                "Property Size (SqFt)",
                min_value=100,
                max_value=20000,
                value=1500
            )

            price = st.number_input(
                "Current Price (Lakhs)",
                min_value=0.1,
                max_value=100000.0,
                value=65.0,
                step=1.0
            )

        with col3:

            year_built = st.number_input(
                "Year Built",
                min_value=1900,
                max_value=2026,
                value=2020
            )

            furnished = st.selectbox(
                "Furnished Status",
                furnishing_options
            )

        st.divider()

        st.subheader("3. Building Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            floor_no = st.number_input(
                "Floor Number",
                min_value=0,
                max_value=100,
                value=3
            )

        with col2:

            total_floors = st.number_input(
                "Total Floors",
                min_value=1,
                max_value=100,
                value=5
            )

        with col3:

            facing = st.selectbox(
                "Facing",
                facing_options
            )

        st.divider()

        st.subheader("4. Nearby Facilities")

        col1, col2, col3 = st.columns(3)

        with col1:

            schools = st.number_input(
                "Nearby Schools",
                min_value=0,
                max_value=50,
                value=4
            )

        with col2:

            hospitals = st.number_input(
                "Nearby Hospitals",
                min_value=0,
                max_value=50,
                value=3
            )

        with col3:

            transport = st.selectbox(
                "Public Transport Accessibility",
                transport_options
            )

        st.divider()

        st.subheader("5. Amenities and Facilities")

        selected_amenities = st.multiselect(
            "Select Amenities",
            options=amenities_options,
            default=[]
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            parking = st.selectbox(
                "Parking Space",
                parking_options
            )

        with col2:

            security = st.selectbox(
                "Security",
                security_options
            )

        with col3:

            owner = st.selectbox(
                "Owner Type",
                owner_options
            )

        availability = st.selectbox(
            "Availability Status",
            availability_options
        )

        st.divider()

        submitted = st.form_submit_button(
            "🚀 Analyze Property",
            use_container_width=True
        )

    # Process prediction

    if submitted:

        if floor_no > total_floors:

            st.error(
                "Floor number cannot exceed total floors."
            )

        else:

            property_data = {

                "State": state,

                "City": city,

                "Locality": locality,

                "Property_Type": property_type,

                "BHK": bhk,

                "Size_in_SqFt": size,

                "Price_in_Lakhs": price,

                "Year_Built": year_built,

                "Furnished_Status": furnished,

                "Floor_No": floor_no,

                "Total_Floors": total_floors,

                "Nearby_Schools": schools,

                "Nearby_Hospitals": hospitals,

                "Public_Transport_Accessibility": transport,

                "Parking_Space": parking,

                "Security": security,

                "Amenities": selected_amenities,

                "Facing": facing,

                "Owner_Type": owner,

                "Availability_Status": availability

            }

            try:

                with st.spinner(
                    "Analyzing property and calculating peer benchmarks..."
                ):

                    results = predict_property(
                        property_data
                    )

                st.success(
                    "Property analysis completed successfully!"
                )

                st.divider()

                st.subheader(
                    "📊 Investment Analysis Results"
                )

                # Use the transparent peer-price comparison as the
                # primary assessment. The classifier is shown separately
                # as an experimental model output, not as financial advice.
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Benchmark-Based Assessment",
                    results["benchmark_rule_result"]
                )

                col2.metric(
                    "Property Price per SqFt",
                    f'₹{results["property_price_per_sqft"] * 100000:,.0f}'
                )

                col3.metric(
                    "Peer Median Price per SqFt",
                    f'₹{results["peer_median_price_per_sqft"] * 100000:,.0f}'
                )

                st.caption(
                    "The benchmark compares the asking price per square foot "
                    "with the training-data median for comparable properties. "
                    "It indicates relative affordability only; it does not "
                    "prove that a property will generate a good return."
                )

                with st.expander("Experimental ML classifier output"):
                    st.write(
                        "**Classifier prediction:**",
                        results["investment_status"]
                    )
                    st.write(
                        "**Model probability for the Good Investment class:**",
                        f'{results["good_investment_probability"]:.2f}%'
                    )
                    st.warning(
                        "The classifier currently has limited test performance "
                        "(F1 score about 0.51 and ROC-AUC about 0.50). "
                        "Treat this output as experimental, not as a reliable "
                        "investment recommendation."
                    )

                st.divider()

                st.subheader(
                    "📈 Five-Year Price Scenario"
                )

                st.caption(
                    "Illustrative estimate using an assumed "
                    f'{results["annual_appreciation_rate"]:.0f}% '
                    f'annual appreciation rate for '
                    f'{results["forecast_years"]} years.'
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Current Property Price",
                    format_lakhs(
                        results["current_price_lakhs"]
                    )
                )

                col2.metric(
                    "Estimated Price After 5 Years",
                    format_lakhs(
                        results["scenario_future_price_lakhs"]
                    )
                )

                col3.metric(
                    "Expected Scenario Appreciation",
                    format_lakhs(
                        results["expected_appreciation_lakhs"]
                    )
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Appreciation Percentage",
                    f'{results["expected_appreciation_percentage"]:.2f}%'
                )

                col2.metric(
                    "Assumed Annual Appreciation",
                    f'{results["annual_appreciation_rate"]:.0f}%'
                )

                # Current price versus assumption-based scenario
                comparison = pd.DataFrame({
                    "Price Type": [
                        "Current Price",
                        "Five-Year Scenario Price",
                    ],
                    "Price (Lakhs)": [
                        results["current_price_lakhs"],
                        results["scenario_future_price_lakhs"],
                    ],
                })

                st.bar_chart(
                    comparison.set_index("Price Type")
                )

                st.info(
                    "This five-year value is a formula-based scenario "
                    "that assumes constant annual appreciation. It is "
                    "not an independent machine-learning forecast, a "
                    "verified market forecast, or a guaranteed return."
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# ==========================================================
# 9. MARKET ANALYTICS
# ==========================================================

elif page == "Market Analytics":

    st.title("📊 Real Estate Market Analytics")

    st.write(
        "Explore property prices, locations, "
        "property types and market distributions."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        state_filter = st.selectbox(
            "Select State",
            ["All"] + get_options("State", "All")
        )

    with col2:

        type_filter = st.selectbox(
            "Select Property Type",
            ["All"] + get_options(
                "Property_Type", "All"
            )
        )

    filtered_df = df.copy()

    if state_filter != "All":

        filtered_df = filtered_df[
            filtered_df["State"] == state_filter
        ]

    if type_filter != "All":

        filtered_df = filtered_df[
            filtered_df["Property_Type"] == type_filter
        ]

    st.metric(
        "Filtered Properties",
        f"{len(filtered_df):,}"
    )

    st.divider()

    if filtered_df.empty:

        st.warning(
            "No properties match the selected filters. "
            "Please choose a different state or property type."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Average Price by City")

            if {"City", "Price_in_Lakhs"}.issubset(filtered_df.columns):

                city_price = (
                    filtered_df.groupby("City")["Price_in_Lakhs"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(15)
                    .rename("Average Price (₹ Lakhs)")
                )

                st.bar_chart(city_price)

                st.caption(
                    "Average asking price by city, expressed in lakh rupees."
                )

        with col2:

            st.subheader("BHK Distribution")

            if "BHK" in filtered_df.columns:

                bhk_distribution = (
                    filtered_df["BHK"]
                    .value_counts()
                    .sort_index()
                    .rename_axis("BHK")
                    .rename("Number of Properties")
                )

                st.bar_chart(bhk_distribution)

        st.divider()

        st.subheader("Price Distribution")

        if "Price_in_Lakhs" in filtered_df.columns:

            prices = pd.to_numeric(
                filtered_df["Price_in_Lakhs"],
                errors="coerce"
            ).dropna()

            if not prices.empty:

                counts, bin_edges = np.histogram(
                    prices,
                    bins=20
                )

                bin_labels = [
                    f"₹{bin_edges[i]:,.0f}–₹{bin_edges[i + 1]:,.0f} L"
                    for i in range(len(bin_edges) - 1)
                ]

                price_distribution = pd.Series(
                    counts,
                    index=bin_labels,
                    name="Number of Properties"
                )

                st.bar_chart(price_distribution)

                st.caption(
                    "Each bar represents the number of properties "
                    "within a price range. Prices are in lakh rupees."
                )

            else:

                st.info("No valid property prices are available for this filter.")

        st.divider()

        st.subheader("Property Price per SqFt")

        if {"City", "Price_per_SqFt"}.issubset(filtered_df.columns):

            city_price_sqft = (
                filtered_df.assign(
                    Price_per_SqFt=pd.to_numeric(
                        filtered_df["Price_per_SqFt"],
                        errors="coerce"
                    )
                )
                .groupby("City")["Price_per_SqFt"]
                .median()
                .dropna()
                .sort_values(ascending=False)
                .head(20)
                * 100000
            )

            if not city_price_sqft.empty:

                city_price_sqft = city_price_sqft.rename(
                    "Median Price (₹ per SqFt)"
                )

                st.line_chart(city_price_sqft)

                st.caption(
                    "Median price per square foot by city. "
                    "The source dataset stores this value in lakh rupees; "
                    "the chart converts it to rupees."
                )

            else:

                st.info("No valid price-per-square-foot values are available.")

        st.divider()

        st.subheader("Filtered Property Dataset")

        st.dataframe(
            filtered_df.head(500),
            use_container_width=True
        )

        st.caption(
            "The source dataset's Price_per_SqFt column is stored in "
            "lakh rupees per square foot (₹ Lakhs/SqFt)."
        )


# ==========================================================
# 10. MODEL PERFORMANCE
# ==========================================================

elif page == "Model Performance":

    st.title("🤖 Machine Learning Model Performance")

    st.write(
        "Performance evaluation of the trained "
        "classification and regression algorithms."
    )

    st.divider()

    # Classification metrics

    st.subheader("Classification Model")

    cls_path = (
        EVALUATION_DIR
        / "classification_evaluation_metrics.csv"
    )

    cls_metrics = get_report(cls_path)

    if cls_metrics is not None:

        row = cls_metrics.iloc[0]

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Accuracy",
            f'{row["Accuracy"] * 100:.2f}%'
        )

        col2.metric(
            "Precision",
            f'{row["Precision"] * 100:.2f}%'
        )

        col3.metric(
            "Recall",
            f'{row["Recall"] * 100:.2f}%'
        )

        col4.metric(
            "F1 Score",
            f'{row["F1_Score"] * 100:.2f}%'
        )

        col5.metric(
            "ROC-AUC",
            f'{row["ROC_AUC"]:.4f}'
        )

    else:

        st.warning(
            "Classification metrics not found."
        )

    # Classification graphs

    cm_path = (
        EVALUATION_DIR
        / "classification_confusion_matrix.png"
    )

    roc_path = (
        EVALUATION_DIR
        / "classification_roc_curve.png"
    )

    col1, col2 = st.columns(2)

    with col1:

        if cm_path.exists():

            st.image(
                str(cm_path),
                caption="Classification Confusion Matrix",
                use_container_width=True
            )

    with col2:

        if roc_path.exists():

            st.image(
                str(roc_path),
                caption="Classification ROC Curve",
                use_container_width=True
            )

    st.divider()

    st.divider()

    st.subheader("Regression / Five-Year Scenario")

    st.info(
        "The project's Future_Price_5Y target is generated from the "
        "current property price using an assumed 8% annual appreciation "
        "rate for five years. Because this target is formula-derived, "
        "regression metrics such as R², MAE and RMSE do not demonstrate "
        "independent market-forecasting ability. The dashboard therefore "
        "does not present those regression metrics as model performance. "
        "The Property Prediction page shows the calculation transparently "
        "as an illustrative scenario instead."
    )

    st.caption(
        "Classification metrics above evaluate the benchmark-based labels "
        "on the held-out test set. The label means that the property's "
        "price per square foot is at or below its comparable peer median."
    )


# 11. FOOTER
# ==========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">

    Real Estate Investment Advisor

    Built with Python, Machine Learning and Streamlit

    Academic Project | 2026

    </div>
    """,
    unsafe_allow_html=True
)
