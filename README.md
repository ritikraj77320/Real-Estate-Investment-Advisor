# 🏡 Real Estate Investment Advisor

### AI-Powered Property Investment Analysis and Market Insights for India

An end-to-end Machine Learning and Streamlit application designed to explore Indian residential real estate data, analyze property prices, compare price-per-square-foot benchmarks, and generate illustrative five-year property price scenarios.

<p align="center">

**🌐 Live Application:**  
https://property-investment-advisor.streamlit.app/

**💻 GitHub Repository:**  
https://github.com/ritikraj77320/Real-Estate-Investment-Advisor

</p>

---

## 🚀 Live Demo

Explore the deployed application:

👉 **[Real Estate Investment Advisor – Open Application](https://property-investment-advisor.streamlit.app/)**

The application is hosted using **Streamlit Community Cloud**, allowing users to access its interactive dashboards and property analysis features through a web browser.

---

## 📌 Project Overview

The Real Estate Investment Advisor is an academic Machine Learning project that focuses on analyzing residential property data in India.

The application works with approximately:

- **250,000** property records
- **42** cities
- **20** states

It combines data preprocessing, exploratory data analysis, feature engineering, machine learning model development, evaluation, and interactive visualization.

The main objective is to provide an interactive platform where users can explore property-market information and understand how a property's price per square foot compares with a comparable-property benchmark.

---

## ✨ Key Features

### 1. 🏠 Home Dashboard
- Displays overall dataset statistics.
- Shows total properties, cities, and states covered.
- Presents average property prices.
- Visualizes property distribution by property type.
- Displays average prices by property type.
- Provides an overview of the available dataset.

### 2. 📊 Property Prediction
- Accepts user-provided property information.
- Calculates property price per square foot.
- Compares the entered price with a comparable-property median benchmark.
- Displays a benchmark-based affordability assessment.
- Shows experimental Machine Learning classification output separately.
- Provides an illustrative five-year price scenario based on an assumed annual appreciation rate of 8%.

### 3. 📈 Market Analytics
- Filters property records by state and property type.
- Visualizes average property prices across cities.
- Displays BHK distribution.
- Presents property price distribution.
- Analyzes property prices per square foot.
- Provides filtered property data for exploration.

### 4. 🤖 Model Performance
- Displays classification model evaluation metrics.
- Presents a confusion matrix.
- Visualizes the ROC curve.
- Explains the limitations of the classification model.
- Avoids presenting formula-derived regression results as independently validated forecasting performance.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Model Serialization | Joblib |
| Data Visualization | Matplotlib, Seaborn |
| Web Application | Streamlit |
| Development Environment | Visual Studio Code |
| Version Control | Git and GitHub |
| Deployment | Streamlit Community Cloud |

---

## 📂 Project Structure

```text
Real-Estate-Investment-Advisor/
│
├── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── classification_best_model.joblib
│   ├── classification_metadata.json
│   ├── regression_best_model.joblib
│   ├── regression_metadata.json
│   ├── xgboost.joblib
│   └── other model artifacts
│
├── reports/
│   ├── eda_results/
│   ├── evaluation/
│   ├── figures/
│   └── model_results/
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_classification.py
│   ├── train_regression.py
│   ├── evaluate_models.py
│   └── prediction.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: Some large datasets and model artifacts may be excluded from GitHub or supplied separately. The deployed application requires the data and model files referenced by its runtime code.*

---

## ⚙️ Installation and Setup

Follow these instructions to run the project locally.

### Step 1: Clone the repository

```bash
git clone https://github.com/ritikraj77320/Real-Estate-Investment-Advisor.git
```

### Step 2: Navigate to the project directory

```bash
cd Real-Estate-Investment-Advisor
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```bat
.venv\Scripts\activate.bat
```

### Step 5: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 6: Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

### Required Application Assets

Make sure the following files are available at the paths expected by the application:

- `data/processed/india_housing_features.csv`
- `models/classification_best_model.joblib`
- `data/processed/investment_benchmarks_train.csv` (where required by the prediction workflow)

---

## 🧠 Machine Learning Workflow

The project follows an end-to-end Machine Learning workflow:

1. Data Collection and Loading
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Classification Model Training
6. Regression Model Training
7. Model Evaluation
8. Model Serialization
9. Prediction Integration
10. Streamlit Application Development
11. Cloud Deployment

### Models and Libraries

- Logistic Regression
- Random Forest
- XGBoost
- Linear Regression
- Scikit-learn
- Joblib

---

## 📉 Classification Model Performance

The current classification model produced the following held-out test results:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 49.98% |
| Precision | 49.89% |
| Recall | 52.63% |
| F1 Score | 51.22% |
| ROC-AUC | 0.4997 |

**Interpretation:** These results indicate weak classification performance, close to random discrimination. Therefore, the classifier is presented as an experimental academic component rather than a reliable investment recommendation engine.

---

## 📐 Property Price Scenario

The application includes an illustrative five-year property price calculation using an assumed annual appreciation rate of 8%.

The calculation uses compound appreciation:

```text
Future Price = Current Price × (1 + 0.08)^5
```

This is a formula-based scenario, not an independently trained or validated market forecast. Actual property appreciation can differ significantly depending on location, market demand, infrastructure, property condition, and economic conditions.

---

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

**Live URL:**  
https://property-investment-advisor.streamlit.app/

Deployment configuration:

- Repository: `ritikraj77320/Real-Estate-Investment-Advisor`
- Branch: `main`
- Main application file: `app.py`
- Hosting platform: Streamlit Community Cloud

Changes pushed to the connected GitHub repository can trigger application updates through Streamlit Community Cloud.

---

## ⚠️ Important Disclaimer

This project is developed for educational, academic, and demonstration purposes.

- It does not provide professional financial or investment advice.
- The benchmark-based assessment is a relative affordability comparison, not a verified property valuation.
- The classification model has limited predictive performance.
- The five-year price scenario uses an assumed 8% annual appreciation rate.
- Actual property prices and investment returns are subject to market risks and uncertainty.

Users should conduct independent research and consult qualified professionals before making real estate investment decisions.

---

## 🔮 Future Improvements

- Improve classification accuracy through better feature selection and target design.
- Develop a genuine property price forecasting model using independently observed future market data.
- Incorporate rental yield and return-on-investment analysis.
- Add more detailed city-wise and locality-wise market comparisons.
- Improve model validation and hyperparameter tuning.
- Integrate reliable and regularly updated property-market datasets.
- Add additional interactive visualizations and investment indicators.

---

## 👨‍💻 Author

**Ritik Raj**

### Project Links

- 🌐 **Live Application:** https://property-investment-advisor.streamlit.app/
- 💻 **GitHub Repository:** https://github.com/ritikraj77320/Real-Estate-Investment-Advisor

---

⭐ If you find this project interesting, feel free to explore the repository and the live application.