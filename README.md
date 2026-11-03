# Real Estate Investment Advisor

An academic machine-learning and Streamlit project for exploring Indian residential-property data, comparing a property's price per square foot with comparable-property benchmarks, and displaying an illustrative five-year price scenario.

> **Important:** This application is an educational decision-support demonstration, not financial advice, a verified property valuation, or a guarantee of investment returns. The classification model has limited test performance and should be treated as experimental. The five-year value is a formula-based scenario using an assumed 8% annual appreciation rate; it is not an independently validated market forecast.

## Project overview

The project uses a dataset of approximately 250,000 property records across 42 cities and 20 states (as represented in the current dashboard). It includes data preparation, exploratory data analysis, feature engineering, model training/evaluation, and a Streamlit interface.

### Main application pages

1. **Home Dashboard** – dataset summary, property-type summaries, city-level average prices, and sample records.
2. **Property Prediction** – accepts property details, compares the property's price per square foot with a comparable-property training benchmark, and displays the experimental classifier output separately.
3. **Market Analytics** – filters records by state and property type and visualizes city prices, BHK counts, price ranges, and price per square foot.
4. **Model Performance** – reports classification metrics, confusion matrix, and ROC curve. Formula-derived regression metrics are intentionally not presented as independent forecasting performance.

## Technology stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Joblib
- Matplotlib and Seaborn (analysis/evaluation plots)
- Streamlit

## Repository structure

```text
Real_Estate_Project/
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   ├── eda_results/
│   ├── evaluation/
│   ├── figures/
│   └── model_results/
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_classification.py
│   ├── train_regression.py
│   ├── evaluate_models.py
│   └── prediction.py
├── requirements.txt
└── README.md
```

The exact files included in a submission may vary depending on whether datasets and trained model artifacts are delivered separately due to file size.

## Setup

### 1. Clone or download the project

```bash
git clone <https://github.com/ritikraj77320/Real-Estate-Investment-Advisor.git>
cd Real_Estate_Project
```

If you received the project as a ZIP, extract it and open the extracted project folder in VS Code.

### 2. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Check required local assets

Before launching, confirm that these project assets are present at the paths expected by the code:

- `data/processed/india_housing_features.csv`
- `models/classification_best_model.joblib`
- `data/processed/investment_benchmarks_train.csv` (used by the current prediction workflow, if required by the checked-out `src/prediction.py`)

Large datasets and serialized model files may need to be supplied separately if they are not stored in Git. Do not commit secrets, private credentials, or local environment files.

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

Streamlit will print a local URL (normally `http://localhost:8501`) in the terminal.

## Model and interpretation notes

### Benchmark-based assessment

The primary property assessment compares the entered property's price per square foot with a median benchmark calculated from comparable training records. The benchmark is a relative affordability indicator only. It does not account for every factor that affects investment performance, such as legal title, construction quality, rental yield, liquidity, taxes, maintenance, financing, or future local demand.

### Experimental classification model

The current held-out test results are approximately:

| Metric | Result |
|---|---:|
| Accuracy | 49.98% |
| Precision | 49.89% |
| Recall | 52.63% |
| F1 score | 51.22% |
| ROC-AUC | 0.4997 |

These results indicate weak discriminative performance, close to random ranking. The classifier is shown for academic demonstration and must not be interpreted as a dependable investment recommendation.

### Five-year price scenario

The application illustrates compound appreciation using an assumed annual rate of 8% over five years:

`Scenario price = Current price × (1 + 0.08)^5`

This is a user-visible assumption-based calculation, not an independently trained or validated market forecast.

## Reproducing the workflow

The `src/` scripts cover preprocessing, exploratory analysis, feature engineering, classification/regression training, and model evaluation. Run scripts from the project root so their relative paths resolve correctly. Review each script's configured input/output paths before rerunning it; training may overwrite existing model and report artifacts.

## Data and model artifact note

Check the dataset's original source, license, and redistribution terms before publishing it. Trained model files and CSV datasets can be large; GitHub rejects individual files over its standard size limit. Use Git LFS or provide the large assets through an approved separate submission location, and document exactly where they can be obtained. Never upload `.env`, passwords, API keys, or other secrets.

## Limitations and future improvements

- Improve the classification target and feature signal, then evaluate using a validation set and a genuinely untouched test set.
- Validate the benchmark across cities/property types and report sample sizes for peer groups.
- Build a forecasting target from independently observed future sale/listing data rather than a formula derived from current price.
- Add data-source documentation, data validation, and model/version tracking.
- Add stronger handling for unseen categories and missing or inconsistent user inputs.

## Author / project links

- **GitHub repository:** `<https://github.com/ritikraj77320/Real-Estate-Investment-Advisor.git>`

Replace the placeholder above with the actual public repository URL before publishing this README.
