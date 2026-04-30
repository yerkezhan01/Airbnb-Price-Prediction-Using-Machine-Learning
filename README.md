# SDS 311: Airbnb Pricing Regression Analysis

Regression analysis project for Airbnb pricing in Amsterdam.

https://ru.airbnb.com/amsterdam-netherlands/stays?_set_bev_on_new_domain=1777534742_EAMDJmOGQ2NDBlYm&set_everest_cookie_on_new_domain=1777534742.EANWVmYmYwMmE3YmY2NT.bONyNvz9JbiIQckxjjcLqqBnjkr74t-Hh-9fJ41jDt4

## Project Structure

```
├── data/                    # Source data
│   ├── listings_gz.csv     # Detailed listing information
│   ├── listings.csv        # Brief listing information
│   ├── reviews.csv         # Reviews
│   ├── reviews_gz.csv      # Detailed reviews
│   └── neighbourhoods.csv  # Neighborhoods
├── results/                 # Analysis results
│   ├── plots/              # EDA and diagnostic plots
│   ├── model_comparison.csv
│   ├── cv_results.csv
│   └── final_model_coefficients.csv
├── airbnb_regression_analysis.py  # Main analysis script
└── README.md
```

## Description

The project includes:

1. **EDA (Exploratory Data Analysis)** - exploratory data analysis
2. **Modeling** - building multiple linear regression models
3. **Diagnostics** - checking models for multicollinearity, outliers, residual normality
4. **Model Selection** - comparing models using F-tests and cross-validation

## Results

### Best Model

- **Model 8 (Random Forest)**: R² = 0.98
- **Model 6 (Ridge)**: R² = 0.92
- **Model 7 (Lasso)**: R² = 0.92

### Quality Metrics

- Mean RMSE for Ridge (5-fold CV): ~43 EUR
- Random Forest R² (Test Set): ~0.98

### Improvements Applied

- IQR method for outlier treatment
- Features from review data
- Merged data from multiple sources
- Feature engineering (host experience, amenities count, distance to center, etc.)
- Polynomial features and interactions
- Ridge and Lasso regularization

## Running

### Option 1: Locally (without Docker)

1. Install required Python packages:
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn joblib
```

2. Run the main script:
```bash
python airbnb_regression_analysis.py
```

### Option 2: With Docker (recommended for reproducibility)

1. Make sure Docker is installed:
```bash
docker --version
```

2. Run analysis in container:
```bash
docker-compose up python-analysis
```

Or build and run manually:
```bash
docker build -t sds311-python .
docker run -v $(pwd)/results:/app/results sds311-python
```

### Future Services

The project is prepared for expansion:
- **Next.js application**: `docker-compose --profile web up nextjs-app`

