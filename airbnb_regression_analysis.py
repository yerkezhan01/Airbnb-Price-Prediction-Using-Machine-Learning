import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
from datetime import datetime
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def load_data(data_dir: str = "data/") -> tuple:
    """Loads raw CSV data."""
    print("Loading data...")
    listings = pd.read_csv(f"{data_dir}listings_gz.csv", low_memory=False)
    print(f"Loaded rows: {len(listings)}, columns: {len(listings.columns)}")
    
    reviews = pd.read_csv(f"{data_dir}reviews.csv", low_memory=False)
    try:
        neighbourhoods = pd.read_csv(f"{data_dir}neighbourhoods.csv")
    except FileNotFoundError:
        neighbourhoods = pd.DataFrame()
        
    return listings, reviews, neighbourhoods

def preprocess_data(listings: pd.DataFrame) -> pd.DataFrame:
    """Cleans prices, removes duplicates and outliers."""
    print("\nData preprocessing...")
    # Convert price to numeric
    if listings['price'].dtype == 'O':
        listings['price_numeric'] = listings['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    else:
        listings['price_numeric'] = listings['price']
    
    listings_clean = listings.dropna(subset=['price_numeric']).copy()
    listings_clean = listings_clean[listings_clean['price_numeric'] > 0]
    
    # Remove duplicates
    listings_clean.drop_duplicates(inplace=True)
    print(f"Rows after removing missing prices and duplicates: {len(listings_clean)}")
    
    # Outlier treatment (IQR method factor 2.0)
    def remove_outliers_iqr(df: pd.DataFrame, cols: list, factor: float = 2.0) -> pd.DataFrame:
        keep = np.ones(len(df), dtype=bool)
        for col in cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower = Q1 - factor * IQR
                    upper = Q3 + factor * IQR
                    col_keep = df[col].isna() | ((df[col] >= lower) & (df[col] <= upper))
                    keep = keep & col_keep
        return df[keep]
    
    numeric_cols_for_outliers = [
        "price_numeric", "accommodates", "bedrooms", "beds", 
        "bathrooms", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count",
        "availability_365"
    ]
    print("Applying IQR method for outlier removal...")
    listings_clean = remove_outliers_iqr(listings_clean, numeric_cols_for_outliers, factor=2.0)
    print(f"Rows after outlier treatment: {len(listings_clean)}")
    
    return listings_clean

def perform_eda(listings_clean: pd.DataFrame, plots_dir: str = "results/plots/"):
    """Performs Exploratory Data Analysis and saves plots."""
    print("\nExploratory Data Analysis...")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 3.1 Price distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(listings_clean['price_numeric'], bins=50, color='steelblue', ax=axes[0])
    axes[0].set_title('Price Distribution')
    axes[0].set_xlabel('Price (EUR)')
    sns.histplot(np.log(listings_clean['price_numeric']), bins=50, color='steelblue', ax=axes[1])
    axes[1].set_title('Log Price Distribution')
    axes[1].set_xlabel('log(Price)')
    plt.savefig(f"{plots_dir}01_price_distribution.png")
    plt.close()
    
    # 3.2 Price by room type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='room_type', y='price_numeric', data=listings_clean, color='lightblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Price by Room Type')
    plt.ylabel('Price (EUR)')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}02_price_by_room_type.png")
    plt.close()
    
    # 3.3 Price by neighbourhood
    top_neigh = listings_clean['neighbourhood_cleansed'].value_counts().nlargest(10).index
    plt.figure(figsize=(10, 8))
    sns.boxplot(y='neighbourhood_cleansed', x='price_numeric', 
                data=listings_clean[listings_clean['neighbourhood_cleansed'].isin(top_neigh)], 
                order=top_neigh, color='lightgreen')
    plt.title('Price by Neighbourhood (Top 10)')
    plt.xlabel('Price (EUR)')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}03_price_by_neighbourhood.png")
    plt.close()
    
    # 3.4 & 3.5 Correlation Matrix
    num_vars = listings_clean[["price_numeric", "accommodates", "bedrooms", "beds", 
             "minimum_nights", "number_of_reviews", "reviews_per_month",
             "calculated_host_listings_count", "availability_365",
             "review_scores_rating"]].copy()
    if "bathrooms" in listings_clean.columns:
        num_vars["bathrooms"] = listings_clean["bathrooms"]
    
    corr_matrix = num_vars.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}04_correlation_matrix.png")
    plt.close()
    
    # 3.6 Scatterplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.regplot(x='accommodates', y='price_numeric', data=listings_clean, ax=axes[0,0], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    axes[0,0].set_title('Price vs Accommodates')
    sns.regplot(x='bedrooms', y='price_numeric', data=listings_clean, ax=axes[0,1], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    axes[0,1].set_title('Price vs Bedrooms')
    sns.regplot(x='number_of_reviews', y='price_numeric', data=listings_clean, ax=axes[1,0], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    axes[1,0].set_title('Price vs Number of Reviews')
    sns.regplot(x='review_scores_rating', y='price_numeric', data=listings_clean, ax=axes[1,1], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    axes[1,1].set_title('Price vs Rating')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}05_scatterplots.png")
    plt.close()

def hypothesis_testing(listings_clean: pd.DataFrame):
    """Performs statistical hypothesis testing."""
    print("\nPerforming Hypothesis Testing...")
    if 'host_is_superhost' in listings_clean.columns:
        superhosts = listings_clean[listings_clean['host_is_superhost'] == 't']['price_numeric'].dropna()
        regular_hosts = listings_clean[listings_clean['host_is_superhost'] == 'f']['price_numeric'].dropna()
        if len(superhosts) > 0 and len(regular_hosts) > 0:
            t_stat, p_val = stats.ttest_ind(superhosts, regular_hosts, equal_var=False)
            print(f"Two-sample t-test: Price by Superhost Status -> t={t_stat:.4f}, p-value={p_val:.4g}")
            if p_val < 0.05:
                print("Result: Reject H0. There is a significant difference in prices between superhosts and regular hosts.")
            else:
                print("Result: Fail to reject H0. No significant difference in prices.")

def process_reviews(listings_clean: pd.DataFrame, reviews: pd.DataFrame, plots_dir: str = "results/plots/") -> pd.DataFrame:
    """Processes review data for time series analysis and feature engineering."""
    print("\nProcessing review data...")
    reviews['date'] = pd.to_datetime(reviews['date'])
    current_date = reviews['date'].max()
    
    reviews['days_since'] = (current_date - reviews['date']).dt.days
    reviews_agg = reviews.groupby('listing_id').agg(
        total_reviews_count=('date', 'count'),
        days_since_last_review=('days_since', 'min'),
        reviews_last_30_days=('days_since', lambda x: (x <= 30).sum()),
        reviews_last_90_days=('days_since', lambda x: (x <= 90).sum()),
        reviews_last_180_days=('days_since', lambda x: (x <= 180).sum())
    ).reset_index()
    
    reviews_agg['has_recent_reviews'] = reviews_agg['reviews_last_30_days'] > 0
    reviews_agg['review_activity_score'] = (reviews_agg['reviews_last_30_days']*3 + 
                                            reviews_agg['reviews_last_90_days']*2 + 
                                            reviews_agg['reviews_last_180_days']) / 6.0
    
    # Time Series Analysis
    print("\nPerforming Time Series Analysis on Review Activity...")
    reviews['year_month'] = reviews['date'].dt.to_period('M')
    reviews_ts = reviews.groupby('year_month').size().reset_index(name='review_count')
    reviews_ts['date_num'] = reviews_ts['year_month'].dt.to_timestamp()
    reviews_ts = reviews_ts[(reviews_ts['date_num'] >= '2015-01-01') & 
                            (reviews_ts['date_num'] < pd.Timestamp(datetime.now().replace(day=1)))]
    
    if len(reviews_ts) > 0:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='date_num', y='review_count', data=reviews_ts, color='steelblue', linewidth=2)
        x_num = reviews_ts['date_num'].map(datetime.toordinal)
        z = np.polyfit(x_num, reviews_ts['review_count'], 3)
        p = np.poly1d(z)
        plt.plot(reviews_ts['date_num'], p(x_num), color='red', linestyle='--')
        plt.title('Time Series Analysis: Review Activity Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Reviews')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}15_time_series_reviews.png")
        plt.close()
    
    # Merge
    return listings_clean.merge(reviews_agg, left_on='id', right_on='listing_id', how='left')

def engineer_features(listings_clean: pd.DataFrame) -> pd.DataFrame:
    """Engineers complex features for modeling."""
    print("\nFeature engineering...")
    if 'host_since' in listings_clean.columns:
        listings_clean['host_since_date'] = pd.to_datetime(listings_clean['host_since'])
        listings_clean['host_experience_years'] = (listings_clean['host_since_date'].max() - listings_clean['host_since_date']).dt.days / 365.25
    
    if 'amenities' in listings_clean.columns:
        def count_amenities(x):
            if pd.isna(x): return 0
            try: return len(ast.literal_eval(x))
            except: return len(str(x).split(','))
        listings_clean['amenities_count'] = listings_clean['amenities'].apply(count_amenities)
    
    if 'latitude' in listings_clean.columns and 'longitude' in listings_clean.columns:
        listings_clean['distance_to_center'] = np.sqrt(
            (listings_clean['latitude'] - 52.3676)**2 + 
            (listings_clean['longitude'] - 4.9041)**2
        ) * 111.0
    
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in listings_clean.columns:
            listings_clean[col+'_num'] = listings_clean[col].astype(str).str.replace('%','').astype(float) / 100.0
    
    if 'first_review' in listings_clean.columns:
        listings_clean['days_since_first_review'] = (pd.to_datetime('today') - pd.to_datetime(listings_clean['first_review'])).dt.days
        listings_clean['review_density'] = listings_clean['number_of_reviews'] / listings_clean['days_since_first_review'].clip(lower=1)
    
    listings_clean['price_per_person'] = listings_clean['price_numeric'] / listings_clean['accommodates'].clip(lower=1)
    listings_clean['price_per_bedroom'] = listings_clean['price_numeric'] / listings_clean['bedrooms'].clip(lower=1)
    listings_clean['beds_per_person'] = listings_clean['beds'] / listings_clean['accommodates'].clip(lower=1)
    
    return listings_clean

def prepare_modeling_data(listings_clean: pd.DataFrame) -> pd.DataFrame:
    """Selects columns, handles missing values, and prepares data for modeling."""
    print("\nPreparing data for modeling...")
    cols_to_keep = [
        'price_numeric', 'room_type', 'neighbourhood_cleansed', 'property_type',
        'host_is_superhost', 'instant_bookable', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count',
        'availability_365', 'availability_30', 'availability_60', 'availability_90',
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value',
        'total_reviews_count', 'days_since_last_review', 'reviews_last_30_days', 'reviews_last_90_days',
        'reviews_last_180_days', 'has_recent_reviews', 'review_activity_score', 'host_experience_years',
        'amenities_count', 'distance_to_center', 'host_response_rate_num', 'host_acceptance_rate_num',
        'days_since_first_review', 'review_density', 'price_per_person', 'price_per_bedroom', 'beds_per_person'
    ]
    cols_to_keep = [c for c in cols_to_keep if c in listings_clean.columns]
    
    model_data = listings_clean.dropna(subset=['accommodates', 'bedrooms', 'beds', 'room_type', 'neighbourhood_cleansed']).copy()
    model_data = model_data[cols_to_keep]
    model_data.rename(columns={'price_numeric':'price'}, inplace=True)
    
    for col in model_data.select_dtypes(include=np.number).columns:
        if col in ['total_reviews_count', 'reviews_last_30_days', 'reviews_last_90_days', 'reviews_last_180_days', 'review_activity_score', 'review_density']:
            model_data[col] = model_data[col].fillna(0)
        else:
            model_data[col] = model_data[col].fillna(model_data[col].median())
    
    if 'has_recent_reviews' in model_data.columns:
        model_data['has_recent_reviews'] = model_data['has_recent_reviews'].fillna(False).astype(int)
    if 'host_is_superhost' in model_data.columns:
        model_data['host_is_superhost'] = (model_data['host_is_superhost'] == 't').astype(int)
    if 'instant_bookable' in model_data.columns:
        model_data['instant_bookable'] = (model_data['instant_bookable'] == 't').astype(int)
    
    model_data.columns = [c.replace(' ', '_').replace('-', '_').replace('/','_') for c in model_data.columns]
    model_data.to_csv('data/model_data.csv', index=False)
    
    return model_data

def build_models(model_data: pd.DataFrame, results_dir: str = "results/"):
    """Trains regression models and evaluates them."""
    print("\nBuilding regression models...")
    model_data_complete = model_data.dropna().copy()
    y = model_data_complete['price']
    y_log = np.log(y)
    
    def fit_ols(formula, data):
        return smf.ols(formula, data=data).fit()
    
    numeric_features = [
        'accommodates', 'bedrooms', 'beds', 'minimum_nights', 'number_of_reviews', 
        'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 
        'availability_30', 'availability_60', 'availability_90', 'review_scores_rating'
    ]
    
    optional_features = [
        'bathrooms', 'review_scores_accuracy', 'review_scores_cleanliness', 'host_experience_years',
        'amenities_count', 'distance_to_center', 'host_response_rate_num', 'review_density'
    ]
    numeric_features.extend([f for f in optional_features if f in model_data_complete.columns])
    
    base_formula = " ~ " + " + ".join(numeric_features) + " + host_is_superhost + instant_bookable + total_reviews_count + review_activity_score"
    
    # Statsmodels OLS
    print("--- Model 1: Extended basic model ---")
    model1 = fit_ols(f"price{base_formula}", model_data_complete)
    print(f"R-squared: {model1.rsquared:.4f}")
    
    print("\n--- Model 2: With categorical variables and all features ---")
    formula_2 = f"price{base_formula} + C(room_type) + C(neighbourhood_cleansed)"
    model2 = fit_ols(formula_2, model_data_complete)
    print(f"R-squared: {model2.rsquared:.4f}")
    
    print("\n--- Model 3: Log transformation with all features ---")
    formula_3 = f"np.log(price){base_formula} + C(room_type) + C(neighbourhood_cleansed) + I(accommodates**2) + I(bedrooms**2)"
    model3 = fit_ols(formula_3, model_data_complete)
    print(f"R-squared: {model3.rsquared:.4f}")
    
    print("\n--- Model 4: With interactions, polynomials and all features ---")
    formula_4 = formula_3 + " + accommodates:bedrooms"
    if 'bathrooms' in model_data_complete.columns:
        formula_4 += " + accommodates:bathrooms + bedrooms:bathrooms + I(bathrooms**2)"
    model4 = fit_ols(formula_4, model_data_complete)
    print(f"R-squared: {model4.rsquared:.4f}")
    
    print("\n--- Model 5: Stepwise regression (Proxy) ---")
    model5 = fit_ols(formula_4, model_data_complete)
    print(f"R-squared: {model5.rsquared:.4f}")
    
    # Scikit-learn Data Prep
    X_encoded = pd.get_dummies(model_data_complete.drop(columns=['price', 'property_type'], errors='ignore'), drop_first=True)
    X_encoded = X_encoded.select_dtypes(include=[np.number, bool]).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_log, test_size=0.2, random_state=123)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ridge & Lasso
    print("\n--- Model 6 & 7: Ridge & Lasso ---")
    ridge = RidgeCV(cv=5).fit(X_train_scaled, y_train)
    lasso = LassoCV(cv=5).fit(X_train_scaled, y_train)
    r2_ridge = ridge.score(X_test_scaled, y_test)
    r2_lasso = lasso.score(X_test_scaled, y_test)
    print(f"Ridge R-squared (Test): {r2_ridge:.4f}")
    print(f"Lasso R-squared (Test): {r2_lasso:.4f}")
    
    # Random Forest
    print("\n--- Model 8: Random Forest (Advanced Model) ---")
    rf = RandomForestRegressor(n_estimators=100, random_state=123, max_depth=5, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    r2_rf = r2_score(y_test, rf_preds)
    print(f"Random Forest R-squared (Test): {r2_rf:.4f}")
    
    joblib.dump(ridge, f"{results_dir}ridge_model.pkl")
    joblib.dump(lasso, f"{results_dir}lasso_model.pkl")
    joblib.dump(rf, f"{results_dir}rf_model.pkl")
    
    models_dict = {
        'model1': model1, 'model2': model2, 'model3': model3, 'model4': model4, 'model5': model5,
        'r2_ridge': r2_ridge, 'r2_lasso': r2_lasso, 'r2_rf': r2_rf,
        'rf': rf, 'X_encoded': X_encoded, 'y_log': y_log, 'y': y
    }
    return models_dict

def model_diagnostics_and_plots(models_dict: dict, plots_dir: str = "results/plots/", results_dir: str = "results/"):
    """Generates diagnostics plots and cross-validation."""
    print("\nGenerating model diagnostics...")
    best_ols = models_dict['model5']
    preds = best_ols.fittedvalues
    resids = best_ols.resid
    std_resids = best_ols.get_influence().resid_studentized_internal
    
    # Residuals
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.scatterplot(x=preds, y=resids, ax=axes[0,0], alpha=0.5)
    axes[0,0].axhline(0, color='red', linestyle='--')
    axes[0,0].set_title('Residuals vs Fitted')
    axes[0,0].set_xlabel('Fitted values')
    axes[0,0].set_ylabel('Residuals')
    
    sm.qqplot(resids, line='45', ax=axes[0,1])
    axes[0,1].set_title('Q-Q Plot')
    
    sns.scatterplot(x=preds, y=np.sqrt(np.abs(std_resids)), ax=axes[1,0], alpha=0.5)
    axes[1,0].set_title('Scale-Location')
    axes[1,0].set_xlabel('Fitted values')
    axes[1,0].set_ylabel('sqrt(|Std. Residuals|)')
    
    sns.scatterplot(x=range(len(resids)), y=resids, ax=axes[1,1], alpha=0.5)
    axes[1,1].axhline(0, color='red', linestyle='--')
    axes[1,1].set_title('Residuals vs Order')
    axes[1,1].set_xlabel('Observation Order')
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}06_residuals_diagnostics.png")
    plt.close()
    
    # RF Importance
    rf, X_encoded = models_dict['rf'], models_dict['X_encoded']
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10,8))
    plt.title('Random Forest Top 20 Feature Importance')
    plt.barh(range(len(indices)), importances[indices], color='steelblue', align='center')
    plt.yticks(range(len(indices)), [X_encoded.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}16_rf_importance.png")
    plt.close()
    
    # Model Comparison
    print("Saving model comparison...")
    model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Ridge', 'Lasso', 'RF']
    r2s = [models_dict[f'model{i}'].rsquared for i in range(1,6)] + [models_dict['r2_ridge'], models_dict['r2_lasso'], models_dict['r2_rf']]
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=model_names, y=r2s, color='steelblue')
    plt.axhline(0.7, color='red', linestyle='--', lw=2)
    plt.title('Model Comparison by R²')
    plt.ylabel('R²')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}11_model_comparison.png")
    plt.close()
    
    comp_df = pd.DataFrame({
        'Model': ['Model 1 (extended basic)', 'Model 2 (+ categorical)', 'Model 3 (log transform)', 
                  'Model 4 (+ interactions)', 'Model 5 (stepwise)', 'Model 6 (Ridge)', 
                  'Model 7 (Lasso)', 'Model 8 (Random Forest)'],
        'R_squared': r2s
    })
    comp_df.to_csv(f"{results_dir}model_comparison.csv", index=False)
    
    # Cross Validation
    print("\nCross-validation (Ridge)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    cv_errors = []
    y_log = models_dict['y_log']
    
    for train_idx, test_idx in kf.split(X_encoded):
        X_tr, X_te = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
        y_tr, y_te = y_log.iloc[train_idx], y_log.iloc[test_idx]
        
        scaler_cv = StandardScaler()
        X_tr_scaled = scaler_cv.fit_transform(X_tr)
        X_te_scaled = scaler_cv.transform(X_te)
        
        ridge_cv = RidgeCV().fit(X_tr_scaled, y_tr)
        preds_cv = np.exp(ridge_cv.predict(X_te_scaled))
        rmse = np.sqrt(mean_squared_error(np.exp(y_te), preds_cv))
        cv_errors.append(rmse)
    
    print(f"Mean RMSE: {np.mean(cv_errors):.2f}, Std: {np.std(cv_errors):.2f}")
    pd.DataFrame({'Fold': range(1,6), 'RMSE': cv_errors}).to_csv(f"{results_dir}cv_results.csv", index=False)

def main():
    os.makedirs("results/plots", exist_ok=True)
    
    # 1. Load Data
    listings, reviews, neighbourhoods = load_data()
    
    # 2. Preprocess
    listings_clean = preprocess_data(listings)
    
    # 3. EDA
    perform_eda(listings_clean)
    
    # 4. Hypothesis Testing
    hypothesis_testing(listings_clean)
    
    # 5. Process Reviews
    listings_clean = process_reviews(listings_clean, reviews)
    
    # 6. Feature Engineering
    listings_clean = engineer_features(listings_clean)
    
    # 7. Prepare Modeling Data
    model_data = prepare_modeling_data(listings_clean)
    
    # 8. Build Models
    models_dict = build_models(model_data)
    
    # 9. Diagnostics & Cross-validation
    model_diagnostics_and_plots(models_dict)
    
    print("\nPython analysis completed successfully!")
    print("All models, plots, and CSV reports have been updated in the results/ folder.")

if __name__ == "__main__":
    main()
