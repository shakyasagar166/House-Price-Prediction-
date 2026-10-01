"""
Model Training and Property Valuation Inference Module for House Price Prediction.
Trains regression models, evaluates R2, MAE, RMSE metrics, and serializes production pipelines.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from src.data_preprocessing import run_preprocessing_pipeline
from src.feature_engineering import engineer_features, get_preprocessor_pipeline, get_feature_lists
from src.visualization import plot_feature_importance, plot_model_residuals


def prepare_datasets(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Prepares train and test splits with feature engineering applied."""
    fe_df = engineer_features(df)
    num_cols, cat_cols = get_feature_lists()
    all_features = num_cols + cat_cols

    X = fe_df[all_features]
    y = fe_df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, num_cols, cat_cols


def train_and_compare_models(X_train, y_train, num_cols, cat_cols):
    """Trains Ridge, Random Forest, and Gradient Boosting pipelines."""
    models = {
        'Ridge Regression': Ridge(alpha=10.0, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=140, learning_rate=0.08, max_depth=4, random_state=42)
    }

    results = {}
    best_name = None
    best_score = -1.0
    best_pipeline = None

    for name, reg in models.items():
        preprocessor = get_preprocessor_pipeline(cat_cols, num_cols)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', reg)
        ])

        # 5-Fold Cross Validation on R2 Score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        mean_cv_r2 = float(np.mean(cv_scores))

        pipeline.fit(X_train, y_train)
        results[name] = {
            'pipeline': pipeline,
            'cv_r2': mean_cv_r2
        }
        print(f"   [{name}] 5-Fold Cross-Val R2 Score: {mean_cv_r2:.4f}")

        if mean_cv_r2 > best_score:
            best_score = mean_cv_r2
            best_name = name
            best_pipeline = pipeline

    print(f"\n-> Best Valuation Model: {best_name} (Cross-Val R2: {best_score:.4f})")
    return best_name, best_pipeline, results


def evaluate_pipeline(pipeline, X_test, y_test):
    """Evaluates the regression pipeline on unseen test properties."""
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    metrics = {
        'R2 Score': float(r2),
        'MAE ($)': float(mae),
        'RMSE ($)': float(rmse),
        'MAPE (%)': float(mape)
    }

    print("\n================ TEST SET VALUATION PERFORMANCE ================")
    print(f"   R2 Score (Variance Explained) : {r2:.4f}")
    print(f"   Mean Absolute Error (MAE)     : ${mae:,.2f}")
    print(f"   Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"   Mean Absolute Percentage Error: {mape:.2f}%")
    print("==================================================================")

    return metrics, y_pred


def extract_feature_importances(pipeline, num_cols, cat_cols):
    """Extracts transformed feature names and importance weights."""
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']

    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + encoded_cat_names

    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
    elif hasattr(regressor, 'coef_'):
        importances = np.abs(regressor.coef_)
    else:
        importances = np.ones(len(all_feature_names)) / len(all_feature_names)

    return all_feature_names, importances


def save_model(pipeline, filepath: str = "models/model.pkl"):
    """Saves serialized production pipeline to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"[OK] Production valuation model saved to: {filepath}")


def load_model(filepath: str = "models/model.pkl"):
    """Loads serialized pipeline from disk."""
    if not os.path.exists(filepath):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "models", "model.pkl")
    return joblib.load(filepath)


def predict_house_price(model_pipeline, property_dict: dict) -> dict:
    """
    Inference helper for web dashboard:
    Accepts raw property features, engineers attributes, and returns estimated price,
    confidence range, price per sqft, and investment grade.
    """
    df = pd.DataFrame([property_dict])
    fe_df = engineer_features(df)

    num_cols, cat_cols = get_feature_lists()
    X = fe_df[num_cols + cat_cols]

    est_price = float(model_pipeline.predict(X)[0])
    est_price = max(80000.0, est_price)

    # 5% confidence interval
    lower_bound = est_price * 0.95
    upper_bound = est_price * 1.05

    sqft = float(property_dict.get('square_feet', 1500))
    price_per_sqft = est_price / max(1.0, sqft)

    # Investment Grade analysis
    if property_dict.get('neighborhood') in ['Downtown', 'Harbor View', 'Lakeside'] and property_dict.get('school_rating', 5) >= 7:
        grade = "Prime Investment (Grade A+)"
        advice = "High appreciation potential driven by top school district and prime waterfront / urban accessibility."
    elif property_dict.get('crime_rate_index') == 'Low':
        grade = "Solid Growth (Grade A)"
        advice = "Stable family residential area with low crime and consistent rental demand."
    else:
        grade = "Moderate Return (Grade B)"
        advice = "Attractive entry price point with steady yield potential."

    return {
        'estimated_price': est_price,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'price_per_sqft': price_per_sqft,
        'investment_grade': grade,
        'market_advice': advice
    }


def run_training_pipeline():
    """End-to-end execution: preprocess, train, evaluate, and save."""
    print("-> Step 1: Preprocessing raw housing records...")
    cleaned_df = run_preprocessing_pipeline()

    print("\n-> Step 2: Extracting valuation features & splitting train/test...")
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_datasets(cleaned_df)
    print(f"   Training properties: {len(X_train)} | Test properties: {len(X_test)}")

    print("\n-> Step 3: Benchmarking regression algorithms...")
    best_name, best_pipeline, _ = train_and_compare_models(X_train, y_train, num_cols, cat_cols)

    print("\n-> Step 4: Evaluating on test properties...")
    metrics, y_pred = evaluate_pipeline(best_pipeline, X_test, y_test)

    print("\n-> Step 5: Serializing production valuation pipeline...")
    save_model(best_pipeline, "models/model.pkl")

    return best_pipeline, metrics


if __name__ == "__main__":
    run_training_pipeline()
