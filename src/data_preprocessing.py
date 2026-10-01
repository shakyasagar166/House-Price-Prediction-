"""
Data Preprocessing Module for House Price Prediction & Real Estate Valuation.
Handles loading raw property records, type conversions, missing value imputation, and categorical standardization.
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(filepath: str = "data/raw/dataset.csv") -> pd.DataFrame:
    """Loads raw property records dataset."""
    if not os.path.exists(filepath):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "raw", "dataset.csv")
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw housing dataset:
    1. Trims whitespace from column names.
    2. Converts numeric string columns (with possible blanks) to numeric.
    3. Imputes missing square footage and lot sizes with medians.
    4. Encodes binary amenities (has_pool, has_garden) to 1/0.
    """
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    # Numeric fields that may contain blank space strings
    numeric_fields = [
        'square_feet', 'bedrooms', 'bathrooms', 'year_built',
        'lot_size_sqft', 'garage_spaces', 'floors',
        'distance_to_city_center_km', 'school_rating', 'price'
    ]

    for col in numeric_fields:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col].astype(str).str.strip(), errors='coerce')
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    # Standardize binary columns
    for col in ['has_pool', 'has_garden']:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    return cleaned


def save_processed_data(df: pd.DataFrame, filepath: str = "data/processed/cleaned_data.csv") -> None:
    """Saves cleaned property records to processed directory."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[OK] Cleaned property dataset saved to: {filepath} ({len(df)} records)")


def run_preprocessing_pipeline(raw_path: str = "data/raw/dataset.csv",
                               processed_path: str = "data/processed/cleaned_data.csv") -> pd.DataFrame:
    """Executes the data preprocessing pipeline."""
    print("-> Loading raw housing dataset...")
    raw_df = load_raw_data(raw_path)
    print(f"   Raw shape: {raw_df.shape}")

    print("-> Cleaning and imputing property attributes...")
    cleaned_df = clean_data(raw_df)
    print(f"   Cleaned shape: {cleaned_df.shape}")

    save_processed_data(cleaned_df, processed_path)
    return cleaned_df


if __name__ == "__main__":
    run_preprocessing_pipeline()
