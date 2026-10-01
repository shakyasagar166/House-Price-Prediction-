"""
Feature Engineering Module for House Price Prediction & Real Estate Valuation.
Constructs real-estate valuation indices, amenity scores, structural ratios, and ColumnTransformer pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs high-signal property valuation features:
    1. property_age: 2026 - year_built
    2. total_amenity_score: has_pool + has_garden + garage_spaces
    3. bed_to_bath_ratio: bedrooms / (bathrooms + 0.1)
    4. lot_coverage_ratio: square_feet / (lot_size_sqft + 1.0)
    5. location_desirability_index: school_rating / (distance_to_city_center_km + 1.0)
    6. is_luxury_segment: flag for luxury villas or large square footage (>3200)
    """
    feat_df = df.copy()

    # 1. Structural Age
    feat_df['property_age'] = 2026 - feat_df['year_built']

    # 2. Amenity Score
    feat_df['total_amenity_score'] = (
        feat_df['has_pool'].astype(int) +
        feat_df['has_garden'].astype(int) +
        feat_df['garage_spaces'].astype(int)
    )

    # 3. Spatial & Ratios
    feat_df['bed_to_bath_ratio'] = feat_df['bedrooms'] / (feat_df['bathrooms'] + 0.1)
    feat_df['lot_coverage_ratio'] = feat_df['square_feet'] / (feat_df['lot_size_sqft'] + 1.0)

    # 4. Location Desirability Interaction
    feat_df['location_desirability_index'] = feat_df['school_rating'] / (feat_df['distance_to_city_center_km'] + 1.0)

    # 5. Luxury Segment Flag
    is_large = (feat_df['square_feet'] >= 3200).astype(int)
    is_villa = (feat_df['property_type'] == 'Luxury Villa').astype(int)
    feat_df['is_luxury_segment'] = ((is_large + is_villa) > 0).astype(int)

    return feat_df


def get_feature_lists():
    """Returns numerical and categorical feature subsets for regression modeling."""
    numerical_features = [
        'square_feet',
        'bedrooms',
        'bathrooms',
        'property_age',
        'lot_size_sqft',
        'garage_spaces',
        'floors',
        'distance_to_city_center_km',
        'school_rating',
        'total_amenity_score',
        'bed_to_bath_ratio',
        'lot_coverage_ratio',
        'location_desirability_index'
    ]

    categorical_features = [
        'neighborhood',
        'property_type',
        'renovation_status',
        'crime_rate_index',
        'has_pool',
        'has_garden',
        'is_luxury_segment'
    ]

    return numerical_features, categorical_features


def get_preprocessor_pipeline(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    """Creates a unified ColumnTransformer for scaling continuous features and one-hot encoding categories."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor
