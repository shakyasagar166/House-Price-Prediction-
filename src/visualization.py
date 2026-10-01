"""
Visualization Module for House Price Prediction & Real Estate Valuation.
Generates property price distributions, neighborhood valuation charts, regression scatter plots, and residual diagnostics.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def plot_price_distribution(df, save_path=None):
    """Plots housing market price histogram and KDE curve."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df['price'] / 1000, bins=25, kde=True, color='#0284c7', edgecolor='#0f172a', ax=ax)
    ax.set_title('Property Valuation Distribution ($ in Thousands)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Price ($K)', fontsize=10)
    ax.set_ylabel('Number of Properties', fontsize=10)

    median_price = df['price'].median() / 1000
    ax.axvline(median_price, color='#ef4444', linestyle='--', label=f'Median: ${median_price:,.0f}K')
    ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    return fig


def plot_price_by_neighborhood(df, save_path=None):
    """Plots median property prices across neighborhoods."""
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df.groupby('neighborhood')['price'].median().sort_values(ascending=False).index

    sns.barplot(data=df, x='price', y='neighborhood', order=order, color='#0ea5e9', ax=ax, edgecolor='#0f172a')
    ax.set_title('Median Property Price by Neighborhood ($)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Median Valuation ($)', fontsize=10)
    ax.set_ylabel('Neighborhood', fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    return fig


def plot_sqft_vs_price(df, save_path=None):
    """Plots square footage vs price scatter with regression trendline."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        data=df,
        x='square_feet',
        y=df['price'] / 1000,
        scatter_kws={'alpha': 0.5, 'color': '#0284c7'},
        line_kws={'color': '#ef4444', 'linewidth': 2},
        ax=ax
    )
    ax.set_title('Square Footage vs Valuation ($K)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Living Area (Square Feet)', fontsize=10)
    ax.set_ylabel('Property Price ($K)', fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    return fig


def plot_model_residuals(y_true, y_pred, save_path=None):
    """Plots Predicted vs Actual and Residual Distribution side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Actual vs Predicted
    ax1.scatter(y_true / 1000, y_pred / 1000, alpha=0.5, color='#0284c7', edgecolors='none')
    min_val = min(y_true.min(), y_pred.min()) / 1000
    max_val = max(y_true.max(), y_pred.max()) / 1000
    ax1.plot([min_val, max_val], [min_val, max_val], color='#ef4444', lw=2, linestyle='--', label='Perfect Fit')
    ax1.set_title('Actual vs Predicted Valuation ($K)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Actual Price ($K)', fontsize=10)
    ax1.set_ylabel('Predicted Price ($K)', fontsize=10)
    ax1.legend()

    # Residuals
    residuals = (y_true - y_pred) / 1000
    sns.histplot(residuals, kde=True, color='#0ea5e9', ax=ax2)
    ax2.axvline(0, color='#ef4444', linestyle='--')
    ax2.set_title('Residual Error Distribution ($K)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Prediction Residual Error ($K)', fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    return fig


def plot_feature_importance(feature_names, importances, top_n=12, save_path=None):
    """Plots top predictive regression features."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = [importances[i] for i in indices]

    sns.barplot(x=top_importances, y=top_features, color='#0284c7', ax=ax, edgecolor='#0f172a')
    ax.set_title(f'Top {top_n} Property Price Valuation Drivers', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Relative Feature Importance', fontsize=10)

    for i, v in enumerate(top_importances):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=8.5, fontweight='semibold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    return fig
