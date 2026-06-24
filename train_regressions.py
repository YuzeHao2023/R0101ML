#!/usr/bin/env python3
"""
Train multiple regression models on R0101.csv and visualize results.

Usage:
  python train_regressions.py --csv R0101.csv --out outputs

This script will try to handle multi-row headers by joining header rows.
It selects feature columns containing PRES, ATM, or TEMP (case-insensitive)
and uses `MASSFRA` as the target label.
"""
import argparse
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

sns.set(style="whitegrid")
warnings.filterwarnings("ignore")


def read_csv_flexible(path: str) -> pd.DataFrame:
    # Try reading with normal header first
    df = pd.read_csv(path)
    if "MASSFRA" in df.columns:
        return df

    # If not found, attempt to read with multiple header rows and join them
    raw = pd.read_csv(path, header=None)
    # Heuristic: if first 1-5 rows contain many non-numeric strings, treat them as header
    max_header_rows = min(6, len(raw))
    for h in range(1, max_header_rows + 1):
        try:
            df_try = pd.read_csv(path, header=list(range(h)))
            # join multiindex columns into single strings
            if isinstance(df_try.columns, pd.MultiIndex):
                cols = [" ".join([str(x).strip() for x in col if str(x) != 'nan']).strip() for col in df_try.columns.values]
                df_try.columns = cols
            if "MASSFRA" in df_try.columns:
                return df_try
        except Exception:
            continue

    # Fallback: read without header and promote first few rows to header by concatenation
    raw = pd.read_csv(path, header=None)
    # Use first 4 rows as header if available
    header_rows = min(4, len(raw))
    header = []
    for c in range(raw.shape[1]):
        parts = [str(raw.iloc[r, c]).strip() for r in range(header_rows) if str(raw.iloc[r, c]).strip() not in ['nan', 'None', '']]
        header.append(" ".join(parts).strip() or f"col_{c}")
    df = raw.iloc[header_rows:].copy()
    df.columns = header
    df = df.reset_index(drop=True)
    return df


def select_features(df: pd.DataFrame):
    cols = list(df.columns)
    target = None
    for c in cols:
        if str(c).strip().upper() == "MASSFRA":
            target = c
            break
    if target is None:
        # try fuzzy match
        for c in cols:
            if "MASS" in str(c).upper() and "FRA" in str(c).upper():
                target = c
                break
    if target is None:
        raise ValueError("Could not find target column 'MASSFRA' in CSV columns")

    feature_keys = ["PRES", "ATM", "TEMP"]
    features = [c for c in cols if any(k in str(c).upper() for k in feature_keys) and c != target]
    if not features:
        # fallback: use all numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c != target]

    return features, target


def prepare_xy(df: pd.DataFrame, features, target):
    X = df[features].apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(df[target], errors='coerce')
    data = pd.concat([X, y], axis=1).dropna()
    X = data[features].values
    y = data[target].values
    return X, y


def get_models():
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=0),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=0),
        'SVR': SVR()
    }
    if _HAS_XGB:
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=0, verbosity=0)
    return models


def evaluate_model(name, model, X_train, X_test, y_train, y_test, out_dir: Path):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Scatter plot actual vs predicted
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{name} Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(out_dir / f'{name}_actual_vs_predicted.png')
    plt.close()

    # Residuals
    plt.figure(figsize=(6, 4))
    sns.histplot(y_test - y_pred, kde=True)
    plt.title(f'{name} Residuals')
    plt.xlabel('Residual')
    plt.tight_layout()
    plt.savefig(out_dir / f'{name}_residuals.png')
    plt.close()

    return {'model': name, 'mse': float(mse), 'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='R0101.csv', help='Path to CSV file')
    parser.add_argument('--out', type=str, default='outputs', help='Output directory for figures and results')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.csv} ...")
    df = read_csv_flexible(args.csv)
    print("Columns found:", df.columns.tolist())

    features, target = select_features(df)
    print('Selected features:', features)
    print('Target:', target)

    X, y = prepare_xy(df, features, target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = get_models()
    results = []
    for name, model in models.items():
        print('Training', name)
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test, out_dir)
        results.append(res)

    res_df = pd.DataFrame(results).sort_values('rmse')
    res_df.to_csv(out_dir / 'results_summary.csv', index=False)

    # Plot comparison of RMSE and R2
    plt.figure(figsize=(8, 4))
    ax = plt.subplot(1, 2, 1)
    sns.barplot(x='rmse', y='model', data=res_df, ax=ax)
    ax.set_title('RMSE by model')

    ax2 = plt.subplot(1, 2, 2)
    sns.barplot(x='r2', y='model', data=res_df, ax=ax2)
    ax2.set_title('R2 by model')
    plt.tight_layout()
    plt.savefig(out_dir / 'model_comparison.png')
    plt.close()

    print('Done. Results saved to', out_dir)


if __name__ == '__main__':
    main()
