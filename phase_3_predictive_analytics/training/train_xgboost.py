import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import ttest_rel
import logging
import joblib
import xgboost as xgb
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ===== Shared helpers (copied from LR script) =====

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns:
        raise ValueError("CSV missing 'timestamp' column")
    df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df[df['datetime'].notna()].sort_values('datetime').drop_duplicates('datetime')
    df = df.set_index('datetime').asfreq('h')
    for col in ['nox_gt', 'co_gt', 'no2_gt']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df[['nox_gt','co_gt','no2_gt']] = df[['nox_gt','co_gt','no2_gt']].interpolate(method='time', limit=3)
    hours = df.index.hour
    for col in ['nox_gt', 'co_gt', 'no2_gt']:
        if col in df.columns:
            s = df[col]
            mean_by_hour = s.groupby(hours).transform('mean')
            std_by_hour = s.groupby(hours).transform('std')
            k = 1.5
            seasonal_fill = mean_by_hour.clip(lower=mean_by_hour - k*std_by_hour,
                                              upper=mean_by_hour + k*std_by_hour)
            df[col] = s.where(s.notna(), seasonal_fill).fillna(s.median())
    df.reset_index(inplace=True)
    return df


def chronological_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return df.iloc[:n_train].copy(), df.iloc[n_train:n_train+n_val].copy(), df.iloc[n_train+n_val:].copy()


def build_minimal_features(df: pd.DataFrame, horizon_h: int = 6, add_exog: bool = True, add_temporal: bool = True):
    d = df.copy()
    target_col = 'nox_gt'
    if target_col not in d.columns:
        raise ValueError("Missing 'nox_gt' in CSV")

    lags = [1, 3, 6, 12, 18, 24, 48]
    for L in lags:
        d[f'nox_lag{L}'] = d[target_col].shift(L)
    d['nox_mean_24h'] = d[target_col].shift(1).rolling(window=24, min_periods=6).mean()

    feature_names = [f'nox_lag{L}' for L in lags] + ['nox_mean_24h']
    if add_exog:
        if 'co_gt' in d.columns:
            d['co_lag1'] = d['co_gt'].shift(1)
            feature_names.append('co_lag1')
        if 'no2_gt' in d.columns:
            d['no2_lag1'] = d['no2_gt'].shift(1)
            feature_names.append('no2_lag1')
    if add_temporal:
        hr = d['datetime'].dt.hour
        d['sin_hour'] = np.sin(2 * np.pi * hr / 24)
        d['cos_hour'] = np.cos(2 * np.pi * hr / 24)
        feature_names += ['sin_hour', 'cos_hour']

    y = d[target_col].shift(-horizon_h)
    X = d[feature_names]
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask]


def naive_baseline(series: pd.Series, horizon_h: int, index_like: pd.Index) -> np.ndarray:
    s = pd.to_numeric(series, errors='coerce')
    if len(s) <= horizon_h:
        return np.full(len(index_like), np.nan)
    return np.full(len(index_like), s.iloc[-horizon_h-1])


def significance_test(y_true, y_pred_model, y_pred_baseline):
    errors_model = np.abs(y_true - y_pred_model)
    errors_baseline = np.abs(y_true - y_pred_baseline)
    mask = np.isfinite(errors_model) & np.isfinite(errors_baseline)
    if np.sum(mask) < 2:
        return {'t_statistic': np.nan, 'p_value': np.nan, 'significant': False}
    t_stat, p_value = ttest_rel(errors_baseline[mask], errors_model[mask])
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': int(p_value < 0.05),
        'n_samples': int(np.sum(mask))
    }


def mae(y_true, y_pred): return float(np.nanmean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1e-12, denom)
    return float(200.0 * np.nanmean(np.abs(y_true - y_pred) / denom))


# ===== Main training =====

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost for NOx h=6')
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    P3_DIR = os.path.dirname(CURRENT_DIR)
    PROJECT_ROOT = os.path.dirname(P3_DIR)
    default_csv = os.path.join(PROJECT_ROOT, 'phase_1_streaming_infrastructure','data','processed','air_quality_clean.csv')
    default_artifacts = os.path.join(P3_DIR, 'models')
    parser.add_argument('--csv', default=default_csv)
    parser.add_argument('--artifacts', default=default_artifacts)
    args = parser.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)
    model_dir = os.path.join(args.artifacts, 'xgb')
    os.makedirs(model_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(CURRENT_DIR, 'xgb_training.log')
    logger = logging.getLogger('xgb_training')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler()
    fh = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)

    logger.info("Starting XGBoost training (h=6)")
    df = load_csv(args.csv)
    train_df, val_df, test_df = chronological_split(df)

    # Build features
    X_full, y_full = build_minimal_features(df, horizon_h=6)
    dts = pd.to_datetime(df.loc[X_full.index, 'datetime'])

    def slice_by_range(start_dt, end_dt):
        mask = (dts >= start_dt) & (dts <= end_dt)
        return X_full[mask], y_full[mask]

    X_train, y_train = slice_by_range(train_df['datetime'].min(), train_df['datetime'].max())
    X_val, y_val = slice_by_range(val_df['datetime'].min(), val_df['datetime'].max())
    X_test, y_test = slice_by_range(test_df['datetime'].min(), test_df['datetime'].max())

    # Hyperparam grid
    param_grid = [
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
        {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 5},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6}
    ]

    best = {"params": None, "val_rmse": float('inf'), "model": None, "y_val_pred": None}
    for params in tqdm(param_grid, desc="XGB search"):
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            **params
        )
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        cur_rmse = rmse(y_val, y_val_pred)
        if cur_rmse < best["val_rmse"]:
            best.update({"params": params, "val_rmse": cur_rmse, "model": model, "y_val_pred": y_val_pred})
            logger.info(f"New best: {params}, val_rmse={cur_rmse:.3f}")

    # Refit on train+val
    X_trval = pd.concat([X_train, X_val]); y_trval = pd.concat([y_train, y_val])
    final_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, **best["params"])
    final_model.fit(X_trval, y_trval)
    y_test_pred = final_model.predict(X_test)

    # Baselines
    baseline_val = naive_baseline(val_df['nox_gt'], 6, y_val.index)
    baseline_test = naive_baseline(test_df['nox_gt'], 6, y_test.index)

    # Metrics
    metrics = {
        "val": {
            "mae": mae(y_val, best["y_val_pred"]),
            "rmse": rmse(y_val, best["y_val_pred"]),
            "r2": float(r2_score(y_val, best["y_val_pred"])),
            "smape": smape(y_val, best["y_val_pred"]),
            "mae_baseline": mae(y_val, baseline_val),
            "rmse_baseline": rmse(y_val, baseline_val),
            "significance_test": significance_test(y_val, best["y_val_pred"], baseline_val)
        },
        "test": {
            "mae": mae(y_test, y_test_pred),
            "rmse": rmse(y_test, y_test_pred),
            "r2": float(r2_score(y_test, y_test_pred)),
            "smape": smape(y_test, y_test_pred),
            "mae_baseline": mae(y_test, baseline_test),
            "rmse_baseline": rmse(y_test, baseline_test),
            "significance_test": significance_test(y_test, y_test_pred, baseline_test)
        },
        "features": X_train.columns.tolist(),
        "data_summary": {
            "target": "nox_gt",
            "horizon_h": 6,
            "features": X_train.shape[1],
            "rows_after_feature_build": len(X_full),
            "train": {"rows": len(X_train)},
            "validation": {"rows": len(X_val)},
            "test": {"rows": len(X_test)}
        }
    }

    # Save artifacts
    joblib.dump(final_model, os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(model_dir, "features.json"), "w") as f:
        json.dump({"features": X_train.columns.tolist(), "params": best["params"]}, f, indent=2)
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Artifacts saved for XGB (model, features, metrics).")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
