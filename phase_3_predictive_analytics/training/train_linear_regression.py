import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import ttest_rel
import logging
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns:
        raise ValueError("CSV missing 'timestamp' column")
    df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df[df['datetime'].notna()].sort_values('datetime').drop_duplicates('datetime')
    # enforce hourly cadence and interpolate small gaps (≤3h)
    df = df.set_index('datetime').asfreq('h')
    for col in ['nox_gt', 'co_gt', 'no2_gt']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df[['nox_gt','co_gt','no2_gt']] = df[['nox_gt','co_gt','no2_gt']].interpolate(method='time', limit=3)
    # Fill remaining NaNs using seasonal hourly means bounded by seasonal bands
    hours = df.index.hour
    for col in ['nox_gt', 'co_gt', 'no2_gt']:
        if col in df.columns:
            s = df[col]
            mean_by_hour = s.groupby(hours).transform('mean')
            std_by_hour = s.groupby(hours).transform('std')
            k = 1.5
            band_low = mean_by_hour - k * std_by_hour
            band_high = mean_by_hour + k * std_by_hour
            seasonal_fill = mean_by_hour.clip(lower=band_low, upper=band_high)
            s_filled = s.where(s.notna(), seasonal_fill)
            # Final fallback to overall median if any NaNs remain
            s_filled = s_filled.fillna(s.median())
            df[col] = s_filled
    df.reset_index(inplace=True)
    return df


def chronological_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return df.iloc[:n_train].copy(), df.iloc[n_train:n_train+n_val].copy(), df.iloc[n_train+n_val:].copy()


def build_minimal_features(df: pd.DataFrame, horizon_h: int = 6, add_exog: bool = True, add_temporal: bool = True):
    d = df.copy()
    # Core target
    target_col = 'nox_gt'
    if target_col not in d.columns:
        raise ValueError("Missing 'nox_gt' in CSV")

    # Minimal NOx lags
    lags = [1, 3, 6, 12, 18, 24, 48]
    for L in lags:
        d[f'nox_lag{L}'] = d[target_col].shift(L)

    # Rolling mean 24h (closed-left via shift(1))
    d['nox_mean_24h'] = d[target_col].shift(1).rolling(window=24, min_periods=6).mean()

    # Optional tiny exog set
    feature_names = [f'nox_lag{L}' for L in lags] + ['nox_mean_24h']

    if add_exog:
        if 'co_gt' in d.columns:
            d['co_lag1'] = d['co_gt'].shift(1)
            feature_names.append('co_lag1')
        if 'no2_gt' in d.columns:
            d['no2_lag1'] = d['no2_gt'].shift(1)
            feature_names.append('no2_lag1')

    # Temporal encodings for LR
    if add_temporal:
        hr = d['datetime'].dt.hour
        d['sin_hour'] = np.sin(2 * np.pi * hr / 24)
        d['cos_hour'] = np.cos(2 * np.pi * hr / 24)
        feature_names += ['sin_hour', 'cos_hour']

    # Target shift
    y = d[target_col].shift(-horizon_h)

    # Select predictors: only the features we created (avoid dragging unrelated NaNs)
    X = d[feature_names]

    # Drop rows with NaNs
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    return X, y


def naive_baseline(series: pd.Series, horizon_h: int, index_like: pd.Index) -> np.ndarray:
    # y_hat(t+6) = y(t) → align as constant equal to last available naive
    s = pd.to_numeric(series, errors='coerce')
    if len(s) <= horizon_h:
        return np.full(len(index_like), np.nan)
    return np.full(len(index_like), s.iloc[-horizon_h-1])


def significance_test(y_true, y_pred_model, y_pred_baseline):
    """Test if model is significantly better than baseline using paired t-test"""
    errors_model = np.abs(y_true - y_pred_model)
    errors_baseline = np.abs(y_true - y_pred_baseline)
    
    # Remove any NaN values for valid comparison
    mask = np.isfinite(errors_model) & np.isfinite(errors_baseline)
    if np.sum(mask) < 2:
        return {'t_statistic': np.nan, 'p_value': np.nan, 'significant': False}
    
    errors_model_clean = errors_model[mask]
    errors_baseline_clean = errors_baseline[mask]
    
    # Paired t-test: test if baseline errors are significantly higher than model errors
    t_stat, p_value = ttest_rel(errors_baseline_clean, errors_model_clean)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': int(p_value < 0.05),
        'n_samples': int(np.sum(mask))
    }


def mae(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.nanmean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    denom = (np.abs(y_true) + np.abs(y_pred))
    # avoid division by zero
    denom = np.where(denom == 0, 1e-12, denom)
    return float(200.0 * np.nanmean(np.abs(y_true - y_pred) / denom))


def main():
    parser = argparse.ArgumentParser(description='Train Linear Regression/Ridge/Lasso for NOx h=6')
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    P3_DIR = os.path.dirname(CURRENT_DIR)  # phase_3_predictive_analytics
    PROJECT_ROOT = os.path.dirname(P3_DIR)
    default_csv = os.path.join(PROJECT_ROOT, 'phase_1_streaming_infrastructure','data','processed','air_quality_clean.csv')
    default_artifacts = os.path.join(P3_DIR, 'models')
    parser.add_argument('--csv', default=default_csv)
    parser.add_argument('--artifacts', default=default_artifacts)
    parser.add_argument('--model', choices=['auto','linear','ridge','lasso'], default='auto')
    parser.add_argument('--alphas', type=str, default='', help='Comma-separated alphas for ridge/lasso (optional)')
    args = parser.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)

    # Prepare logging (console + file in training folder)
    model_dir = os.path.join(args.artifacts, 'lr')
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(CURRENT_DIR, 'lr_training.log')
    logger = logging.getLogger('lr_training')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info('Starting LR training (h=6)')
    logger.info(f'CSV: {args.csv}')
    logger.info(f'Artifacts dir: {args.artifacts}')

    df = load_csv(args.csv)
    logger.info(f'Loaded data: rows={len(df)}, time=[{df["datetime"].min()} .. {df["datetime"].max()}]')
    train_df, val_df, test_df = chronological_split(df)
    logger.info(f'Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')

    # Build features on full data to preserve history, then slice by time if needed
    logger.info('Building minimal features on full dataset...')
    X_full, y_full = build_minimal_features(df, horizon_h=6)
    dts = pd.to_datetime(df.loc[X_full.index, 'datetime'])
    logger.info(f'Feature rows after build: {len(X_full)}; feature count={X_full.shape[1]}')

    def slice_by_range(start_dt, end_dt):
        mask = (dts >= start_dt) & (dts <= end_dt)
        return X_full[mask], y_full[mask]

    train_start, train_end = train_df['datetime'].min(), train_df['datetime'].max()
    val_start, val_end = val_df['datetime'].min(), val_df['datetime'].max()
    test_start, test_end = test_df['datetime'].min(), test_df['datetime'].max()

    X_train, y_train = slice_by_range(train_start, train_end)
    X_val, y_val = slice_by_range(val_start, val_end)
    X_test, y_test = slice_by_range(test_start, test_end)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        logger.error('Insufficient rows after feature build. Check gaps/NaNs.')
        print(json.dumps({'error': 'Insufficient rows after feature build. Check gaps/NaNs.'}, indent=2))
        return

    # Scale predictors (fit on train)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    logger.info('Scaling complete.')

    # Model selection (validation)
    model_choice = args.model
    best = {'algo': None, 'alpha': None, 'val_rmse': float('inf'), 'model': None, 'y_val_pred': None}
    # Define candidates
    candidates = []
    if model_choice == 'auto':
        # default alpha grids
        ridge_alphas = [0.01, 0.1, 1.0, 10.0]
        lasso_alphas = [0.0001, 0.001, 0.01, 0.1]
        candidates.append(('linear', None))
        candidates += [('ridge', a) for a in ridge_alphas]
        candidates += [('lasso', a) for a in lasso_alphas]
    elif model_choice == 'linear':
        candidates = [('linear', None)]
    else:
        alpha_list = [float(a.strip()) for a in args.alphas.split(',') if a.strip()]
        if not alpha_list:
            alpha_list = [0.01, 0.1, 1.0, 10.0] if model_choice == 'ridge' else [0.0001, 0.001, 0.01, 0.1]
        candidates = [(model_choice, a) for a in alpha_list]

    logger.info(f'Model selection over {len(candidates)} candidates: {candidates}')
    for algo, alpha in tqdm(candidates, desc='Model selection', leave=False):
        if algo == 'linear':
            model = LinearRegression()
        elif algo == 'ridge':
            model = Ridge(alpha=alpha)
        else:
            model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train_s, y_train)
        y_val_pred_tmp = model.predict(X_val_s)
        cur = rmse(y_val, y_val_pred_tmp)
        if cur < best['val_rmse']:
            best.update({'algo': algo, 'alpha': alpha, 'val_rmse': cur, 'model': model, 'y_val_pred': y_val_pred_tmp})
            logger.info(f'New best: algo={algo}, alpha={alpha}, val_rmse={cur:.3f}')

    # After selecting alpha, refit on train+val and evaluate on test
    X_trval = pd.concat([X_train, X_val])
    y_trval = pd.concat([y_train, y_val])
    scaler_final = StandardScaler()
    X_trval_s = scaler_final.fit_transform(X_trval)
    X_test_s_final = scaler_final.transform(X_test)

    if best['algo'] == 'linear':
        final_model = LinearRegression()
    elif best['algo'] == 'ridge':
        final_model = Ridge(alpha=best['alpha'])
    else:
        final_model = Lasso(alpha=best['alpha'], max_iter=10000)
    logger.info(f'Refitting final model on Train+Val: algo={best["algo"]}, alpha={best["alpha"]}')
    final_model.fit(X_trval_s, y_trval)
    y_test_pred = final_model.predict(X_test_s_final)
    y_val_pred = best['y_val_pred']

    # Baseline
    baseline_val = naive_baseline(val_df['nox_gt'], 6, y_val.index)
    baseline_test = naive_baseline(test_df['nox_gt'], 6, y_test.index)

    # Data summary
    full_rows = int(len(X_full))
    train_rows = int(len(X_train))
    val_rows = int(len(X_val))
    test_rows = int(len(X_test))
    feature_count = int(X_train.shape[1])
    data_summary = {
        'target': 'nox_gt',
        'horizon_h': 6,
        'features': feature_count,
        'rows_after_feature_build': full_rows,
        'train': {
            'rows': train_rows,
            'start': str(train_start),
            'end': str(train_end)
        },
        'validation': {
            'rows': val_rows,
            'start': str(val_start),
            'end': str(val_end)
        },
        'test': {
            'rows': test_rows,
            'start': str(test_start),
            'end': str(test_end)
        }
    }

    # Statistical significance testing
    val_significance = significance_test(y_val, y_val_pred, baseline_val)
    test_significance = significance_test(y_test, y_test_pred, baseline_test)

    metrics = {
        'val': {
            'mae': mae(y_val, y_val_pred),
            'rmse': rmse(y_val, y_val_pred),
            'r2': float(r2_score(y_val, y_val_pred)),
            'smape': smape(y_val, y_val_pred),
            'mae_baseline': mae(y_val, baseline_val),
            'rmse_baseline': rmse(y_val, baseline_val),
            'significance_test': val_significance
        },
        'test': {
            'mae': mae(y_test, y_test_pred),
            'rmse': rmse(y_test, y_test_pred),
            'r2': float(r2_score(y_test, y_test_pred)),
            'smape': smape(y_test, y_test_pred),
            'mae_baseline': mae(y_test, baseline_test),
            'rmse_baseline': rmse(y_test, baseline_test),
            'significance_test': test_significance
        },
        'features': X_train.columns.tolist(),
        'data_summary': data_summary
    }

    # Save artifacts
    import joblib
    # Organize artifacts in fixed LR folder, regardless of selected algo
    joblib.dump(final_model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(scaler_final, os.path.join(model_dir, 'scaler.pkl'))
    with open(os.path.join(model_dir, 'features.json'), 'w') as f:
        json.dump({'features': X_train.columns.tolist(), 'selected_algo': best['algo'], 'selected_alpha': best['alpha']}, f, indent=2)
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info('Artifacts saved (model, scaler, features, metrics, log).')

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()


