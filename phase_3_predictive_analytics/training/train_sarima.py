import argparse
import json
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    # enforce hourly cadence and interpolate small gaps (≤3h) on target
    df = df.set_index('datetime').asfreq('h')
    df['nox_gt'] = pd.to_numeric(df['nox_gt'], errors='coerce').interpolate(method='time', limit=3)
    # Fill remaining NaNs using seasonal bands per hour for target
    hrs = df.index.hour
    s = df['nox_gt']
    mean_by_hour = s.groupby(hrs).transform('mean')
    std_by_hour = s.groupby(hrs).transform('std')
    k = 1.5
    band_low = mean_by_hour - k * std_by_hour
    band_high = mean_by_hour + k * std_by_hour
    seasonal_fill = mean_by_hour.clip(lower=band_low, upper=band_high)
    df['nox_gt'] = s.where(s.notna(), seasonal_fill).fillna(s.median())
    df.reset_index(inplace=True)
    return df


def chronological_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return df.iloc[:n_train].copy(), df.iloc[n_train:n_train+n_val].copy(), df.iloc[n_train+n_val:].copy()


def series_from(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors='coerce')
    ts = pd.Series(s.values, index=pd.to_datetime(df['datetime'], errors='coerce'))
    return ts.asfreq('h')


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
    denom = np.where(denom == 0, 1e-12, denom)
    return float(200.0 * np.nanmean(np.abs(y_true - y_pred) / denom))


def naive_baseline(series: pd.Series, horizon_h: int, length: int) -> np.ndarray:
    # Repeat last observed value as naive forecast; approximate for comparison
    last_val = series.dropna().iloc[-1]
    return np.full(length, last_val)


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


def fit_and_forecast(ts: pd.Series, order, seasonal_order, steps: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=steps)
    mean = fc.predicted_mean.values
    conf = fc.conf_int(alpha=0.05).values
    return mean, (conf[:, 0], conf[:, 1])


def main():
    parser = argparse.ArgumentParser(description='Train SARIMA for NOx h=6')
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    P3_DIR = os.path.dirname(CURRENT_DIR)
    PROJECT_ROOT = os.path.dirname(P3_DIR)
    default_csv = os.path.join(PROJECT_ROOT, 'phase_1_streaming_infrastructure','data','processed','air_quality_clean.csv')
    default_artifacts = os.path.join(P3_DIR, 'models')
    parser.add_argument('--csv', default=default_csv)
    parser.add_argument('--artifacts', default=default_artifacts)
    args = parser.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)

    # Prepare logging (console + file in training folder)
    model_dir = os.path.join(args.artifacts, 'sarima')
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(CURRENT_DIR, 'sarima_training.log')
    logger = logging.getLogger('sarima_training')
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

    logger.info('Starting SARIMA training (h=6)')
    logger.info(f'CSV: {args.csv}')
    logger.info(f'Artifacts dir: {args.artifacts}')

    df = load_csv(args.csv)
    logger.info(f'Loaded data: rows={len(df)}, time=[{df["datetime"].min()} .. {df["datetime"].max()}]')
    train_df, val_df, test_df = chronological_split(df)
    logger.info(f'Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')

    y_train = series_from(train_df, 'nox_gt')
    y_val = series_from(val_df, 'nox_gt')
    y_test = series_from(test_df, 'nox_gt')

    # Candidate orders around daily seasonality (s=24)
    candidates = [
        ((1, 0, 1), (1, 1, 1, 24)),
        ((2, 0, 2), (1, 1, 1, 24)),
        ((1, 1, 1), (1, 1, 0, 24)),
        ((1, 0, 2), (0, 1, 1, 24)),
    ]

    # Select best by validation RMSE on a simple forecast from end of train for len(val)
    best = {'order': None, 'seasonal_order': None, 'val_rmse': float('inf')}
    logger.info(f'Evaluating {len(candidates)} SARIMA candidates: {candidates}')
    for order, seas in tqdm(candidates, desc='SARIMA selection', leave=False):
        try:
            y_pred_val, _ = fit_and_forecast(y_train, order, seas, steps=len(y_val))
            cur = rmse(y_val.values, y_pred_val)
            if cur < best['val_rmse']:
                best.update({'order': order, 'seasonal_order': seas, 'val_rmse': cur})
                logger.info(f'New best: order={order}, seasonal_order={seas}, val_rmse={cur:.3f}')
        except Exception:
            logger.warning(f'Candidate failed: order={order}, seasonal_order={seas}')
            continue

    if best['order'] is None:
        print(json.dumps({'error': 'SARIMA training failed for all candidates'}, indent=2))
        return

    # Refit on train+val and forecast test horizon path (len(test))
    y_trainval = series_from(pd.concat([train_df, val_df]), 'nox_gt')
    test_steps = len(y_test)
    logger.info(f'Refitting on Train+Val with best config and forecasting Test ({test_steps} steps)...')
    y_pred_test, (lo_test, hi_test) = fit_and_forecast(y_trainval, best['order'], best['seasonal_order'], steps=test_steps)

    # Baseline for test
    baseline_test = naive_baseline(y_trainval, horizon_h=6, length=test_steps)

    # Pairwise valid mask for metrics (drop NaNs)
    y_true_all = y_test.values.astype(float)
    y_pred_all = np.asarray(y_pred_test, dtype=float)
    mask = np.isfinite(y_true_all) & np.isfinite(y_pred_all)
    y_true = y_true_all[mask]
    y_pred = y_pred_all[mask]
    base_masked = baseline_test[:len(y_pred)] if len(baseline_test) >= len(y_pred) else baseline_test
    base_masked = np.asarray(base_masked, dtype=float)
    if len(base_masked) != len(y_pred):
        # align lengths conservatively
        m = min(len(base_masked), len(y_pred))
        base_masked = base_masked[:m]
        y_true = y_true[:m]
        y_pred = y_pred[:m]

    # Statistical significance testing
    test_significance = significance_test(y_true, y_pred, base_masked)

    # Data summary similar to LR
    data_summary = {
        'target': 'nox_gt',
        'horizon_h': 6,
        'train': {
            'rows': int(len(y_train.dropna())),
            'start': str(train_df['datetime'].min()),
            'end': str(train_df['datetime'].max())
        },
        'validation': {
            'rows': int(len(y_val.dropna())),
            'start': str(val_df['datetime'].min()),
            'end': str(val_df['datetime'].max())
        },
        'test': {
            'rows': int(len(y_test.dropna())),
            'start': str(test_df['datetime'].min()),
            'end': str(test_df['datetime'].max())
        }
    }

    metrics = {
        'val': {
            'rmse': float(best['val_rmse'])
        },
        'test': {
            'mae': mae(y_true, y_pred) if len(y_pred) > 0 else None,
            'rmse': rmse(y_true, y_pred) if len(y_pred) > 0 else None,
            'r2': float(r2_score(y_true, y_pred)) if len(y_pred) > 1 else None,
            'smape': smape(y_true, y_pred) if len(y_pred) > 0 else None,
            'mae_baseline': mae(y_true, base_masked) if len(y_pred) > 0 else None,
            'rmse_baseline': rmse(y_true, base_masked) if len(y_pred) > 0 else None,
            'significance_test': test_significance
        },
        'model': {
            'order': best['order'],
            'seasonal_order': best['seasonal_order'],
            'seasonal_period': 24,
            'horizon_h': 6
        },
        'data_summary': data_summary
    }

    # Persist final fitted model on train+val
    model = SARIMAX(y_trainval, order=best['order'], seasonal_order=best['seasonal_order'], enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    # Organize artifacts in per-model subfolder
    model_dir = os.path.join(args.artifacts, 'sarima')
    os.makedirs(model_dir, exist_ok=True)
    res.save(os.path.join(model_dir, 'model.pkl'))
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(metrics['model'], f, indent=2)
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info('Artifacts saved (model, config, metrics, log).')

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()


