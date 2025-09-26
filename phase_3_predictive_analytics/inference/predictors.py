import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def _build_minimal_features_for_inference(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    d = df.copy()
    if 'datetime' not in d.columns:
        d['datetime'] = pd.to_datetime(d['timestamp'], errors='coerce')
    d = d.sort_values('datetime')
    # ensure hourly cadence and fill gaps (same as training policy)
    d = d.set_index('datetime').asfreq('h')
    for col in ['nox_gt', 'co_gt', 'no2_gt']:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors='coerce')
    d[['nox_gt','co_gt','no2_gt']] = d[['nox_gt','co_gt','no2_gt']].interpolate(method='time', limit=3)
    hours = d.index.hour
    for col in ['nox_gt', 'co_gt', 'no2_gt']:
        if col in d.columns:
            s = d[col]
            mean_by_hour = s.groupby(hours).transform('mean')
            std_by_hour = s.groupby(hours).transform('std')
            k = 1.5
            seasonal_fill = (mean_by_hour).clip(lower=mean_by_hour - k * std_by_hour,
                                                upper=mean_by_hour + k * std_by_hour)
            d[col] = s.where(s.notna(), seasonal_fill).fillna(s.median())

    d.reset_index(inplace=True)

    # lags and rolling
    lags = [1, 3, 6, 12, 18, 24, 48]
    for L in lags:
        d[f'nox_lag{L}'] = d['nox_gt'].shift(L)
    d['nox_mean_24h'] = d['nox_gt'].shift(1).rolling(window=24, min_periods=6).mean()
    d['co_lag1'] = d['co_gt'].shift(1) if 'co_gt' in d.columns else np.nan
    d['no2_lag1'] = d['no2_gt'].shift(1) if 'no2_gt' in d.columns else np.nan
    hrs = pd.to_datetime(d['datetime']).dt.hour
    d['sin_hour'] = np.sin(2 * np.pi * hrs / 24)
    d['cos_hour'] = np.cos(2 * np.pi * hrs / 24)

    # last valid timestamp where all features exist
    feature_cols = [f'nox_lag{L}' for L in lags] + ['nox_mean_24h','co_lag1','no2_lag1','sin_hour','cos_hour']
    mask = d[feature_cols].notna().all(axis=1)
    d_valid = d[mask]
    if d_valid.empty:
        return pd.DataFrame(), pd.NaT
    t = d_valid['datetime'].max()
    x_row = d_valid.loc[d_valid['datetime'] == t, feature_cols]
    return x_row, pd.to_datetime(t)


def predict_lr_next_6(df_filtered: pd.DataFrame, models_dir: str) -> pd.DataFrame:
    """Predict next 6 hours with LR family model. Returns dataframe with columns: datetime, lr_pred."""
    lr_dir = os.path.join(models_dir, 'lr')
    model = joblib.load(os.path.join(lr_dir, 'model.pkl'))
    scaler = joblib.load(os.path.join(lr_dir, 'scaler.pkl'))
    with open(os.path.join(lr_dir, 'features.json')) as f:
        feats = json.load(f)['features']

    x_row, t = _build_minimal_features_for_inference(df_filtered)
    if x_row.empty or pd.isna(t):
        return pd.DataFrame()
    # standardize and predict one 6h-ahead point (we’ll generate future timestamps 1..6h)
    x_s = scaler.transform(x_row[feats])
    y_hat = float(model.predict(x_s)[0])
    future_times = pd.date_range(start=t + pd.Timedelta(hours=1), periods=6, freq='h')
    # simple persistence of same point across 6 steps or repeat single-step? For now, repeat horizon value at t+6 only.
    # Better: build recursive or direct multi-step; keeping simple per scope.
    preds = pd.DataFrame({
        'datetime': future_times,
        'lr_pred': [np.nan]*5 + [y_hat]
    })
    return preds


def predict_sarima_next_6(df_filtered: pd.DataFrame, models_dir: str) -> pd.DataFrame:
    """Predict next 6 hours with SARIMA. Returns dataframe with: datetime, sarima_mean, sarima_lo, sarima_hi."""
    sarima_dir = os.path.join(models_dir, 'sarima')
    res = SARIMAXResults.load(os.path.join(sarima_dir, 'model.pkl'))
    # Forecast 6 steps from end of model’s sample; assumes model trained on train+val
    fc = res.get_forecast(steps=6)
    mean = fc.predicted_mean.values
    conf = fc.conf_int(alpha=0.05).values
    # Future timestamps from last in df_filtered (or from model’s index if needed)
    if 'datetime' in df_filtered.columns and not df_filtered.empty:
        last_t = pd.to_datetime(df_filtered['datetime']).max()
    else:
        last_t = pd.to_datetime(res.data.row_labels[-1])
    future_times = pd.date_range(start=last_t + pd.Timedelta(hours=1), periods=6, freq='h')
    out = pd.DataFrame({
        'datetime': future_times,
        'sarima_mean': mean,
        'sarima_lo': conf[:, 0],
        'sarima_hi': conf[:, 1]
    })
    return out


