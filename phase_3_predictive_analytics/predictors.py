import json
import os
import time
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Set up simple logging
import os
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'inference.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
    """Predict next 6 hours with LR family model using aligned anchors (no recursive feedback).
    For each horizon k=1..6, use features at s = t-(6-k) so the h=6 model predicts y(s+6)=y(t+k)."""
    
    start_time = time.time()
    logger.info(f"LR prediction request - Input rows: {len(df_filtered)}")
    
    try:
        # Load model artifacts
        lr_dir = os.path.join(models_dir, 'lr')
        model_path = os.path.join(lr_dir, 'model.pkl')
        scaler_path = os.path.join(lr_dir, 'scaler.pkl')
        features_path = os.path.join(lr_dir, 'features.json')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            logger.warning(f"LR model files missing - {model_path}")
            return pd.DataFrame()
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(features_path) as f:
            feats = json.load(f)['features']
        
        logger.info(f"LR model loaded successfully from {model_path}")

        # Prepare working frame with gap filling like training
        d = df_filtered.copy()
        if 'datetime' not in d.columns:
            d['datetime'] = pd.to_datetime(d['timestamp'], errors='coerce')
        d = d.sort_values('datetime')
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

        last_time = d.index.max()
        if pd.isna(last_time):
            logger.error("LR prediction failed - No valid timestamps in dataset")
            return pd.DataFrame()

        # Build full feature frame once
        dd = d.copy()
        lags = [1, 3, 6, 12, 18, 24, 48]
        for L in lags:
            dd[f'nox_lag{L}'] = dd['nox_gt'].shift(L)
        dd['nox_mean_24h'] = dd['nox_gt'].shift(1).rolling(window=24, min_periods=6).mean()
        dd['co_lag1'] = dd['co_gt'].shift(1) if 'co_gt' in dd.columns else np.nan
        dd['no2_lag1'] = dd['no2_gt'].shift(1) if 'no2_gt' in dd.columns else np.nan
        hrs2 = dd.index.hour
        dd['sin_hour'] = np.sin(2 * np.pi * hrs2 / 24)
        dd['cos_hour'] = np.cos(2 * np.pi * hrs2 / 24)
        dd = dd.reset_index()

        # Anchors: s_k = t - (6 - k) for k=1..6, predict y(s_k+6) => y(t+k)
        anchors = [last_time - pd.Timedelta(hours=(6 - k)) for k in range(1, 7)]
        rows = []
        for s in anchors:
            xr = dd.loc[dd['datetime'] == s]
            if xr.empty:
                logger.error(f"LR prediction failed - No data for anchor {s}")
                return pd.DataFrame()
            xr = xr[feats] if all(f in xr.columns for f in feats) else xr.reindex(columns=feats, fill_value=np.nan)
            rows.append(xr)
        X_feat = pd.concat(rows, axis=0)
        if X_feat.isna().any().any():
            logger.error("LR prediction failed - NaN values in features")
            return pd.DataFrame()
        X_s = scaler.transform(X_feat)
        y_hats = model.predict(X_s)
        future_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=6, freq='h')
        
        result_df = pd.DataFrame({'datetime': future_times, 'lr_pred': y_hats})
        
        # Log successful prediction
        execution_time = time.time() - start_time
        logger.info(f"LR prediction success - Generated {len(y_hats)} predictions, mean: {np.mean(y_hats):.2f}, time: {execution_time:.3f}s")
        
        return result_df
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"LR prediction error - {str(e)}, time: {execution_time:.3f}s")
        return pd.DataFrame()


def predict_sarima_next_6(df_filtered: pd.DataFrame, models_dir: str) -> pd.DataFrame:
    """Predict next 6 hours with SARIMA. Returns dataframe with: datetime, sarima_mean, sarima_lo, sarima_hi."""
    
    start_time = time.time()
    logger.info(f"SARIMA prediction request - Input rows: {len(df_filtered)}")
    
    try:
        # Load model
        sarima_dir = os.path.join(models_dir, 'sarima')
        model_path = os.path.join(sarima_dir, 'model.pkl')
        
        if not os.path.exists(model_path):
            logger.warning(f"SARIMA model file missing - {model_path}")
            return pd.DataFrame()
        
        res = SARIMAXResults.load(model_path)
        logger.info(f"SARIMA model loaded successfully from {model_path}")
        
        # Generate forecast
        fc = res.get_forecast(steps=6)
        mean = fc.predicted_mean.values
        conf = fc.conf_int(alpha=0.05).values
        
        # Determine last timestamp
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
        
        # Log successful prediction
        execution_time = time.time() - start_time
        logger.info(f"SARIMA prediction success - Generated {len(mean)} predictions, mean: {np.mean(mean):.2f}, time: {execution_time:.3f}s")
        
        return out
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"SARIMA prediction error - {str(e)}, time: {execution_time:.3f}s")
        return pd.DataFrame()


