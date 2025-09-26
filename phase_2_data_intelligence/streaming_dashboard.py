#!/usr/bin/env python3
"""
Streamlit Dashboard for Real-Time Air Quality Analytics

This dashboard provides real-time visualization of air quality data
processed by the streaming analytics engine.

Author: Santiago Bolaños Vega
Date: 2025-09-22
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
import sys
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import pearsonr
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sys, os
# Ensure project root is on path to import phase_3 modules when running from Phase 2
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
try:
    from phase_3_predictive_analytics.predictors import predict_lr_next_6, predict_sarima_next_6, predict_xgb_next_6
except ModuleNotFoundError:
    # Fallback to relative import if package-style import fails
    from ..phase_3_predictive_analytics.predictors import predict_lr_next_6, predict_sarima_next_6, predict_xgb_next_6  # type: ignore

# Import analytics module from same directory
from streaming_analytics import StreamingAnalytics

# Page configuration
st.set_page_config(
    page_title="Air Quality Streaming Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_analytics():
    """Initialize the streaming analytics engine."""
    csv_file = "../phase_1_streaming_infrastructure/data/processed/air_quality_clean.csv"
    return StreamingAnalytics(csv_file)

@st.cache_data(ttl=2)
def load_filtered_data(csv_file: str, range_key: str) -> pd.DataFrame:
    """Load CSV and filter by selected time range using dataset max timestamp as reference."""
    if not os.path.exists(csv_file):
        return pd.DataFrame()

    df = pd.read_csv(csv_file)
    if 'timestamp' not in df.columns:
        return pd.DataFrame()

    df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df[df['datetime'].notna()].copy()
    if df.empty:
        return df

    end_time = df['datetime'].max()
    ranges = {
        'Today': pd.Timedelta(days=1),
        'Last 7 days': pd.Timedelta(days=7),
        'Last 30 days': pd.Timedelta(days=30),
        'Last 365 days': pd.Timedelta(days=365),
        'All data': None,
    }
    delta = ranges.get(range_key)
    if delta is None:
        return df.sort_values('datetime')

    start_time = end_time - delta
    return df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)].sort_values('datetime')

def create_stacked_timeseries(df: pd.DataFrame):
    """Render four full-width stacked time series plots for pollutants."""
    if df.empty:
        st.info("No data available for time series visualization.")
        return

    pollutants = [
        ('co_gt', 'CO (mg/m³)', '#1f77b4'),
        ('nox_gt', 'NOx (ppb)', '#ff7f0e'),
        ('no2_gt', 'NO2 (µg/m³)', '#2ca02c'),
        ('c6h6_gt', 'Benzene (µg/m³)', '#d62728'),
    ]

    for col, title, color in pollutants:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce')
            fig = go.Figure(
                data=[go.Scatter(x=df['datetime'], y=series, mode='lines', line=dict(color=color, width=2))]
            )
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title=title,
                height=300,
                margin=dict(l=40, r=20, t=60, b=40)
            )
            st.plotly_chart(fig, width='stretch', key=f"timeseries_{col}")

def create_correlation_heatmap_from_df(df: pd.DataFrame):
    """Create correlation heatmap for pollutants from filtered DataFrame."""
    if df.empty:
        return None
    cols = ['co_gt', 'nox_gt', 'no2_gt', 'c6h6_gt']
    present = [c for c in cols if c in df.columns]
    if len(present) < 2:
        return None
    numeric_df = df[present].apply(pd.to_numeric, errors='coerce').dropna()
    if len(numeric_df) < 2:
        return None
    corr_matrix = numeric_df.corr()

    # Compute pairwise Pearson p-values matrix
    cols = list(numeric_df.columns)
    pval_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if i < j:
                r, p = pearsonr(numeric_df[ci], numeric_df[cj])
                pval_matrix.loc[ci, cj] = p
                pval_matrix.loc[cj, ci] = p
            elif i == j:
                pval_matrix.loc[ci, cj] = 0.0

    # FDR (Benjamini-Hochberg) correction over upper triangle (excluding diagonal)
    upper_mask = np.triu(np.ones_like(pval_matrix.values, dtype=bool), k=1)
    pvals = pval_matrix.values[upper_mask]
    if pvals.size > 0:
        order = np.argsort(pvals)
        ranked_p = pvals[order]
        m = float(len(ranked_p))
        qvals_sorted = ranked_p * m / (np.arange(1, len(ranked_p) + 1))
        # enforce monotonicity
        qvals_sorted = np.minimum.accumulate(qvals_sorted[::-1])[::-1]
        qvals = np.empty_like(pvals)
        qvals[order] = np.clip(qvals_sorted, 0, 1)
        qval_matrix = pd.DataFrame(np.ones_like(pval_matrix.values), index=cols, columns=cols, dtype=float)
        qval_array = qval_matrix.values
        qval_array[upper_mask] = qvals
        # Symmetrize by taking elementwise min with transpose
        qval_sym = np.minimum(qval_array, qval_array.T)
        qval_matrix.iloc[:, :] = qval_sym
    else:
        qval_matrix = pval_matrix.copy()

    # Build two heatmaps: correlations and q-values (FDR-adjusted p-values)
    corr_fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Pollutant Correlation Matrix (Selected Range)"
    )
    corr_fig.update_layout(height=400)

    qval_fig = px.imshow(
        qval_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis_r",
        title="Correlation Significance (FDR-adjusted q-values)"
    )
    qval_fig.update_layout(height=400)

    return corr_fig, qval_fig

def create_anomaly_plot(analytics):
    """Create anomaly detection visualization."""
    anomalies = analytics.current_stats.get('anomalies', [])
    
    if not anomalies:
        return None
    
    df_anomalies = pd.DataFrame(anomalies)
    
    fig = go.Figure()
    
    for pollutant in df_anomalies['pollutant'].unique():
        pollutant_data = df_anomalies[df_anomalies['pollutant'] == pollutant]
        
        fig.add_trace(go.Scatter(
            x=pollutant_data['timestamp'],
            y=pollutant_data['value'],
            mode='markers',
            name=f'{pollutant} Anomalies',
            marker=dict(
                size=10,
                color=pollutant_data['z_score'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Z-Score")
            ),
            hovertemplate=f'<b>{pollutant} Anomaly</b><br>' +
                         'Time: %{x}<br>' +
                         'Value: %{y}<br>' +
                         'Z-Score: %{marker.color}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Detected Anomalies",
        xaxis_title="Time",
        yaxis_title="Pollutant Value (Mixed Units)",
        height=400
    )
    
    return fig

def create_statistics_summary(df: pd.DataFrame):
    """Create comprehensive statistics summary for all pollutants."""
    if df.empty:
        return None, None, None
    
    # Define pollutant columns, display names, and units
    pollutant_info = {
        'co_gt': {'name': 'CO (Carbon Monoxide)', 'unit': 'mg/m³', 'color': '#1f77b4'},
        'nox_gt': {'name': 'NOx (Nitrogen Oxides)', 'unit': 'ppb', 'color': '#ff7f0e'},
        'no2_gt': {'name': 'NO2 (Nitrogen Dioxide)', 'unit': 'µg/m³', 'color': '#2ca02c'},
        'c6h6_gt': {'name': 'Benzene', 'unit': 'µg/m³', 'color': '#d62728'}
    }
    
    pollutant_columns = list(pollutant_info.keys())
    pollutant_names = [info['name'] for info in pollutant_info.values()]
    pollutant_colors = [info['color'] for info in pollutant_info.values()]
    
    # Calculate comprehensive statistics for each pollutant
    stats_data = []
    
    for col in pollutant_columns:
        if col in df.columns:
            info = pollutant_info[col]
            name = info['name']
            unit = info['unit']
            color = info['color']
            
            # Convert to numeric and remove NaN values
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            
            if len(values) > 0:
                stats = {
                    'Pollutant': name,
                    'Unit': unit,
                    'Count': len(values),
                    'Mean': values.mean(),
                    'Median': values.median(),
                    'Std Dev': values.std(),
                    'Variance': values.var(),
                    'Min': values.min(),
                    'Max': values.max(),
                    'Q1 (25%)': values.quantile(0.25),
                    'Q3 (75%)': values.quantile(0.75),
                    'IQR': values.quantile(0.75) - values.quantile(0.25),
                    'Skewness': values.skew(),
                    'Kurtosis': values.kurtosis(),
                    'Range': values.max() - values.min(),
                    'CV (%)': (values.std() / values.mean() * 100) if values.mean() != 0 else 0
                }
                stats_data.append(stats)
    
    if not stats_data:
        return None, None, None
    
    # Create statistics DataFrame
    stats_df = pd.DataFrame(stats_data)
    
    # Create box plot for distribution visualization
    box_fig = go.Figure()
    
    for col in pollutant_columns:
        if col in df.columns:
            info = pollutant_info[col]
            name = info['name']
            unit = info['unit']
            color = info['color']
            
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(values) > 0:
                box_fig.add_trace(go.Box(
                    y=values,
                    name=f"{name} ({unit})",
                    marker_color=color,
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8
                ))
    
    box_fig.update_layout(
        title="Pollutant Distributions (Box Plots) - Note: Different Units",
        yaxis_title="Concentration",
        height=500,
        showlegend=True,
        boxmode='group'
    )
    
    # Create summary metrics visualization
    metrics_fig = go.Figure()
    
    # Prepare data for metrics chart
    pollutants = stats_df['Pollutant'].tolist()
    means = stats_df['Mean'].tolist()
    stds = stats_df['Std Dev'].tolist()
    
    # Add mean values
    metrics_fig.add_trace(go.Bar(
        name='Mean',
        x=pollutants,
        y=means,
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(pollutants)],
        error_y=dict(type='data', array=stds, visible=True),
        text=[f'{m:.1f}' for m in means],
        textposition='auto'
    ))
    
    metrics_fig.update_layout(
        title="Average Pollutant Levels with Standard Deviation - ⚠️ Different Units",
        xaxis_title="Pollutant",
        yaxis_title="Concentration (Mixed Units)",
        height=400,
        showlegend=False
    )
    
    return stats_df, box_fig, metrics_fig

def create_temporal_patterns_from_df(df: pd.DataFrame):
    """Create temporal pattern analysis directly from filtered DataFrame."""
    if df.empty or 'datetime' not in df.columns:
        return None
    
    # Define pollutants and their colors
    pollutants = [
        ('co_gt', 'CO (mg/m³)', '#1f77b4'),
        ('nox_gt', 'NOx (ppb)', '#ff7f0e'),
        ('no2_gt', 'NO2 (µg/m³)', '#2ca02c'),
        ('c6h6_gt', 'Benzene (µg/m³)', '#d62728')
    ]
    
    # Check which pollutants are available
    available_pollutants = []
    for pollutant_code, pollutant_name, color in pollutants:
        if pollutant_code in df.columns:
            available_pollutants.append((pollutant_code, pollutant_name, color))
    
    if not available_pollutants:
        return None
    
    # Extract hour and day of week from datetime
    df_analysis = df.copy()
    df_analysis['hour'] = df_analysis['datetime'].dt.hour
    df_analysis['day_of_week'] = df_analysis['datetime'].dt.dayofweek
    
    # Create separate figures for independent legends
    from plotly.subplots import make_subplots
    
    # Hourly patterns figure
    hourly_fig = go.Figure()
    for pollutant_code, pollutant_name, color in available_pollutants:
        # Convert to numeric and remove NaN values
        pollutant_values = pd.to_numeric(df_analysis[pollutant_code], errors='coerce')
        df_analysis[f'{pollutant_code}_numeric'] = pollutant_values
        
        # Group by hour and calculate mean
        hourly_means = df_analysis.groupby('hour')[f'{pollutant_code}_numeric'].mean()
        
        if not hourly_means.empty:
            hours = hourly_means.index.tolist()
            values = hourly_means.values.tolist()
            
            hourly_fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=values,
                    mode='lines+markers',
                    name=pollutant_name,
                    line=dict(color=color, width=2)
                )
            )
    
    hourly_fig.update_layout(
        title="Hourly Patterns",
        xaxis_title="Hour of Day",
        yaxis_title="Average Pollutant Level (Mixed Units)",
        height=400,
        showlegend=True
    )
    
    # Daily patterns figure
    daily_fig = go.Figure()
    for pollutant_code, pollutant_name, color in available_pollutants:
        # Group by day of week and calculate mean
        daily_means = df_analysis.groupby('day_of_week')[f'{pollutant_code}_numeric'].mean()
        
        if not daily_means.empty:
            days = daily_means.index.tolist()
            values = daily_means.values.tolist()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            daily_fig.add_trace(
                go.Scatter(
                    x=[day_names[d] for d in days],
                    y=values,
                    mode='lines+markers',
                    name=pollutant_name,
                    line=dict(color=color, width=2)
                )
            )
    
    daily_fig.update_layout(
        title="Daily Patterns",
        xaxis_title="Day of Week",
        yaxis_title="Average Pollutant Level (Mixed Units)",
        height=400,
        showlegend=True
    )
    
    # Seasonal patterns figure
    seasonal_fig = go.Figure()
    
    # Add seasonal background colors - more distinct and vibrant
    seasonal_colors = {
        'Winter': 'rgba(100, 149, 237, 0.25)',  # Cornflower blue
        'Spring': 'rgba(50, 205, 50, 0.25)',    # Lime green
        'Summer': 'rgba(255, 165, 0, 0.25)',    # Orange
        'Fall': 'rgba(255, 69, 0, 0.25)'         # Red orange
    }
    
    # Add seasonal background rectangles with slight overlaps to cover entire year
    seasonal_fig.add_vrect(x0="Dec", x1="Mar", fillcolor=seasonal_colors['Winter'], 
                          layer="below", line_width=0, annotation_text="Winter<br>(Dec-Feb)", 
                          annotation_position="top left", annotation_font_size=10)
    seasonal_fig.add_vrect(x0="Mar", x1="Jun", fillcolor=seasonal_colors['Spring'], 
                          layer="below", line_width=0, annotation_text="Spring<br>(Mar-May)", 
                          annotation_position="top left", annotation_font_size=10)
    seasonal_fig.add_vrect(x0="Jun", x1="Sep", fillcolor=seasonal_colors['Summer'], 
                          layer="below", line_width=0, annotation_text="Summer<br>(Jun-Aug)", 
                          annotation_position="top left", annotation_font_size=10)
    seasonal_fig.add_vrect(x0="Sep", x1="Dec", fillcolor=seasonal_colors['Fall'], 
                          layer="below", line_width=0, annotation_text="Fall<br>(Sep-Nov)", 
                          annotation_position="top left", annotation_font_size=10)
    
    for pollutant_code, pollutant_name, color in available_pollutants:
        # Group by month and calculate mean
        monthly_means = df_analysis.groupby(df_analysis['datetime'].dt.month)[f'{pollutant_code}_numeric'].mean()
        
        if not monthly_means.empty:
            months = monthly_means.index.tolist()
            values = monthly_means.values.tolist()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            seasonal_fig.add_trace(
                go.Scatter(
                    x=[month_names[m-1] for m in months],
                    y=values,
                    mode='lines+markers',
                    name=pollutant_name,
                    line=dict(color=color, width=3),
                    marker=dict(size=8)
                )
            )
    
    seasonal_fig.update_layout(
        title="Seasonal Patterns (Monthly) - How pollutants vary across seasons",
        xaxis_title="Month",
        yaxis_title="Average Pollutant Level (Mixed Units)",
        height=450,
        showlegend=True,
        hovermode='x unified',
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text="<b>Seasonal Coverage:</b><br>• Winter: Dec-Mar (includes Dec-Feb)<br>• Spring: Mar-Jun (includes Mar-May)<br>• Summer: Jun-Sep (includes Jun-Aug)<br>• Fall: Sep-Dec (includes Sep-Nov)",
                showarrow=False,
                font=dict(size=10, color="gray"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        ]
    )
    
    return hourly_fig, daily_fig, seasonal_fig

def create_temporal_patterns(analytics):
    """Create temporal pattern analysis for all pollutants."""
    patterns = analytics.current_stats.get('temporal_patterns', {})
    
    if not patterns:
        return None
    
    # Define pollutants and their colors
    pollutants = [
        ('co_gt', 'CO (mg/m³)', '#1f77b4'),
        ('nox_gt', 'NOx (ppb)', '#ff7f0e'),
        ('no2_gt', 'NO2 (µg/m³)', '#2ca02c'),
        ('c6h6_gt', 'Benzene (µg/m³)', '#d62728')
    ]
    
    # Check which patterns are available
    available_patterns = []
    for pollutant_code, pollutant_name, color in pollutants:
        hourly_key = f'hourly_{pollutant_code}_pattern'
        daily_key = f'daily_{pollutant_code}_pattern'
        if hourly_key in patterns or daily_key in patterns:
            available_patterns.append((pollutant_code, pollutant_name, color))
    
    if not available_patterns:
        return None
    
    # Create subplots - 2 columns for hourly and daily patterns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Hourly Patterns', 'Daily Patterns'),
        horizontal_spacing=0.1
    )
    
    # Hourly patterns
    for pollutant_code, pollutant_name, color in available_patterns:
        hourly_key = f'hourly_{pollutant_code}_pattern'
        if hourly_key in patterns:
            hours = list(patterns[hourly_key].keys())
            values = list(patterns[hourly_key].values())
            
            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=values,
                    mode='lines+markers',
                    name=f'Hourly {pollutant_name}',
                    line=dict(color=color, width=2)
                ),
                row=1, col=1
            )
    
    # Daily patterns
    for pollutant_code, pollutant_name, color in available_patterns:
        daily_key = f'daily_{pollutant_code}_pattern'
        if daily_key in patterns:
            days = list(patterns[daily_key].keys())
            values = list(patterns[daily_key].values())
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            fig.add_trace(
                go.Scatter(
                    x=[day_names[d] for d in days],
                    y=values,
                    mode='lines+markers',
                    name=f'Daily {pollutant_name}',
                    line=dict(color=color, width=2)
                ),
                row=1, col=2
            )
    
    fig.update_layout(
        height=800,
        title_text="Temporal Patterns in Pollutant Levels",
        title_x=0.5,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
    fig.update_xaxes(title_text="Day of Week", row=2, col=1)
    fig.update_yaxes(title_text="Average Pollutant Level", row=1, col=1)
    fig.update_yaxes(title_text="Average Pollutant Level", row=2, col=1)
    
    return fig

def detect_outliers_zscore(df: pd.DataFrame, columns: list, threshold: float = 3.0) -> pd.DataFrame:
    """Detect outliers using Z-score method."""
    outliers = []
    
    for col in columns:
        if col in df.columns:
            # Convert to numeric but keep original indices
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            
            # Find valid (non-NaN) values
            valid_mask = numeric_values.notna()
            if valid_mask.sum() < 10:  # Need minimum data points
                continue
            
            # Get valid values and their original indices
            valid_values = numeric_values[valid_mask]
            valid_indices = valid_values.index
            
            # Calculate Z-scores on valid values
            z_scores = np.abs(stats.zscore(valid_values))
            outlier_mask = z_scores > threshold
            
            if outlier_mask.any():
                # Get outlier indices in the original DataFrame
                outlier_indices = valid_indices[outlier_mask]
                outlier_values = valid_values[outlier_mask]
                outlier_z_scores = z_scores[outlier_mask]
                
                for i, idx in enumerate(outlier_indices):
                    # Get timestamp using the original DataFrame index
                    timestamp = 'unknown'
                    try:
                        if 'datetime' in df.columns and idx in df.index:
                            timestamp = df.loc[idx, 'datetime']
                        elif 'datetime' in df.columns and idx < len(df):
                            timestamp = df.iloc[idx]['datetime']
                    except (IndexError, KeyError):
                        timestamp = 'unknown'
                    
                    outliers.append({
                        'index': idx,
                        'column': col,
                        'value': float(outlier_values.iloc[i]),
                        'z_score': float(outlier_z_scores[i]),
                        'timestamp': timestamp,
                        'method': 'z_score'
                    })
    
    return pd.DataFrame(outliers)

def detect_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Detect outliers using Interquartile Range (IQR) method."""
    outliers = []
    
    for col in columns:
        if col in df.columns:
            # Convert to numeric but keep original indices
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            
            # Find valid (non-NaN) values
            valid_mask = numeric_values.notna()
            if valid_mask.sum() < 10:  # Need minimum data points
                continue
            
            # Get valid values and their original indices
            valid_values = numeric_values[valid_mask]
            valid_indices = valid_values.index
            
            # Calculate IQR bounds
            Q1 = valid_values.quantile(0.25)
            Q3 = valid_values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            outlier_mask = (valid_values < lower_bound) | (valid_values > upper_bound)
            
            if outlier_mask.any():
                # Get outlier indices in the original DataFrame
                outlier_indices = valid_indices[outlier_mask]
                outlier_values = valid_values[outlier_mask]
                
                for i, idx in enumerate(outlier_indices):
                    # Get timestamp using the original DataFrame index
                    timestamp = 'unknown'
                    try:
                        if 'datetime' in df.columns and idx in df.index:
                            timestamp = df.loc[idx, 'datetime']
                        elif 'datetime' in df.columns and idx < len(df):
                            timestamp = df.iloc[idx]['datetime']
                    except (IndexError, KeyError):
                        timestamp = 'unknown'
                    
                    outliers.append({
                        'index': idx,
                        'column': col,
                        'value': float(outlier_values.iloc[i]),
                        'iqr_score': float((outlier_values.iloc[i] - Q1) / IQR) if IQR > 0 else 0,
                        'timestamp': timestamp,
                        'method': 'iqr'
                    })
    
    return pd.DataFrame(outliers)

def detect_outliers_isolation_forest(df: pd.DataFrame, columns: list, contamination: float = 0.1) -> pd.DataFrame:
    """Detect outliers using Isolation Forest method."""
    outliers = []
    
    # Prepare data for Isolation Forest
    numeric_cols = [col for col in columns if col in df.columns]
    if len(numeric_cols) < 2:
        return pd.DataFrame(outliers)
    
    # Get numeric data
    data = df[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if len(data) < 10:
        return pd.DataFrame(outliers)
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = iso_forest.fit_predict(data_scaled)
    
    # Get outlier indices
    outlier_indices = data[outlier_labels == -1].index
    
    for idx in outlier_indices:
        if idx < len(df):
            # Get timestamp using the original DataFrame index
            timestamp = 'unknown'
            try:
                if 'datetime' in df.columns and idx in df.index:
                    timestamp = df.loc[idx, 'datetime']
                elif 'datetime' in df.columns and idx < len(df):
                    timestamp = df.iloc[idx]['datetime']
            except (IndexError, KeyError):
                timestamp = 'unknown'
            
            outliers.append({
                'index': idx,
                'columns': ', '.join(numeric_cols),
                'timestamp': timestamp,
                'method': 'isolation_forest',
                'anomaly_score': float(iso_forest.decision_function(data_scaled[data.index == idx])[0])
            })
    
    return pd.DataFrame(outliers)

def create_outlier_visualization(df: pd.DataFrame, outliers_df: pd.DataFrame, method: str):
    """Create visualization showing outliers on time series."""
    if outliers_df.empty or df.empty:
        return None
    
    # Get pollutant columns
    pollutant_cols = ['co_gt', 'nox_gt', 'no2_gt', 'c6h6_gt']
    available_cols = [col for col in pollutant_cols if col in df.columns]
    
    if not available_cols:
        return None
    
    fig = make_subplots(
        rows=len(available_cols), cols=1,
        subplot_titles=[f'{col.upper()} Outliers ({method})' for col in available_cols],
        vertical_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, col in enumerate(available_cols):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce')
            
            # Add main time series
            fig.add_trace(
                go.Scatter(
                    x=df['datetime'],
                    y=values,
                    mode='lines',
                    name=f'{col}',
                    line=dict(color=colors[i], width=1),
                    opacity=0.7
                ),
                row=i+1, col=1
            )
            
            # Add outliers
            col_outliers = outliers_df[outliers_df['column'] == col] if 'column' in outliers_df.columns else pd.DataFrame()
            if not col_outliers.empty:
                outlier_values = []
                outlier_times = []
                
                for _, outlier_row in col_outliers.iterrows():
                    idx = outlier_row['index']
                    outlier_value = outlier_row['value']
                    outlier_time = outlier_row['timestamp']
                    
                    # Use the actual outlier values and times from detection
                    outlier_values.append(outlier_value)
                    outlier_times.append(outlier_time)
                
                fig.add_trace(
                    go.Scatter(
                        x=outlier_times,
                        y=outlier_values,
                        mode='markers',
                        name=f'{col} outliers',
                        marker=dict(color='red', size=8, symbol='x'),
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
    
    fig.update_layout(
        height=300 * len(available_cols),
        title_text=f"Outlier Detection Results ({method})",
        title_x=0.5,
        showlegend=False
    )
    
    return fig

def main():
    """Main dashboard function."""
    
    # Initialize session state for efficient updates
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Initialize active tab state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0  # Default to first tab (Time Series)
    
    # Header
    st.markdown('<h1 class="main-header">🌍 Air Quality Streaming Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize analytics
    analytics = initialize_analytics()
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Time range selector
    csv_path = "../phase_1_streaming_infrastructure/data/processed/air_quality_clean.csv"
    
    # Initialize time range in session state
    if 'time_range' not in st.session_state:
        st.session_state.time_range = "Last 30 days"
    
    range_key = st.sidebar.selectbox(
        "Time range",
        options=["Today", "Last 7 days", "Last 30 days", "Last 365 days", "All data"],
        index=["Today", "Last 7 days", "Last 30 days", "Last 365 days", "All data"].index(st.session_state.time_range)
    )
    
    # Update session state when time range changes
    if range_key != st.session_state.time_range:
        st.session_state.time_range = range_key

    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (every 5 seconds)", value=True)
    
    # Refresh interval
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.session_state.last_refresh = time.time()
        st.rerun()
    
    # Process new data for summary
    current_stats = analytics.process_new_data()
    summary_stats = analytics.get_summary_stats()
    
    # Load filtered DataFrame for visualizations
    df_filtered = load_filtered_data(csv_path, range_key)
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Records",
            value=f"{summary_stats['total_records']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="⏰ Last Update",
            value=summary_stats['last_update'][:19] if summary_stats['last_update'] else "Never",
            delta=None
        )
    
    with col3:
        st.metric(
            label="🚨 Recent Anomalies",
            value=summary_stats['recent_anomalies'],
            delta=None
        )
    
    with col4:
        file_status = "✅ Connected" if summary_stats['file_exists'] else "❌ Disconnected"
        st.metric(
            label="📁 Data Source",
            value=file_status,
            delta=None
        )
    
    # Simple tab selection - let Streamlit handle the state
    tab_names = ["📈 Time Series", "📊 Statistics", "🔗 Correlations", "🚨 Anomalies", "📊 Patterns", "🔍 Outliers", "🔮 Forecasts"]
    
    # Tab selector in sidebar
    selected_tab = st.sidebar.radio(
        "Select View",
        options=tab_names,
        key="tab_selector"
    )
    
    # Conditional rendering based on selected tab
    if selected_tab == "📈 Time Series":
        st.header("Real-Time Pollutant Levels")
        
        # Stacked, full-width time series plots for selected range
        if df_filtered.empty:
            st.info("No data available for the selected range. Make sure Phase 1 consumer is running.")
        else:
            st.caption(f"Showing: {range_key} | Records: {len(df_filtered):,}")
            create_stacked_timeseries(df_filtered)
    
    elif selected_tab == "📊 Statistics":
        st.header("Pollutant Statistics Summary")
        
        if df_filtered.empty:
            st.info("No data available for statistical analysis in the selected range.")
        else:
            st.caption(f"Statistical analysis for: {range_key} | Records: {len(df_filtered):,}")
            
            # Get statistics summary
            stats_result = create_statistics_summary(df_filtered)
            if stats_result[0] is not None:
                stats_df, box_fig, metrics_fig = stats_result
                
                # Display summary metrics
                st.subheader("📈 Summary Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", f"{len(df_filtered):,}")
                with col2:
                    st.metric("Pollutants Analyzed", len(stats_df))
                with col3:
                    st.metric("Data Points", f"{stats_df['Count'].sum():,}")
                with col4:
                    avg_cv = stats_df['CV (%)'].mean()
                    st.metric("Average CV", f"{avg_cv:.1f}%")
                
                # Display charts
                st.subheader("📊 Visualizations")
                
                # Add unit warning
                st.warning("⚠️ **Important**: Pollutants are measured in different units - direct comparison may be misleading!")
                
                st.plotly_chart(metrics_fig, width='stretch', key="stats_metrics")
                st.plotly_chart(box_fig, width='stretch', key="stats_boxplot")
                
                # Detailed statistics table (restored)
                st.subheader("📋 Detailed Statistics")
                display_stats = stats_df.copy()
                numeric_cols = [
                    'Mean', 'Median', 'Std Dev', 'Variance', 'Min', 'Max',
                    'Q1 (25%)', 'Q3 (75%)', 'IQR', 'Skewness', 'Kurtosis', 'Range', 'CV (%)'
                ]
                for col in numeric_cols:
                    if col in display_stats.columns:
                        display_stats[col] = display_stats[col].round(3)
                st.dataframe(display_stats, width='stretch')
            else:
                st.info("Insufficient data for statistical analysis.")
    
    elif selected_tab == "🔗 Correlations":
        st.header("Pollutant Correlations")
        
        # Correlation heatmap for selected range with significance
        result = create_correlation_heatmap_from_df(df_filtered)
        if result:
            if isinstance(result, tuple):
                corr_fig, qval_fig = result
                cols = st.columns(2)
                with cols[0]:
                    st.plotly_chart(corr_fig, width='stretch', key="corr_heatmap")
                with cols[1]:
                    st.plotly_chart(qval_fig, width='stretch', key="corr_qvals")
            else:
                st.plotly_chart(result, width='stretch', key="corr_heatmap")
        else:
            st.info("Insufficient data for correlation analysis in the selected range.")
    
    elif selected_tab == "🚨 Anomalies":
        st.header("Anomaly Detection")
        
        # Anomaly plot
        anomaly_fig = create_anomaly_plot(analytics)
        if anomaly_fig:
            st.plotly_chart(anomaly_fig, width='stretch', key="anomaly_scatter")
        else:
            st.success("No anomalies detected in recent data.")
        
        # Anomaly details
        anomalies = analytics.current_stats.get('anomalies', [])
        if anomalies:
            st.subheader("Anomaly Details")
            anomaly_df = pd.DataFrame(anomalies)
            st.dataframe(anomaly_df, width='stretch')
    
    elif selected_tab == "📊 Patterns":
        st.header("Temporal Patterns")
        
        if df_filtered.empty:
            st.info("No data available for temporal pattern analysis in the selected range.")
        else:
            st.caption(f"Analyzing patterns in: {range_key} | Records: {len(df_filtered):,}")
            
            # Use the new function that works with filtered data
            patterns_result = create_temporal_patterns_from_df(df_filtered)
            if patterns_result:
                hourly_fig, daily_fig, seasonal_fig = patterns_result
                st.plotly_chart(hourly_fig, width='stretch', key="patterns_hourly")
                st.plotly_chart(daily_fig, width='stretch', key="patterns_daily")
                st.plotly_chart(seasonal_fig, width='stretch', key="patterns_seasonal")
            else:
                st.info("Insufficient data for temporal pattern analysis.")
    
    elif selected_tab == "🔍 Outliers":
        st.header("Outlier Detection")
        
        if df_filtered.empty:
            st.info("No data available for outlier analysis in the selected range.")
        else:
            st.caption(f"Analyzing outliers in: {range_key} | Records: {len(df_filtered):,}")
            
            # Outlier detection configuration
            st.info("**How outliers are calculated:** Each method analyzes pollutant values (CO, NOx, NO2, Benzene) to identify unusual readings that deviate significantly from normal patterns.")
            
            # Initialize session state
            if 'detection_method' not in st.session_state:
                st.session_state.detection_method = "Z-Score"
            if 'calculation_mode' not in st.session_state:
                st.session_state.calculation_mode = "Per Pollutant"
            if 'threshold' not in st.session_state:
                st.session_state.threshold = 3.0
            if 'contamination' not in st.session_state:
                st.session_state.contamination = 0.1
            
            # Use form to prevent reruns
            with st.form("outlier_detection_form"):
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    detection_method = st.selectbox(
                        "Detection Method",
                        ["Z-Score", "IQR", "Isolation Forest"],
                        index=["Z-Score", "IQR", "Isolation Forest"].index(st.session_state.detection_method),
                        help="Z-Score: Statistical outliers (>3σ), IQR: Values outside 1.5×IQR, Isolation Forest: ML-based anomaly detection"
                    )
                
                with col2:
                    if detection_method == "Z-Score":
                        threshold = st.slider("Z-Score Threshold", 2.0, 5.0, st.session_state.threshold, 0.1)
                    elif detection_method == "Isolation Forest":
                        contamination = st.slider("Contamination Rate", 0.01, 0.3, st.session_state.contamination, 0.01)
                
                with col3:
                    calculation_mode = st.radio(
                        "Calculation Mode",
                        ["Per Pollutant", "Multivariate"],
                        index=["Per Pollutant", "Multivariate"].index(st.session_state.calculation_mode),
                        help="Per Pollutant: Analyze each pollutant separately. Multivariate: Analyze all pollutants together (Isolation Forest only)."
                    )
                
                # Submit button
                submitted = st.form_submit_button("Update Outlier Detection")
                
                # Update session state when form is submitted
                if submitted:
                    st.session_state.detection_method = detection_method
                    st.session_state.calculation_mode = calculation_mode
                    if detection_method == "Z-Score":
                        st.session_state.threshold = threshold
                    elif detection_method == "Isolation Forest":
                        st.session_state.contamination = contamination
                    st.rerun()
            
            # Use session state values for detection
            detection_method = st.session_state.detection_method
            calculation_mode = st.session_state.calculation_mode
            threshold = st.session_state.threshold
            contamination = st.session_state.contamination
            
            # Detect outliers
            pollutant_cols = ['co_gt', 'nox_gt', 'no2_gt', 'c6h6_gt']
            available_cols = [col for col in pollutant_cols if col in df_filtered.columns]
            
            if len(available_cols) >= 1:
                if calculation_mode == "Per Pollutant":
                    # Analyze each pollutant separately
                    if detection_method == "Z-Score":
                        outliers_df = detect_outliers_zscore(df_filtered, available_cols, threshold)
                    elif detection_method == "IQR":
                        outliers_df = detect_outliers_iqr(df_filtered, available_cols)
                    else:  # Isolation Forest
                        outliers_df = detect_outliers_isolation_forest(df_filtered, available_cols, contamination)
                else:  # Multivariate
                    if detection_method == "Isolation Forest":
                        outliers_df = detect_outliers_isolation_forest(df_filtered, available_cols, contamination)
                    else:
                        st.warning("Multivariate mode is only available for Isolation Forest method.")
                        outliers_df = pd.DataFrame()
                
                # Display results
                if not outliers_df.empty:
                    st.success(f"Found {len(outliers_df)} outliers using {detection_method}")
                    
                    # Show debug info
                    with st.expander("Debug Information"):
                        st.write(f"**Data Info:**")
                        st.write(f"- Total records: {len(df_filtered)}")
                        st.write(f"- Available columns: {available_cols}")
                        st.write(f"- Data range: {df_filtered['datetime'].min()} to {df_filtered['datetime'].max()}")
                        
                        for col in available_cols:
                            if col in df_filtered.columns:
                                col_data = pd.to_numeric(df_filtered[col], errors='coerce')
                                valid_count = col_data.notna().sum()
                                st.write(f"- {col}: {valid_count} valid values, range: {col_data.min():.2f} to {col_data.max():.2f}")
                        
                        st.write(f"**Outlier Detection Results:**")
                        st.write(f"- Method: {detection_method}")
                        st.write(f"- Total outliers found: {len(outliers_df)}")
                        
                        # Show sample outliers for verification
                        if not outliers_df.empty:
                            st.write("**Sample Outliers:**")
                            sample_outliers = outliers_df.head(5)
                            for _, row in sample_outliers.iterrows():
                                st.write(f"- {row['column']}: value={row['value']:.2f}, timestamp={row['timestamp']}")
                                if detection_method == "Z-Score":
                                    st.write(f"  Z-score: {row['z_score']:.2f}")
                                elif detection_method == "IQR":
                                    st.write(f"  IQR score: {row['iqr_score']:.2f}")
                    
                    # Show outlier visualization
                    outlier_fig = create_outlier_visualization(df_filtered, outliers_df, detection_method)
                    if outlier_fig:
                        st.plotly_chart(outlier_fig, width='stretch', key="outliers_ts")
                    
                    # Show outlier details table
                    st.subheader("Outlier Details")
                    st.dataframe(outliers_df, width='stretch')
                    
                    # Download button
                    csv = outliers_df.to_csv(index=False)
                    st.download_button(
                        label="Download Outliers CSV",
                        data=csv,
                        file_name=f"outliers_{detection_method.lower()}_{range_key.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info(f"No outliers detected using {detection_method}")
                    
                    # Show debug info when no outliers found
                    with st.expander("Debug Information (No Outliers Found)"):
                        st.write(f"**Data Info:**")
                        st.write(f"- Total records: {len(df_filtered)}")
                        st.write(f"- Available columns: {available_cols}")
                        
                        for col in available_cols:
                            if col in df_filtered.columns:
                                col_data = pd.to_numeric(df_filtered[col], errors='coerce')
                                valid_count = col_data.notna().sum()
                                if valid_count > 0:
                                    mean_val = col_data.mean()
                                    std_val = col_data.std()
                                    st.write(f"- {col}: {valid_count} valid values, mean: {mean_val:.2f}, std: {std_val:.2f}")
                                    
                                    if detection_method == "Z-Score":
                                        max_z = np.abs((col_data - mean_val) / std_val).max()
                                        st.write(f"  Max Z-score: {max_z:.2f} (threshold: {threshold})")
                                    elif detection_method == "IQR":
                                        Q1 = col_data.quantile(0.25)
                                        Q3 = col_data.quantile(0.75)
                                        IQR = Q3 - Q1
                                        st.write(f"  IQR: {IQR:.2f}, bounds: [{Q1-1.5*IQR:.2f}, {Q3+1.5*IQR:.2f}]")
            else:
                st.warning("Insufficient pollutant data for outlier detection")

    elif selected_tab == "🔮 Forecasts":
        st.header("Next 6 Hours Forecasts (NOx)")

        if df_filtered.empty:
            st.info("No data available to generate forecasts.")
        else:
            # Resolve models dir from project root to avoid cwd issues
            models_dir = os.path.join(_PROJECT_ROOT, "phase_3_predictive_analytics", "models")
            lr_df = pd.DataFrame()
            sar_df = pd.DataFrame()
            xgb_df = pd.DataFrame()
            lr_model_path = os.path.join(models_dir, "lr", "model.pkl")
            sar_model_path = os.path.join(models_dir, "sarima", "model.pkl")
            xgb_model_path = os.path.join(models_dir, "xgb", "model.pkl")

            
            # Load model metrics
            lr_metrics_path = os.path.join(models_dir, "lr", "metrics.json")
            sar_metrics_path = os.path.join(models_dir, "sarima", "metrics.json")
            xgb_metrics_path = os.path.join(models_dir, "xgb", "metrics.json")
            
            # Display model performance metrics
            st.subheader("Model Performance Metrics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Linear Regression Model**")
                if os.path.exists(lr_metrics_path):
                    try:
                        import json
                        with open(lr_metrics_path, 'r') as f:
                            lr_metrics = json.load(f)
                        
                        test_metrics = lr_metrics.get('test', {})
                        baseline_metrics = lr_metrics.get('test', {})
                        
                        st.metric("MAE", f"{test_metrics.get('mae', 0):.1f}", 
                                delta=f"{baseline_metrics.get('mae_baseline', 0) - test_metrics.get('mae', 0):.1f} improvement")
                        st.metric("RMSE", f"{test_metrics.get('rmse', 0):.1f}", 
                                delta=f"{baseline_metrics.get('rmse_baseline', 0) - test_metrics.get('rmse', 0):.1f} improvement")
                        st.metric("R²", f"{test_metrics.get('r2', 0):.3f}")
                        st.metric("sMAPE", f"{test_metrics.get('smape', 0):.1f}%")
                        
                        # Statistical significance
                        sig_test = test_metrics.get('significance_test', {})
                        if sig_test and 'significant' in sig_test:
                            significance_text = "✅ Significant" if sig_test['significant'] == 1 else "❌ Not significant"
                            p_value = sig_test.get('p_value', 0)
                            st.metric("Statistical Significance", significance_text, 
                                    delta=f"p={p_value:.4f}")
                        
                        # Model info
                        data_summary = lr_metrics.get('data_summary', {})
                        st.caption(f"Features: {data_summary.get('features', 0)} | Test samples: {data_summary.get('test', {}).get('rows', 0)}")
                        
                    except Exception as e:
                        st.error(f"Failed to load LR metrics: {e}")
                else:
                    st.info("LR metrics not available")
            
            with col2:
                st.markdown("**SARIMA Model**")
                if os.path.exists(sar_metrics_path):
                    try:
                        import json
                        with open(sar_metrics_path, 'r') as f:
                            sar_metrics = json.load(f)
                        
                        test_metrics = sar_metrics.get('test', {})
                        baseline_metrics = sar_metrics.get('test', {})
                        
                        st.metric("MAE", f"{test_metrics.get('mae', 0):.1f}", 
                                delta=f"{baseline_metrics.get('mae_baseline', 0) - test_metrics.get('mae', 0):.1f} improvement")
                        st.metric("RMSE", f"{test_metrics.get('rmse', 0):.1f}", 
                                delta=f"{baseline_metrics.get('rmse_baseline', 0) - test_metrics.get('rmse', 0):.1f} improvement")
                        st.metric("R²", f"{test_metrics.get('r2', 0):.3f}")
                        st.metric("sMAPE", f"{test_metrics.get('smape', 0):.1f}%")
                        
                        # Statistical significance
                        sig_test = test_metrics.get('significance_test', {})
                        if sig_test and 'significant' in sig_test:
                            significance_text = "✅ Significant" if sig_test['significant'] == 1 else "❌ Not significant"
                            p_value = sig_test.get('p_value', 0)
                            st.metric("Statistical Significance", significance_text, 
                                    delta=f"p={p_value:.4f}")
                        
                        # Model info
                        model_config = sar_metrics.get('model', {})
                        order = model_config.get('order', [])
                        seasonal_order = model_config.get('seasonal_order', [])
                        st.caption(f"Order: {order} × {seasonal_order}")
                        
                    except Exception as e:
                        st.error(f"Failed to load SARIMA metrics: {e}")
                else:
                    st.info("SARIMA metrics not available")
            
            with col3:
                st.markdown("**XGBoost Model**")
                if os.path.exists(xgb_metrics_path):
                    try:
                        import json
                        with open(xgb_metrics_path, 'r') as f:
                            xgb_metrics = json.load(f)
                        
                        test_metrics = xgb_metrics.get('test', {})
                        baseline_metrics = xgb_metrics.get('test', {})
                        
                        st.metric("MAE", f"{test_metrics.get('mae', 0):.1f}", 
                                delta=f"{baseline_metrics.get('mae_baseline', 0) - test_metrics.get('mae', 0):.1f} improvement")
                        st.metric("RMSE", f"{test_metrics.get('rmse', 0):.1f}", 
                                delta=f"{baseline_metrics.get('rmse_baseline', 0) - test_metrics.get('rmse', 0):.1f} improvement")
                        st.metric("R²", f"{test_metrics.get('r2', 0):.3f}")
                        st.metric("sMAPE", f"{test_metrics.get('smape', 0):.1f}%")
                        
                        # Statistical significance
                        sig_test = test_metrics.get('significance_test', {})
                        if sig_test and 'significant' in sig_test:
                            significance_text = "✅ Significant" if sig_test['significant'] == 1 else "❌ Not significant"
                            p_value = sig_test.get('p_value', 0)
                            st.metric("Statistical Significance", significance_text, 
                                    delta=f"p={p_value:.4f}")
                        
                        # Model info
                        data_summary = xgb_metrics.get('data_summary', {})
                        st.caption(f"Features: {data_summary.get('features', 0)} | Test samples: {data_summary.get('test', {}).get('rows', 0)}")
                        
                    except Exception as e:
                        st.error(f"Failed to load XGB metrics: {e}")
                else:
                    st.info("XGB metrics not available")
            
            st.markdown("---")

            # Generate predictions with safeguards
            try:
                if os.path.exists(lr_model_path):
                    lr_df = predict_lr_next_6(df_filtered, models_dir)
                else:
                    st.info("LR model not found. Train it first to enable LR forecasts.")
            except Exception as e:
                st.warning(f"LR forecast failed: {e}")

            try:
                if os.path.exists(sar_model_path):
                    sar_df = predict_sarima_next_6(df_filtered, models_dir)
                else:
                    st.info("SARIMA model not found. Train it first to enable SARIMA forecasts.")
            except Exception as e:
                st.warning(f"SARIMA forecast failed: {e}")

            try:
                if os.path.exists(xgb_model_path):
                    xgb_df = predict_xgb_next_6(df_filtered, models_dir)
                else:
                    st.info("XGB model not found. Train it first to enable XGB forecasts.")
            except Exception as e:
                st.warning(f"XGB forecast failed: {e}")

            if (lr_df is None or lr_df.empty) and (sar_df is None or sar_df.empty) and (xgb_df is None or xgb_df.empty):
                st.info("Forecast models not found or insufficient history to predict.")
            else:
                # Plot recent NOx plus forecasts
                nox_series = pd.to_numeric(df_filtered.get('no2_gt') if 'nox_gt' not in df_filtered.columns else df_filtered['nox_gt'], errors='coerce')
                recent_mask = df_filtered['datetime'] >= (df_filtered['datetime'].max() - pd.Timedelta(hours=48))
                base_x = df_filtered.loc[recent_mask, 'datetime']
                base_y = nox_series.loc[recent_mask]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=base_x, y=base_y, mode='lines', name='NOx (last 48h)', line=dict(color='#2ca02c', width=2)))

                if lr_df is not None and not lr_df.empty:
                    fig.add_trace(go.Scatter(x=lr_df['datetime'], y=lr_df['lr_pred'], mode='lines+markers', name='LR t+6', line=dict(color='#9467bd', dash='dash')))

                if sar_df is not None and not sar_df.empty:
                    fig.add_trace(go.Scatter(x=sar_df['datetime'], y=sar_df['sarima_mean'], mode='lines+markers', name='SARIMA mean', line=dict(color='#000000', dash='dot')))
                    fig.add_trace(go.Scatter(x=sar_df['datetime'], y=sar_df['sarima_lo'], mode='lines', name='PI low', line=dict(color='rgba(0,0,0,0.3)', dash='dot'), showlegend=False))
                    fig.add_trace(go.Scatter(x=sar_df['datetime'], y=sar_df['sarima_hi'], mode='lines', name='PI high', line=dict(color='rgba(0,0,0,0.3)', dash='dot'), fill='tonexty', fillcolor='rgba(0,0,0,0.08)', showlegend=False))

                if xgb_df is not None and not xgb_df.empty:
                    fig.add_trace(go.Scatter(
                        x=xgb_df['datetime'],
                        y=xgb_df['xgb_pred'],
                        mode='lines+markers',
                        name='XGBoost',
                        line=dict(color='#1f77b4', dash='dashdot')  # choose a distinct style
                    ))


                fig.update_layout(title="NOx Forecasts for Next 6 Hours", xaxis_title="Time", yaxis_title="NOx", height=400)
                st.plotly_chart(fig, width='stretch', key="forecasts_chart")

                # Table view
                st.subheader("Forecast Table")
                table = pd.DataFrame()
                if lr_df is not None and not lr_df.empty:
                    table = lr_df.rename(columns={'lr_pred': 'LR t+6'})
                if sar_df is not None and not sar_df.empty:
                    if table.empty:
                        table = sar_df.copy()
                    else:
                        table = table.merge(sar_df, on='datetime', how='outer')
                if xgb_df is not None and not xgb_df.empty:
                    if table.empty:
                        table = xgb_df.copy()
                    else:
                        table = table.merge(xgb_df, on='datetime', how='outer')
                if not table.empty:
                    st.dataframe(table, width='stretch')
                else:
                    st.info("No forecast rows available.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Dashboard Status**: " + 
        ("🟢 Active" if summary_stats['file_exists'] else "🔴 Inactive") +
        f" | **Last Update**: {summary_stats['last_update'] or 'Never'}"
    )
    
    # Auto-refresh logic - proper timing control
    if auto_refresh:
        current_time = time.time()
        time_since_refresh = current_time - st.session_state.last_refresh
        
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh = current_time
            st.rerun()
        else:
            # Show countdown timer
            remaining_time = refresh_interval - time_since_refresh
            st.sidebar.caption(f"⏱️ Next refresh in {remaining_time:.1f}s")
            
            # Create a placeholder for auto-refresh
            placeholder = st.empty()
            placeholder.markdown("")
            
            # Sleep for the remaining time and then refresh
            time.sleep(remaining_time)
            st.rerun()

if __name__ == "__main__":
    main()
