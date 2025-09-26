# Phase 2: Advanced Environmental Data Intelligence and Pattern Analysis

- **Name:** Santiago Bolaños Vega
- **Course:** Fundamentals of Operationalizing AI
- **Date:** September 23, 2025

## Objective
The objective of Phase 2 is to deliver a real-time streaming analytics dashboard that performs comprehensive exploratory data analysis (EDA) on clean air quality data. The dashboard visualizes temporal patterns, quantifies cross-pollutant relationships, detects anomalies and outliers, and integrates predictive forecasting capabilities while handling data quality issues in a resilient manner.

## Why Streamlit?
Streamlit was selected as the framework for this phase due to the following advantages:  
- Rapid iteration and interactivity with minimal boilerplate  
- Built-in session state and forms that preserve user selections and prevent disruptive reruns  
- First-class Plotly integration for interactive visualization  
- Simple local development and deployment workflow  
- Seamless compatibility with Phase 3 predictive models  

## 1) Data Flow
The data flow across components is as follows:  
1. Phase 1 continuously generates `air_quality_clean.csv`.  
2. `streaming_analytics.py` maintains sliding windows, computes summary statistics, correlations, and anomalies.  
3. `streaming_dashboard.py` polls the CSV file at controlled intervals, caches filtered data, and renders the visualizations.  
4. Phase 3 predictive models are loaded and integrated to provide real-time forecasts.  
5. Auto-refresh and manual refresh ensure the dashboard remains current without unnecessary reruns.  

### Project Structure
```
phase_2_data_intelligence/
├── streaming_analytics.py    # Core streaming analytics engine (sliding windows, anomalies)
├── streaming_dashboard.py    # Streamlit dashboard (visualization & interaction)
```
## 2) Dashboard Features

### 2.1 Time Series
- Full-width stacked time series for `CO (mg/m³)`, `NOx (ppb)`, `NO2 (µg/m³)`, and `Benzene (µg/m³)`  
- Range selector: Today, Last 7/30/365 days, or All data  
- Session state used to preserve tab position during range changes  

### 2.2 Statistics
- Summary metrics per pollutant: count, mean, median, standard deviation, variance, minimum, maximum, quartiles, interquartile range, skewness, kurtosis, range, and coefficient of variation  
- Visualizations: bar plots with error bars (mean ± standard deviation) and box plots  
- Unit awareness with clear labeling; warnings for mixed units  

### 2.3 Correlations
- Correlation heatmap across pollutants for the selected range  
- Side-by-side heatmap of statistical significance (Benjamini–Hochberg FDR-adjusted q-values)  
- Interpretation guide provided: darker q-values represent stronger evidence of correlation  

### 2.4 Anomalies
- Anomalies detected by the streaming analytics engine using Z-scores within sliding windows  
- Results include timestamps, pollutant values, and corresponding z-scores  
- Designed to complement outlier detection but focused on real-time deviation monitoring  

### 2.5 Patterns
- Hourly patterns (mean by hour) and day-of-week patterns (stacked visualization)  
- Seasonal identification (monthly means) with annotations and seasonal bands  

### 2.6 Outliers
- Detection methods: Z-Score threshold, IQR, and Isolation Forest (with contamination parameter)  
- Per-pollutant outlier identification with timestamps and markers on plots  
- Configurable form to adjust detection parameters; selections persist via session state  

### 2.7 Forecasts
The Forecasts tab directly addresses the requirements of **Phase 3: Predictive Analytics Model Development and Deployment**, while being embedded within the Phase 2 dashboard for real-time demonstration. This view ensures predictive modeling results are operationalized in the same interactive environment as the EDA.

- Real-time forecasting of NOx concentrations for the next six hours using Linear Regression and SARIMA models  
- Model performance metrics: MAE, RMSE, R², sMAPE with baseline comparison and significance testing  
- Forecast overlays on recent NOx time series with confidence intervals  
- Interactive forecast table with predictions, timestamps, and confidence levels  

**Phase 3 Integration Features included in this view:**  
- Automatic detection and loading of trained Linear Regression and SARIMA models  
- Real-time display of accuracy metrics with baseline comparison  
- Statistical significance testing (paired t-tests against naive baseline)  
- Comprehensive logging of prediction requests, results, failures, and performance metrics  
- Graceful error handling to ensure dashboard stability when models are unavailable  

By embedding Phase 3 outputs into the Phase 2 dashboard, the system provides an end-to-end demonstration of streaming data analysis combined with predictive forecasting, ensuring continuity across the project phases.

### 2.8 Data Quality Handling
- Phase 1 preprocessing ensures invalid timestamps are excluded  
- Phase 2 applies `errors='coerce'` and filters `NaT` values for additional protection  
- Environmental parameter validation:  
  - Temperature: negative values preserved to reflect winter conditions  
  - Humidity: negative values excluded, no artificial upper limits imposed  

### 2.9 Performance and Refresh
- Auto-refresh with precise timing controls; manual refresh available  
- `st.cache_data(ttl=2)` used for filtered data; `st.cache_resource` for analytics instance  
- Unique chart keys implemented to prevent visual artifacts across tabs  

## 3) How to Run
1. **Prerequisites:**  
   - `conda activate kafka-air-quality`  
   - `pip install -r requirements.txt`  
2. **Data Pipeline:**  
   - Ensure Phase 1 producer/consumer pipeline is generating `phase_1_streaming_infrastructure/data/processed/air_quality_clean.csv`  
3. **Model Training (Optional):**  
   - Train Phase 3 models with `python phase_3_predictive_analytics/train.py`  
   - Models will be automatically loaded if available  
4. **Launch Dashboard:**  
   - `streamlit run phase_2_data_intelligence/streaming_dashboard.py`  
   - Open the provided local URL in a browser  

## 4) Notes and Rationale
- Streamlit was selected for rapid development, interactivity, and session management capabilities  
- CSV polling provides a pragmatic "near real-time" experience without binding the UI directly to a Kafka client; direct ingestion could be added in future iterations  
- Distinction between anomalies and outliers: anomalies reflect deviations in a streaming context, while outliers represent statistical or machine learning detections over a user-selected range  
- Forecast integration demonstrates the transition from exploratory analysis to predictive modeling, bridging Phase 2 and Phase 3 deliverables within one unified system  

## 5) Dependencies
- **Core:** streamlit, plotly, pandas, numpy  
- **Analytics:** scipy, scikit-learn, statsmodels  
- **Visualization:** seaborn, matplotlib (optional for extended EDA)  
- **Phase 3 Integration:** joblib (for model loading), logging (for inference tracking)  

## Acknowledgments

This documentation and the associated code were generated with assistance from Claude AI to ensure technical accuracy, proper grammar, and professional formatting. All content has been validated and reviewed by the author to reflect the actual implementation and architectural decisions made during the development of the Phase 2 data intelligence dashboard and its integration with Phase 3 predictive analytics.