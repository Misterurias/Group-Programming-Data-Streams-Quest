# Phase 1: Preprocessing Methodology

- **Name:** Santiago Bolaños Vega  
- **Course:** Fundamentals of Operationalizing AI  
- **Date:** September 23, 2025  

## Objective
This document defines a robust, streaming-friendly preprocessing pipeline that converts raw sensor messages into clean, validated, and feature-rich records suitable for analysis and modeling. The methodology ensures data quality while preserving temporal continuity and enabling real-time processing.

## 1) Data Characteristics and Challenges

### Dataset Overview
- **Source:** UCI Air Quality Dataset (roadside monitoring, 2004–2005)  
- **Format:** Semicolon-separated values with decimal commas (European locale)  
- **Challenges:** Missing sentinel values, mixed data types, occasional outliers, temporal continuity requirements  

### Data Quality Framework
The preprocessing pipeline implements a multi-layered quality control system:  
- **Range-based outlier detection** using predefined physical limits for each air quality parameter  
- **Complete record exclusion** for rows with all missing values  
- **Timestamp validation** excluding date-only records (`YYYY-MM-DD` format)  
- **Comprehensive exclusion tracking** for audit and quality monitoring  

### Business Impact
Poor data quality directly impacts downstream analytics:  
- Incorrect parsing of decimal commas yields wrong magnitudes, leading to invalid insights  
- Missing values and outliers can bias predictive models  
- Sensor drift and failures must be detectable early to maintain system reliability  

## 2) Preprocessing Pipeline Architecture

The pipeline follows a sequential, streaming-friendly approach designed for real-time data processing. Each step builds upon the previous one, ensuring data quality and temporal integrity.

### Step 1: Data Assembly
- Collect Kafka JSON messages and convert to a structured DataFrame  
- Preserve original message metadata for traceability  
- Handle batch processing efficiently for streaming workloads  

### Step 2: Type Normalization
- Replace decimal commas with dots, extract first numeric token, and convert to numerics  
- Parse timestamps into datetime; exclude incomplete or invalid timestamps  
- Rationale: ensures consistent numeric values and strictly valid timestamps for downstream analysis  

### Step 3: Missing Value Handling
- Convert sentinel values (`-200`) to `NaN`  
- Exclude rows with all values missing  
- Imputation strategy:  
  - **Pollutants (CO, NOx, NO2, Benzene):** forward-fill then mean imputation  
  - **Environmental variables (Temperature, RH, AH):** linear interpolation then mean  
  - **Other fields:** mean imputation  
- Rationale: preserves temporal continuity while avoiding large analytical gaps  

### Step 4: Outlier Detection and Temporal Validation
- **Range-based detection** using domain-specific physical limits:  
  - CO: 0–50 mg/m³  
  - NOx: 0–1000 ppb  
  - NO2: 0–500 µg/m³  
  - Benzene: 0–50 µg/m³  
  - Temperature: unrestricted (includes negatives)  
  - RH: ≥0% (no upper bound)  
  - AH: ≥0 g/m³ (no upper bound)  
- **Temporal validation:**  
  - Exclude records with date-only timestamps or unparseable formats  
  - Maintain strict chronological order  
- **Exclusion tracking:**  
  - Save dropped records with `exclusion_reason` codes in `data/processed/air_quality_excluded.csv`  

### Step 5: Feature Engineering
- Temporal features: hour, day_of_week, month, season  
- Derived indices:  
  - **Air Quality Index (proxy):** CO/NOx/NO2 scaled to 0–100  
  - **Comfort Index:** Temperature + RH combined, scaled to 0–100  
- Rationale: improves interpretability and supports exploratory analysis without heavy computation  

### Step 6: Quality Validation and Scoring
- Validation checks: range validation, chemical consistency (NOx ≥ NO2), completeness scoring  
- Quality score: starts at 1.0, deductions for missingness, violations, or inconsistencies, clipped to [0,1]  
- Rationale: provides quantitative reliability assessment, enabling filtering and alerting  

### Step 7: Persistence and Output Management
- **Clean data:** `data/processed/air_quality_clean.csv`  
- **Excluded records:** `data/processed/air_quality_excluded.csv`  
- **Test artifacts:** `tests/artifacts/processed/air_quality_test.csv`  
- Files append after headers on first batch; directories auto-created by consumer  
- Rationale: separation of clean vs excluded data ensures transparency and traceability  

## 3) Field-Specific Processing Details

### Numeric Fields
**Target:** co_gt, pt08_s1_co, nmhc_gt, c6h6_gt, pt08_s2_nmhc, nox_gt, pt08_s3_nox, no2_gt, pt08_s4_no2, pt08_s5_o3, temperature, relative_humidity, absolute_humidity  

**Processing:**  
1. Locale normalization: comma → dot, extract decimal token  
2. Type conversion: robust `to_numeric(errors='coerce')`  
3. Missing values: replace sentinels with NaN, impute field-specifically  
4. Outlier filtering: enforce predefined physical ranges  

### Temporal Fields
**Target:** datetime, timestamp  

**Processing:**  
1. Format validation: only fully qualified timestamps retained  
2. Parsing: standardized datetime conversion  
3. Exclusion: remove date-only or invalid entries  
4. Derived features: hour, day_of_week, month, season  

## 4) Business Justification and Impact

### Methodology Benefits
- Locale normalization prevents systemic numeric bias  
- Strategic imputation minimizes data loss while preserving temporal continuity  
- Outlier removal mitigates sensor spikes that could distort models  
- Quality indices support interpretability and lightweight monitoring  
- Quality scoring enables trust in downstream analytics and automated alerts  

### Operational Advantages
- Streaming compatibility: lightweight, real-time friendly design  
- Audit compliance: exclusion tracking supports transparency and regulation  
- Debugging support: detailed logs accelerate issue resolution  
- Scalability: efficient enough for high-throughput environments  

## 5) Observability and Monitoring

### Logging Strategy
- Per-message logs from producer to consumer  
- Conversion samples, NaN summaries, and processing metrics  
- Batch-level timing and validation summaries  

### Quality Monitoring
- Real-time dashboards for exclusion rates, scores, and throughput  
- Alerting thresholds for degradation or failures  
- Historical analysis for sensor drift and long-term environmental patterns  

## Acknowledgments

This documentation and the associated code were generated with assistance from Claude AI to ensure technical accuracy, proper grammar, and professional formatting. All content has been validated and reviewed by the author to reflect the actual implementation and architectural decisions made during the development of the Phase 1 preprocessing methodology.