#!/usr/bin/env python3
"""
Data Preprocessing Module for Air Quality Data

This module handles data cleaning, transformation, and preprocessing
for air quality sensor data from Kafka streams.

Author: Santiago Bolaños Vega
Date: 2025-09-21
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

class DataPreprocessor:
    """
    Handles data preprocessing for air quality sensor data.
    
    This class is responsible for:
    - Handling missing values (-200)
    - Data type conversion
    - Data cleaning and transformation
    - Feature engineering
    """
    
    def __init__(self, missing_value_indicator: float = -200):
        """
        Initialize the data preprocessor.
        
        Args:
            missing_value_indicator: Value used to indicate missing data
        """
        self.missing_value_indicator = missing_value_indicator
        self.logger = logging.getLogger(__name__)
        
        # Define expected ranges for air quality parameters
        self.expected_ranges = {
            'co_gt': (0, 50),           # CO in mg/m³
            'nox_gt': (0, 1000),        # NOx in ppb
            'no2_gt': (0, 500),         # NO2 in µg/m³
            'benzene_gt': (0, 50),      # Benzene in µg/m³
            'temperature': (None, None),    # Temperature in °C (no limits - can be negative)
            'relative_humidity': (0, None), # RH in % (no upper limit)
            'absolute_humidity': (0, None)   # AH in g/m³ (no upper limit)
        }
        
        # Track excluded records
        self.excluded_records = []
        self.exclusion_reasons = []
        
        self.logger.info("Data preprocessor initialized")
    
    def process_batch(self, messages: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process a batch of messages from Kafka.
        
        Args:
            messages: List of Kafka messages containing sensor data
            
        Returns:
            DataFrame with processed data
        """
        if not messages:
            return pd.DataFrame()
        
        self.logger.info(f"Processing batch of {len(messages)} messages")
        # Log a preview of incoming raw messages before any transformation
        try:
            preview_count = min(len(messages), 10)
            self.logger.info(
                f"About to preprocess {len(messages)} messages; sample (up to {preview_count}): "
                f"{messages[:preview_count]}"
            )
        except Exception:
            pass
        
        # Convert messages to DataFrame
        df = self._messages_to_dataframe(messages)
        # Log one raw record for visibility
        try:
            raw_sample = df.head(1).to_dict(orient='records')
            self.logger.debug(f"Preprocessing input sample (raw): {raw_sample}")
        except Exception:
            pass
        
        # Apply preprocessing steps
        # 1) Normalize and convert numeric types FIRST (handles decimal commas)
        df = self._convert_data_types(df)
        # 2) Then handle missing values and imputation/interpolation on numeric dtypes
        df = self._handle_missing_values(df)
        df = self._clean_outliers(df)
        df = self._add_derived_features(df)
        
        self.logger.info(f"Preprocessing completed for {len(df)} records")
        return df
    
    def _messages_to_dataframe(self, messages: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert Kafka messages to DataFrame."""
        records = []
        
        for message in messages:
            try:
                # Extract sensor data from message
                sensor_data = message.get('sensor_data', {})
                metadata = message.get('metadata', {})
                
                record = {
                    'timestamp': message.get('timestamp'),
                    'date': message.get('date'),
                    'time': message.get('time'),
                    'co_gt': sensor_data.get('co_gt'),
                    'pt08_s1_co': sensor_data.get('pt08_s1_co'),
                    'nmhc_gt': sensor_data.get('nmhc_gt'),
                    'c6h6_gt': sensor_data.get('c6h6_gt'),
                    'pt08_s2_nmhc': sensor_data.get('pt08_s2_nmhc'),
                    'nox_gt': sensor_data.get('nox_gt'),
                    'pt08_s3_nox': sensor_data.get('pt08_s3_nox'),
                    'no2_gt': sensor_data.get('no2_gt'),
                    'pt08_s4_no2': sensor_data.get('pt08_s4_no2'),
                    'pt08_s5_o3': sensor_data.get('pt08_s5_o3'),
                    'temperature': sensor_data.get('temperature'),
                    'relative_humidity': sensor_data.get('relative_humidity'),
                    'absolute_humidity': sensor_data.get('absolute_humidity'),
                    'source': metadata.get('source'),
                    'simulation': metadata.get('simulation')
                }
                
                records.append(record)
                
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                continue
        
        df = pd.DataFrame(records)
        try:
            self.logger.debug(
                f"Messages→DataFrame: columns={list(df.columns)} size={len(df)}"
            )
        except Exception:
            pass
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values (-200) using appropriate strategies.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        df_processed = df.copy()
        
        # Count missing values before processing
        missing_before = (df_processed == self.missing_value_indicator).sum().sum()
        
        # Drop completely empty rows (all -200 values)
        completely_empty = (df_processed == self.missing_value_indicator).all(axis=1)
        dropped_count = completely_empty.sum()
        if dropped_count > 0:
            # Store excluded empty rows
            empty_records = df_processed[completely_empty].copy()
            empty_records['exclusion_reason'] = 'completely_empty'
            empty_records['exclusion_details'] = 'All values are missing sentinel values'
            self.excluded_records.extend(empty_records.to_dict('records'))
            self.exclusion_reasons.extend(['completely_empty'] * len(empty_records))
            
            df_processed = df_processed[~completely_empty]
            self.logger.info(f"Dropped {dropped_count} completely empty rows")
        
        # Handle missing values for each column
        for column in df_processed.columns:
            if column in ['timestamp', 'date', 'time', 'source', 'simulation']:
                continue  # Skip non-numeric columns
            
            # Replace -200 with NaN for proper handling
            df_processed[column] = df_processed[column].replace(
                self.missing_value_indicator, np.nan
            )
            
            # Apply appropriate imputation strategy
            if column in ['co_gt', 'nox_gt', 'no2_gt', 'benzene_gt']:
                # For air quality parameters: forward fill, then mean
                df_processed[column] = df_processed[column].fillna(method='ffill')
                df_processed[column] = df_processed[column].fillna(df_processed[column].mean())
            
            elif column in ['temperature', 'relative_humidity', 'absolute_humidity']:
                # For environmental parameters: linear interpolation
                df_processed[column] = df_processed[column].interpolate(method='linear')
                df_processed[column] = df_processed[column].fillna(df_processed[column].mean())
            
            else:
                # For other numeric columns: mean imputation
                df_processed[column] = df_processed[column].fillna(df_processed[column].mean())
        
        # Count missing values after processing
        missing_after = df_processed.isnull().sum().sum()
        
        self.logger.info(f"Missing values handled: {missing_before} → {missing_after}")
        try:
            nan_counts = df_processed.isnull().sum().sort_values(ascending=False)
            self.logger.debug(
                f"NaN counts by column after missing handling (top 8): {nan_counts.head(8).to_dict()}"
            )
        except Exception:
            pass
        
        return df_processed
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types to appropriate formats.
        
        Args:
            df: DataFrame with mixed data types
            
        Returns:
            DataFrame with proper data types
        """
        df_processed = df.copy()
        
        # Convert timestamp to datetime with exclusion for invalid/incomplete
        if 'timestamp' in df_processed.columns:
            self.logger.info("Validating and converting timestamps to datetime")

            # Identify date-only timestamps (YYYY-MM-DD) to exclude
            ts_str_series = df_processed['timestamp'].astype(str).str.strip()
            date_only_mask = ts_str_series.str.len().eq(10) & ts_str_series.str.count('-').eq(2)

            date_only_count = int(date_only_mask.sum())
            if date_only_count > 0:
                date_only_records = df_processed[date_only_mask].copy()
                date_only_records['exclusion_reason'] = 'timestamp_incomplete'
                date_only_records['exclusion_details'] = 'Timestamp missing time component (YYYY-MM-DD)'
                self.excluded_records.extend(date_only_records.to_dict('records'))
                self.exclusion_reasons.extend(['timestamp_incomplete'] * date_only_count)
                df_processed = df_processed[~date_only_mask]
                self.logger.info(f"Excluded {date_only_count} records with incomplete timestamps")

            # Convert remaining to datetime; coerce invalid to NaT
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], errors='coerce')

            # Exclude any rows where timestamp could not be parsed
            invalid_mask = df_processed['timestamp'].isna()
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                invalid_records = df_processed[invalid_mask].copy()
                invalid_records['exclusion_reason'] = 'timestamp_invalid'
                invalid_records['exclusion_details'] = 'Unparseable timestamp format'
                self.excluded_records.extend(invalid_records.to_dict('records'))
                self.exclusion_reasons.extend(['timestamp_invalid'] * invalid_count)
                df_processed = df_processed[~invalid_mask]
                self.logger.warning(f"Excluded {invalid_count} records with invalid timestamps")

            if len(df_processed) > 0:
                self.logger.info(
                    f"Timestamps converted successfully; sample: {df_processed['timestamp'].head(3).tolist()}"
                )
        
        # Convert numeric columns
        numeric_columns = [
            'co_gt', 'pt08_s1_co', 'nmhc_gt', 'c6h6_gt', 'pt08_s2_nmhc',
            'nox_gt', 'pt08_s3_nox', 'no2_gt', 'pt08_s4_no2', 'pt08_s5_o3',
            'temperature', 'relative_humidity', 'absolute_humidity'
        ]
        # Snapshot before conversion for debugging
        try:
            sample_before = (
                df_processed[[c for c in numeric_columns if c in df_processed.columns]]
                .head(3).astype(str).to_dict(orient='records')
            )
            self.logger.debug(f"Numeric fields sample BEFORE conversion: {sample_before}")
        except Exception:
            pass
        
        for column in numeric_columns:
            if column in df_processed.columns:
                # Normalize numeric strings safely:
                # 1) Ensure string type
                # 2) Replace decimal comma with dot (European → US)
                # 3) Extract the FIRST valid numeric token to avoid concatenation issues
                # 4) Convert to numeric
                series_str = df_processed[column].astype(str)
                series_str = series_str.str.replace(',', '.', regex=False)
                token = series_str.str.extract(r'(-?\d+(?:\.\d+)?)', expand=False)
                df_processed[column] = pd.to_numeric(token, errors='coerce')

        # Snapshot after conversion for debugging
        try:
            sample_after = (
                df_processed[[c for c in numeric_columns if c in df_processed.columns]]
                .head(3).to_dict(orient='records')
            )
            self.logger.debug(f"Numeric fields sample AFTER conversion: {sample_after}")
        except Exception:
            pass
        
        self.logger.info("Data types converted successfully")
        return df_processed
    
    def _clean_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean outliers using range validation method.
        
        Args:
            df: DataFrame with potential outliers
            
        Returns:
            DataFrame with outliers cleaned
        """
        df_processed = df.copy()
        outliers_removed = 0
        
        for column, (min_val, max_val) in self.expected_ranges.items():
            if column in df_processed.columns:
                # Identify outliers - handle None for no limits
                outlier_mask = pd.Series([False] * len(df_processed), index=df_processed.index)
                
                if min_val is not None:
                    outlier_mask = outlier_mask | (df_processed[column] < min_val)
                if max_val is not None:
                    outlier_mask = outlier_mask | (df_processed[column] > max_val)
                
                outlier_records = df_processed[outlier_mask].copy()
                
                if len(outlier_records) > 0:
                    # Add exclusion reason
                    if min_val is None and max_val is None:
                        range_desc = "no limits"
                    elif min_val is None:
                        range_desc = f"[−∞, {max_val}]"
                    elif max_val is None:
                        range_desc = f"[{min_val}, ∞)"
                    else:
                        range_desc = f"[{min_val}, {max_val}]"
                    
                    outlier_records['exclusion_reason'] = f'{column}_out_of_range'
                    outlier_records['exclusion_details'] = f'{column} outside range {range_desc}'
                    
                    # Store excluded records
                    self.excluded_records.extend(outlier_records.to_dict('records'))
                    self.exclusion_reasons.extend([f'{column}_out_of_range'] * len(outlier_records))
                
                # Remove outliers
                before_count = len(df_processed)
                df_processed = df_processed[~outlier_mask]
                after_count = len(df_processed)
                outliers_removed += (before_count - after_count)
        
        if outliers_removed > 0:
            self.logger.info(f"Removed {outliers_removed} outlier records")
        
        return df_processed
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for analysis.
        
        Args:
            df: DataFrame with basic features
            
        Returns:
            DataFrame with additional derived features
        """
        df_processed = df.copy()
        
        # Add temporal features
        if 'timestamp' in df_processed.columns:
            df_processed['hour'] = df_processed['timestamp'].dt.hour
            df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
            df_processed['month'] = df_processed['timestamp'].dt.month
            df_processed['season'] = df_processed['month'].apply(self._get_season)
        
        # Add air quality indices
        if all(col in df_processed.columns for col in ['co_gt', 'nox_gt', 'no2_gt']):
            df_processed['air_quality_index'] = self._calculate_air_quality_index(df_processed)
        
        # Add environmental comfort index
        if all(col in df_processed.columns for col in ['temperature', 'relative_humidity']):
            df_processed['comfort_index'] = self._calculate_comfort_index(df_processed)
        
        self.logger.info("Derived features added successfully")
        return df_processed
    
    def _get_season(self, month: int) -> str:
        """Convert month to season."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    def _calculate_air_quality_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate a simple air quality index."""
        # Simple weighted combination of pollutants
        co_norm = df['co_gt'] / df['co_gt'].max() if df['co_gt'].max() > 0 else 0
        nox_norm = df['nox_gt'] / df['nox_gt'].max() if df['nox_gt'].max() > 0 else 0
        no2_norm = df['no2_gt'] / df['no2_gt'].max() if df['no2_gt'].max() > 0 else 0
        
        return (co_norm * 0.4 + nox_norm * 0.3 + no2_norm * 0.3) * 100
    
    def _calculate_comfort_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate environmental comfort index."""
        # Simple comfort index based on temperature and humidity
        temp_score = 1 - abs(df['temperature'] - 22) / 30  # Optimal temp around 22°C
        humidity_score = 1 - abs(df['relative_humidity'] - 50) / 50  # Optimal RH around 50%
        
        return (temp_score * 0.6 + humidity_score * 0.4) * 100
    
    def get_preprocessing_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about preprocessing results.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with preprocessing statistics
        """
        stats = {
            'total_records': len(df),
            'missing_values_handled': (df == self.missing_value_indicator).sum().sum(),
            'outliers_removed': 0,  # Would need to track this during processing
            'derived_features_added': len([col for col in df.columns if col in 
                                        ['hour', 'day_of_week', 'month', 'season', 
                                         'air_quality_index', 'comfort_index']]),
            'data_types_converted': len(df.select_dtypes(include=[np.number]).columns),
            'preprocessing_timestamp': pd.Timestamp.now().isoformat()
        }
        
        return stats
    
    def save_excluded_records(self, output_file: str) -> None:
        """
        Save excluded records to a CSV file.
        
        Args:
            output_file: Path to save excluded records CSV
        """
        if not self.excluded_records:
            self.logger.info("No excluded records to save")
            return
        
        try:
            excluded_df = pd.DataFrame(self.excluded_records)
            excluded_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(self.excluded_records)} excluded records to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save excluded records: {e}")
    
    def get_exclusion_summary(self) -> Dict[str, Any]:
        """
        Get summary of excluded records.
        
        Returns:
            Dictionary with exclusion statistics
        """
        if not self.exclusion_reasons:
            return {'total_excluded': 0, 'exclusion_breakdown': {}}
        
        from collections import Counter
        exclusion_counts = Counter(self.exclusion_reasons)
        
        return {
            'total_excluded': len(self.excluded_records),
            'exclusion_breakdown': dict(exclusion_counts),
            'exclusion_rate': len(self.excluded_records) / (len(self.excluded_records) + len(self.excluded_records)) if self.excluded_records else 0
        }
