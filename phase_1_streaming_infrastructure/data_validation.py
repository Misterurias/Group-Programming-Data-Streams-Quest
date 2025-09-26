#!/usr/bin/env python3
"""
Data Validation Module for Air Quality Data

This module handles data quality validation and assessment
for air quality sensor data.

Author: Santiago Bolaños Vega
Date: 2025-09-21
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

class DataValidator:
    """
    Handles data validation and quality assessment for air quality data.
    
    This class is responsible for:
    - Data range validation
    - Consistency checks
    - Quality scoring
    - Data integrity assessment
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.logger = logging.getLogger(__name__)
        
        # Define validation rules
        self.validation_rules = {
            'co_gt': {'min': 0, 'max': 50, 'unit': 'mg/m³'},
            'nox_gt': {'min': 0, 'max': 1000, 'unit': 'ppb'},
            'no2_gt': {'min': 0, 'max': 500, 'unit': 'µg/m³'},
            'benzene_gt': {'min': 0, 'max': 50, 'unit': 'µg/m³'},
            'temperature': {'min': -20, 'max': 50, 'unit': '°C'},
            'relative_humidity': {'min': 0, 'max': 100, 'unit': '%'},
            'absolute_humidity': {'min': 0, 'max': 10, 'unit': 'g/m³'}
        }
        
        self.logger.info("Data validator initialized")
    
    def validate_batch(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate a batch of processed data.
        
        Args:
            df: DataFrame with processed data
            
        Returns:
            Tuple of (validated_dataframe, validation_results)
        """
        if df.empty:
            return df, self._empty_validation_result()
        
        self.logger.info(f"Validating batch of {len(df)} records")
        
        # Perform validation checks
        range_validation = self._validate_ranges(df)
        consistency_validation = self._validate_consistency(df)
        completeness_validation = self._validate_completeness(df)
        
        # Calculate quality scores
        quality_scores = self._calculate_quality_scores(df)
        
        # Add quality scores to dataframe
        df_validated = df.copy()
        df_validated['quality_score'] = quality_scores
        
        # Compile validation results
        validation_results = {
            'total_records': len(df),
            'valid_records': len(df_validated[df_validated['quality_score'] >= 0.7]),
            'range_violations': range_validation['violations'],
            'consistency_violations': consistency_validation['violations'],
            'completeness_score': completeness_validation['score'],
            'average_quality_score': quality_scores.mean(),
            'quality_distribution': self._get_quality_distribution(quality_scores),
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.logger.info(f"Validation completed: {validation_results['valid_records']}/{validation_results['total_records']} valid records")
        
        return df_validated, validation_results
    
    def _validate_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that values are within expected ranges."""
        violations = {}
        
        for column, rules in self.validation_rules.items():
            if column in df.columns:
                min_val, max_val = rules['min'], rules['max']
                
                # Count violations
                below_min = (df[column] < min_val).sum()
                above_max = (df[column] > max_val).sum()
                
                violations[column] = {
                    'below_min': int(below_min),
                    'above_max': int(above_max),
                    'total_violations': int(below_min + above_max),
                    'min_value': float(df[column].min()) if not df[column].empty else None,
                    'max_value': float(df[column].max()) if not df[column].empty else None
                }
        
        return {'violations': violations}
    
    def _validate_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate logical consistency between related fields."""
        violations = []
        
        # Check NOx >= NO2 (NOx includes NO2)
        if 'nox_gt' in df.columns and 'no2_gt' in df.columns:
            nox_no2_inconsistent = df['nox_gt'] < df['no2_gt']
            if nox_no2_inconsistent.any():
                violations.append({
                    'check': 'nox_no2_consistency',
                    'description': 'NOx should be >= NO2',
                    'violations': int(nox_no2_inconsistent.sum())
                })
        
        # Check temperature and humidity relationship
        if 'temperature' in df.columns and 'relative_humidity' in df.columns:
            # High humidity with very low temperature might be suspicious
            temp_humidity_suspicious = (df['temperature'] < 0) & (df['relative_humidity'] > 90)
            if temp_humidity_suspicious.any():
                violations.append({
                    'check': 'temp_humidity_consistency',
                    'description': 'Very low temperature with high humidity',
                    'violations': int(temp_humidity_suspicious.sum())
                })
        
        # Check air quality parameter relationships
        if all(col in df.columns for col in ['co_gt', 'nox_gt', 'benzene_gt']):
            # All parameters zero might indicate sensor failure
            all_zero = (df['co_gt'] == 0) & (df['nox_gt'] == 0) & (df['benzene_gt'] == 0)
            if all_zero.any():
                violations.append({
                    'check': 'all_parameters_zero',
                    'description': 'All air quality parameters are zero',
                    'violations': int(all_zero.sum())
                })
        
        return {'violations': violations}
    
    def _validate_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data completeness."""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness_score = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
        
        return {
            'score': completeness_score,
            'missing_cells': int(missing_cells),
            'total_cells': int(total_cells)
        }
    
    def _calculate_quality_scores(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate quality scores for each record.
        
        Args:
            df: DataFrame with processed data
            
        Returns:
            Series with quality scores (0.0 to 1.0)
        """
        scores = pd.Series(1.0, index=df.index)  # Start with perfect score
        
        # Penalize missing values
        missing_penalty = df.isnull().sum(axis=1) / len(df.columns) * 0.3
        scores -= missing_penalty
        
        # Penalize range violations
        for column, rules in self.validation_rules.items():
            if column in df.columns:
                min_val, max_val = rules['min'], rules['max']
                range_violations = (df[column] < min_val) | (df[column] > max_val)
                scores -= range_violations * 0.2
        
        # Penalize consistency violations
        if 'nox_gt' in df.columns and 'no2_gt' in df.columns:
            consistency_violations = df['nox_gt'] < df['no2_gt']
            scores -= consistency_violations * 0.1
        
        # Ensure scores are between 0 and 1
        scores = scores.clip(0.0, 1.0)
        
        return scores
    
    def _get_quality_distribution(self, quality_scores: pd.Series) -> Dict[str, int]:
        """Get distribution of quality scores."""
        distribution = {
            'excellent': int((quality_scores >= 0.9).sum()),
            'good': int(((quality_scores >= 0.7) & (quality_scores < 0.9)).sum()),
            'fair': int(((quality_scores >= 0.5) & (quality_scores < 0.7)).sum()),
            'poor': int((quality_scores < 0.5).sum())
        }
        return distribution
    
    def _empty_validation_result(self) -> Dict[str, Any]:
        """Return empty validation result for empty dataframe."""
        return {
            'total_records': 0,
            'valid_records': 0,
            'range_violations': {},
            'consistency_violations': [],
            'completeness_score': 0.0,
            'average_quality_score': 0.0,
            'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0},
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def get_validation_summary(self, validation_results: Dict[str, Any]) -> str:
        """
        Get a human-readable validation summary.
        
        Args:
            validation_results: Results from validate_batch
            
        Returns:
            String summary of validation results
        """
        summary = f"""
Validation Summary:
- Total Records: {validation_results['total_records']}
- Valid Records: {validation_results['valid_records']}
- Average Quality Score: {validation_results['average_quality_score']:.3f}
- Completeness Score: {validation_results['completeness_score']:.3f}

Quality Distribution:
- Excellent (≥0.9): {validation_results['quality_distribution']['excellent']}
- Good (0.7-0.9): {validation_results['quality_distribution']['good']}
- Fair (0.5-0.7): {validation_results['quality_distribution']['fair']}
- Poor (<0.5): {validation_results['quality_distribution']['poor']}
"""
        return summary
