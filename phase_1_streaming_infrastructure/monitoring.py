#!/usr/bin/env python3
"""
Monitoring Module for Air Quality Data Processing

This module handles monitoring, metrics, and alerting
for the air quality data processing pipeline.

Author: Santiago Bolaños Vega
Date: 2025-09-21
"""

import json
import time
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

class MonitoringSystem:
    """
    Handles monitoring, metrics, and alerting for the data processing pipeline.
    
    This class is responsible for:
    - Processing statistics tracking
    - Data quality monitoring
    - Performance metrics
    - Alert generation
    """
    
    def __init__(self, log_file: str = 'data/processed/processing_log.json'):
        """
        Initialize the monitoring system.
        
        Args:
            log_file: Path to log file for storing metrics
        """
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        
        # Initialize metrics
        self.metrics = {
            'start_time': datetime.now(),
            'total_messages_processed': 0,
            'total_batches_processed': 0,
            'total_errors': 0,
            'total_processing_time': 0.0,
            'average_batch_size': 0.0,
            'average_processing_time_per_batch': 0.0,
            'data_quality_stats': {
                'total_records': 0,
                'valid_records': 0,
                'average_quality_score': 0.0,
                'range_violations': 0,
                'consistency_violations': 0
            },
            'alerts': []
        }
        
        self.logger.info("Monitoring system initialized")
    
    def record_batch_processing(self, 
                               batch_size: int, 
                               processing_time: float,
                               validation_results: Dict[str, Any],
                               preprocessing_stats: Dict[str, Any]) -> None:
        """
        Record metrics for a processed batch.
        
        Args:
            batch_size: Number of messages in the batch
            processing_time: Time taken to process the batch
            validation_results: Results from data validation
            preprocessing_stats: Statistics from preprocessing
        """
        # Update basic metrics
        self.metrics['total_messages_processed'] += batch_size
        self.metrics['total_batches_processed'] += 1
        self.metrics['total_processing_time'] += processing_time
        
        # Calculate averages
        self.metrics['average_batch_size'] = (
            self.metrics['total_messages_processed'] / 
            self.metrics['total_batches_processed']
        )
        self.metrics['average_processing_time_per_batch'] = (
            self.metrics['total_processing_time'] / 
            self.metrics['total_batches_processed']
        )
        
        # Update data quality stats
        self._update_quality_stats(validation_results)
        
        # Check for alerts
        self._check_alerts(batch_size, processing_time, validation_results)
        
        # Log batch processing
        self.logger.info(f"Batch processed: {batch_size} messages in {processing_time:.3f}s")
    
    def _update_quality_stats(self, validation_results: Dict[str, Any]) -> None:
        """Update data quality statistics."""
        quality_stats = self.metrics['data_quality_stats']
        
        # Update totals
        quality_stats['total_records'] += validation_results.get('total_records', 0)
        quality_stats['valid_records'] += validation_results.get('valid_records', 0)
        quality_stats['range_violations'] += sum(
            v.get('total_violations', 0) 
            for v in validation_results.get('range_violations', {}).values()
        )
        quality_stats['consistency_violations'] += len(
            validation_results.get('consistency_violations', [])
        )
        
        # Calculate running average quality score
        if quality_stats['total_records'] > 0:
            current_avg = validation_results.get('average_quality_score', 0)
            total_records = quality_stats['total_records']
            batch_records = validation_results.get('total_records', 0)
            
            # Weighted average
            quality_stats['average_quality_score'] = (
                (quality_stats['average_quality_score'] * (total_records - batch_records) + 
                 current_avg * batch_records) / total_records
            )
    
    def _check_alerts(self, 
                     batch_size: int, 
                     processing_time: float,
                     validation_results: Dict[str, Any]) -> None:
        """Check for conditions that require alerts."""
        current_time = datetime.now()
        
        # Alert: High processing time
        if processing_time > 10.0:  # More than 10 seconds
            self._add_alert('HIGH_PROCESSING_TIME', 
                          f"Batch processing took {processing_time:.2f}s", 
                          current_time)
        
        # Alert: Low data quality
        avg_quality = validation_results.get('average_quality_score', 0)
        if avg_quality < 0.5:
            self._add_alert('LOW_DATA_QUALITY', 
                          f"Average quality score: {avg_quality:.3f}", 
                          current_time)
        
        # Alert: High error rate
        error_rate = validation_results.get('total_records', 0) - validation_results.get('valid_records', 0)
        if error_rate > batch_size * 0.1:  # More than 10% errors
            self._add_alert('HIGH_ERROR_RATE', 
                          f"Error rate: {error_rate}/{batch_size}", 
                          current_time)
        
        # Alert: Range violations
        range_violations = sum(
            v.get('total_violations', 0) 
            for v in validation_results.get('range_violations', {}).values()
        )
        if range_violations > batch_size * 0.05:  # More than 5% range violations
            self._add_alert('HIGH_RANGE_VIOLATIONS', 
                          f"Range violations: {range_violations}/{batch_size}", 
                          current_time)
    
    def _add_alert(self, alert_type: str, message: str, timestamp: datetime) -> None:
        """Add an alert to the metrics."""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': timestamp.isoformat(),
            'severity': self._get_alert_severity(alert_type)
        }
        
        self.metrics['alerts'].append(alert)
        self.logger.warning(f"ALERT [{alert_type}]: {message}")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get severity level for alert type."""
        severity_map = {
            'HIGH_PROCESSING_TIME': 'WARNING',
            'LOW_DATA_QUALITY': 'CRITICAL',
            'HIGH_ERROR_RATE': 'CRITICAL',
            'HIGH_RANGE_VIOLATIONS': 'WARNING'
        }
        return severity_map.get(alert_type, 'INFO')
    
    def record_error(self, error_type: str, error_message: str) -> None:
        """
        Record an error in the metrics.
        
        Args:
            error_type: Type of error
            error_message: Error message
        """
        self.metrics['total_errors'] += 1
        self.logger.error(f"ERROR [{error_type}]: {error_message}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        current_time = datetime.now()
        uptime = current_time - self.metrics['start_time']
        
        metrics = self.metrics.copy()
        # Convert datetime objects to ISO strings for JSON serialization
        metrics['start_time'] = self.metrics['start_time'].isoformat()
        metrics['uptime_seconds'] = uptime.total_seconds()
        metrics['uptime_human'] = str(uptime)
        metrics['current_time'] = current_time.isoformat()
        
        # Calculate throughput
        if uptime.total_seconds() > 0:
            metrics['messages_per_second'] = (
                metrics['total_messages_processed'] / uptime.total_seconds()
            )
        else:
            metrics['messages_per_second'] = 0
        
        return metrics
    
    def save_metrics(self) -> None:
        """Save metrics to log file."""
        try:
            metrics = self.get_current_metrics()
            
            # Load existing logs if file exists
            try:
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logs = []
            
            # Add current metrics
            logs.append(metrics)
            
            # Save updated logs
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            self.logger.info(f"Metrics saved to {self.log_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality metrics."""
        return self.metrics['data_quality_stats'].copy()
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.metrics['alerts']
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
        
        return recent_alerts
    
    def print_status(self) -> None:
        """Print current status to console."""
        metrics = self.get_current_metrics()
        
        print("\n" + "="*50)
        print("AIR QUALITY PROCESSING STATUS")
        print("="*50)
        print(f"Uptime: {metrics['uptime_human']}")
        print(f"Messages Processed: {metrics['total_messages_processed']}")
        print(f"Batches Processed: {metrics['total_batches_processed']}")
        print(f"Total Errors: {metrics['total_errors']}")
        print(f"Throughput: {metrics['messages_per_second']:.2f} msg/sec")
        print(f"Average Batch Size: {metrics['average_batch_size']:.1f}")
        print(f"Average Processing Time: {metrics['average_processing_time_per_batch']:.3f}s")
        
        print("\nData Quality:")
        quality_stats = metrics['data_quality_stats']
        print(f"  Total Records: {quality_stats['total_records']}")
        print(f"  Valid Records: {quality_stats['valid_records']}")
        print(f"  Average Quality Score: {quality_stats['average_quality_score']:.3f}")
        print(f"  Range Violations: {quality_stats['range_violations']}")
        print(f"  Consistency Violations: {quality_stats['consistency_violations']}")
        
        # Show recent alerts
        recent_alerts = self.get_recent_alerts(1)  # Last hour
        if recent_alerts:
            print(f"\nRecent Alerts ({len(recent_alerts)}):")
            for alert in recent_alerts[-5:]:  # Show last 5 alerts
                print(f"  [{alert['severity']}] {alert['message']}")
        
        print("="*50)
