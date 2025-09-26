#!/usr/bin/env python3
"""
Air Quality Data Consumer for Apache Kafka

This module implements a production-grade Kafka consumer that processes
air quality sensor data with comprehensive preprocessing, validation,
and monitoring capabilities.

Author: Santiago Bolaños Vega
Date: 2025-09-21
"""

import json
import logging
import time
import os
import sys
from typing import Dict, List, Any, Optional
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration and modules
from config.kafka_config import KAFKA_CONFIG, CONSUMER_CONFIG, DATA_CONFIG, LOGGING_CONFIG
from data_preprocessing import DataPreprocessor
from data_validation import DataValidator
from monitoring import MonitoringSystem

class AirQualityConsumer:
    """
    Production-grade Kafka consumer for air quality sensor data.
    
    This class handles:
    - Kafka message consumption
    - Data preprocessing and validation
    - Data storage and persistence
    - Monitoring and alerting
    """
    
    def __init__(self, 
                 bootstrap_servers: str = None,
                 topic_name: str = None,
                 consumer_group: str = None,
                 batch_size: int = None,
                 output_file: str = None,
                 excluded_file: str = None):
        """
        Initialize the Air Quality Consumer.
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic_name: Kafka topic name
            consumer_group: Consumer group ID
            batch_size: Number of messages to process in each batch
            output_file: Output file for processed data
        """
        # Use configuration defaults if parameters not provided
        self.bootstrap_servers = bootstrap_servers or KAFKA_CONFIG['bootstrap_servers']
        self.topic_name = topic_name or KAFKA_CONFIG['topic_name']
        self.consumer_group = consumer_group or CONSUMER_CONFIG['group_id']
        self.batch_size = batch_size or CONSUMER_CONFIG['max_poll_records']
        self.output_file = output_file or 'data/processed/air_quality_clean.csv'
        self.excluded_file = excluded_file or 'data/processed/air_quality_excluded.csv'
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.validator = DataValidator()
        self.monitor = MonitoringSystem()
        
        # Initialize Kafka consumer
        self.consumer = None
        self._setup_kafka_consumer()
        
        # Initialize storage
        self.first_batch = True
        self._ensure_output_directory()
        
        self.logger.info("Air Quality Consumer initialized successfully")
        self.logger.info(f"Topic: {self.topic_name}")
        self.logger.info(f"Consumer Group: {self.consumer_group}")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Output File: {self.output_file}")
    
    def _setup_logging(self):
        """Configure structured logging for the consumer."""
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG['level']),
            format=LOGGING_CONFIG['format'],
            handlers=[
                logging.FileHandler('consumer.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_kafka_consumer(self):
        """Initialize Kafka consumer with production-grade configuration."""
        try:
            # Merge default consumer config with custom settings
            consumer_config = CONSUMER_CONFIG.copy()
            consumer_config['bootstrap_servers'] = self.bootstrap_servers
            consumer_config['group_id'] = self.consumer_group
            
            self.consumer = KafkaConsumer(
                self.topic_name,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                **consumer_config
            )
            self.logger.info("Kafka consumer configured successfully")
        except KafkaError as e:
            self.logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    def _ensure_output_directory(self):
        """Ensure output directory exists."""
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Created output directory: {output_dir}")
    
    def _save_batch(self, df: pd.DataFrame) -> None:
        """
        Save processed batch to output file.
        
        Args:
            df: DataFrame with processed data
        """
        try:
            if self.first_batch:
                # First batch: create file with headers
                df.to_csv(self.output_file, mode='w', header=True, index=False)
                self.first_batch = False
                self.logger.info(f"Created output file: {self.output_file}")
            else:
                # Subsequent batches: append without headers
                df.to_csv(self.output_file, mode='a', header=False, index=False)
            
            self.logger.debug(f"Saved {len(df)} records to {self.output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save batch: {e}")
            self.monitor.record_error('SAVE_ERROR', str(e))
    
    def _save_excluded_records(self) -> None:
        """Save excluded records to the excluded file."""
        try:
            self.preprocessor.save_excluded_records(self.excluded_file)
        except Exception as e:
            self.logger.error(f"Failed to save excluded records: {e}")
            self.monitor.record_error('EXCLUDED_SAVE_ERROR', str(e))
    
    def _process_batch(self, messages: List[Any]) -> None:
        """
        Process a batch of messages.
        
        Args:
            messages: List of Kafka messages
        """
        if not messages:
            return
        
        batch_start_time = time.time()
        
        try:
            # Extract message values
            message_values = [msg.value for msg in messages]
            
            # Preprocess data
            processed_data = self.preprocessor.process_batch(message_values)
            
            if processed_data.empty:
                self.logger.warning("No valid data after preprocessing")
                return
            
            # Validate data
            validated_data, validation_results = self.validator.validate_batch(processed_data)
            
            # Get preprocessing statistics
            preprocessing_stats = self.preprocessor.get_preprocessing_stats(processed_data)
            
            # Save processed data
            self._save_batch(validated_data)
            
            # Record metrics
            processing_time = time.time() - batch_start_time
            self.monitor.record_batch_processing(
                len(messages), 
                processing_time, 
                validation_results, 
                preprocessing_stats
            )
            
            # Log batch summary
            self.logger.info(f"Batch processed: {len(messages)} messages → {len(validated_data)} records")
            self.logger.info(f"Quality score: {validation_results['average_quality_score']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            self.monitor.record_error('BATCH_PROCESSING_ERROR', str(e))
    
    def start_consuming(self, max_messages: Optional[int] = None):
        """
        Start consuming messages from Kafka.
        
        Args:
            max_messages: Maximum number of messages to process (None for unlimited)
        """
        self.logger.info("Starting message consumption")
        self.logger.info(f"Max messages: {max_messages or 'unlimited'}")
        
        messages_processed = 0
        batch_count = 0
        
        try:
            while True:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if not message_batch:
                    continue
                
                # Process each partition
                for topic_partition, messages in message_batch.items():
                    if messages:
                        # Process the batch
                        self._process_batch(messages)
                        messages_processed += len(messages)
                        batch_count += 1
                        
                        # Check if we've reached the limit
                        if max_messages and messages_processed >= max_messages:
                            self.logger.info(f"Reached message limit: {max_messages}")
                            return
                        
                        # Print status every 10 batches
                        if batch_count % 10 == 0:
                            self.monitor.print_status()
                
                # Handle keyboard interrupt
                try:
                    pass  # Check for interrupt
                except KeyboardInterrupt:
                    self.logger.info("Consumption interrupted by user")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Consumption interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error during consumption: {e}")
            self.monitor.record_error('CONSUMPTION_ERROR', str(e))
        finally:
            # Save final metrics
            self.monitor.save_metrics()
            
            # Print final status
            self.monitor.print_status()
            
            # Save excluded records
            self._save_excluded_records()
            
            # Print exclusion summary
            exclusion_summary = self.preprocessor.get_exclusion_summary()
            if exclusion_summary['total_excluded'] > 0:
                self.logger.info(f"Exclusion Summary:")
                self.logger.info(f"  Total excluded: {exclusion_summary['total_excluded']}")
                self.logger.info(f"  Breakdown: {exclusion_summary['exclusion_breakdown']}")
                self.logger.info(f"  Excluded records saved to: {self.excluded_file}")
            
            self.logger.info(f"Consumption completed: {messages_processed} messages processed")
    
    def close(self):
        """Close the Kafka consumer."""
        if self.consumer:
            self.consumer.close()
            self.logger.info("Kafka consumer closed")


def main():
    """Main function to run the consumer."""
    try:
        # Initialize consumer with configuration defaults
        consumer = AirQualityConsumer()
        
        # Start consuming (unlimited - keep reading the stream)
        consumer.start_consuming()
        
    except KeyboardInterrupt:
        print("\nConsumer stopped by user")
    except Exception as e:
        print(f"Consumer failed: {e}")
    finally:
        if 'consumer' in locals():
            consumer.close()


if __name__ == "__main__":
    main()
