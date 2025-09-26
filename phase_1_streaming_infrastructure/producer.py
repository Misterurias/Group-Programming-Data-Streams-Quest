#!/usr/bin/env python3
"""
Air Quality Data Producer for Apache Kafka

This module implements a production-grade Kafka producer that simulates real-time
air quality sensor data streaming by replaying the UCI Air Quality dataset with
temporal simulation.

Author: Santiago Bolaños Vega
Date: 2025-09-21
"""

import json
import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config.kafka_config import KAFKA_CONFIG, PRODUCER_CONFIG, DATA_CONFIG, LOGGING_CONFIG

class AirQualityProducer:
    """
    Production-grade Kafka producer for air quality sensor data.
    
    This class handles:
    - Reading UCI Air Quality dataset
    - Simulating real-time data streaming
    - Error handling and retry logic
    - Structured logging
    - Data validation
    """
    
    def __init__(self, 
                 bootstrap_servers: str = None,
                 topic_name: str = None,
                 data_file: str = None,
                 simulation_speed: float = None,
                 batch_size: int = None):
        """
        Initialize the Air Quality Producer.
        
        Args:
            bootstrap_servers: Kafka broker addresses (uses config default if None)
            topic_name: Kafka topic name for air quality data (uses config default if None)
            data_file: Path to UCI Air Quality dataset (uses config default if None)
            simulation_speed: Speed multiplier (uses config default if None)
            batch_size: Number of records to process in each batch (uses config default if None)
        """
        # Use configuration defaults if parameters not provided
        self.bootstrap_servers = bootstrap_servers or KAFKA_CONFIG['bootstrap_servers']
        self.topic_name = topic_name or KAFKA_CONFIG['topic_name']
        self.data_file = data_file or DATA_CONFIG['data_file']
        self.simulation_speed = simulation_speed or DATA_CONFIG['simulation_speed']
        self.batch_size = batch_size or DATA_CONFIG['batch_size']
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize Kafka producer
        self.producer = None
        self._setup_kafka_producer()
        
        # Load and prepare data
        self.data = None
        self._load_data()
        
        self.logger.info(f"Air Quality Producer initialized successfully")
        self.logger.info(f"Topic: {self.topic_name}")
        self.logger.info(f"Bootstrap servers: {self.bootstrap_servers}")
        self.logger.info(f"Simulation speed: {self.simulation_speed}x")
        self.logger.info(f"Total records: {len(self.data)}")
    
    def _setup_logging(self):
        """Configure structured logging for the producer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('producer.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_kafka_producer(self):
        """Initialize Kafka producer with production-grade configuration."""
        try:
            # Merge default producer config with any custom settings
            producer_config = PRODUCER_CONFIG.copy()
            producer_config['bootstrap_servers'] = self.bootstrap_servers
            
            self.producer = KafkaProducer(
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                **producer_config
            )
            self.logger.info("Kafka producer configured successfully")
        except KafkaError as e:
            self.logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def _load_data(self):
        """Load and prepare the UCI Air Quality dataset."""
        try:
            # Read CSV with semicolon separator
            self.data = pd.read_csv(self.data_file, sep=';')
            
            # Basic data validation
            self._validate_data()
            
            # Convert date and time columns
            self._prepare_temporal_data()
            
            self.logger.info(f"Dataset loaded successfully: {len(self.data)} records")
            self.logger.info(f"Date range: {self.data['datetime'].min()} to {self.data['datetime'].max()}")
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _validate_data(self):
        """Perform basic data validation."""
        required_columns = ['Date', 'Time', 'CO(GT)', 'NOx(GT)', 'C6H6(GT)']
        
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")
        
        self.logger.info("Data validation passed")
    
    def _prepare_temporal_data(self):
        """Prepare temporal data for streaming simulation."""
        # Combine Date and Time columns
        self.data['datetime'] = pd.to_datetime(
            self.data['Date'] + ' ' + self.data['Time'], 
            format='%d/%m/%Y %H.%M.%S'
        )
        
        # Sort by datetime to ensure chronological order
        self.data = self.data.sort_values('datetime').reset_index(drop=True)
        
        self.logger.info("Temporal data prepared for streaming")
    
    def _create_message(self, row: pd.Series) -> Dict[str, Any]:
        """
        Create a Kafka message from a data row.
        
        Args:
            row: Pandas Series containing one record
            
        Returns:
            Dictionary representing the message
        """
        message = {
            'timestamp': row['datetime'].isoformat(),
            'date': row['Date'],
            'time': row['Time'],
            'sensor_data': {
                'co_gt': row['CO(GT)'],
                'pt08_s1_co': row['PT08.S1(CO)'],
                'nmhc_gt': row['NMHC(GT)'],
                'c6h6_gt': row['C6H6(GT)'],
                'pt08_s2_nmhc': row['PT08.S2(NMHC)'],
                'nox_gt': row['NOx(GT)'],
                'pt08_s3_nox': row['PT08.S3(NOx)'],
                'no2_gt': row['NO2(GT)'],
                'pt08_s4_no2': row['PT08.S4(NO2)'],
                'pt08_s5_o3': row['PT08.S5(O3)'],
                'temperature': row['T'],
                'relative_humidity': row['RH'],
                'absolute_humidity': row['AH']
            },
            'metadata': {
                'source': 'UCI_Air_Quality_Dataset',
                'simulation': True,
                'original_timestamp': row['datetime'].isoformat()
            }
        }
        
        return message
    
    def _calculate_delay(self, current_idx: int) -> float:
        """
        Calculate delay between messages based on temporal simulation.
        
        Args:
            current_idx: Current record index
            
        Returns:
            Delay in seconds
        """
        if current_idx == 0:
            return 0
        
        # Calculate time difference between consecutive records
        current_time = self.data.iloc[current_idx]['datetime']
        previous_time = self.data.iloc[current_idx - 1]['datetime']
        time_diff = (current_time - previous_time).total_seconds()
        
        # Apply simulation speed multiplier
        delay = time_diff / self.simulation_speed
        
        # Cap delay at reasonable limits (max 10 seconds for simulation)
        return min(delay, 10.0)
    
    def send_message(self, message: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Send a single message to Kafka.
        
        Args:
            message: Message data
            key: Optional message key
            
        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            future = self.producer.send(
                self.topic_name,
                value=message,
                key=key
            )
            
            # Wait for confirmation (optional, for reliability)
            record_metadata = future.get(timeout=10)
            
            self.logger.debug(f"Message sent to {record_metadata.topic} "
                           f"partition {record_metadata.partition} "
                           f"offset {record_metadata.offset}")
            
            return True
            
        except KafkaError as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending message: {e}")
            return False
    
    def start_streaming(self, start_idx: int = 0, end_idx: Optional[int] = None):
        """
        Start streaming air quality data with temporal simulation.
        
        Args:
            start_idx: Starting record index
            end_idx: Ending record index (None for all records)
        """
        if end_idx is None:
            end_idx = len(self.data)
        
        self.logger.info(f"Starting data streaming from record {start_idx} to {end_idx}")
        self.logger.info(f"Simulation speed: {self.simulation_speed}x")
        
        start_time = time.time()
        messages_sent = 0
        errors = 0
        
        try:
            for idx in range(start_idx, end_idx):
                row = self.data.iloc[idx]
                message = self._create_message(row)
                
                # Use timestamp as key for partitioning
                key = str(row['datetime'].timestamp())
                
                # Send message
                if self.send_message(message, key):
                    messages_sent += 1
                    
                    # Log progress every 100 messages
                    if messages_sent % 100 == 0:
                        self.logger.info(f"Sent {messages_sent} messages")
                else:
                    errors += 1
                
                # Calculate and apply delay for temporal simulation
                delay = self._calculate_delay(idx)
                if delay > 0:
                    time.sleep(delay)
                
                # Handle keyboard interrupt gracefully
                if idx % 1000 == 0:
                    try:
                        # Check for interrupt every 1000 messages
                        pass
                    except KeyboardInterrupt:
                        self.logger.info("Streaming interrupted by user")
                        break
        
        except KeyboardInterrupt:
            self.logger.info("Streaming interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error during streaming: {e}")
        finally:
            # Flush remaining messages
            self.producer.flush()
            
            # Log final statistics
            total_time = time.time() - start_time
            self.logger.info(f"Streaming completed:")
            self.logger.info(f"  Messages sent: {messages_sent}")
            self.logger.info(f"  Errors: {errors}")
            self.logger.info(f"  Total time: {total_time:.2f} seconds")
            self.logger.info(f"  Average rate: {messages_sent/total_time:.2f} messages/second")
    
    def close(self):
        """Close the Kafka producer."""
        if self.producer:
            self.producer.close()
            self.logger.info("Kafka producer closed")


def main():
    """Main function to run the producer."""
    try:
        # Initialize producer with configuration defaults
        producer = AirQualityProducer()
        
        # Start streaming
        producer.start_streaming()
        
    except KeyboardInterrupt:
        print("\nProducer stopped by user")
    except Exception as e:
        print(f"Producer failed: {e}")
    finally:
        if 'producer' in locals():
            producer.close()


if __name__ == "__main__":
    main()
