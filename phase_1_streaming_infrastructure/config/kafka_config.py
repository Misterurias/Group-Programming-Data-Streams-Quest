#!/usr/bin/env python3
"""
Kafka Configuration Settings

This module contains all Kafka-related configuration parameters
for the air quality streaming system.
"""

# Kafka Connection Settings
KAFKA_CONFIG = {
    'bootstrap_servers': 'localhost:9092',
    'topic_name': 'air-quality-data',
    'consumer_group': 'air-quality-consumers',
    'auto_offset_reset': 'earliest',  # Start from beginning
    'enable_auto_commit': True,
    'auto_commit_interval_ms': 1000,
}

# Producer Settings
PRODUCER_CONFIG = {
    'acks': 'all',  # Wait for all replicas
    'retries': 3,
    'retry_backoff_ms': 100,
    'batch_size': 16384,
    'linger_ms': 10,
    'compression_type': 'gzip',
    'max_block_ms': 10000,
    'request_timeout_ms': 30000,
    'delivery_timeout_ms': 120000,
    'enable_idempotence': True,
}

# Consumer Settings
CONSUMER_CONFIG = {
    'group_id': 'air-quality-consumers',
    'auto_offset_reset': 'earliest',
    'enable_auto_commit': True,
    'auto_commit_interval_ms': 1000,
    'session_timeout_ms': 30000,
    'heartbeat_interval_ms': 3000,
    'max_poll_records': 10,  # Default batch size
    'fetch_min_bytes': 1,
    'fetch_max_wait_ms': 500,
}

# Data Processing Settings
DATA_CONFIG = {
    'data_file': '../data/AirQualityUCI.csv',
    'simulation_speed': 1000000.0,  # 10,000x faster than real-time for ultra-fast processing
    'batch_size': 100,
    'missing_value_indicator': -200,
    'log_interval': 100,  # Log progress every N messages
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'air_quality_streaming.log',
    'max_bytes': 10485760,  # 10MB
    'backup_count': 5,
}
