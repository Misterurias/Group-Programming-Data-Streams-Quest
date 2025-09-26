# Phase 1: Setup, Architecture, and Troubleshooting Guide

- **Name:** Santiago Bolaños Vega
- **Course:** Fundamentals of Operationalizing AI
- **Date:** September 23, 2025

## Objective
The objective of Phase 1 is to build a real-time streaming pipeline for the UCI Air Quality dataset using Apache Kafka.  
The pipeline includes:  
- A Python producer and consumer  
- Micro-batching for controlled throughput  
- Robust preprocessing and validation  
- Structured logging for observability  
- Simple persistence (CSV) for testing and verification  


## 1) Complete Setup Procedures

### 1.1 System Requirements
The program was developed in a device with the following configurations.
- macOS (Apple Silicon M3 Pro validated)
- ≥ 8 GB RAM (18 GB used here)
- Homebrew installed
- Conda (Anaconda/Miniconda)

### 1.2 Conda Environment
First, it is important to set up an environment to keep all the dependencies in place. 
```bash
conda create -n kafka-air-quality python=3.12 -y
conda activate kafka-air-quality
pip install -r requirements.txt
```

### 1.3 Apache Kafka (Homebrew, KRaft)
Then, it's important to install Kafka to be able to run the producer and consumer apps.
```bash
brew install kafka
brew services start kafka
# To stop: brew services stop kafka
```

### 1.4 Dataset
The input dataset is the UCI Air Quality dataset:
- Place `AirQualityUCI.csv` in `data/` (raw).
- Note: semicolon-separated (;) and decimal commas (,).
- Contains hourly air quality measurements from 2004-2005

### 1.5 Project Structure (Phase 1)
Get familiar with the project structure. 
```
phase_1_streaming_infrastructure/
  config/
    kafka_config.py
  producer.py
  consumer.py
  data_preprocessing.py
  data_validation.py
  monitoring.py
  tests/
    fast_test_producer.py
    fast_test_consumer.py
    test_consumer.py
    quick_test.py
    artifacts/
      processed/
  data/
    processed/
```

### 1.6 Configuration Highlights
In the config file there are some important configurations for the producer and the consumer.
- `KAFKA_CONFIG.topic_name = 'air-quality-data'` - This is the Kafka topic name used for streaming air quality data
- `DATA_CONFIG.simulation_speed` - Controls temporal playback speed for simulation purposes. Since the original dataset has hourly measurements, this allows faster processing during testing
- `CONSUMER_CONFIG.max_poll_records` - Micro-batch size determining how many records are processed simultaneously by the consumer

### 1.7 Quick End-to-End Test
Run this code to check the correct behavior of the producer and consumer:
```bash
cd phase_1_streaming_infrastructure
python tests/quick_test.py
```
Expected: unit tests pass; producer sends 10 messages; consumer processes 10; CSV at `tests/artifacts/processed/air_quality_test.csv`.

## 2) Architectural Decisions

### Separation of Concerns

The pipeline is split by responsibility. The producer focuses on ingestion: it reads the semicolon/decimal-comma CSV, preserves the raw values, simulates timing, and publishes JSON to Kafka. Keeping the producer simple and stateless makes it easy to run fast and scale out.

The consumer owns data quality and persistence. It reads in small batches, applies preprocessing and validation, writes results, and emits metrics. Concentrating the "clean-and-check" logic here fits how streams are typically consumed by multiple downstreams and allows evolution of quality rules without touching ingestion.

### Preprocessing, Validation, and Observability

Preprocessing and validation live in dedicated modules. This keeps the logic testable on its own, encourages reuse, and makes changes safer. Monitoring and structured logging are part of the design from the start so observers can see what's happening during a live stream, not only after something breaks.

### Temporal Simulation and Micro-batching

Temporal simulation uses the dataset's timestamps scaled by `simulation_speed`. Real spacing is maintained when realism is desired and turned way up (with capped records) when quick feedback is needed. Micro-batching adds a sweet spot between throughput and latency: small batches reduce Kafka I/O overhead while still feeling near-real-time.

### Persistence

For Phase 1, persistence is a simple append-to-CSV approach. It's easy to inspect, grade, and version. Tests write to a `tests/artifacts` path to avoid mixing with "real" outputs and to keep cleanup trivial.

### Locale and Message Format

Because the dataset uses decimal commas, it is read with the proper separator and locale normalization is deferred to preprocessing. This preserves provenance in the messages while ensuring consumers produce consistent numerics for analysis. Detailed, per-message logs are maintained during this phase to maximize traceability; if logs grow too large, rotation can be added without changing verbosity.




## 3) Runbook 

### Production-like Runbook
1. Start Kafka
   - `brew services start kafka`
2. Ensure topic exists (idempotent)
   - `kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic air-quality-data --partitions 1 --replication-factor 1`
3. Activate environment
   - `conda activate kafka-air-quality`
4. Start the consumer (runs continuously)
   - `python consumer.py`
   - Output will append to `data/processed/air_quality_clean.csv` (or your configured path)
   - **Excluded records** will be saved to `data/processed/air_quality_excluded.csv` with detailed exclusion reasons (including `timestamp_incomplete` and `timestamp_invalid`)
5. Start the producer (full stream)
   - `python producer.py`
   - Uses temporal simulation; adjust `simulation_speed` in `config/kafka_config.py`
   - **Producer runs indefinitely** (no message limit) - processes entire dataset and continues for new messages
6. Monitor health
   - Tail logs: `tail -f consumer.log` and `tail -f producer.log`
   - Check consumer group lag: `kafka-consumer-groups --bootstrap-server localhost:9092 --group air-quality-consumers --describe`
7. Graceful shutdown
   - Ctrl+C in producer/consumer terminals; both handle cleanup and close Kafka clients
8. Rotate/retain artifacts
   - Periodically archive `data/processed/*.csv` and rotate logs if they grow large


## 4) Troubleshooting Guide

### 4.1 Common Errors
- "ModuleNotFoundError: kafka": install `kafka-python` in the active Conda env.
- Kafka connection errors: ensure `brew services start kafka`; broker at `localhost:9092`.
- Import paths in tests: tests adjust `sys.path` to import modules from parent.

### 4.2 Timeouts in Tests
- Reduce records (`end_idx`) and/or increase `simulation_speed`.
- Quick test uses 120s timeout and only 10 records.

### 4.3 CSV Not Found After Test
- Consumer writes to `tests/artifacts/processed/air_quality_test.csv`.
- Ensure directory exists; quick test creates/validates it.

### 4.4 Decimal Parsing Errors
- Symptom: "Could not convert string '2,6...' to numeric".
- Fix: preprocessing replaces decimal commas, extracts first numeric token, then converts.

### 4.5 Metrics Save Error
- Serialize timestamps as ISO strings in monitoring.

### 4.6 No Messages Consumed
- Confirm producer ran first, topic name matches, and consumer group is configured.

### 4.7 Data Quality Issues
- **High exclusion rate**: Check `data/processed/air_quality_excluded.csv` for exclusion reasons and patterns
- **Missing records**: Verify that excluded records are being tracked properly in the preprocessing logs
- **Unexpected filtering**: Review outlier ranges in `data_preprocessing.py` if too many records are being filtered
- **Timestamp integrity**: Records with date-only timestamps (YYYY-MM-DD) or unparseable timestamps are excluded by design to keep the clean dataset strictly time-qualified. See `exclusion_reason` values `timestamp_incomplete` and `timestamp_invalid` in the excluded CSV.

## 5) Configuration Values and Rationale

### Key Configuration Parameters

**Kafka Settings:**
- `topic_name = 'air-quality-data'` - Simple, descriptive topic name for air quality data streaming
- `bootstrap_servers = 'localhost:9092'` - Standard Kafka port for local development
- `consumer_group = 'air-quality-consumers'` - Consumer group for coordinated message processing
- `auto_offset_reset = 'earliest'` - Start processing from the beginning of the topic

**Producer Configuration:**
- `acks = 'all'` - Wait for all replicas to acknowledge before considering message sent
- `retries = 3` - Number of retry attempts for failed message sends
- `batch_size = 16384` - Producer batch size in bytes for efficiency
- `compression_type = 'gzip'` - Compress messages to reduce network overhead

**Consumer Configuration:**
- `max_poll_records = 10` - Micro-batch size balancing throughput and memory usage
- `session_timeout_ms = 30000` - 30-second timeout for consumer session
- `fetch_max_wait_ms = 500` - Maximum wait time for fetching messages

**Data Processing:**
- `simulation_speed = 1000000.0` - Ultra-fast processing (1,000,000x faster than real-time) for rapid development
- `log_interval = 100` - Log progress every 100 processed messages

### Rationale for Configuration Choices

**Ultra-Fast Simulation Speed (1,000,000x):** The original dataset spans over a year with hourly measurements. An ultra-fast speedup allows complete dataset processing in seconds rather than hours, enabling rapid iteration during development and testing.

**Micro-batching (10 records):** This batch size provides a good balance between latency and throughput. Smaller batches reduce memory usage and processing time per batch, while larger batches would increase latency and memory requirements.

**Producer Reliability Settings:** Using `acks='all'` ensures message durability by waiting for all replicas to acknowledge. Retry mechanisms and compression reduce network overhead while maintaining reliability.

**Consumer Session Management:** 30-second session timeouts provide sufficient buffer for processing while maintaining responsiveness. The 500ms fetch wait optimizes latency vs throughput trade-offs.

## Acknowledgments

This documentation and the associated code were generated with assistance from Claude AI to ensure technical accuracy, proper grammar, and professional formatting. All content has been validated and reviewed by the author to reflect the actual implementation and architectural decisions made during the development of the Phase 1 streaming infrastructure.
