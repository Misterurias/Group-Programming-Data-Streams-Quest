#!/usr/bin/env python3

"""
Fast Test Consumer for Air Quality Data

This is a fast version of the consumer for testing purposes.
It processes messages quickly and stops after a limited number.

Author: Santiago Bolaños Vega
Date: 2025-09-21
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consumer import AirQualityConsumer

def main():
    """Main function to run the fast test consumer."""
    try:
        # Ensure test artifacts directory exists
        artifacts_dir = os.path.join('tests', 'artifacts', 'processed')
        os.makedirs(artifacts_dir, exist_ok=True)

        # Initialize consumer with fast test settings
        consumer = AirQualityConsumer(
            batch_size=10,  # Smaller batches for faster processing
            output_file=os.path.join('tests', 'artifacts', 'processed', 'air_quality_test.csv')
        )
        
        print("🚀 Starting FAST TEST Consumer...")
        print(f"   Topic: {consumer.topic_name}")
        print(f"   Consumer Group: {consumer.consumer_group}")
        print(f"   Batch Size: {consumer.batch_size}")
        print(f"   Output File: {consumer.output_file}")
        print("   Processing: First 10 messages only")
        print("   Press Ctrl+C to stop")
        
        # Start consuming with limited messages
        consumer.start_consuming(max_messages=10)
        
    except KeyboardInterrupt:
        print("\n⏹️  Fast test consumer stopped by user")
    except Exception as e:
        print(f"❌ Fast test consumer failed: {e}")
    finally:
        if 'consumer' in locals():
            consumer.close()

if __name__ == "__main__":
    main()
