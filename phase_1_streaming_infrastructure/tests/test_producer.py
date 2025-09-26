#!/usr/bin/env python3
"""
Test script for the Air Quality Producer

This script tests the producer functionality with a small subset of data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from producer import AirQualityProducer

def test_producer():
    """Test the producer with a small dataset."""
    print("Testing Air Quality Producer...")
    
    try:
        # Initialize producer with test settings (overriding config defaults)
        producer = AirQualityProducer(
            topic_name='air-quality-test',  # Use test topic
            simulation_speed=1.0,  # Real-time speed for testing
            batch_size=10
        )
        
        print("✅ Producer initialized successfully")
        
        # Test sending first 5 messages
        print("Testing message sending...")
        for i in range(5):
            row = producer.data.iloc[i]
            message = producer._create_message(row)
            key = str(row['datetime'].timestamp())
            
            if producer.send_message(message, key):
                print(f"✅ Message {i+1} sent successfully")
            else:
                print(f"❌ Failed to send message {i+1}")
        
        print("✅ Producer test completed successfully")
        
    except Exception as e:
        print(f"❌ Producer test failed: {e}")
        return False
    finally:
        if 'producer' in locals():
            producer.close()
    
    return True

if __name__ == "__main__":
    success = test_producer()
    sys.exit(0 if success else 1)
