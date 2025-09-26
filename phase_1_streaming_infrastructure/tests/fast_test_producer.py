#!/usr/bin/env python3
"""
Fast Test Producer for Air Quality Data

This is a fast version of the producer for testing purposes.
It processes only a small number of messages at high speed.

Author: Santiago Bolaños Vega
Date: 2025-09-21
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from producer import AirQualityProducer

def main():
    """Main function to run the fast test producer."""
    try:
        # Initialize producer with ultra-fast test settings
        producer = AirQualityProducer(
            simulation_speed=1000.0,  # 1000x faster than real-time
            batch_size=100
        )
        
        print("🚀 Starting ULTRA-FAST TEST Producer...")
        print(f"   Topic: {producer.topic_name}")
        print(f"   Simulation Speed: {producer.simulation_speed}x")
        print(f"   Processing: First 10 messages only")
        print("   Press Ctrl+C to stop")
        
        # Start streaming with very limited messages
        producer.start_streaming(start_idx=0, end_idx=10)
        
    except KeyboardInterrupt:
        print("\n⏹️  Fast test producer stopped by user")
    except Exception as e:
        print(f"❌ Fast test producer failed: {e}")
    finally:
        if 'producer' in locals():
            producer.close()

if __name__ == "__main__":
    main()
