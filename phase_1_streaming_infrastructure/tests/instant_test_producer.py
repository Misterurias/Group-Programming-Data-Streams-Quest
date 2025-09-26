#!/usr/bin/env python3
"""
Instant Test Producer for Air Quality Data

This is an instant version of the producer for testing purposes.
It sends messages immediately without any temporal delays.

Author: Santiago Bolaños Vega
Date: 2025-09-21
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from producer import AirQualityProducer

class InstantTestProducer(AirQualityProducer):
    """Instant producer that sends messages without delays."""
    
    def _calculate_delay(self, current_idx: int) -> float:
        """Override to remove all delays for instant testing."""
        return 0.0  # No delays - send messages instantly

def main():
    """Main function to run the instant test producer."""
    try:
        # Initialize producer with instant test settings
        producer = InstantTestProducer(
            simulation_speed=1000.0,  # Not used since we override delays
            batch_size=100
        )
        
        print("🚀 Starting INSTANT TEST Producer...")
        print(f"   Topic: {producer.topic_name}")
        print(f"   Processing: First 20 messages only")
        print("   No delays - instant sending")
        
        # Start streaming with very limited messages and no delays
        producer.start_streaming(start_idx=0, end_idx=20)
        
    except KeyboardInterrupt:
        print("\n⏹️  Instant test producer stopped by user")
    except Exception as e:
        print(f"❌ Instant test producer failed: {e}")
    finally:
        if 'producer' in locals():
            producer.close()

if __name__ == "__main__":
    main()
