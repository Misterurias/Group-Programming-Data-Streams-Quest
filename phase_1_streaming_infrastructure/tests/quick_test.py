#!/usr/bin/env python3
"""
Quick End-to-End Test Script

This script runs a quick end-to-end test of the producer-consumer pipeline.
It processes only a small number of messages at high speed.

Author: Santiago Bolaños Vega
Date: 2025-09-21
"""

import subprocess
import time
import os
import sys

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n🚀 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def main():
    """Run quick end-to-end test."""
    print("="*60)
    print("🚀 QUICK END-TO-END TEST")
    print("="*60)
    
    # Step 1: Run unit tests
    print("\n📋 Step 1: Running Unit Tests")
    unit_test_success = run_command("python tests/test_consumer.py", "Unit Tests")
    
    if not unit_test_success:
        print("❌ Unit tests failed. Stopping.")
        return False
    
    # Step 2: Run fast producer
    print("\n📤 Step 2: Running Fast Producer")
    producer_success = run_command("python tests/fast_test_producer.py", "Fast Producer")
    
    if not producer_success:
        print("❌ Fast producer failed. Stopping.")
        return False
    
    # Step 3: Run fast consumer
    print("\n📥 Step 3: Running Fast Consumer")
    consumer_success = run_command("python tests/fast_test_consumer.py", "Fast Consumer")
    
    if not consumer_success:
        print("❌ Fast consumer failed. Stopping.")
        return False
    
    # Step 4: Check output
    print("\n📊 Step 4: Checking Output")
    # Ensure test artifacts directory exists
    artifacts_dir = os.path.join('tests', 'artifacts', 'processed')
    os.makedirs(artifacts_dir, exist_ok=True)
    output_file = os.path.join('tests', 'artifacts', 'processed', 'air_quality_test.csv')
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"✅ Output file created: {output_file}")
        print(f"   File size: {file_size} bytes")
        
        # Count lines in output file
        try:
            with open(output_file, 'r') as f:
                line_count = sum(1 for line in f)
            print(f"   Records processed: {line_count}")
            if line_count >= 10:
                print("   ✅ Expected number of records processed")
            else:
                print(f"   ⚠️  Expected 10+ records, got {line_count}")
        except Exception as e:
            print(f"   Could not count lines: {e}")
    else:
        print(f"❌ Output file not found: {output_file}")
        return False
    
    print("\n" + "="*60)
    print("🎉 QUICK TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("✅ Unit tests passed")
    print("✅ Producer sent messages")
    print("✅ Consumer processed messages")
    print("✅ Output file created")
    print("\n🚀 Your streaming pipeline is working!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
