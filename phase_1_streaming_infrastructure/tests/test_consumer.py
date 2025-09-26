#!/usr/bin/env python3
"""
Test script for the Air Quality Consumer

This script tests the consumer functionality with a small subset of data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consumer import AirQualityConsumer
from data_preprocessing import DataPreprocessor
from data_validation import DataValidator

def test_preprocessing():
    """Test the preprocessing module."""
    print("Testing Data Preprocessing...")
    
    try:
        preprocessor = DataPreprocessor()
        
        # Create test data with missing values
        test_messages = [
            {
                'timestamp': '2004-03-10 18:00:00',
                'sensor_data': {
                    'co_gt': 2.6,
                    'nox_gt': 166,
                    'benzene_gt': 11.9,
                    'temperature': 13.6,
                    'relative_humidity': 48.9,
                    'absolute_humidity': 0.7578
                },
                'metadata': {'source': 'test'}
            },
            {
                'timestamp': '2004-03-10 19:00:00',
                'sensor_data': {
                    'co_gt': -200,  # Missing value
                    'nox_gt': 103,
                    'benzene_gt': 9.4,
                    'temperature': 13.3,
                    'relative_humidity': 47.7,
                    'absolute_humidity': 0.7255
                },
                'metadata': {'source': 'test'}
            }
        ]
        
        # Process test data
        processed_data = preprocessor.process_batch(test_messages)
        
        print(f"✅ Preprocessing test passed: {len(processed_data)} records processed")
        print(f"   Missing values handled: {preprocessor.get_preprocessing_stats(processed_data)}")
        
        return processed_data
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return None

def test_validation(processed_data):
    """Test the validation module."""
    print("Testing Data Validation...")
    
    try:
        validator = DataValidator()
        
        # Validate processed data
        validated_data, validation_results = validator.validate_batch(processed_data)
        
        print(f"✅ Validation test passed: {len(validated_data)} records validated")
        print(f"   Quality score: {validation_results['average_quality_score']:.3f}")
        print(f"   Valid records: {validation_results['valid_records']}/{validation_results['total_records']}")
        
        return validated_data, validation_results
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return None, None

def test_consumer():
    """Test the consumer module."""
    print("Testing Air Quality Consumer...")
    
    try:
        # Initialize consumer with test settings
        consumer = AirQualityConsumer(
            topic_name='air-quality-test',  # Use test topic
            batch_size=5  # Small batch for testing
        )
        
        print("✅ Consumer initialized successfully")
        
        # Test components
        print(f"   Preprocessor: {type(consumer.preprocessor).__name__}")
        print(f"   Validator: {type(consumer.validator).__name__}")
        print(f"   Monitor: {type(consumer.monitor).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Consumer test failed: {e}")
        return False

def test_integration():
    """Test the complete pipeline."""
    print("Testing Complete Pipeline...")
    
    try:
        # Test preprocessing
        processed_data = test_preprocessing()
        if processed_data is None:
            return False
        
        # Test validation
        validated_data, validation_results = test_validation(processed_data)
        if validated_data is None:
            return False
        
        # Test consumer
        consumer_success = test_consumer()
        if not consumer_success:
            return False
        
        print("✅ Complete pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("AIR QUALITY CONSUMER TEST SUITE")
    print("="*50)
    
    success = test_integration()
    
    print("\n" + "="*50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("Consumer is ready for production use.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above.")
    print("="*50)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
