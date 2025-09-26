#!/usr/bin/env python3
"""
Simple training script for Phase 3 Predictive Analytics.
Runs Linear Regression, SARIMA, and XGBoost model training.
"""

import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_training_script(script_path: str, script_name: str):
    """Run a training script and log the results."""
    logger.info(f"Starting {script_name} training...")
    
    try:
        # Run the training script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.path.dirname(script_path))
        
        if result.returncode == 0:
            logger.info(f"✅ {script_name} training completed successfully")
            if result.stdout:
                logger.info(f"Output: {result.stdout.strip()}")
        else:
            logger.error(f"❌ {script_name} training failed")
            logger.error(f"Error: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to run {script_name}: {str(e)}")
        return False
    
    return True

def main():
    """Main training function."""
    logger.info("🚀 Starting Phase 3 Model Training")
    logger.info("=" * 50)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_dir = os.path.join(script_dir, 'training')
    
    # Define training scripts
    scripts = [
        {
            'path': os.path.join(training_dir, 'train_linear_regression.py'),
            'name': 'Linear Regression'
        },
        {
            'path': os.path.join(training_dir, 'train_sarima.py'),
            'name': 'SARIMA'
        },
        {
            'path': os.path.join(training_dir, 'train_xgboost.py'),
            'name': 'XGBoost'
        }
    ]
    
    # Track results
    results = []
    
    # Run each training script
    for script in scripts:
        if os.path.exists(script['path']):
            success = run_training_script(script['path'], script['name'])
            results.append((script['name'], success))
        else:
            logger.error(f"❌ Training script not found: {script['path']}")
            results.append((script['name'], False))
    
    # Summary
    logger.info("=" * 50)
    logger.info("📊 Training Summary:")
    
    successful = 0
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"  {name}: {status}")
        if success:
            successful += 1
    
    logger.info(f"\n🎯 Completed: {successful}/{len(results)} models trained successfully")
    
    if successful == len(results):
        logger.info("🎉 All models trained successfully!")
        return 0
    else:
        logger.warning("⚠️  Some models failed to train. Check logs above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
