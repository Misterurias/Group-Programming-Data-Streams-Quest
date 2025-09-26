#!/usr/bin/env python3
"""
Simple training script for Phase 3 Predictive Analytics.
Runs Linear Regression, SARIMA, and XGBoost model training.
"""

import os
import sys
import subprocess
import logging
import json
import urllib.parse
from typing import Optional

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore

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
        exit_code = 0
    else:
        logger.warning("⚠️  Some models failed to train. Check logs above.")
        exit_code = 1

    # Optional: Auto-register best model to MLflow Model Registry
    try:
        if os.getenv("REGISTER_BEST", "0") != "1":
            logger.info("Model auto-registration disabled (set REGISTER_BEST=1 to enable).")
            return exit_code

        if mlflow is None or MlflowClient is None:
            logger.warning("MLflow not available; cannot perform auto-registration.")
            return exit_code

        tracking_uri = mlflow.get_tracking_uri()
        logger.info(f"MLflow tracking URI: {tracking_uri}")

        # Skip if file:// store (no registry support)
        if tracking_uri and tracking_uri.startswith("file:"):
            logger.warning("Registry requires an MLflow server with DB backend. Skipping registration for file:// store.")
            return exit_code

        client = MlflowClient()
        exp = client.get_experiment_by_name("AirQuality-NOx-6h")
        if exp is None:
            logger.warning("Experiment 'AirQuality-NOx-6h' not found. Skipping registration.")
            return exit_code

        # Find best run by lowest test_rmse
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'FINISHED' and metrics.test_rmse > 0",
            order_by=["metrics.test_rmse ASC"],
            max_results=50,
        )
        if not runs:
            logger.warning("No finished runs with test_rmse found. Skipping registration.")
            return exit_code

        best = runs[0]
        best_run_id = best.info.run_id
        best_rmse = best.data.metrics.get("test_rmse", float("inf"))
        algo = best.data.params.get("algorithm", "unknown")
        logger.info(f"Best run: {best_run_id} algo={algo} test_rmse={best_rmse:.3f}")

        model_name = os.getenv("MODEL_NAME", "AirQualityNOx6h")
        model_uri = f"runs:/{best_run_id}/model"

        logger.info(f"Registering model: name={model_name}, uri={model_uri}")
        registered = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Promote to Production
        client.transition_model_version_stage(
            name=model_name,
            version=registered.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"Promoted {model_name} v{registered.version} to Production")

    except Exception as e:
        logger.warning(f"Auto-registration skipped due to error: {e}")

    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
