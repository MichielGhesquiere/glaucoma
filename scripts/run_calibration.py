#!/usr/bin/env python3
"""
Quick calibration script for your existing multi-task models.
This will calibrate all your trained models and save them with '_calibrated' suffix.
"""

import subprocess
import sys
import os

def run_calibration():
    """Run temperature scaling calibration on existing models."""
    
    # Default arguments for your setup
    base_cmd = [
        sys.executable, "calibrate_models.py",
        "--backbone", "resnet18",  # Change this to match your trained models
        "--compare_tasks",  # Calibrate both single-task and multi-task if you have comparison results
        "--batch_size", "128",
        "--num_workers", "4",
        "--max_iter", "50"
    ]
    
    print("🔥 Starting temperature scaling calibration...")
    print(f"Command: {' '.join(base_cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(base_cmd, check=True, capture_output=False)
        print("\n✅ Calibration completed successfully!")
        print("\nCalibrated models are saved with '_calibrated' suffix")
        print("These models include temperature scaling for better calibration.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Calibration failed with error code {e.returncode}")
        print("Check the error messages above for details.")
        
    except FileNotFoundError:
        print("❌ Could not find calibrate_models.py script")
        print("Make sure you're running this from the scripts directory.")

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("calibrate_models.py"):
        print("❌ Please run this script from the scripts directory where calibrate_models.py is located")
        sys.exit(1)
    
    print("Temperature Scaling Calibration for Glaucoma Models")
    print("=" * 50)
    print("This will:")
    print("• Load your existing trained models")
    print("• Apply temperature scaling calibration using validation data")
    print("• Save calibrated models with '_calibrated' suffix")
    print("• Improve ECE and Brier score without retraining")
    print("-" * 50)
    
    # Ask for confirmation
    response = input("Continue with calibration? (y/N): ")
    if response.lower() in ['y', 'yes']:
        run_calibration()
    else:
        print("Calibration cancelled.")
