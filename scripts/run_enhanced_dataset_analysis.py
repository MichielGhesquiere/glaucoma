#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script to run the enhanced dataset analysis with vCDR analysis.

This script demonstrates how to use the enhanced dataset_analysis.py script
to perform comprehensive analysis including:
1. Basic dataset statistics
2. Pixel distribution analysis
3. UMAP embeddings visualization  
4. vCDR analysis with optimal threshold calculation using U-Net segmentation

Usage:
    python run_enhanced_dataset_analysis.py
"""

import subprocess
import sys
import os

def run_enhanced_analysis():
    """Run the enhanced dataset analysis with all features enabled."""
    
    # Define the command to run
    cmd = [
        sys.executable, "dataset_analysis.py",
        
        # Basic configuration
        "--output_dir", "dataset_analysis_results",
        "--data_type", "raw",
        "--base_data_root", r"D:\glaucoma\data",
        
        # Enable all advanced analyses
        "--pixel_analysis",
        "--umap_analysis", 
        "--vcdr_analysis",
        
        # For comprehensive analysis, process ALL samples (remove max limit)
        "--vcdr_max_samples", "None",
        
        # Model paths
        "--model_path", r"D:\glaucoma\models\VFM_Fundus_weights.pth",
        "--unet_model_path", r"D:\glaucoma\multitask_training_logs\best_multitask_model_epoch_25.pth",
        
        # Dataset configuration
        "--use_airogs",
        "--airogs_num_rg", "3000",
        "--airogs_num_nrg", "3000",
        
        # External datasets
        "--eval_papilla",
        "--eval_chaksu", 
        "--eval_acrima",
        "--eval_hygd",
        
        # Random seed for reproducibility
        "--seed", "42"
    ]
    
    print("="*80)
    print("ENHANCED DATASET ANALYSIS")
    print("="*80)
    print("Running comprehensive dataset analysis with:")
    print("  ✓ Basic dataset statistics")
    print("  ✓ Pixel distribution analysis") 
    print("  ✓ UMAP embeddings visualization")
    print("  ✓ vCDR analysis with optimal thresholds (ALL SAMPLES)")
    print("="*80)
    print()
    
    # Change to the scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Run the analysis
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Check the 'dataset_analysis_results' directory for:")
        print("  📊 Basic statistics and summary tables")
        print("  📈 Distribution plots and visualizations")
        print("  🗺️ UMAP embeddings visualization")  
        print("  👁️ vCDR analysis with optimal thresholds")
        print("  📋 ROC curves and performance metrics")
        print("="*80)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Analysis failed with error code {e.returncode}")
        print("Check the error messages above for details.")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Could not find dataset_analysis.py script")
        print("Make sure you're running this from the scripts directory.")
        sys.exit(1)

def run_vcdr_only_analysis():
    """Run only the vCDR analysis for faster testing."""
    
    cmd = [
        sys.executable, "dataset_analysis.py",
        
        # Basic configuration
        "--output_dir", "dataset_analysis_results",
        "--data_type", "raw", 
        "--base_data_root", r"D:\glaucoma\data",
        
        # Enable only vCDR analysis (limited samples for testing)
        "--vcdr_analysis",
        "--no-pixel_analysis",
        "--no-umap_analysis",
        "--vcdr_max_samples", "500",  # Limit for testing
        
        # U-Net model path
        "--unet_model_path", r"D:\glaucoma\multitask_training_logs\best_multitask_model_epoch_25.pth",
        
        # Dataset configuration (minimal for testing)
        "--no-use_airogs",
        "--eval_papilla",
        "--no-eval_chaksu",
        "--no-eval_acrima", 
        "--no-eval_hygd"
    ]
    
    print("Running vCDR-only analysis with LIMITED samples for testing...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ vCDR analysis completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ vCDR analysis failed with error code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhanced dataset analysis")
    parser.add_argument("--vcdr-only", action="store_true", 
                       help="Run only vCDR analysis for faster testing")
    
    args = parser.parse_args()
    
    if args.vcdr_only:
        run_vcdr_only_analysis()
    else:
        run_enhanced_analysis()
