#!/usr/bin/env python3
"""
Quick evaluation runner for Multi-Source Domain Adaptation models.

This script provides a simple interface to evaluate all pre-trained models
with commonly used settings.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

def run_evaluation():
    """Run the evaluation script with standard settings."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Multi-Source Domain Adaptation Evaluation')
    parser.add_argument('--model_dir', default='experiments/multisource',
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', default='evaluation_results/multisource',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--n_bootstraps', type=int, default=1000,
                       help='Number of bootstrap samples for confidence intervals')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--enable_subgroup_analysis', action='store_true', default=True,
                       help='Enable subgroup analysis')
    parser.add_argument('--age_threshold', type=int, default=65,
                       help='Age threshold for subgroup analysis')
    parser.add_argument('--min_subgroup_size', type=int, default=20,
                       help='Minimum subgroup size for analysis')
    parser.add_argument('--use_tta', action='store_true',
                       help='Enable Test-Time Augmentation during evaluation')
    parser.add_argument('--test_datasets', nargs='*',
                       help='Specific datasets to test on (if not specified, tests on all)')
    parser.add_argument('--exclude_datasets', nargs='*',
                       help='Datasets to exclude from testing')
    
    args = parser.parse_args()
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    eval_script = script_dir / "evaluate_multisource.py"
    
    # Build command with parsed arguments
    cmd = [
        sys.executable, str(eval_script),
        "--model_dir", args.model_dir,
        "--output_dir", args.output_dir,
        "--batch_size", str(args.batch_size),
        "--n_bootstraps", str(args.n_bootstraps),
        "--num_workers", str(args.num_workers),
        "--age_threshold", str(args.age_threshold),
        "--min_subgroup_size", str(args.min_subgroup_size)
    ]
    
    # Add optional flags
    if args.enable_subgroup_analysis:
        cmd.append("--enable_subgroup_analysis")
    
    if args.use_tta:
        cmd.append("--use_tta")
        
    if args.test_datasets:
        cmd.extend(["--test_datasets"] + args.test_datasets)
        
    if args.exclude_datasets:
        cmd.extend(["--exclude_datasets"] + args.exclude_datasets)
    
    print("Running Multi-Source Domain Adaptation Evaluation...")
    if args.use_tta:
        print("🔄 TEST-TIME AUGMENTATION (TTA) ENABLED")
        print("   - Will apply horizontal flips, rotations, and intensity augmentations")
        print("   - Results will be saved to TTA-specific directory")
        print("   - This may increase evaluation time significantly")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        # Run the evaluation
        result = subprocess.run(cmd, check=True)
        
        print("-" * 80)
        print("Evaluation completed successfully!")
        
        output_dir = args.output_dir
        if args.use_tta and not args.output_dir.endswith('-TTA'):
            output_dir = args.output_dir.replace('multisource', 'multisource-TTA')
            
        print(f"Results saved to: {output_dir}/")
        print("- detailed_evaluation_results.csv: Full results with all metrics")
        print("- evaluation_summary_table.csv: Formatted summary table")
        print("- subgroup_analysis_*.png: Subgroup analysis plots (if metadata available)")
        
        if args.use_tta:
            print("\n🔄 TTA was used - results may show improved performance due to augmentation averaging")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(run_evaluation())
