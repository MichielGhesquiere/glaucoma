#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare Multi-Source Domain Fine-Tuning Experiments

This script compares results from multiple experiments to evaluate the effectiveness
of different domain adaptation techniques.

Usage:
    python compare_experiments.py ./results_baseline_vfm ./results_vfm_swa
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob

def load_experiment_results(result_dir):
    """Load results from an experiment directory."""
    result_dir = Path(result_dir)
    
    # Find the most recent results file
    detailed_files = list(result_dir.glob("**/detailed_results_*.csv"))
    if not detailed_files:
        raise FileNotFoundError(f"No detailed results found in {result_dir}")
    
    # Get the most recent file
    latest_file = max(detailed_files, key=os.path.getctime)
    
    # Load results
    df = pd.read_csv(latest_file)
    
    # Extract experiment name from directory
    experiment_name = result_dir.name
    df['experiment'] = experiment_name
    
    print(f"Loaded {len(df)} results from {experiment_name}")
    return df

def compare_experiments(experiment_dirs, output_dir="./comparison_results"):
    """Compare multiple experiments and generate comparison visualizations."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all experiment results
    all_results = []
    for exp_dir in experiment_dirs:
        try:
            df = load_experiment_results(exp_dir)
            all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load results from {exp_dir}: {e}")
    
    if not all_results:
        print("No valid experiment results found!")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    combined_path = os.path.join(output_dir, "combined_results.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined results saved to: {combined_path}")
    
    # Generate comparison report
    generate_comparison_report(combined_df, output_dir)
    
    # Generate comparison visualizations
    create_comparison_visualizations(combined_df, output_dir)
    
    return combined_df

def generate_comparison_report(df, output_dir):
    """Generate a detailed comparison report."""
    
    # Calculate summary statistics by experiment
    metrics = ['auc', 'sensitivity_at_95_specificity', 'ece', 'accuracy']
    
    summary_stats = df.groupby('experiment')[metrics].agg(['mean', 'std']).round(4)
    
    # Flatten column names
    summary_stats.columns = [f"{col[0]}_{col[1]}" for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, "experiment_comparison_summary.csv")
    summary_stats.to_csv(summary_path, index=False)
    
    # Generate text report
    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-SOURCE DOMAIN FINE-TUNING EXPERIMENT COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Experiments compared: {len(df['experiment'].unique())}\n")
        f.write(f"Total results: {len(df)}\n")
        f.write(f"Datasets tested: {len(df['target_dataset'].unique())}\n\n")
        
        f.write("SUMMARY STATISTICS (Mean ± Std):\n")
        f.write("-" * 50 + "\n")
        
        for _, row in summary_stats.iterrows():
            experiment = row['experiment']
            f.write(f"\n{experiment}:\n")
            f.write(f"  AUC:                    {row['auc_mean']:.4f} ± {row['auc_std']:.4f}\n")
            f.write(f"  Sens@95%Spec:          {row['sensitivity_at_95_specificity_mean']:.4f} ± {row['sensitivity_at_95_specificity_std']:.4f}\n")
            f.write(f"  ECE:                   {row['ece_mean']:.4f} ± {row['ece_std']:.4f}\n")
            f.write(f"  Accuracy:              {row['accuracy_mean']:.2f}% ± {row['accuracy_std']:.2f}%\n")
        
        # Calculate improvements
        f.write("\n" + "=" * 50 + "\n")
        f.write("PERFORMANCE IMPROVEMENTS:\n")
        f.write("=" * 50 + "\n")
        
        if len(summary_stats) >= 2:
            # Assume first experiment is baseline
            baseline = summary_stats.iloc[0]
            for i in range(1, len(summary_stats)):
                improved = summary_stats.iloc[i]
                f.write(f"\n{improved['experiment']} vs {baseline['experiment']}:\n")
                
                auc_improvement = improved['auc_mean'] - baseline['auc_mean']
                sens_improvement = improved['sensitivity_at_95_specificity_mean'] - baseline['sensitivity_at_95_specificity_mean']
                ece_improvement = baseline['ece_mean'] - improved['ece_mean']  # Lower ECE is better
                acc_improvement = improved['accuracy_mean'] - baseline['accuracy_mean']
                
                f.write(f"  AUC improvement:        {auc_improvement:+.4f}\n")
                f.write(f"  Sens@95Spec improvement: {sens_improvement:+.4f}\n")
                f.write(f"  ECE improvement:        {ece_improvement:+.4f} (lower is better)\n")
                f.write(f"  Accuracy improvement:   {acc_improvement:+.2f}%\n")
        
        # Detailed results by dataset
        f.write("\n" + "=" * 50 + "\n")
        f.write("DETAILED RESULTS BY DATASET:\n")
        f.write("=" * 50 + "\n")
        
        for dataset in df['target_dataset'].unique():
            f.write(f"\nTarget Dataset: {dataset}\n")
            f.write("-" * 30 + "\n")
            
            dataset_results = df[df['target_dataset'] == dataset]
            for experiment in dataset_results['experiment'].unique():
                exp_results = dataset_results[dataset_results['experiment'] == experiment]
                if len(exp_results) > 0:
                    row = exp_results.iloc[0]
                    f.write(f"  {experiment}:\n")
                    f.write(f"    AUC: {row['auc']:.4f}, Sens@95Spec: {row['sensitivity_at_95_specificity']:.4f}, ")
                    f.write(f"ECE: {row['ece']:.4f}, Acc: {row['accuracy']:.1f}%\n")
    
    print(f"Comparison report saved to: {report_path}")

def create_comparison_visualizations(df, output_dir):
    """Create comparison visualizations."""
    
    metrics = ['auc', 'sensitivity_at_95_specificity', 'ece', 'accuracy']
    metric_names = ['AUC', 'Sensitivity @ 95% Specificity', 'Expected Calibration Error', 'Accuracy (%)']
    
    # 1. Bar plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i // 2, i % 2]
        
        # Calculate mean and std for each experiment
        summary = df.groupby('experiment')[metric].agg(['mean', 'std']).reset_index()
        
        # Create bar plot with error bars
        x_pos = np.arange(len(summary))
        bars = ax.bar(x_pos, summary['mean'], yerr=summary['std'], 
                     capsize=5, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'][:len(summary)])
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + summary['std'].iloc[j],
                   f'{summary["mean"].iloc[j]:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Experiment', fontsize=10)
        ax.set_ylabel(name, fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary['experiment'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Experiment Performance Comparison (Mean ± Std)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comparison_path = os.path.join(output_dir, "experiment_comparison_bars.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar comparison saved to: {comparison_path}")
    
    # 2. Box plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i // 2, i % 2]
        
        # Create box plot
        experiments = df['experiment'].unique()
        data_for_box = [df[df['experiment'] == exp][metric].values for exp in experiments]
        
        box_plot = ax.boxplot(data_for_box, labels=experiments, patch_artist=True)
        
        # Color the boxes
        colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(experiments)]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Experiment', fontsize=10)
        ax.set_ylabel(name, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Experiment Performance Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    box_path = os.path.join(output_dir, "experiment_comparison_boxes.png")
    plt.savefig(box_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Box plot comparison saved to: {box_path}")
    
    # 3. Dataset-specific comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create grouped bar plot for AUC by dataset
    datasets = df['target_dataset'].unique()
    experiments = df['experiment'].unique()
    
    x = np.arange(len(datasets))
    width = 0.35 if len(experiments) == 2 else 0.25
    
    for i, experiment in enumerate(experiments):
        exp_data = []
        for dataset in datasets:
            subset = df[(df['experiment'] == experiment) & (df['target_dataset'] == dataset)]
            if len(subset) > 0:
                exp_data.append(subset['auc'].iloc[0])
            else:
                exp_data.append(0)
        
        offset = (i - len(experiments)/2 + 0.5) * width
        bars = ax.bar(x + offset, exp_data, width, label=experiment, alpha=0.7)
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Target Dataset', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('AUC Comparison by Target Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    dataset_path = os.path.join(output_dir, "experiment_comparison_by_dataset.png")
    plt.savefig(dataset_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dataset comparison saved to: {dataset_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare Multi-Source Domain Fine-Tuning Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('experiment_dirs', nargs='+', 
                       help="Directories containing experiment results")
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                       help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    print("Starting experiment comparison...")
    print(f"Comparing experiments: {args.experiment_dirs}")
    
    # Run comparison
    combined_df = compare_experiments(args.experiment_dirs, args.output_dir)
    
    if combined_df is not None:
        print(f"\nComparison complete! Results saved to: {args.output_dir}")
        print("\nFiles generated:")
        print(f"  - combined_results.csv: All results combined")
        print(f"  - experiment_comparison_summary.csv: Summary statistics")
        print(f"  - comparison_report.txt: Detailed text report")
        print(f"  - experiment_comparison_bars.png: Bar chart comparison")
        print(f"  - experiment_comparison_boxes.png: Box plot comparison")
        print(f"  - experiment_comparison_by_dataset.png: Dataset-specific comparison")

if __name__ == "__main__":
    main()
