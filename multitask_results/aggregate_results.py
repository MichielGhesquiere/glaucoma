"""
Aggregate and visualize results from all trained models.

This script collects results from all fold directories and creates comprehensive
visualizations comparing:
1. AUC, Sensitivity, Specificity between single-task and multi-task models
2. ECE and Brier scores for original and calibrated models
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from datetime import datetime
from sklearn.metrics import roc_curve

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def find_result_files(base_dir):
    """
    Find all result JSON files and prediction CSV files in the directory structure.
    
    Returns:
        dict: Dictionary with file types as keys and lists of file paths as values
    """
    result_files = {
        'comparison': [],  # comparison_*.json files
        'calibrated': [],  # *_calibrated_results.json files
        'individual': [],   # results_*_singletask.json and results_*_multitask.json files
        'predictions': []   # predictions_*.csv files
    }
    
    base_path = Path(base_dir)
    
    # Search for all JSON files and CSV files
    for json_file in base_path.rglob("*.json"):
        file_name = json_file.name.lower()
        
        if file_name.startswith('comparison_') and file_name.endswith('.json'):
            result_files['comparison'].append(json_file)
        elif '_calibrated_results.json' in file_name:
            result_files['calibrated'].append(json_file)
        elif (file_name.startswith('results_') and 
              (file_name.endswith('_singletask.json') or file_name.endswith('_multitask.json'))):
            result_files['individual'].append(json_file)
    
    # Search for prediction CSV files
    for csv_file in base_path.rglob("predictions_*.csv"):
        result_files['predictions'].append(csv_file)
    
    print(f"Found {len(result_files['comparison'])} comparison files")
    print(f"Found {len(result_files['calibrated'])} calibrated result files")
    print(f"Found {len(result_files['individual'])} individual result files")
    print(f"Found {len(result_files['predictions'])} prediction files")
    
    return result_files

def calculate_sensitivity_at_specificity(y_true, y_scores, target_specificity=0.95):
    """
    Calculate sensitivity at a given specificity level.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        target_specificity: Target specificity level (default 0.95)
        
    Returns:
        float: Sensitivity at the target specificity, or np.nan if not achievable
    """
    try:
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Calculate specificity = 1 - FPR
        specificity = 1 - fpr
        
        # Find the point where specificity is closest to target
        # But only consider points where specificity >= target_specificity
        valid_indices = specificity >= target_specificity
        
        if not np.any(valid_indices):
            # If we can't achieve the target specificity, return nan
            return np.nan
        
        # Among valid points, find the one with highest sensitivity
        valid_sensitivities = tpr[valid_indices]
        max_sensitivity = np.max(valid_sensitivities)
        
        return max_sensitivity
        
    except Exception as e:
        print(f"Error calculating sensitivity at specificity: {e}")
        return np.nan

def load_prediction_results(prediction_files):
    """
    Load prediction CSV files and calculate sensitivity at 95% specificity.
    
    Returns:
        pd.DataFrame: DataFrame with prediction-based results including sens@95%spec
    """
    data = []
    
    for file_path in prediction_files:
        try:
            # Load predictions
            predictions_df = pd.read_csv(file_path)
            
            # Extract metadata from file path
            # e.g., fold_CHAKSU_vfm_comparison/multitask/predictions_vfm_multitask.csv
            parent_folder = file_path.parent.parent.name  # fold_CHAKSU_vfm_comparison
            task_folder = file_path.parent.name  # multitask or singletask
            file_name = file_path.name  # predictions_vfm_multitask.csv
            
            # Extract test dataset from parent folder
            test_dataset_match = re.search(r'fold_([^_]+(?:_all)?)_', parent_folder)
            test_dataset = test_dataset_match.group(1) if test_dataset_match else 'unknown'
            test_dataset = standardize_dataset_name(test_dataset)
            
            # Extract backbone from filename
            backbone_match = re.search(r'predictions_([^_]+)_', file_name)
            backbone = backbone_match.group(1) if backbone_match else 'unknown'
            
            # Task mode from folder name
            task_mode = task_folder
            
            # Filter for valid binary classification data
            valid_data = predictions_df[
                (predictions_df['has_binary_label'] == True) & 
                (predictions_df['true_binary_label'].notna()) & 
                (predictions_df['pred_binary_prob'].notna())
            ].copy()
            
            if len(valid_data) == 0:
                print(f"No valid prediction data in {file_path}")
                continue
            
            # Calculate metrics
            y_true = valid_data['true_binary_label'].astype(int)
            y_scores = valid_data['pred_binary_prob'].astype(float)
            
            # Calculate sensitivity at 95% specificity
            sens_at_95_spec = calculate_sensitivity_at_specificity(y_true, y_scores, 0.95)
            
            # Also calculate standard metrics at 0.5 threshold for comparison
            y_pred = (y_scores >= 0.5).astype(int)
            
            # Calculate standard metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred)  # This is sensitivity
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc = roc_auc_score(y_true, y_scores)
            
            # Calculate specificity manually
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            data.append({
                'test_dataset': test_dataset,
                'backbone': backbone,
                'task_mode': task_mode,
                'auc': auc,
                'sensitivity': recall,
                'specificity': specificity,
                'sensitivity_at_95_spec': sens_at_95_spec,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'ece': np.nan,  # Not calculated from predictions
                'brier_score': np.nan,  # Not calculated from predictions
                'calibrated': False,
                'file_path': str(file_path),
                'data_source': 'predictions'
            })
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return pd.DataFrame(data)

def load_comparison_results(comparison_files):
    """
    Load and aggregate comparison results from comparison_*.json files.
    
    Returns:
        pd.DataFrame: DataFrame with comparison results
    """
    data = []
    
    for file_path in comparison_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            # Extract test dataset and backbone from filename or content
            file_name = file_path.name
            if 'comparison_' in file_name:
                backbone = file_name.replace('comparison_', '').replace('.json', '')
            else:
                backbone = result.get('backbone', 'unknown')
            
            test_dataset = result.get('test_dataset', 'unknown')
            test_dataset = standardize_dataset_name(test_dataset)  # Standardize dataset name
            
            # Extract metrics for both single-task and multi-task
            singletask_metrics = result.get('singletask_results', {}).get('metrics', {})
            multitask_metrics = result.get('multitask_results', {}).get('metrics', {})
            
            # Add single-task row
            if singletask_metrics:
                data.append({
                    'test_dataset': test_dataset,
                    'backbone': backbone,
                    'task_mode': 'singletask',
                    'auc': singletask_metrics.get('auc', np.nan),
                    'sensitivity': singletask_metrics.get('sensitivity', np.nan),
                    'specificity': singletask_metrics.get('specificity', np.nan),
                    'sensitivity_at_95_spec': np.nan,  # Not available from JSON
                    'accuracy': singletask_metrics.get('accuracy', np.nan),
                    'precision': singletask_metrics.get('precision', np.nan),
                    'recall': singletask_metrics.get('recall', np.nan),
                    'f1': singletask_metrics.get('f1', np.nan),
                    'ece': singletask_metrics.get('ece', np.nan),
                    'brier_score': singletask_metrics.get('brier_score', np.nan),
                    'calibrated': False,
                    'file_path': str(file_path),
                    'data_source': 'comparison'
                })
            
            # Add multi-task row
            if multitask_metrics:
                data.append({
                    'test_dataset': test_dataset,
                    'backbone': backbone,
                    'task_mode': 'multitask',
                    'auc': multitask_metrics.get('auc', np.nan),
                    'sensitivity': multitask_metrics.get('sensitivity', np.nan),
                    'specificity': multitask_metrics.get('specificity', np.nan),
                    'sensitivity_at_95_spec': np.nan,  # Not available from JSON
                    'accuracy': multitask_metrics.get('accuracy', np.nan),
                    'precision': multitask_metrics.get('precision', np.nan),
                    'recall': multitask_metrics.get('recall', np.nan),
                    'f1': multitask_metrics.get('f1', np.nan),
                    'ece': multitask_metrics.get('ece', np.nan),
                    'brier_score': multitask_metrics.get('brier_score', np.nan),
                    'calibrated': False,
                    'file_path': str(file_path),
                    'data_source': 'comparison'
                })
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return pd.DataFrame(data)

def load_calibrated_results(calibrated_files):
    """
    Load and aggregate calibrated results from *_calibrated_results.json files.
    
    Returns:
        pd.DataFrame: DataFrame with calibrated results
    """
    data = []
    
    for file_path in calibrated_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            test_dataset = result.get('test_dataset', 'unknown')
            test_dataset = standardize_dataset_name(test_dataset)  # Standardize dataset name
            backbone = result.get('backbone', 'unknown')
            task_mode = result.get('task_mode', 'unknown')
            
            # Original metrics
            original_metrics = result.get('original_test_metrics', {})
            calibrated_metrics = result.get('calibrated_test_metrics', {})
            
            # Add original (uncalibrated) row
            if original_metrics:
                data.append({
                    'test_dataset': test_dataset,
                    'backbone': backbone,
                    'task_mode': task_mode,
                    'auc': original_metrics.get('auc', np.nan),
                    'sensitivity': original_metrics.get('sensitivity', np.nan),
                    'specificity': original_metrics.get('specificity', np.nan),
                    'sensitivity_at_95_spec': np.nan,  # Not available from JSON
                    'accuracy': original_metrics.get('accuracy', np.nan),
                    'precision': original_metrics.get('precision', np.nan),
                    'recall': original_metrics.get('recall', np.nan),
                    'f1': original_metrics.get('f1', np.nan),
                    'ece': original_metrics.get('ece', np.nan),
                    'brier_score': original_metrics.get('brier_score', np.nan),
                    'calibrated': False,
                    'file_path': str(file_path),
                    'data_source': 'calibrated'
                })
            
            # Add calibrated row
            if calibrated_metrics:
                data.append({
                    'test_dataset': test_dataset,
                    'backbone': backbone,
                    'task_mode': task_mode,
                    'auc': calibrated_metrics.get('auc', np.nan),
                    'sensitivity': calibrated_metrics.get('sensitivity', np.nan),
                    'specificity': calibrated_metrics.get('specificity', np.nan),
                    'sensitivity_at_95_spec': np.nan,  # Not available from JSON
                    'accuracy': calibrated_metrics.get('accuracy', np.nan),
                    'precision': calibrated_metrics.get('precision', np.nan),
                    'recall': calibrated_metrics.get('recall', np.nan),
                    'f1': calibrated_metrics.get('f1', np.nan),
                    'ece': calibrated_metrics.get('ece', np.nan),
                    'brier_score': calibrated_metrics.get('brier_score', np.nan),
                    'calibrated': True,
                    'file_path': str(file_path),
                    'data_source': 'calibrated'
                })
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return pd.DataFrame(data)

def load_individual_results(individual_files):
    """
    Load and aggregate individual results from results_*_singletask.json and results_*_multitask.json files.
    
    Returns:
        pd.DataFrame: DataFrame with individual results
    """
    data = []
    
    for file_path in individual_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            # Extract information from filename
            file_name = file_path.name.lower()
            
            # Extract task mode from filename
            if '_singletask.json' in file_name:
                task_mode = 'singletask'
            elif '_multitask.json' in file_name:
                task_mode = 'multitask'
            else:
                continue
            
            # Extract backbone from filename (e.g., results_resnet18_multitask.json -> resnet18)
            backbone_match = re.search(r'results_([^_]+)_', file_name)
            backbone = backbone_match.group(1) if backbone_match else 'unknown'
            
            # Extract test dataset from parent folder name
            parent_folder = file_path.parent.name
            # Handle both fold_DATASET_ and fold_DATASET_all_ patterns
            test_dataset_match = re.search(r'fold_([^_]+(?:_all)?)_', parent_folder)
            test_dataset = test_dataset_match.group(1) if test_dataset_match else 'unknown'
            test_dataset = standardize_dataset_name(test_dataset)  # Standardize dataset name
            
            # Extract metrics
            metrics = result.get('metrics', {})
            
            if metrics:
                data.append({
                    'test_dataset': test_dataset,
                    'backbone': backbone,
                    'task_mode': task_mode,
                    'auc': metrics.get('auc', np.nan),
                    'sensitivity': metrics.get('sensitivity', np.nan),
                    'specificity': metrics.get('specificity', np.nan),
                    'sensitivity_at_95_spec': np.nan,  # Not available from JSON
                    'accuracy': metrics.get('accuracy', np.nan),
                    'precision': metrics.get('precision', np.nan),
                    'recall': metrics.get('recall', np.nan),
                    'f1': metrics.get('f1', np.nan),
                    'ece': metrics.get('ece', np.nan),
                    'brier_score': metrics.get('brier_score', np.nan),
                    'calibrated': False,
                    'file_path': str(file_path),
                    'data_source': 'individual'
                })
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return pd.DataFrame(data)

def standardize_dataset_name(dataset_name):
    """
    Standardize dataset names by removing _all suffix and handling variations.
    
    Args:
        dataset_name: Original dataset name
        
    Returns:
        Standardized dataset name
    """
    if not dataset_name or dataset_name == 'unknown':
        return dataset_name
    
    # Remove _all suffix
    if dataset_name.endswith('_all'):
        dataset_name = dataset_name[:-4]
    
    # Handle specific dataset name mappings
    name_mappings = {
        'OIA-ODIR-test': 'OIA-ODIR-test',
        'OIA-ODIR-train': 'OIA-ODIR-train',
        'REFUGE1': 'REFUGE1',
        'AIROGS': 'AIROGS',
        'CHAKSU': 'CHAKSU',
        'HYGD': 'HYGD'
    }
    
    return name_mappings.get(dataset_name, dataset_name)

def create_task_comparison_plots(df, output_dir):
    """
    Create plots comparing single-task vs multi-task performance for different backbones.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plots
    """
    # Filter for non-calibrated results only for task comparison
    df_original = df[df['calibrated'] == False].copy()
    
    if df_original.empty:
        print("No original (non-calibrated) results found for task comparison")
        return
    
    # Metrics to plot
    metrics = ['auc', 'sensitivity', 'specificity', 'sensitivity_at_95_spec']
    metric_labels = ['AUC', 'Sensitivity@0.5', 'Specificity@0.5', 'Sensitivity@95%Spec']
    
    # Get unique backbones
    backbones = sorted(df_original['backbone'].unique())
    print(f"Found backbones: {backbones}")
    
    for metric, metric_label in zip(metrics, metric_labels):
        if metric not in df_original.columns or df_original[metric].isna().all():
            print(f"Skipping {metric} - no data available")
            continue
            
        plt.figure(figsize=(16, 10))
        
        # Create subplot for each backbone
        n_backbones = len(backbones)
        if n_backbones == 1:
            # Single backbone - use the old format
            pivot_data = df_original.pivot_table(
                index='test_dataset', 
                columns='task_mode', 
                values=metric, 
                aggfunc='mean'
            )
            
            x = np.arange(len(pivot_data.index))
            width = 0.35
            
            singletask_values = pivot_data.get('singletask', pd.Series([np.nan] * len(pivot_data)))
            multitask_values = pivot_data.get('multitask', pd.Series([np.nan] * len(pivot_data)))
            
            plt.bar(x - width/2, singletask_values, width, label='Single-task', alpha=0.8)
            plt.bar(x + width/2, multitask_values, width, label='Multi-task', alpha=0.8)
            
            plt.xlabel('Test Dataset')
            plt.ylabel(metric_label)
            plt.title(f'{metric_label} Comparison: Single-task vs Multi-task')
            plt.xticks(x, pivot_data.index, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        else:
            # Multiple backbones - create grouped bars for each backbone and task combination
            # Create a pivot table with multi-level columns
            pivot_data = df_original.pivot_table(
                index='test_dataset',
                columns=['backbone', 'task_mode'],
                values=metric,
                aggfunc='mean'
            )
            
            if pivot_data.empty:
                print(f"No data available for {metric} comparison")
                continue
            
            x = np.arange(len(pivot_data.index))
            n_groups = len(pivot_data.columns)
            width = 0.8 / n_groups
            
            colors = plt.cm.Set1(np.linspace(0, 1, n_groups))
            
            for i, (backbone, task_mode) in enumerate(pivot_data.columns):
                if (backbone, task_mode) in pivot_data.columns:
                    values = pivot_data[(backbone, task_mode)]
                    label = f'{backbone} {task_mode}'
                    offset = (i - (n_groups-1)/2) * width
                    plt.bar(x + offset, values, width, label=label, alpha=0.8, color=colors[i])
            
            plt.xlabel('Test Dataset')
            plt.ylabel(metric_label)
            plt.title(f'{metric_label} Comparison: Single-task vs Multi-task by Backbone')
            plt.xticks(x, pivot_data.index, rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(output_dir, f'task_comparison_{metric}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {metric_label} comparison plot to: {save_path}")
    
    # Create a combined plot showing all four combinations (ResNet vs VFM, Single vs Multi)
    if len(backbones) >= 2 and 'sensitivity_at_95_spec' in df_original.columns:
        create_combined_backbone_comparison(df_original, output_dir)

def create_combined_backbone_comparison(df, output_dir):
    """
    Create a focused comparison plot showing ResNet vs VFM for both single-task and multi-task.
    
    Args:
        df: DataFrame with results (non-calibrated only)
        output_dir: Directory to save plots
    """
    # Focus on the key metric: sensitivity at 95% specificity
    metric = 'sensitivity_at_95_spec'
    
    # Filter for data that has this metric
    df_with_metric = df[df[metric].notna()].copy()
    
    if df_with_metric.empty:
        print("No sensitivity@95%spec data available for combined comparison")
        return
    
    # Create pivot table with backbone and task_mode
    pivot_data = df_with_metric.pivot_table(
        index='test_dataset',
        columns=['backbone', 'task_mode'],
        values=metric,
        aggfunc='mean'
    )
    
    if pivot_data.empty:
        print("No pivot data available for combined comparison")
        return
    
    plt.figure(figsize=(16, 10))
    
    x = np.arange(len(pivot_data.index))
    width = 0.2
    
    # Define colors for each combination
    colors = {
        ('resnet18', 'singletask'): '#1f77b4',    # Blue
        ('resnet18', 'multitask'): '#aec7e8',     # Light blue
        ('vfm', 'singletask'): '#ff7f0e',         # Orange  
        ('vfm', 'multitask'): '#ffbb78'           # Light orange
    }
    
    labels = {
        ('resnet18', 'singletask'): 'ResNet18 Single-task',
        ('resnet18', 'multitask'): 'ResNet18 Multi-task',
        ('vfm', 'singletask'): 'VFM Single-task',
        ('vfm', 'multitask'): 'VFM Multi-task'
    }
    
    # Plot bars for each combination
    offset = -1.5
    for backbone in ['resnet18', 'vfm']:
        for task_mode in ['singletask', 'multitask']:
            if (backbone, task_mode) in pivot_data.columns:
                values = pivot_data[(backbone, task_mode)]
                color = colors.get((backbone, task_mode), 'gray')
                label = labels.get((backbone, task_mode), f'{backbone} {task_mode}')
                
                plt.bar(x + offset * width, values, width, 
                       label=label, alpha=0.8, color=color)
                offset += 1
    
    plt.xlabel('Test Dataset')
    plt.ylabel('Sensitivity at 95% Specificity')
    plt.title('Sensitivity@95%Spec: ResNet18 vs VFM Comparison\n(Single-task vs Multi-task)')
    plt.xticks(x, pivot_data.index, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(output_dir, 'combined_backbone_task_comparison_sens95spec.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined backbone comparison plot to: {save_path}")
    
    # Also create a summary table
    summary_table = pivot_data.round(4)
    summary_path = os.path.join(output_dir, 'backbone_task_comparison_table.csv')
    summary_table.to_csv(summary_path)
    print(f"Saved comparison table to: {summary_path}")

def create_calibration_plots(df, output_dir):
    """
    Create plots comparing calibration metrics before and after temperature scaling,
    including backbone information.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plots
    """
    # Filter for datasets that have both original and calibrated results
    df_with_calibration = df.copy()
    
    if df_with_calibration.empty:
        print("No calibration results found")
        return
    
    # Create calibration label that includes backbone information
    df_with_calibration['calibration_label'] = df_with_calibration.apply(
        lambda row: f"{row['backbone']}_{row['task_mode']}_{'calibrated' if row['calibrated'] else 'original'}", 
        axis=1
    )
    
    # Metrics to plot
    calibration_metrics = ['ece', 'brier_score']
    metric_labels = ['Expected Calibration Error (ECE)', 'Brier Score']
    
    for metric, metric_label in zip(calibration_metrics, metric_labels):
        if metric not in df_with_calibration.columns:
            print(f"Skipping {metric} - column not found")
            continue
            
        # Filter data for this metric - keep rows where metric has valid data
        metric_data = df_with_calibration[df_with_calibration[metric].notna()].copy()
        
        if metric_data.empty:
            print(f"Skipping {metric} - no valid data available")
            continue
            
        print(f"Plotting {metric} with {len(metric_data)} valid data points")
        
        plt.figure(figsize=(18, 12))
        
        # Create a pivot table for easier plotting
        pivot_data = metric_data.pivot_table(
            index='test_dataset', 
            columns='calibration_label', 
            values=metric, 
            aggfunc='mean'
        )
        
        # Define the order of columns we want (including both backbones)
        desired_columns = [
            'resnet18_singletask_original', 'resnet18_singletask_calibrated',
            'resnet18_multitask_original', 'resnet18_multitask_calibrated',
            'vfm_singletask_original', 'vfm_singletask_calibrated',
            'vfm_multitask_original', 'vfm_multitask_calibrated'
        ]
        
        # Filter to only include columns that exist in our data
        available_columns = [col for col in desired_columns if col in pivot_data.columns]
        if available_columns:
            pivot_data = pivot_data[available_columns]
        else:
            print(f"No matching columns found for {metric}")
            continue
        
        # Create grouped bar plot
        x = np.arange(len(pivot_data.index))
        width = 0.1
        n_groups = len(available_columns)
        
        # Define colors for each combination
        colors = {
            'resnet18_singletask_original': '#1f77b4',      # Blue
            'resnet18_singletask_calibrated': '#aec7e8',    # Light blue
            'resnet18_multitask_original': '#ff7f0e',       # Orange
            'resnet18_multitask_calibrated': '#ffbb78',     # Light orange
            'vfm_singletask_original': '#2ca02c',           # Green
            'vfm_singletask_calibrated': '#98df8a',         # Light green
            'vfm_multitask_original': '#d62728',            # Red
            'vfm_multitask_calibrated': '#ff9896'           # Light red
        }
        
        labels = {
            'resnet18_singletask_original': 'ResNet18 Single-task Original',
            'resnet18_singletask_calibrated': 'ResNet18 Single-task Calibrated',
            'resnet18_multitask_original': 'ResNet18 Multi-task Original',
            'resnet18_multitask_calibrated': 'ResNet18 Multi-task Calibrated',
            'vfm_singletask_original': 'VFM Single-task Original',
            'vfm_singletask_calibrated': 'VFM Single-task Calibrated',
            'vfm_multitask_original': 'VFM Multi-task Original',
            'vfm_multitask_calibrated': 'VFM Multi-task Calibrated'
        }
        
        for i, col in enumerate(available_columns):
            if col in pivot_data.columns:
                offset = (i - (n_groups-1)/2) * width
                color = colors.get(col, f'C{i}')
                label = labels.get(col, col.replace('_', ' ').title())
                plt.bar(x + offset, pivot_data[col], width, label=label, alpha=0.8, color=color)
        
        plt.xlabel('Test Dataset')
        plt.ylabel(metric_label)
        plt.title(f'{metric_label}: Original vs Calibrated Models (ResNet18 vs VFM)')
        plt.xticks(x, pivot_data.index, rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(output_dir, f'calibration_comparison_{metric}_with_backbones.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {metric_label} calibration plot with backbones to: {save_path}")
        
        # Also create a separate plot for each backbone for cleaner visualization
        create_backbone_specific_calibration_plots(metric_data, metric, metric_label, output_dir)

def create_backbone_specific_calibration_plots(metric_data, metric, metric_label, output_dir):
    """
    Create separate calibration plots for each backbone for cleaner visualization.
    
    Args:
        metric_data: DataFrame with calibration metric data
        metric: Name of the metric (e.g., 'ece', 'brier_score')
        metric_label: Display label for the metric
        output_dir: Directory to save plots
    """
    backbones = metric_data['backbone'].unique()
    
    for backbone in backbones:
        backbone_data = metric_data[metric_data['backbone'] == backbone].copy()
        
        if backbone_data.empty:
            continue
            
        # Create calibration label for this backbone
        backbone_data['calibration_label'] = backbone_data.apply(
            lambda row: f"{row['task_mode']}_{'calibrated' if row['calibrated'] else 'original'}", 
            axis=1
        )
        
        plt.figure(figsize=(14, 8))
        
        # Create pivot table
        pivot_data = backbone_data.pivot_table(
            index='test_dataset',
            columns='calibration_label',
            values=metric,
            aggfunc='mean'
        )
        
        if pivot_data.empty:
            continue
        
        # Define column order
        desired_columns = [
            'singletask_original', 'singletask_calibrated',
            'multitask_original', 'multitask_calibrated'
        ]
        
        available_columns = [col for col in desired_columns if col in pivot_data.columns]
        if not available_columns:
            continue
            
        pivot_data = pivot_data[available_columns]
        
        # Create grouped bar plot
        x = np.arange(len(pivot_data.index))
        width = 0.2
        n_groups = len(available_columns)
        
        colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']  # Blue, light blue, orange, light orange
        labels = ['Single-task Original', 'Single-task Calibrated', 'Multi-task Original', 'Multi-task Calibrated']
        
        for i, col in enumerate(available_columns):
            if col in pivot_data.columns:
                offset = (i - (n_groups-1)/2) * width
                color = colors[i] if i < len(colors) else f'C{i}'
                label = labels[i] if i < len(labels) else col.replace('_', ' ').title()
                plt.bar(x + offset, pivot_data[col], width, label=label, alpha=0.8, color=color)
        
        plt.xlabel('Test Dataset')
        plt.ylabel(metric_label)
        plt.title(f'{metric_label}: {backbone.upper()} - Original vs Calibrated Models')
        plt.xticks(x, pivot_data.index, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(output_dir, f'calibration_comparison_{metric}_{backbone}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {metric_label} calibration plot for {backbone.upper()} to: {save_path}")

def create_summary_table(df, output_dir):
    """
    Create summary tables with aggregated statistics.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save tables
    """
    # Overall summary
    summary_stats = []
    
    for task_mode in ['singletask', 'multitask']:
        for calibrated in [False, True]:
            subset = df[(df['task_mode'] == task_mode) & (df['calibrated'] == calibrated)]
            
            if subset.empty:
                continue
                
            calibration_label = 'Calibrated' if calibrated else 'Original'
            
            stats = {
                'Configuration': f"{task_mode.title()} {calibration_label}",
                'N_datasets': len(subset['test_dataset'].unique()),
                'AUC_mean': subset['auc'].mean(),
                'AUC_std': subset['auc'].std(),
                'Sensitivity_mean': subset['sensitivity'].mean(),
                'Sensitivity_std': subset['sensitivity'].std(),
                'Specificity_mean': subset['specificity'].mean(),
                'Specificity_std': subset['specificity'].std(),
                'Sens@95Spec_mean': subset['sensitivity_at_95_spec'].mean(),
                'Sens@95Spec_std': subset['sensitivity_at_95_spec'].std(),
                'ECE_mean': subset['ece'].mean(),
                'ECE_std': subset['ece'].std(),
                'Brier_mean': subset['brier_score'].mean(),
                'Brier_std': subset['brier_score'].std()
            }
            summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Round numerical columns
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    
    # Save summary table
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary statistics to: {summary_path}")
    
    # Detailed results table
    detailed_path = os.path.join(output_dir, 'detailed_results.csv')
    df.to_csv(detailed_path, index=False)
    print(f"Saved detailed results to: {detailed_path}")
    
    return summary_df

def analyze_calibration_effectiveness(df, output_dir):
    """
    Analyze when temperature scaling is effective vs ineffective.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save analysis
    """
    # Filter for calibrated results that have both original and calibrated versions
    calibrated_data = []
    
    for _, group in df.groupby(['test_dataset', 'backbone', 'task_mode']):
        original = group[group['calibrated'] == False]
        calibrated = group[group['calibrated'] == True]
        
        if len(original) == 1 and len(calibrated) == 1:
            orig_row = original.iloc[0]
            calib_row = calibrated.iloc[0]
            
            # Calculate improvements
            ece_improvement = orig_row['ece'] - calib_row['ece']  # Positive = improvement
            brier_improvement = orig_row['brier_score'] - calib_row['brier_score']  # Positive = improvement
            
            calibrated_data.append({
                'test_dataset': orig_row['test_dataset'],
                'backbone': orig_row['backbone'],
                'task_mode': orig_row['task_mode'],
                'original_ece': orig_row['ece'],
                'calibrated_ece': calib_row['ece'],
                'ece_improvement': ece_improvement,
                'ece_improved': ece_improvement > 0.005,  # Threshold for meaningful improvement
                'original_brier': orig_row['brier_score'],
                'calibrated_brier': calib_row['brier_score'],
                'brier_improvement': brier_improvement,
                'brier_improved': brier_improvement > 0.005,
                'original_auc': orig_row['auc'],
                'original_sensitivity': orig_row['sensitivity'],
                'original_specificity': orig_row['specificity'],
                'well_calibrated_originally': orig_row['ece'] < 0.05  # Already well calibrated
            })
    
    calib_df = pd.DataFrame(calibrated_data)
    
    if calib_df.empty:
        print("No calibration data available for analysis")
        return
    
    # Save detailed calibration analysis
    calib_analysis_path = os.path.join(output_dir, 'calibration_effectiveness_analysis.csv')
    calib_df.to_csv(calib_analysis_path, index=False)
    print(f"Saved calibration effectiveness analysis to: {calib_analysis_path}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("CALIBRATION EFFECTIVENESS ANALYSIS")
    print("="*60)
    
    total_models = len(calib_df)
    ece_improved_count = calib_df['ece_improved'].sum()
    brier_improved_count = calib_df['brier_improved'].sum()
    well_calibrated_count = calib_df['well_calibrated_originally'].sum()
    
    print(f"Total models analyzed: {total_models}")
    print(f"ECE improved: {ece_improved_count}/{total_models} ({100*ece_improved_count/total_models:.1f}%)")
    print(f"Brier improved: {brier_improved_count}/{total_models} ({100*brier_improved_count/total_models:.1f}%)")
    print(f"Already well-calibrated (ECE < 0.05): {well_calibrated_count}/{total_models} ({100*well_calibrated_count/total_models:.1f}%)")
    
    # Analyze by test dataset (proxy for distribution shift)
    print(f"\nCalibration effectiveness by test dataset:")
    dataset_analysis = calib_df.groupby('test_dataset').agg({
        'ece_improvement': ['mean', 'std', 'count'],
        'ece_improved': 'sum',
        'original_ece': 'mean',
        'well_calibrated_originally': 'sum'
    }).round(4)
    
    # Flatten column names
    dataset_analysis.columns = ['_'.join(col).strip() for col in dataset_analysis.columns]
    dataset_analysis['ece_improved_pct'] = (dataset_analysis['ece_improved_sum'] / dataset_analysis['ece_improvement_count'] * 100).round(1)
    dataset_analysis['well_calibrated_pct'] = (dataset_analysis['well_calibrated_originally_sum'] / dataset_analysis['ece_improvement_count'] * 100).round(1)
    
    print(dataset_analysis[['ece_improvement_mean', 'ece_improved_pct', 'original_ece_mean', 'well_calibrated_pct']])
    
    # Correlation analysis
    if len(calib_df) > 3:
        print(f"\nCorrelation analysis:")
        print(f"Original ECE vs ECE improvement: {calib_df['original_ece'].corr(calib_df['ece_improvement']):.3f}")
        print(f"Original AUC vs ECE improvement: {calib_df['original_auc'].corr(calib_df['ece_improvement']):.3f}")
        
        # Hypothesis: Models with poor original calibration benefit more from temperature scaling
        poorly_calibrated = calib_df[calib_df['original_ece'] > 0.1]
        well_calibrated = calib_df[calib_df['original_ece'] <= 0.05]
        
        if len(poorly_calibrated) > 0 and len(well_calibrated) > 0:
            print(f"\nPoorly calibrated models (ECE > 0.1): {len(poorly_calibrated)}")
            print(f"  Average ECE improvement: {poorly_calibrated['ece_improvement'].mean():.4f}")
            print(f"  Success rate: {poorly_calibrated['ece_improved'].mean()*100:.1f}%")
            
            print(f"\nWell calibrated models (ECE ≤ 0.05): {len(well_calibrated)}")
            print(f"  Average ECE improvement: {well_calibrated['ece_improvement'].mean():.4f}")
            print(f"  Success rate: {well_calibrated['ece_improved'].mean()*100:.1f}%")
    
    return calib_df

def create_calibration_effectiveness_plots(calib_df, output_dir):
    """
    Create plots analyzing calibration effectiveness.
    
    Args:
        calib_df: DataFrame with calibration analysis
        output_dir: Directory to save plots
    """
    if calib_df.empty:
        return
    
    # Plot 1: ECE improvement vs original ECE
    plt.figure(figsize=(12, 8))
    
    # Color by whether calibration helped
    colors = ['red' if not improved else 'green' for improved in calib_df['ece_improved']]
    
    plt.scatter(calib_df['original_ece'], calib_df['ece_improvement'], 
               c=colors, alpha=0.7, s=100)
    
    # Add dataset labels
    for _, row in calib_df.iterrows():
        plt.annotate(row['test_dataset'], 
                    (row['original_ece'], row['ece_improvement']),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=0.05, color='blue', linestyle='--', alpha=0.5, label='Well-calibrated threshold')
    
    plt.xlabel('Original ECE')
    plt.ylabel('ECE Improvement (Original - Calibrated)')
    plt.title('Temperature Scaling Effectiveness vs Original Calibration')
    plt.legend(['No improvement', 'Improvement', 'Zero improvement', 'Well-calibrated threshold'])
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'calibration_effectiveness_scatter.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration effectiveness scatter plot to: {save_path}")
    
    # Plot 2: Success rate by test dataset
    plt.figure(figsize=(14, 8))
    
    dataset_success = calib_df.groupby('test_dataset').agg({
        'ece_improved': ['sum', 'count'],
        'original_ece': 'mean'
    })
    
    dataset_success.columns = ['ece_improved_sum', 'total_count', 'avg_original_ece']
    dataset_success['success_rate'] = dataset_success['ece_improved_sum'] / dataset_success['total_count'] * 100
    dataset_success = dataset_success.sort_values('success_rate')
    
    bars = plt.bar(range(len(dataset_success)), dataset_success['success_rate'], 
                   alpha=0.7, color='skyblue')
    
    # Color bars by average original ECE
    for i, (idx, row) in enumerate(dataset_success.iterrows()):
        if row['avg_original_ece'] < 0.05:
            bars[i].set_color('lightgreen')  # Already well calibrated
        elif row['avg_original_ece'] > 0.15:
            bars[i].set_color('lightcoral')  # Poorly calibrated
    
    plt.xlabel('Test Dataset')
    plt.ylabel('Calibration Success Rate (%)')
    plt.title('Temperature Scaling Success Rate by Test Dataset\n(Green: Well-calibrated, Red: Poorly-calibrated, Blue: Moderate)')
    plt.xticks(range(len(dataset_success)), dataset_success.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'calibration_success_by_dataset.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration success by dataset plot to: {save_path}")

def main():
    """Main function to aggregate and visualize results."""
    
    # Set up directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, f'aggregated_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Aggregating results from: {base_dir}")
    print(f"Saving outputs to: {output_dir}")
    
    # Find all result files
    result_files = find_result_files(base_dir)
    
    # Load comparison results
    comparison_df = pd.DataFrame()
    if result_files['comparison']:
        print("\nLoading comparison results...")
        comparison_df = load_comparison_results(result_files['comparison'])
        print(f"Loaded {len(comparison_df)} comparison records")
    
    # Load calibrated results
    calibrated_df = pd.DataFrame()
    if result_files['calibrated']:
        print("\nLoading calibrated results...")
        calibrated_df = load_calibrated_results(result_files['calibrated'])
        print(f"Loaded {len(calibrated_df)} calibrated records")
    
    # Load individual results
    individual_df = pd.DataFrame()
    if result_files['individual']:
        print("\nLoading individual results...")
        individual_df = load_individual_results(result_files['individual'])
        print(f"Loaded {len(individual_df)} individual records")
    
    # Load prediction results for sensitivity at 95% specificity
    prediction_df = pd.DataFrame()
    if result_files['predictions']:
        print("\nLoading prediction results...")
        prediction_df = load_prediction_results(result_files['predictions'])
        print(f"Loaded {len(prediction_df)} prediction records")
    
    # Combine all results
    all_results = []
    if not comparison_df.empty:
        all_results.append(comparison_df)
    if not calibrated_df.empty:
        all_results.append(calibrated_df)
    if not individual_df.empty:
        all_results.append(individual_df)
    if not prediction_df.empty:
        all_results.append(prediction_df)
    
    if not all_results:
        print("No results found to aggregate!")
        return
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Remove duplicates (prefer predictions results over others for sensitivity@95%spec data)
    # First, prioritize by data source: predictions > calibrated > comparison > individual
    data_source_priority = {'predictions': 4, 'calibrated': 3, 'comparison': 2, 'individual': 1}
    combined_df['priority'] = combined_df['data_source'].map(data_source_priority)
    
    # Sort by priority (higher is better) and keep the last (highest priority) entry
    combined_df = combined_df.sort_values(['test_dataset', 'backbone', 'task_mode', 'calibrated', 'priority'])
    combined_df = combined_df.drop_duplicates(
        subset=['test_dataset', 'backbone', 'task_mode', 'calibrated'], 
        keep='last'
    )
    
    # Drop the priority column
    combined_df = combined_df.drop('priority', axis=1)
    
    print(f"\nCombined dataset has {len(combined_df)} records")
    print(f"Test datasets: {sorted(combined_df['test_dataset'].unique())}")
    print(f"Backbones: {sorted(combined_df['backbone'].unique())}")
    print(f"Task modes: {sorted(combined_df['task_mode'].unique())}")
    
    # Debug: Show dataset distribution
    print(f"\nDataset distribution:")
    dataset_counts = combined_df.groupby(['test_dataset', 'backbone', 'task_mode', 'calibrated']).size().reset_index(name='count')
    for _, row in dataset_counts.iterrows():
        calib_status = "calibrated" if row['calibrated'] else "original"
        print(f"  {row['test_dataset']} - {row['backbone']} - {row['task_mode']} - {calib_status}: {row['count']} records")
    
    # Show availability of sensitivity@95%spec data
    sens95_data = combined_df[combined_df['sensitivity_at_95_spec'].notna()]
    if not sens95_data.empty:
        print(f"\nSensitivity@95%Spec data available for:")
        sens95_counts = sens95_data.groupby(['test_dataset', 'backbone', 'task_mode']).size().reset_index(name='count')
        for _, row in sens95_counts.iterrows():
            print(f"  {row['test_dataset']} - {row['backbone']} - {row['task_mode']}: {row['count']} records")
    else:
        print("\nNo sensitivity@95%spec data available")
    
    # Create visualizations
    print("\nCreating task comparison plots...")
    create_task_comparison_plots(combined_df, output_dir)
    
    print("\nCreating calibration comparison plots...")
    create_calibration_plots(combined_df, output_dir)
    
    print("\nAnalyzing calibration effectiveness...")
    calib_analysis_df = analyze_calibration_effectiveness(combined_df, output_dir)
    
    if calib_analysis_df is not None and not calib_analysis_df.empty:
        print("\nCreating calibration effectiveness plots...")
        create_calibration_effectiveness_plots(calib_analysis_df, output_dir)
    
    print("\nCreating summary tables...")
    summary_df = create_summary_table(combined_df, output_dir)
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print(f"📊 Created plots for task comparison and calibration analysis")
    print(f"📋 Summary statistics and detailed results saved as CSV files")

if __name__ == "__main__":
    main()
