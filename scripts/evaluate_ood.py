#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
External Test Set Evaluation Script for Glaucoma Classification Models.

This script evaluates pre-trained model(s) from a parent experiment directory
on specified external OOD (Out-of-Distribution) test datasets.
It performs the following main steps:
1.  Discovers trained model checkpoints based on a configuration list.
2.  Loads necessary external test datasets (e.g., PAPILLA, CHAKSU).
3.  For each discovered model:
    a.  Builds the model architecture using parameters from its original training run,
        correctly identifying if a sequential head structure is needed.
    b.  Loads the trained weights from the checkpoint.
    c.  Evaluates the model on each external test set using the `run_evaluation` function.
    d.  Optionally extracts features for t-SNE visualization.
4.  Aggregates results across all models and datasets to generate:
    a.  Combined ROC curves per dataset.
    b.  Combined calibration curves per dataset.
    c.  Combined underdiagnosis disparity plots per dataset.
5.  Saves all configurations, logs, and results to a structured output directory.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
import glob
import re # For regex to parse epoch numbers from filenames

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve as sk_roc_curve, auc as sk_auc

# Ensure custom modules can be imported by adding the parent directory of 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.utils.helpers import NpEncoder, set_seed
    from src.models.classification.build_model import build_classifier_model # Ensure this is the updated version
    from src.evaluation.evaluator import run_evaluation
    from src.data.utils import (
        get_eval_transforms,
        create_external_test_dataloader,
        adjust_path_for_data_type
    )
    from src.data.external_loader import load_external_test_data
    from src.evaluation.bias_analysis import extract_features_for_tsne
    from src.evaluation.fairness import calculate_underdiagnosis_disparities
    from src.evaluation.plotting import (
        plot_tsne_by_attribute,
        plot_all_roc_curves,
        plot_all_disparities,
        plot_all_calibration_curves
    )
except ImportError as e:
    print(f"Error importing custom modules: {e}\nEnsure 'src' directory is in PYTHONPATH or accessible via sys.path.")
    print("Current sys.path:", sys.path)
    sys.exit(1)

# --- Global Constants ---
RAW_DIR_NAME_CONST: str = "raw"
PROCESSED_DIR_NAME_CONST: str = "processed"

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s"
)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

def discover_models_from_experiment_dir(experiment_parent_dir: str, data_type_suffix: str = "raw") -> list[dict]:
    """
    Automatically discovers trained models from the experiment parent directory.
    
    Args:
        experiment_parent_dir: Path to the parent directory containing experiment folders
        data_type_suffix: Data type suffix used in directory names (e.g., "raw", "processed")
    
    Returns:
        List of dictionaries containing model information for each discovered model
    """
    logger.info(f"Auto-discovering models from: {experiment_parent_dir}")
    
    # Model architecture mappings
    ARCH_MAPPINGS = {
        'vit': 'vit_base_patch16_224',
        'dinov2': 'dinov2_vitb14', 
        'd2': 'dinov2_vitb14',
        'resnet50': 'resnet50',
        'convnext': 'convnext_base',
        'efficientnet': 'efficientnet_b0'  # Add more as needed
    }
    
    if not os.path.isdir(experiment_parent_dir):
        logger.warning(f"Experiment parent directory not found: {experiment_parent_dir}")
        return []
    
    discovered_models = []
    
    # Pattern to match experiment directories
    # Expected format: {model_short}_{data_type}_{training_tag}_{timestamp}
    pattern = f"*_{data_type_suffix}_*"
    experiment_dirs = glob.glob(os.path.join(experiment_parent_dir, pattern))
    
    for exp_dir in experiment_dirs:
        dir_name = os.path.basename(exp_dir)
        logger.info(f"Processing experiment directory: {dir_name}")
        
        # Check if checkpoints directory exists
        checkpoint_dir = os.path.join(exp_dir, "checkpoints")
        if not os.path.isdir(checkpoint_dir):
            logger.warning(f"No checkpoints directory found in: {dir_name}")
            continue
            
        # Check if any .pth files exist
        pth_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
        if not pth_files:
            logger.warning(f"No .pth checkpoint files found in: {checkpoint_dir}")
            continue
        
        # Parse directory name to extract model info
        # Expected format: {model_short}_{data_type}_{training_components}_{timestamp}
        parts = dir_name.split('_')
        if len(parts) < 3:
            logger.warning(f"Directory name format not recognized: {dir_name}")
            continue
            
        model_short = parts[0].lower()
        
        # Map short name to full architecture name
        model_arch_name = None
        for short_key, full_arch in ARCH_MAPPINGS.items():
            if model_short.startswith(short_key):
                model_arch_name = full_arch
                break
        
        if not model_arch_name:
            logger.warning(f"Could not map model short name '{model_short}' to architecture")
            continue
        
        # Extract training tag (everything between data_type and timestamp)
        # Remove the model_short and data_type_suffix from the beginning
        remaining_parts = parts[2:]  # Skip model_short and data_type_suffix
        
        # The last part is typically a timestamp, so exclude it for training tag
        if len(remaining_parts) > 1 and re.match(r'\d{8}_\d{6}', remaining_parts[-1]):
            training_tag_parts = remaining_parts[:-1]
        else:
            training_tag_parts = remaining_parts
            
        training_tag = '_'.join(training_tag_parts) if training_tag_parts else "Unknown"
        
        # Extract dropout from directory name if present
        dropout_prob = 0.3  # default
        dropout_match = re.search(r'Dropout([\d.]+)', dir_name, re.IGNORECASE)
        if dropout_match:
            dropout_prob = float(dropout_match.group(1))
        
        # Create model configuration
        model_config = {
            "ModelNameInTraining": model_arch_name,
            "TrainingModelShortName": model_short,
            "TrainingTag": training_tag,
            "TrainingDropout": dropout_prob,
            "experiment_dir": exp_dir,
            "checkpoint_dir": checkpoint_dir
        }
        
        discovered_models.append(model_config)
        logger.info(f"Discovered model: {model_short} -> {model_arch_name}, Tag: {training_tag}, Dropout: {dropout_prob}")
    
    logger.info(f"Total models discovered: {len(discovered_models)}")
    return discovered_models

def find_model_checkpoint_path(base_dir: str, dir_pattern: str, model_id_for_log: str) -> tuple[str | None, str | None]:
    """
    Finds the best or latest checkpoint file within a model's experiment directory.

    Args:
        base_dir: The parent directory where model experiment folders are located.
        dir_pattern: Glob pattern to find the specific model's experiment directory.
        model_id_for_log: Identifier for logging purposes.

    Returns:
        A tuple (path_to_checkpoint_file, path_to_experiment_directory), or (None, None) if not found.
    """
    logger.info(f"Searching for training run directory with pattern: '{dir_pattern}' under '{base_dir}' for model '{model_id_for_log}'")
    found_model_dirs = glob.glob(os.path.join(base_dir, dir_pattern))

    if not found_model_dirs:
        logger.warning(f"Experiment subdirectory not found for pattern '{dir_pattern}' in '{base_dir}'. Skipping model {model_id_for_log}.")
        return None, None

    model_actual_experiment_dir = sorted(found_model_dirs, reverse=True)[0] # Get most recent if multiple
    logger.info(f"Found training run directory for '{model_id_for_log}': {model_actual_experiment_dir}")

    checkpoint_dir = os.path.join(model_actual_experiment_dir, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        logger.warning(f"Checkpoints directory not found: {checkpoint_dir} for model {model_id_for_log}. Skipping.")
        return None, None

    checkpoint_file = None
    best_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*_best_model_epoch*.pth"))
    best_checkpoints_sorted = sorted(
        best_checkpoints,
        key=lambda x: int(m.group(1)) if (m := re.search(r'_epoch(\d+)\.pth$', os.path.basename(x))) else -1,
        reverse=True
    )
    if best_checkpoints_sorted:
        checkpoint_file = best_checkpoints_sorted[0]
        logger.info(f"Selected 'best_model' checkpoint (highest epoch): {os.path.basename(checkpoint_file)}")
    else:
        logger.warning(f"No '*_best_model_epoch*.pth' checkpoint found for model {model_id_for_log} in {checkpoint_dir}.")
        final_epoch_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*_final_epoch*.pth"))
        final_epoch_checkpoints_sorted = sorted(
            final_epoch_checkpoints,
            key=lambda x: int(m.group(1)) if (m := re.search(r'_epoch(\d+)\.pth$', os.path.basename(x))) else -1,
            reverse=True
        )
        if final_epoch_checkpoints_sorted:
            checkpoint_file = final_epoch_checkpoints_sorted[0]
            logger.info(f"Falling back to 'final_epoch' checkpoint (highest epoch): {os.path.basename(checkpoint_file)}")
        else:
            logger.warning(f"No '*_final_epoch*.pth' checkpoint found for model {model_id_for_log} in {checkpoint_dir}.")
            all_pth_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
            if all_pth_files:
                checkpoint_file = max(all_pth_files, key=os.path.getmtime)
                logger.info(f"Falling back to latest modified checkpoint: {os.path.basename(checkpoint_file)}")

    if checkpoint_file:
        return checkpoint_file, model_actual_experiment_dir
    else:
        logger.warning(f"No suitable checkpoint file ultimately found for {model_id_for_log} in '{checkpoint_dir}'.")
        return None, None

def process_and_evaluate_ood_ensemble(
    all_model_predictions_per_dataset: dict[str, list[pd.DataFrame]],
    model_display_names: list[str],
    output_dir: str,
    timestamp_str: str
) -> dict[str, dict]:
    """
    Creates and evaluates ensemble predictions for each OOD dataset.
    
    Args:
        all_model_predictions_per_dataset: Dict mapping dataset_name -> list of prediction DataFrames
        model_display_names: List of model names corresponding to the DataFrames
        output_dir: Base output directory
        timestamp_str: Timestamp for file naming
        
    Returns:
        Dict mapping dataset_name -> ensemble_roc_data
    """
    if len(model_display_names) < 2:
        logger.warning("Ensemble evaluation requires at least 2 models. Skipping ensemble.")
        return {}
    
    logger.info(f"\n{'='*10} Processing OOD Ensemble Predictions ({len(model_display_names)} models) {'='*10}")
    
    ensemble_roc_data_per_dataset = {}
    
    for dataset_name, prediction_dfs in all_model_predictions_per_dataset.items():
        if len(prediction_dfs) < 2:
            logger.warning(f"Skipping ensemble for {dataset_name} - insufficient models ({len(prediction_dfs)})")
            continue
            
        logger.info(f"\n--- Creating Ensemble for Dataset: {dataset_name} ---")
        
        # Prepare DataFrames for merging
        prepared_dfs = []
        for i, df_model in enumerate(prediction_dfs):
            model_name = model_display_names[i]
            df_renamed = df_model[['image_path', 'label', 'probability']].copy()
            df_renamed.rename(columns={'probability': f'prob_{model_name}'}, inplace=True)
            df_renamed.set_index('image_path', inplace=True)
            prepared_dfs.append(df_renamed)
        
        # Merge all DataFrames
        from functools import reduce
        ensemble_df = reduce(
            lambda left, right: pd.merge(
                left, right.drop(columns=['label'], errors='ignore'),
                left_index=True, right_index=True, how='outer'
            ),
            prepared_dfs
        )
        
        # Calculate ensemble probability (average)
        prob_cols = [col for col in ensemble_df.columns if col.startswith('prob_')]
        ensemble_df['num_models_contributed'] = ensemble_df[prob_cols].notna().sum(axis=1)
        ensemble_df['ensemble_probability'] = ensemble_df[prob_cols].mean(axis=1, skipna=True)
        
        # Clean and filter
        ensemble_df.dropna(subset=['ensemble_probability', 'label'], inplace=True)
        ensemble_df.reset_index(inplace=True)
        
        if ensemble_df.empty:
            logger.warning(f"Ensemble DataFrame empty for {dataset_name}. Skipping.")
            continue
        
        # Save ensemble results
        ensemble_dataset_dir = os.path.join(output_dir, f"ensemble_results_{dataset_name}_{timestamp_str}")
        os.makedirs(ensemble_dataset_dir, exist_ok=True)
        
        ensemble_csv_path = os.path.join(ensemble_dataset_dir, f"ensemble_predictions_{dataset_name}.csv")
        ensemble_df.to_csv(ensemble_csv_path, index=False)
        logger.info(f"Ensemble predictions for {dataset_name} saved to: {ensemble_csv_path}")
        
        # Calculate ROC for ensemble
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(ensemble_df['label'], ensemble_df['ensemble_probability'])
        roc_auc = auc(fpr, tpr)
        
        ensemble_roc_data_per_dataset[dataset_name] = {
            'fpr': fpr, 'tpr': tpr, 'auc': roc_auc
        }
        
        logger.info(f"Ensemble ROC for {dataset_name}: AUC = {roc_auc:.4f}")
    
    return ensemble_roc_data_per_dataset


def create_ood_summary_tables(
    all_roc_data_per_dataset: dict[str, dict],
    ensemble_roc_data_per_dataset: dict[str, dict],
    output_dir: str,
    timestamp_str: str
) -> None:
    """
    Creates comprehensive summary tables for OOD evaluation results.
    
    Args:
        all_roc_data_per_dataset: Dict mapping dataset -> {model_name: roc_data}
        ensemble_roc_data_per_dataset: Dict mapping dataset -> ensemble_roc_data
        output_dir: Output directory for tables
        timestamp_str: Timestamp for file naming
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    logger.info("Creating comprehensive OOD summary tables...")
    
    # Collect all unique models and datasets
    all_models = set()
    all_datasets = set(all_roc_data_per_dataset.keys())
    
    for dataset_data in all_roc_data_per_dataset.values():
        all_models.update(dataset_data.keys())
    
    all_models = sorted(list(all_models))
    all_datasets = sorted(list(all_datasets))
    
    # Create AUC summary table
    fig, ax = plt.subplots(figsize=(14, max(8, len(all_models) * 0.8 + 3)))
    ax.axis('off')
    
    # Table data preparation
    table_data = []
    header = ['Model'] + all_datasets + ['Mean AUC']
    
    for model in all_models:
        row = [model]
        model_aucs = []
        
        for dataset in all_datasets:
            if dataset in all_roc_data_per_dataset and model in all_roc_data_per_dataset[dataset]:
                auc_val = all_roc_data_per_dataset[dataset][model]['auc']
                row.append(f"{auc_val:.3f}")
                model_aucs.append(auc_val)
            else:
                row.append("N/A")
        
        # Calculate mean AUC across available datasets
        mean_auc = np.mean(model_aucs) if model_aucs else np.nan
        row.append(f"{mean_auc:.3f}" if not np.isnan(mean_auc) else "N/A")
        table_data.append(row)
    
    # Add ensemble row if available
    if ensemble_roc_data_per_dataset:
        ensemble_row = ['🔷 ENSEMBLE']
        ensemble_aucs = []
        
        for dataset in all_datasets:
            if dataset in ensemble_roc_data_per_dataset:
                auc_val = ensemble_roc_data_per_dataset[dataset]['auc']
                ensemble_row.append(f"{auc_val:.3f}")
                ensemble_aucs.append(auc_val)
            else:
                ensemble_row.append("N/A")
        
        mean_ensemble_auc = np.mean(ensemble_aucs) if ensemble_aucs else np.nan
        ensemble_row.append(f"{mean_ensemble_auc:.3f}" if not np.isnan(mean_ensemble_auc) else "N/A")
        table_data.append(ensemble_row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=header,
        cellLoc='center',
        loc='center',
        bbox=[0.02, 0.1, 0.96, 0.8]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header styling
    for i in range(len(header)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.08)
    
    # Row styling
    for i in range(1, len(table_data) + 1):
        row_color = '#F8F9FA' if i % 2 == 0 else '#FFFFFF'
        
        # Special styling for ensemble row
        if i == len(table_data) and ensemble_roc_data_per_dataset:
            row_color = '#E8F4FD'
            for j in range(len(header)):
                table[(i, j)].set_text_props(weight='bold')
        
        for j in range(len(header)):
            table[(i, j)].set_facecolor(row_color)
            table[(i, j)].set_height(0.06)
    
    # Title and subtitle
    plt.suptitle('OOD Evaluation Summary: AUC Performance Across Datasets', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.title(f'Generated: {timestamp_str} | Total Models: {len(all_models)} | Datasets: {len(all_datasets)}',
             fontsize=11, color='gray', y=0.90)
    
    # Save table
    auc_table_path = os.path.join(output_dir, f'ood_auc_summary_table_{timestamp_str}.png')
    plt.savefig(auc_table_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"OOD AUC summary table saved to: {auc_table_path}")


def plot_combined_ood_roc_with_ensemble(
    all_roc_data_per_dataset: dict[str, dict],
    ensemble_roc_data_per_dataset: dict[str, dict],
    output_dir: str,
    timestamp_str: str
) -> None:
    """
    Creates combined ROC plots for each OOD dataset with ensemble curves highlighted.
    """
    import matplotlib.pyplot as plt
    
    for dataset_name, model_roc_data in all_roc_data_per_dataset.items():
        if not model_roc_data:
            continue
            
        plt.figure(figsize=(12, 10))
        
        # Plot individual model curves
        num_models = len(model_roc_data)
        colors = plt.cm.tab10(np.linspace(0, 1, num_models))
        
        sorted_models = sorted(model_roc_data.items(), 
                             key=lambda x: x[1]['auc'] if not np.isnan(x[1]['auc']) else -1, 
                             reverse=True)
        
        for i, (model_name, roc_data) in enumerate(sorted_models):
            if roc_data['fpr'] is not None and roc_data['tpr'] is not None:
                auc_val = roc_data['auc']
                label = f'{model_name} (AUC={auc_val:.3f})' if not np.isnan(auc_val) else f'{model_name} (AUC=N/A)'
                plt.plot(roc_data['fpr'], roc_data['tpr'], 
                        color=colors[i], lw=2, alpha=0.8, label=label)
        
        # Plot ensemble curve if available
        if dataset_name in ensemble_roc_data_per_dataset:
            ensemble_data = ensemble_roc_data_per_dataset[dataset_name]
            auc_val = ensemble_data['auc']
            plt.plot(ensemble_data['fpr'], ensemble_data['tpr'], 
                    color='black', lw=4, linestyle='--', alpha=0.9,
                    label=f'🔷 ENSEMBLE (AUC={auc_val:.3f})')
        
        # Styling
        plt.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - {dataset_name} Dataset (OOD Evaluation)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        roc_plot_path = os.path.join(output_dir, f'combined_roc_{dataset_name.lower().replace(" ", "_")}_{timestamp_str}.png')
        plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Combined ROC plot for {dataset_name} saved to: {roc_plot_path}")


def main(args: argparse.Namespace):
    """
    Main function to orchestrate the OOD model evaluation process with ensemble support.
    """
    start_time = datetime.now()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    output_suffix = args.output_dir_suffix if args.output_dir_suffix else f"multi_model_ood_eval_{args.data_type}"
    eval_output_dir = os.path.join(args.experiment_parent_dir, "external_evaluations", f"{output_suffix}_{eval_timestamp}")
    os.makedirs(eval_output_dir, exist_ok=True)

    log_file_path = os.path.join(eval_output_dir, f"log_main_ood_eval_{eval_timestamp}.log")
    main_file_handler = logging.FileHandler(log_file_path, mode='w')
    main_file_handler.setFormatter(log_formatter)
    logger.addHandler(main_file_handler)

    logger.info(f"--- Multi-Model OOD Evaluation Initializing ---")
    logger.info(f"Evaluation run started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Processing models from parent directory: {args.experiment_parent_dir}")
    logger.info(f"Evaluation outputs will be saved to: {eval_output_dir}")
    logger.info(f"Using PyTorch device: {device}")
    logger.info(f"Base data root for external datasets: {os.path.abspath(args.base_data_root)}")

    # Infer the data type suffix used in directory names
    training_run_data_type_suffix = "raw"
    parent_dir_name = os.path.basename(args.experiment_parent_dir).lower()
    if "processed" in parent_dir_name or "proc" in parent_dir_name:
        # Check what suffix is actually used in the directory names
        experiment_dirs = glob.glob(os.path.join(args.experiment_parent_dir, "*"))
        experiment_dir_names = [os.path.basename(d) for d in experiment_dirs if os.path.isdir(d)]
        
        # Look for common patterns
        if any("_proc_" in name for name in experiment_dir_names):
            training_run_data_type_suffix = "proc"
        elif any("_processed_" in name for name in experiment_dir_names):
            training_run_data_type_suffix = "processed"
        else:
            training_run_data_type_suffix = "processed"  # default fallback
            
    logger.info(f"Inferred training data type suffix for directory search: '{training_run_data_type_suffix}'")

    # Auto-discover models or use provided configuration
    if args.model_config_list:
        try:
            if os.path.exists(args.model_config_list):
                with open(args.model_config_list, 'r') as f: 
                    models_to_evaluate_config = json.load(f)
                logger.info(f"Loaded model configurations from JSON file: {args.model_config_list}")
            else:
                models_to_evaluate_config = json.loads(args.model_config_list)
                logger.info("Loaded model configurations from JSON string argument.")
        except Exception as e:
            logger.critical(f"Error processing --model_config_list '{args.model_config_list}': {e}. Exiting.", exc_info=True)
            logger.removeHandler(main_file_handler); main_file_handler.close(); sys.exit(1)
    else:
        # Auto-discover models from experiment parent directory
        logger.info("No --model_config_list provided. Auto-discovering models from experiment directory...")
        models_to_evaluate_config = discover_models_from_experiment_dir(
            args.experiment_parent_dir, 
            training_run_data_type_suffix
        )
        if not models_to_evaluate_config:
            logger.critical("No models discovered and no model config provided. Exiting.")
            logger.removeHandler(main_file_handler); main_file_handler.close(); sys.exit(1)

    with open(os.path.join(eval_output_dir, "evaluation_run_config.json"), "w") as f_cfg:
        config_to_save = vars(args).copy()
        config_to_save['resolved_model_config_list'] = models_to_evaluate_config
        json.dump(config_to_save, f_cfg, indent=4, cls=NpEncoder)

    all_discovered_checkpoints_info = []
    logger.info(f"\n--- Discovering Checkpoints in Parent Directory: {args.experiment_parent_dir} ---")
    for model_config_entry in models_to_evaluate_config:
        required_keys = ["ModelNameInTraining", "TrainingModelShortName", "TrainingTag", "TrainingDropout"]
        if not all(key in model_config_entry for key in required_keys):
            logger.warning(f"Skipping malformed model config: {model_config_entry}. Missing: {', '.join(k for k in required_keys if k not in model_config_entry)}")
            continue
        
        model_id_for_log = f"{model_config_entry['TrainingModelShortName']}-{model_config_entry['TrainingTag']}"
        
        # Check if we have the experiment directory directly (from auto-discovery)
        if 'experiment_dir' in model_config_entry and 'checkpoint_dir' in model_config_entry:
            full_experiment_dir_path = model_config_entry['experiment_dir']
            checkpoint_dir = model_config_entry['checkpoint_dir']
            
            logger.info(f"Using discovered experiment directory for '{model_id_for_log}': {full_experiment_dir_path}")
            
            # Find the best checkpoint in the directory
            checkpoint_file = None
            best_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*_best_model_epoch*.pth"))
            best_checkpoints_sorted = sorted(
                best_checkpoints,
                key=lambda x: int(m.group(1)) if (m := re.search(r'_epoch(\d+)\.pth$', os.path.basename(x))) else -1,
                reverse=True
            )
            if best_checkpoints_sorted:
                checkpoint_file = best_checkpoints_sorted[0]
                logger.info(f"Selected 'best_model' checkpoint (highest epoch): {os.path.basename(checkpoint_file)}")
            else:
                logger.warning(f"No '*_best_model_epoch*.pth' checkpoint found for model {model_id_for_log} in {checkpoint_dir}.")
                final_epoch_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*_final_epoch*.pth"))
                final_epoch_checkpoints_sorted = sorted(
                    final_epoch_checkpoints,
                    key=lambda x: int(m.group(1)) if (m := re.search(r'_epoch(\d+)\.pth$', os.path.basename(x))) else -1,
                    reverse=True
                )
                if final_epoch_checkpoints_sorted:
                    checkpoint_file = final_epoch_checkpoints_sorted[0]
                    logger.info(f"Falling back to 'final_epoch' checkpoint (highest epoch): {os.path.basename(checkpoint_file)}")
                else:
                    logger.warning(f"No '*_final_epoch*.pth' checkpoint found for model {model_id_for_log} in {checkpoint_dir}.")
                    all_pth_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
                    if all_pth_files:
                        checkpoint_file = max(all_pth_files, key=os.path.getmtime)
                        logger.info(f"Falling back to latest modified checkpoint: {os.path.basename(checkpoint_file)}")
            
            if checkpoint_file:
                checkpoint_file_path = checkpoint_file
            else:
                logger.warning(f"No suitable checkpoint file found for {model_id_for_log} in '{checkpoint_dir}'. Skipping.")
                continue
        else:
            # Use the old method for manually specified configurations
            dir_search_pattern = f"{model_config_entry['TrainingModelShortName']}_{training_run_data_type_suffix}_{model_config_entry['TrainingTag']}*"
            
            checkpoint_file_path, full_experiment_dir_path = find_model_checkpoint_path(
                args.experiment_parent_dir, dir_search_pattern, model_id_for_log
            )
        
        if checkpoint_file_path and full_experiment_dir_path:
            all_discovered_checkpoints_info.append({
                "model_arch_name": model_config_entry['ModelNameInTraining'],
                "dropout_prob": model_config_entry['TrainingDropout'],
                "training_tag": model_config_entry['TrainingTag'],
                "checkpoint_path": checkpoint_file_path,
                "full_experiment_dir": full_experiment_dir_path,
                "model_display_name": f"{model_config_entry['TrainingModelShortName']}_{model_config_entry['TrainingTag']}"
            })

    if not all_discovered_checkpoints_info:
        logger.critical(f"No checkpoints discovered. Exiting.")
        logger.removeHandler(main_file_handler); main_file_handler.close(); sys.exit(1)
    logger.info(f"Successfully discovered {len(all_discovered_checkpoints_info)} checkpoint(s).")

    logger.info("\n--- Loading External Test Datasets ---")
    chaksu_base_dir_for_loader = args.chaksu_base_dir
    if args.data_type == 'processed':
        chaksu_base_dir_for_loader = adjust_path_for_data_type(
            current_path=args.chaksu_base_dir, data_type='processed',
            base_data_dir=args.base_data_root, raw_dir_name=RAW_DIR_NAME_CONST,
            processed_dir_name=PROCESSED_DIR_NAME_CONST
        )
        logger.info(f"Adjusted CHAKSU Base Dir for 'processed' OOD evaluation: {chaksu_base_dir_for_loader}")

    external_test_sets_map = load_external_test_data(
        smdg_metadata_file_raw=args.smdg_metadata_file_raw,
        smdg_image_dir_raw=args.smdg_image_dir_raw,
        chaksu_base_dir_eval=chaksu_base_dir_for_loader,
        chaksu_decision_dir_raw=args.chaksu_decision_dir_raw,
        chaksu_metadata_dir_raw=args.chaksu_metadata_dir_raw,
        data_type=args.data_type,
        base_data_root=args.base_data_root,
        raw_dir_name=RAW_DIR_NAME_CONST,
        processed_dir_name=PROCESSED_DIR_NAME_CONST,
        eval_papilla=args.eval_papilla,
        eval_chaksu=args.eval_chaksu,
        eval_acrima=args.eval_acrima,  # NEW
        eval_hygd=args.eval_hygd,      # NEW
        acrima_image_dir_raw=args.acrima_image_dir_raw,  # NEW
        hygd_image_dir_raw=args.hygd_image_dir_raw,      # NEW
        hygd_labels_file_raw=args.hygd_labels_file_raw   # NEW
    )

    if not external_test_sets_map or all(df.empty for df in external_test_sets_map.values()):
        logger.error("No external test data loaded. Exiting."); 
        logger.removeHandler(main_file_handler); main_file_handler.close(); sys.exit(1)
    logger.info(f"Loaded external test datasets: {list(external_test_sets_map.keys())}")

    all_model_dataset_summaries, all_roc_data_collected = {}, {}
    all_disparity_data_collected, all_calibration_data_collected = {}, {}
    
    # NEW: For ensemble processing
    all_model_predictions_per_dataset = {}  # dataset_name -> list of prediction DataFrames
    model_display_names_for_ensemble = []

    for ckpt_idx, model_info_dict in enumerate(all_discovered_checkpoints_info):
        model_arch_name = model_info_dict['model_arch_name']
        dropout_prob = model_info_dict['dropout_prob']
        training_tag = model_info_dict['training_tag']
        checkpoint_file_path = model_info_dict['checkpoint_path']
        model_display_name_for_plots = model_info_dict['model_display_name']

        logger.info(f"\n===== Evaluating Model {ckpt_idx + 1}/{len(all_discovered_checkpoints_info)}: {model_display_name_for_plots} =====")
        logger.info(f"Arch: {model_arch_name}, Dropout: {dropout_prob}, Checkpoint: {os.path.basename(checkpoint_file_path)}")

        is_custom_sequential_vit_needed = False
        is_cnn_with_sequential_head_needed = False
        model_arch_lower = model_arch_name.lower()

        if 'vit' in model_arch_lower:
            custom_vit_keywords = ['vfm_custom', 'vfm-custom']
            if any(keyword in training_tag.lower() for keyword in custom_vit_keywords):
                is_custom_sequential_vit_needed = True
        elif 'resnet' in model_arch_lower or 'efficientnet' in model_arch_lower:
             # Heuristic: if the training_tag specifically indicates this CNN used a sequential head.
             # Example: if training_tag was "ResNet50_timm_SequentialHead"
             # Or if ALL ResNet50_timm runs in your setup used sequential heads:
            if "resnet50_timm" == training_tag.lower() and 'resnet50' == model_arch_lower:
                 is_cnn_with_sequential_head_needed = True
            # Add more specific conditions for other CNNs if necessary.

        logger.info(f"For model '{model_display_name_for_plots}', is_custom_sequential_vit: {is_custom_sequential_vit_needed}, is_cnn_with_sequential_head: {is_cnn_with_sequential_head_needed}")

        model_instance = build_classifier_model(
            model_name=model_arch_name, 
            num_classes=args.num_classes, 
            dropout_prob=dropout_prob,
            pretrained=False, 
            custom_weights_path=checkpoint_file_path,  
            checkpoint_key='model',  
            is_custom_sequential_head_vit=is_custom_sequential_vit_needed,
            is_cnn_with_sequential_head=is_cnn_with_sequential_head_needed
        )

        try:
            checkpoint = torch.load(checkpoint_file_path, map_location=device, weights_only=args.load_weights_only)
            state_dict_keys_to_try = ['model_state_dict', 'state_dict', 'model']
            state_dict = next((checkpoint[key] for key in state_dict_keys_to_try if key in checkpoint), None)
            if state_dict is None:
                if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    state_dict = checkpoint
                else: raise ValueError("Valid state_dict not found in checkpoint.")
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            
            load_status = model_instance.load_state_dict(state_dict, strict=args.strict_load)
            logger.info(f"Model weights loaded. Missing: {load_status.missing_keys}, Unexpected: {load_status.unexpected_keys}")
            if args.strict_load and (load_status.missing_keys or load_status.unexpected_keys):
                 logger.error(f"Strict loading failed. Skipping model."); continue
            model_instance.to(device).eval()
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}", exc_info=True); continue

        if model_display_name_for_plots not in all_model_dataset_summaries:
            all_model_dataset_summaries[model_display_name_for_plots] = {}
        eval_transforms = get_eval_transforms(args.image_size, model_arch_name)

        # Store model name for ensemble (only once)
        if model_display_name_for_plots not in model_display_names_for_ensemble:
            model_display_names_for_ensemble.append(model_display_name_for_plots)

        for test_set_name, df_test_original in external_test_sets_map.items():
            if df_test_original.empty: continue
            df_test = df_test_original.copy()
            if 'label' in df_test.columns and 'types' not in df_test.columns: df_test.rename(columns={'label':'types'}, inplace=True)
            if 'types' not in df_test.columns: logger.error(f"'types' missing in {test_set_name}. Skipping."); continue

            attributes_for_dataloader = list(set(['dataset_source', 'image_path', 'types'] + \
                                            [attr for attr in args.tsne_attributes_to_visualize if attr in df_test.columns]))
            subgroup_attributes_for_run_eval = []
            if test_set_name == "PAPILLA":
                if 'age' in df_test.columns: subgroup_attributes_for_run_eval.append('age')
                if 'eye' in df_test.columns: subgroup_attributes_for_run_eval.append('eye')
            elif test_set_name == "CHAKSU":
                if 'camera' in df_test.columns: subgroup_attributes_for_run_eval.append('camera')
            elif test_set_name == "HYGD":  # NEW
                if 'quality_score' in df_test.columns: subgroup_attributes_for_run_eval.append('quality_score')
            attributes_for_dataloader.extend(subgroup_attributes_for_run_eval)
            attributes_for_dataloader = list(set(attributes_for_dataloader))

            test_loader = create_external_test_dataloader(
                df_test, args.eval_batch_size, args.num_workers, eval_transforms,
                test_set_name, attributes_for_dataloader
            )
            if not test_loader: logger.warning(f"DataLoader failed for {test_set_name}. Skipping."); continue

            logger.info(f"\n--- Evaluating '{model_display_name_for_plots}' on OOD set: '{test_set_name}' ---")
            age_bins_for_eval, age_labels_for_eval = (args.age_bins, args.age_labels) if 'age' in subgroup_attributes_for_run_eval else (None, None)
            if age_bins_for_eval is None and 'age' in subgroup_attributes_for_run_eval: # Set defaults if age is present but bins not specified
                 age_bins_for_eval = [0, 65, np.inf]
                 age_labels_for_eval = ['<65', '>=65']

            model_dataset_eval_results_dir = os.path.join(eval_output_dir, "individual_model_results",
                                                         model_display_name_for_plots.replace(" ", "_"),
                                                         f"dataset_{test_set_name.replace(' ', '_')}")
            os.makedirs(model_dataset_eval_results_dir, exist_ok=True)

            evaluation_summary_dict, df_predictions_with_metadata = run_evaluation(
                model=model_instance, test_loader=test_loader, device=device,
                results_dir=model_dataset_eval_results_dir,
                experiment_name=f"ood_{test_set_name.lower()}_{model_display_name_for_plots.replace(' ', '_')}",
                subgroup_cols=subgroup_attributes_for_run_eval, age_bins=age_bins_for_eval,
                age_labels=age_labels_for_eval, eye_map=args.eye_map_config,
                metrics_to_compare=args.metrics_to_compare, n_bootstraps=args.n_bootstraps,
                alpha=args.alpha, n_calibration_bins=args.n_calibration_bins,
                min_samples_for_calibration=args.min_samples_for_calibration
            )
            if evaluation_summary_dict is None or df_predictions_with_metadata is None:
                logger.error(f"`run_evaluation` failed for model on {test_set_name}.")
                all_model_dataset_summaries[model_display_name_for_plots][test_set_name] = {"error": "run_evaluation failed"}
                continue
            all_model_dataset_summaries[model_display_name_for_plots][test_set_name] = evaluation_summary_dict

            # Store predictions for ensemble (NEW)
            if 'probability' in df_predictions_with_metadata.columns and 'label' in df_predictions_with_metadata.columns:
                # Prepare DataFrame for ensemble
                ensemble_df = df_predictions_with_metadata[['image_path', 'label', 'probability']].copy()
                
                if test_set_name not in all_model_predictions_per_dataset:
                    all_model_predictions_per_dataset[test_set_name] = []
                all_model_predictions_per_dataset[test_set_name].append(ensemble_df)
                
                # ROC data collection (existing code)
                fpr, tpr, _ = sk_roc_curve(df_predictions_with_metadata['label'], df_predictions_with_metadata['probability'], pos_label=1)
                roc_auc_val = sk_auc(fpr, tpr)
                all_roc_data_collected.setdefault(test_set_name, {})[model_display_name_for_plots] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc_val}
                logger.info(f"ROC data: AUC = {roc_auc_val:.4f}")
            
            if evaluation_summary_dict.get('calibration_analysis', {}).get('Overall', {}).get('error') is None:
                all_calibration_data_collected.setdefault(test_set_name, {})[model_display_name_for_plots] = \
                    evaluation_summary_dict['calibration_analysis']['Overall']

            if 'predicted_label' in df_predictions_with_metadata.columns and 'label' in df_predictions_with_metadata.columns:
                subgroup_defs_disp = {}
                if 'age_group' in df_predictions_with_metadata.columns and age_labels_for_eval and len(age_labels_for_eval) >= 2:
                    age_grps = df_predictions_with_metadata['age_group'].astype(str).unique()
                    if str(age_labels_for_eval[0]) in age_grps and str(age_labels_for_eval[-1]) in age_grps:
                        subgroup_defs_disp["age"] = {"col": "age_group", "favored": str(age_labels_for_eval[0]), "disfavored": str(age_labels_for_eval[-1])}
                if 'eye_group' in df_predictions_with_metadata.columns and \
                   'OD' in df_predictions_with_metadata['eye_group'].astype(str).unique() and \
                   'OS' in df_predictions_with_metadata['eye_group'].astype(str).unique():
                    subgroup_defs_disp["eye"] = {"col": "eye_group", "favored": "OD", "disfavored": "OS"}
                cam_col = 'camera_group' if 'camera_group' in df_predictions_with_metadata.columns else 'camera'
                if test_set_name == "CHAKSU" and cam_col in df_predictions_with_metadata.columns:
                    uc = sorted(df_predictions_with_metadata[cam_col].astype(str).unique())
                    if len(uc) >= 2: subgroup_defs_disp["camera"] = {"col": cam_col, "favored": uc[0], "disfavored": uc[1]}
                
                if subgroup_defs_disp:
                    disparities = calculate_underdiagnosis_disparities(
                        df_predictions_with_metadata, subgroup_defs_disp,
                        positive_label_value=1, label_col='label', pred_label_col='predicted_label'
                    )
                    all_disparity_data_collected.setdefault(test_set_name, {}).setdefault(model_display_name_for_plots, {}).update(disparities)

            if args.run_tsne and df_predictions_with_metadata is not None:
                logger.info(f"Attempting t-SNE for '{model_display_name_for_plots}' on OOD set '{test_set_name}'...")
                features_tsne, labels_tsne, metadata_tsne = extract_features_for_tsne(
                    model_instance, test_loader, device, model_arch_name,
                    attributes_to_collect=attributes_for_dataloader
                )
                if features_tsne is not None and metadata_tsne is not None:
                    tsne_dir = os.path.join(model_dataset_eval_results_dir, "tsne_visualizations")
                    color_attrs = [attr for attr_config in args.tsne_attributes_to_visualize
                                   for attr in [attr_config, f"{attr_config}_group"] if attr in metadata_tsne.columns]
                    for color_attr in list(set(color_attrs)): # Unique attributes for coloring
                        plot_tsne_by_attribute(
                            features_tsne, labels_tsne, metadata_tsne, color_attr,
                            tsne_dir, f"{test_set_name}_{model_display_name_for_plots}",
                            args.tsne_perplexity, args.tsne_n_iter, args.tsne_learning_rate,
                            args.tsne_max_samples_plot, args.tsne_min_samples_per_group_viz
                        )
        del model_instance
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # NEW: Process ensemble predictions
    logger.info("\n--- Processing Ensemble Predictions ---")
    ensemble_roc_data_per_dataset = process_and_evaluate_ood_ensemble(
        all_model_predictions_per_dataset,
        model_display_names_for_ensemble,
        eval_output_dir,
        eval_timestamp
    )

    # Generate comprehensive summary outputs
    logger.info("\n--- Generating Comprehensive Summary Tables and Plots ---")
    
    # Create summary tables
    if all_roc_data_collected:
        create_ood_summary_tables(
            all_roc_data_collected,
            ensemble_roc_data_per_dataset,
            eval_output_dir,
            eval_timestamp
        )
        
        # Enhanced ROC plots with ensemble
        plot_combined_ood_roc_with_ensemble(
            all_roc_data_collected,
            ensemble_roc_data_per_dataset,
            eval_output_dir,
            eval_timestamp
        )
    else:
        logger.warning("No ROC data collected. Skipping comprehensive summary tables and enhanced ROC plots.")

    # Original plotting functions (keeping existing functionality)
    logger.info("\n--- Generating Original Combined Summary Plots ---")
    if all_roc_data_collected: plot_all_roc_curves(all_roc_data_collected, eval_output_dir)
    if all_disparity_data_collected: plot_all_disparities(all_disparity_data_collected, eval_output_dir)
    if all_calibration_data_collected: plot_all_calibration_curves(all_calibration_data_collected, eval_output_dir)

    summary_path = os.path.join(eval_output_dir, 'all_models_all_datasets_comprehensive_summary.json')
    with open(summary_path, 'w') as f: json.dump(all_model_dataset_summaries, f, indent=4, cls=NpEncoder)
    logger.info(f"Comprehensive evaluation summaries saved to: {summary_path}")

    logger.removeHandler(main_file_handler); main_file_handler.close()
    logger.info(f"\n{'='*5} OOD Evaluation Script Finished. Total execution time: {datetime.now() - start_time} {'='*5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate one or more Glaucoma Classification Model(s) on External OOD Test Sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--experiment_parent_dir', type=str, required=True, help="Path to parent directory of model experiment folders.")
    parser.add_argument('--model_config_list', type=str, default=None, help="Optional JSON string or path to JSON file defining specific models to evaluate. If not provided, models will be auto-discovered from the experiment directory.")
    parser.add_argument('--output_dir_suffix', type=str, default=None, help="Optional suffix for the main OOD evaluation output directory.")
    
    model_load_group = parser.add_argument_group("Model Loading Parameters")
    model_load_group.add_argument('--num_classes', type=int, default=2)
    model_load_group.add_argument('--image_size', type=int, default=224)
    model_load_group.add_argument('--strict_load', action='store_true', default=False, help="Use strict=True for state_dict loading.")
    model_load_group.add_argument('--load_weights_only', action=argparse.BooleanOptionalAction, default=True, help="Use torch.load weights_only=True.")
    
    data_group = parser.add_argument_group("External Data Configuration")
    data_group.add_argument('--data_type', type=str, default='raw', choices=['raw', 'processed'], help="Target type of OOD image data ('raw' or 'processed').")
    data_group.add_argument('--base_data_root', type=str, default=r'D:\glaucoma\data', help="Absolute base root directory for all datasets.")
    data_group.add_argument('--smdg_metadata_file_raw', type=str, default=os.path.join('raw','SMDG-19','metadata - standardized.csv'))
    data_group.add_argument('--smdg_image_dir_raw', type=str, default=os.path.join('raw','SMDG-19','full-fundus','full-fundus'))
    data_group.add_argument('--chaksu_base_dir', type=str, default=os.path.join('raw','Chaksu','Train','Train','1.0_Original_Fundus_Images'))
    data_group.add_argument('--chaksu_decision_dir_raw', type=str, default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision'))
    data_group.add_argument('--chaksu_metadata_dir_raw', type=str, default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision','Majority'))
    # Add these arguments to your data_group
    data_group.add_argument('--acrima_image_dir_raw', type=str, default=os.path.join('raw','ACRIMA','Database','Images'))
    data_group.add_argument('--hygd_image_dir_raw', type=str, default=os.path.join('raw','HYGD','HYGD','Images'))
    data_group.add_argument('--hygd_labels_file_raw', type=str, default=os.path.join('raw','HYGD','HYGD','Labels.csv'))


    eval_set_group = parser.add_argument_group("OOD Set Selection")
    eval_set_group.add_argument('--eval_papilla', action=argparse.BooleanOptionalAction, default=True)
    eval_set_group.add_argument('--eval_chaksu', action=argparse.BooleanOptionalAction, default=True)
    eval_set_group.add_argument('--eval_acrima', action=argparse.BooleanOptionalAction, default=True)
    eval_set_group.add_argument('--eval_hygd', action=argparse.BooleanOptionalAction, default=True)
    
    gen_eval_group = parser.add_argument_group("General Evaluation Parameters")
    gen_eval_group.add_argument('--eval_batch_size', type=int, default=32)
    gen_eval_group.add_argument('--num_workers', type=int, default=0)
    gen_eval_group.add_argument('--seed', type=int, default=42)
    
    run_eval_params_group = parser.add_argument_group("Parameters for 'run_evaluation' function")
    run_eval_params_group.add_argument('--n_bootstraps', type=int, default=0)
    run_eval_params_group.add_argument('--alpha', type=float, default=0.05)
    run_eval_params_group.add_argument('--n_calibration_bins', type=int, default=10)
    run_eval_params_group.add_argument('--min_samples_for_calibration', type=int, default=30)
    run_eval_params_group.add_argument('--age_bins', type=float, nargs='+', default=None, help="Age bins, e.g., 0 50 65 inf.")
    run_eval_params_group.add_argument('--age_labels', type=str, nargs='+', default=None, help="Labels for age bins.")
    run_eval_params_group.add_argument('--eye_map_config', type=json.loads, default='{"OD":"OD","OS":"OS","Right":"OD","Left":"OS","R":"OD","L":"OS","0":"OD","1":"OS", "nan":"Unknown", "None":"Unknown"}')
    run_eval_params_group.add_argument('--metrics_to_compare', nargs='+', default=['AUC', 'Accuracy', 'F1', 'Precision', 'Recall', 'TPR', 'FPR', 'Specificity', 'NPV'])
    
    tsne_group = parser.add_argument_group("t-SNE Visualization Parameters")
    tsne_group.add_argument('--run_tsne', action=argparse.BooleanOptionalAction, default=False)
    tsne_group.add_argument('--tsne_attributes_to_visualize', nargs='+', default=['dataset_source', 'camera', 'age', 'eye'])
    tsne_group.add_argument('--tsne_perplexity', type=float, default=30.0)
    tsne_group.add_argument('--tsne_n_iter', type=int, default=1000)
    tsne_group.add_argument('--tsne_learning_rate', type=str, default='auto')
    tsne_group.add_argument('--tsne_max_samples_plot', type=int, default=1500)
    tsne_group.add_argument('--tsne_min_samples_per_group_viz', type=int, default=10)

    args = parser.parse_args()
    if args.age_bins: args.age_bins = [np.inf if str(b).lower() == 'inf' else float(b) for b in args.age_bins]
    if args.age_bins and args.age_labels and (len(args.age_labels) != len(args.age_bins) - 1):
        parser.error("If --age_bins and --age_labels are provided, --age_labels must have one less element than --age_bins.")
    args.base_data_root = os.path.abspath(args.base_data_root)
    if not os.path.isdir(args.base_data_root):
        logger.critical(f"Base data root not found: {args.base_data_root}. Exiting."); sys.exit(1)
    main(args)