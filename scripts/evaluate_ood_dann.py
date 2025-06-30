#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
External OOD Test Set Evaluation Script for DANN Models.

This script evaluates pre-trained DANN model(s) from experiment directories
on specified external OOD (Out-of-Distribution) test datasets.
It adapts the standard OOD evaluation workflow for DANN models while
maintaining compatibility with ensemble evaluation and comprehensive reporting.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
import glob
import re
from functools import reduce

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve as sk_roc_curve, auc as sk_auc

# Ensure custom modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.utils.helpers import NpEncoder, set_seed
    # DANN-specific imports
    from src.data.datasets_dann import GlaucomaSubgroupDANNDataset, safe_collate_dann
    from src.models.classification.build_model_dann import build_classifier_model_dann
    
    # Import utilities from standard OOD evaluation script
    from src.data.external_loader import load_external_test_data
    from src.data.utils import adjust_path_for_data_type
    from src.evaluation.bias_analysis import extract_features_for_tsne
    from src.evaluation.fairness import calculate_underdiagnosis_disparities
    from src.evaluation.plotting import (
        plot_tsne_by_attribute,
        plot_all_roc_curves,
        plot_all_disparities,
        plot_all_calibration_curves
    )
    
    # Import from the ID evaluation script for reusable functions
    import evaluate_id_dann as dann_eval_utils
    
except ImportError as e:
    print(f"Error importing modules: {e}\nEnsure all required modules are accessible.")
    sys.exit(1)

# --- Global Constants ---
RAW_DIR_NAME_CONST: str = "raw"
PROCESSED_DIR_NAME_CONST: str = "processed"
DOMAIN_COLUMN_NAME: str = 'dataset_source'

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

DEFAULT_DANN_MODELS_TO_EVALUATE = [
    {"ModelNameInTraining": "vit_base_patch16_224", "TrainingModelShortName": "vit_DANN", "TrainingTag": "VFM_custom_DANN", "TrainingDropout": 0.3},
    {"ModelNameInTraining": "vit_base_patch16_224", "TrainingModelShortName": "vit", "TrainingTag": "DANN", "TrainingDropout": 0.3},
    {"ModelNameInTraining": "resnet50", "TrainingModelShortName": "resnet50", "TrainingTag": "DANN", "TrainingDropout": 0.3},
    {"ModelNameInTraining": "convnext_base", "TrainingModelShortName": "convnext", "TrainingTag": "DANN", "TrainingDropout": 0.3},
]

def find_dann_checkpoint_path(base_dir: str, dir_pattern: str, model_id_for_log: str) -> tuple[str | None, str | None]:
    """
    Finds the best DANN checkpoint file within a model's experiment directory.
    
    Args:
        base_dir: The parent directory where model experiment folders are located.
        dir_pattern: Glob pattern to find the specific model's experiment directory.
        model_id_for_log: Identifier for logging purposes.
    
    Returns:
        A tuple (path_to_checkpoint_file, path_to_experiment_directory), or (None, None) if not found.
    """
    logger.info(f"Searching for DANN training run directory with pattern: '{dir_pattern}' under '{base_dir}' for model '{model_id_for_log}'")
    found_model_dirs = glob.glob(os.path.join(base_dir, dir_pattern))
    
    if not found_model_dirs:
        logger.warning(f"DANN experiment subdirectory not found for pattern '{dir_pattern}' in '{base_dir}'. Skipping model {model_id_for_log}.")
        return None, None
    
    model_actual_experiment_dir = sorted(found_model_dirs, reverse=True)[0]  # Get most recent if multiple
    logger.info(f"Found DANN training run directory for '{model_id_for_log}': {model_actual_experiment_dir}")
    
    checkpoint_dir = os.path.join(model_actual_experiment_dir, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        logger.warning(f"Checkpoints directory not found: {checkpoint_dir} for model {model_id_for_log}. Skipping.")
        return None, None
    
    checkpoint_file = None
    
    # Look for DANN-specific checkpoints first
    dann_best_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*_best_model_epoch*_dann.pth"))
    if dann_best_checkpoints:
        dann_best_checkpoints_sorted = sorted(
            dann_best_checkpoints,
            key=lambda x: int(m.group(1)) if (m := re.search(r'_epoch(\d+)_dann\.pth$', os.path.basename(x))) else -1,
            reverse=True
        )
        checkpoint_file = dann_best_checkpoints_sorted[0]
        logger.info(f"Selected DANN 'best_model' checkpoint (highest epoch): {os.path.basename(checkpoint_file)}")
    else:
        # Fallback to regular best checkpoints
        logger.warning(f"No '*_best_model_epoch*_dann.pth' checkpoint found for model {model_id_for_log}.")
        best_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*_best_model_epoch*.pth"))
        if best_checkpoints:
            best_checkpoints_sorted = sorted(
                best_checkpoints,
                key=lambda x: int(m.group(1)) if (m := re.search(r'_epoch(\d+)\.pth$', os.path.basename(x))) else -1,
                reverse=True
            )
            checkpoint_file = best_checkpoints_sorted[0]
            logger.info(f"Falling back to regular 'best_model' checkpoint: {os.path.basename(checkpoint_file)}")
        else:
            # Final fallback to any .pth file
            all_pth_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
            if all_pth_files:
                checkpoint_file = max(all_pth_files, key=os.path.getmtime)
                logger.info(f"Falling back to latest modified checkpoint: {os.path.basename(checkpoint_file)}")
    
    if checkpoint_file:
        return checkpoint_file, model_actual_experiment_dir
    else:
        logger.warning(f"No suitable checkpoint file found for {model_id_for_log} in '{checkpoint_dir}'.")
        return None, None

def create_dann_external_test_dataloader(df_test: pd.DataFrame, batch_size: int, num_workers: int, 
                                        eval_transforms, test_set_name: str, domain_id_map: dict = None):
    """
    Creates a DataLoader for external test data compatible with DANN models.
    """
    logger.info(f"Creating DANN DataLoader for {test_set_name} with {len(df_test)} samples")
    
    # Ensure required columns exist
    if 'image_path' not in df_test.columns:
        logger.error(f"'image_path' column missing in {test_set_name} DataFrame")
        return None
    
    if 'types' not in df_test.columns:
        logger.error(f"'types' column missing in {test_set_name} DataFrame")
        return None
    
    # Add dataset_source column if not present (for domain information)
    if DOMAIN_COLUMN_NAME not in df_test.columns:
        df_test[DOMAIN_COLUMN_NAME] = test_set_name
    
    # Collect sensitive attributes for DANN dataset
    sensitive_attributes = [DOMAIN_COLUMN_NAME, 'image_path']
    for attr in ['camera', 'age', 'eye', 'quality_score']:
        if attr in df_test.columns:
            sensitive_attributes.append(attr)
    
    try:
        test_dataset = GlaucomaSubgroupDANNDataset(
            df_test,
            transform=eval_transforms,
            sensitive_attributes=sensitive_attributes,
            domain_column=DOMAIN_COLUMN_NAME,
            domain_id_map=domain_id_map,
            label_col='types',
            path_col='image_path'
        )
        
        if len(test_dataset) == 0:
            logger.warning(f"DANN test dataset for {test_set_name} is empty after initialization")
            return None
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=safe_collate_dann,
            persistent_workers=(num_workers > 0)
        )
        
        logger.info(f"Successfully created DANN DataLoader for {test_set_name}")
        return test_loader
        
    except Exception as e:
        logger.error(f"Error creating DANN DataLoader for {test_set_name}: {e}", exc_info=True)
        return None

def evaluate_dann_model_on_ood(model, test_loader, device, model_display_name, test_set_name, output_dir):
    """
    Evaluates a DANN model on an OOD test set.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    all_image_paths = []
    
    logger.info(f"Evaluating DANN model '{model_display_name}' on OOD dataset '{test_set_name}'...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Handle DANN batch format (4 elements including metadata)
            if len(batch) == 4:
                images, labels, domains, metadata = batch
            elif len(batch) == 3:
                images, labels, domains = batch
                metadata = None
            else:
                logger.warning(f"Unexpected batch format with {len(batch)} elements")
                continue
            
            images = images.to(device)
            
            # Forward pass through DANN model
            outputs = model(images)
            
            # Handle DANN output format (class_logits, domain_logits)
            if isinstance(outputs, tuple):
                class_logits, domain_logits = outputs
            else:
                class_logits = outputs
            
            # Get predictions and probabilities
            probs = torch.softmax(class_logits, dim=1)
            predictions = torch.argmax(class_logits, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (glaucoma)
            
            # Extract image paths from metadata if available
            if metadata is not None and len(metadata) > 0:
                # metadata is typically a list of dicts with image_path info
                for item in metadata:
                    if isinstance(item, dict) and 'image_path' in item:
                        all_image_paths.append(item['image_path'])
                    else:
                        all_image_paths.append(f"unknown_{len(all_image_paths)}")
            else:
                # Generate placeholder paths
                batch_size = len(predictions)
                all_image_paths.extend([f"unknown_{len(all_image_paths) + i}" for i in range(batch_size)])
    
    # Create results dataframe
    df_results = pd.DataFrame({
        'image_path': all_image_paths,
        'label': all_labels,
        'predicted_label': all_predictions,
        'probability': all_probs
    })
    
    # Save predictions
    results_path = os.path.join(output_dir, f'{model_display_name}_{test_set_name}_dann_ood_predictions.csv')
    df_results.to_csv(results_path, index=False)
    logger.info(f"DANN OOD predictions saved to: {results_path}")
    
    # Calculate basic metrics
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    accuracy = accuracy_score(all_labels, all_predictions)
    try:
        auc_val = roc_auc_score(all_labels, all_probs)
    except ValueError as e:
        logger.warning(f"Could not calculate AUC: {e}")
        auc_val = np.nan
    
    logger.info(f"DANN OOD Evaluation Results - {model_display_name} on {test_set_name}:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  AUC: {auc_val:.4f}" if not np.isnan(auc_val) else "  AUC: N/A")
    
    # Create evaluation summary
    evaluation_summary = {
        'model_name': model_display_name,
        'dataset': test_set_name,
        'accuracy': accuracy,
        'auc': auc_val,
        'num_samples': len(all_labels),
        'num_positive': sum(all_labels),
        'num_negative': len(all_labels) - sum(all_labels)
    }
    
    return evaluation_summary, df_results

def process_dann_ensemble_ood(
    all_model_predictions_per_dataset: dict[str, list[pd.DataFrame]],
    model_display_names: list[str],
    output_dir: str,
    timestamp_str: str
) -> dict[str, dict]:
    """
    Creates and evaluates ensemble predictions for DANN models on each OOD dataset.
    """
    if len(model_display_names) < 2:
        logger.warning("DANN ensemble evaluation requires at least 2 models. Skipping ensemble.")
        return {}
    
    logger.info(f"\n{'='*10} Processing DANN OOD Ensemble Predictions ({len(model_display_names)} models) {'='*10}")
    
    ensemble_roc_data_per_dataset = {}
    
    for dataset_name, prediction_dfs in all_model_predictions_per_dataset.items():
        if len(prediction_dfs) < 2:
            logger.warning(f"Skipping DANN ensemble for {dataset_name} - insufficient models ({len(prediction_dfs)})")
            continue
            
        logger.info(f"\n--- Creating DANN Ensemble for Dataset: {dataset_name} ---")
        
        # Prepare DataFrames for merging
        prepared_dfs = []
        for i, df_model in enumerate(prediction_dfs):
            model_name = model_display_names[i]
            df_renamed = df_model[['image_path', 'label', 'probability']].copy()
            df_renamed.rename(columns={'probability': f'prob_{model_name}'}, inplace=True)
            df_renamed.set_index('image_path', inplace=True)
            prepared_dfs.append(df_renamed)
        
        # Merge all DataFrames
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
            logger.warning(f"DANN ensemble DataFrame empty for {dataset_name}. Skipping.")
            continue
        
        # Save ensemble results
        ensemble_dataset_dir = os.path.join(output_dir, f"dann_ensemble_results_{dataset_name}_{timestamp_str}")
        os.makedirs(ensemble_dataset_dir, exist_ok=True)
        
        ensemble_csv_path = os.path.join(ensemble_dataset_dir, f"dann_ensemble_predictions_{dataset_name}.csv")
        ensemble_df.to_csv(ensemble_csv_path, index=False)
        logger.info(f"DANN ensemble predictions for {dataset_name} saved to: {ensemble_csv_path}")
        
        # Calculate ROC for ensemble
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(ensemble_df['label'], ensemble_df['ensemble_probability'])
        roc_auc = auc(fpr, tpr)
        
        ensemble_roc_data_per_dataset[dataset_name] = {
            'fpr': fpr, 'tpr': tpr, 'auc': roc_auc
        }
        
        logger.info(f"DANN ensemble ROC for {dataset_name}: AUC = {roc_auc:.4f}")
    
    return ensemble_roc_data_per_dataset

def create_dann_ood_summary_tables(
    all_roc_data_per_dataset: dict[str, dict],
    ensemble_roc_data_per_dataset: dict[str, dict],
    output_dir: str,
    timestamp_str: str
) -> None:
    """
    Creates comprehensive summary tables for DANN OOD evaluation results.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    logger.info("Creating comprehensive DANN OOD summary tables...")
    
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
    header = ['DANN Model'] + all_datasets + ['Mean AUC']
    
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
        ensemble_row = ['🔷 DANN ENSEMBLE']
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
    
    # Header styling (DANN-themed colors)
    for i in range(len(header)):
        table[(0, i)].set_facecolor('#8B4513')  # Brown color for DANN
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.08)
    
    # Row styling
    for i in range(1, len(table_data) + 1):
        row_color = '#FFF8DC' if i % 2 == 0 else '#FFFFFF'  # Light colors
        
        # Special styling for ensemble row
        if i == len(table_data) and ensemble_roc_data_per_dataset:
            row_color = '#F5DEB3'  # Wheat color for ensemble
            for j in range(len(header)):
                table[(i, j)].set_text_props(weight='bold')
        
        for j in range(len(header)):
            table[(i, j)].set_facecolor(row_color)
            table[(i, j)].set_height(0.06)
    
    # Title and subtitle
    plt.suptitle('DANN OOD Evaluation Summary: AUC Performance Across Datasets', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.title(f'Generated: {timestamp_str} | Total DANN Models: {len(all_models)} | Datasets: {len(all_datasets)}',
             fontsize=11, color='gray', y=0.90)
    
    # Save table
    auc_table_path = os.path.join(output_dir, f'dann_ood_auc_summary_table_{timestamp_str}.png')
    plt.savefig(auc_table_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"DANN OOD AUC summary table saved to: {auc_table_path}")

def plot_combined_dann_ood_roc_with_ensemble(
    all_roc_data_per_dataset: dict[str, dict],
    ensemble_roc_data_per_dataset: dict[str, dict],
    output_dir: str,
    timestamp_str: str
) -> None:
    """
    Creates combined ROC plots for each OOD dataset with DANN ensemble curves highlighted.
    """
    import matplotlib.pyplot as plt
    
    for dataset_name, model_roc_data in all_roc_data_per_dataset.items():
        if not model_roc_data:
            continue
            
        plt.figure(figsize=(12, 10))
        
        # Plot individual DANN model curves
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
        
        # Plot DANN ensemble curve if available
        if dataset_name in ensemble_roc_data_per_dataset:
            ensemble_data = ensemble_roc_data_per_dataset[dataset_name]
            auc_val = ensemble_data['auc']
            plt.plot(ensemble_data['fpr'], ensemble_data['tpr'], 
                    color='#8B4513', lw=4, linestyle='--', alpha=0.9,  # Brown for DANN
                    label=f'🔷 DANN ENSEMBLE (AUC={auc_val:.3f})')
        
        # Styling
        plt.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'DANN ROC Curves - {dataset_name} Dataset (OOD Evaluation)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        roc_plot_path = os.path.join(output_dir, f'dann_combined_roc_{dataset_name.lower().replace(" ", "_")}_{timestamp_str}.png')
        plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"DANN combined ROC plot for {dataset_name} saved to: {roc_plot_path}")

def main(args: argparse.Namespace):
    """
    Main function to orchestrate the DANN OOD model evaluation process.
    """
    start_time = datetime.now()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    eval_timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    output_suffix = args.output_dir_suffix if args.output_dir_suffix else f"dann_multi_model_ood_eval_{args.data_type}"
    eval_output_dir = os.path.join(args.experiment_parent_dir, "external_evaluations", f"{output_suffix}_{eval_timestamp}")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    log_file_path = os.path.join(eval_output_dir, f"log_dann_ood_eval_{eval_timestamp}.log")
    main_file_handler = logging.FileHandler(log_file_path, mode='w')
    main_file_handler.setFormatter(log_formatter)
    logger.addHandler(main_file_handler)
    
    logger.info(f"--- DANN Multi-Model OOD Evaluation Initializing ---")
    logger.info(f"Evaluation run started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Processing DANN models from parent directory: {args.experiment_parent_dir}")
    logger.info(f"Evaluation outputs will be saved to: {eval_output_dir}")
    logger.info(f"Using PyTorch device: {device}")
    logger.info(f"Base data root for external datasets: {os.path.abspath(args.base_data_root)}")
    
    # Load model configurations
    models_to_evaluate_config = DEFAULT_DANN_MODELS_TO_EVALUATE
    if args.model_config_list:
        try:
            if os.path.exists(args.model_config_list):
                with open(args.model_config_list, 'r') as f:
                    models_to_evaluate_config = json.load(f)
                logger.info(f"Loaded DANN model configurations from JSON file: {args.model_config_list}")
            else:
                models_to_evaluate_config = json.loads(args.model_config_list)
                logger.info("Loaded DANN model configurations from JSON string argument.")
        except Exception as e:
            logger.critical(f"Error processing --model_config_list '{args.model_config_list}': {e}. Exiting.", exc_info=True)
            logger.removeHandler(main_file_handler)
            main_file_handler.close()
            sys.exit(1)
    
    # Save configuration
    with open(os.path.join(eval_output_dir, "dann_evaluation_run_config.json"), "w") as f_cfg:
        config_to_save = vars(args).copy()
        config_to_save['resolved_model_config_list'] = models_to_evaluate_config
        json.dump(config_to_save, f_cfg, indent=4, cls=NpEncoder)
    
    # Discover DANN checkpoints
    all_discovered_checkpoints_info = []
    logger.info(f"\n--- Discovering DANN Checkpoints in Parent Directory: {args.experiment_parent_dir} ---")
    
    training_run_data_type_suffix = "raw"
    if "processed" in os.path.basename(args.experiment_parent_dir).lower():
        training_run_data_type_suffix = "processed"
    logger.info(f"Inferred training data type suffix for directory search: '{training_run_data_type_suffix}'")
    
    for model_config_entry in models_to_evaluate_config:
        required_keys = ["ModelNameInTraining", "TrainingModelShortName", "TrainingTag", "TrainingDropout"]
        if not all(key in model_config_entry for key in required_keys):
            logger.warning(f"Skipping malformed DANN model config: {model_config_entry}")
            continue
        
        # Look for DANN-specific directory patterns
        dir_search_pattern = f"{model_config_entry['TrainingModelShortName']}_{training_run_data_type_suffix}_{model_config_entry['TrainingTag']}*"
        model_id_for_log = f"DANN_{model_config_entry['TrainingModelShortName']}-{model_config_entry['TrainingTag']}"
        
        checkpoint_file_path, full_experiment_dir_path = find_dann_checkpoint_path(
            args.experiment_parent_dir, dir_search_pattern, model_id_for_log
        )
        
        if checkpoint_file_path and full_experiment_dir_path:
            all_discovered_checkpoints_info.append({
                "model_arch_name": model_config_entry['ModelNameInTraining'],
                "dropout_prob": model_config_entry['TrainingDropout'],
                "training_tag": model_config_entry['TrainingTag'],
                "checkpoint_path": checkpoint_file_path,
                "full_experiment_dir": full_experiment_dir_path,
                "model_display_name": f"DANN_{model_config_entry['TrainingModelShortName']}_{model_config_entry['TrainingTag']}"
            })
    
    if not all_discovered_checkpoints_info:
        logger.critical(f"No DANN checkpoints discovered. Exiting.")
        logger.removeHandler(main_file_handler)
        main_file_handler.close()
        sys.exit(1)
    
    logger.info(f"Successfully discovered {len(all_discovered_checkpoints_info)} DANN checkpoint(s).")
    
    # Load external test datasets (reuse existing function)
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
        eval_oiaodir_test=args.eval_oiaodir_test,
        eval_chaksu=args.eval_chaksu,
        eval_acrima=args.eval_acrima,
        eval_hygd=args.eval_hygd,
        acrima_image_dir_raw=args.acrima_image_dir_raw,
        hygd_image_dir_raw=args.hygd_image_dir_raw,
        hygd_labels_file_raw=args.hygd_labels_file_raw
    )
    
    if not external_test_sets_map or all(df.empty for df in external_test_sets_map.values()):
        logger.error("No external test data loaded. Exiting.")
        logger.removeHandler(main_file_handler)
        main_file_handler.close()
        sys.exit(1)
    
    logger.info(f"Loaded external test datasets: {list(external_test_sets_map.keys())}")
    
    # Initialize data structures for results
    all_model_dataset_summaries = {}
    all_roc_data_collected = {}
    all_model_predictions_per_dataset = {}  # For ensemble
    model_display_names_for_ensemble = []
    
    # Process each DANN model
    for ckpt_idx, model_info_dict in enumerate(all_discovered_checkpoints_info):
        model_arch_name = model_info_dict['model_arch_name']
        dropout_prob = model_info_dict['dropout_prob']
        training_tag = model_info_dict['training_tag']
        checkpoint_file_path = model_info_dict['checkpoint_path']
        model_display_name_for_plots = model_info_dict['model_display_name']
        
        logger.info(f"\n===== Evaluating DANN Model {ckpt_idx + 1}/{len(all_discovered_checkpoints_info)}: {model_display_name_for_plots} =====")
        logger.info(f"Arch: {model_arch_name}, Dropout: {dropout_prob}, Checkpoint: {os.path.basename(checkpoint_file_path)}")
        
        # Load training configuration to get DANN parameters
        orig_results_dir = os.path.join(model_info_dict['full_experiment_dir'], "results")
        train_config_path = os.path.join(orig_results_dir, "config_DANN.json")
        if not os.path.exists(train_config_path):
            train_config_path = os.path.join(orig_results_dir, "training_configuration_dann.json")
        if not os.path.exists(train_config_path):
            train_config_path = os.path.join(orig_results_dir, "training_configuration.json")
        
        if not os.path.exists(train_config_path):
            logger.error(f"Training config missing for {model_display_name_for_plots}. Skipping.")
            continue
        
        try:
            with open(train_config_path, 'r') as f:
                train_args_dict = json.load(f)
            
            # Extract DANN-specific parameters
            was_dann_trained = train_args_dict.get('use_dann', True)  # Assume True for DANN models
            num_doms_train = train_args_dict.get('num_domains', 2)
            grl_lambda_train = train_args_dict.get('dann_grl_base_lambda', 1.0)
            grl_sched_train = train_args_dict.get('dann_lambda_schedule_mode', "paper")
            grl_gamma_train = train_args_dict.get('dann_gamma', 10.0)
            
            # Load domain_id_map from checkpoint
            checkpoint_data = torch.load(checkpoint_file_path, map_location='cpu')
            domain_id_map_from_ckpt = checkpoint_data.get('domain_id_map', None)
            if domain_id_map_from_ckpt:
                logger.info(f"Loaded domain_id_map from checkpoint: {domain_id_map_from_ckpt}")
            num_doms_from_ckpt = checkpoint_data.get('num_domains', num_doms_train)
            
        except Exception as e:
            logger.error(f"Error loading/parsing config for {model_display_name_for_plots}: {e}. Skipping.", exc_info=True)
            continue
        
        # Build DANN model
        model_instance = build_classifier_model_dann(
            model_name=model_arch_name,
            num_classes=args.num_classes,
            dropout_prob=dropout_prob,
            pretrained=False,
            custom_weights_path=checkpoint_file_path,
            checkpoint_key=train_args_dict.get('checkpoint_key', 'model_state_dict'),
            use_dann=was_dann_trained,
            num_domains=num_doms_from_ckpt,
            dann_grl_base_lambda=grl_lambda_train,
            dann_lambda_schedule_mode=grl_sched_train,
            dann_gamma=grl_gamma_train
        )
        
        model_instance.to(device).eval()
        logger.info(f"DANN model loaded successfully: {model_display_name_for_plots}")
        
        # Store model name for ensemble
        if model_display_name_for_plots not in model_display_names_for_ensemble:
            model_display_names_for_ensemble.append(model_display_name_for_plots)
        
        # Initialize model results
        if model_display_name_for_plots not in all_model_dataset_summaries:
            all_model_dataset_summaries[model_display_name_for_plots] = {}
        
        # Get evaluation transforms
        eval_transforms = dann_eval_utils.get_eval_transforms(args.image_size, model_arch_name)
        
        # Evaluate on each external test set
        for test_set_name, df_test_original in external_test_sets_map.items():
            if df_test_original.empty:
                continue
            
            df_test = df_test_original.copy()
            
            # Ensure 'types' column exists
            if 'label' in df_test.columns and 'types' not in df_test.columns:
                df_test.rename(columns={'label': 'types'}, inplace=True)
            if 'types' not in df_test.columns:
                logger.error(f"'types' missing in {test_set_name}. Skipping.")
                continue
            
            logger.info(f"\n--- Evaluating DANN '{model_display_name_for_plots}' on OOD set: '{test_set_name}' ---")
            
            # Create DANN-compatible DataLoader
            test_loader = create_dann_external_test_dataloader(
                df_test, args.eval_batch_size, args.num_workers, eval_transforms,
                test_set_name, domain_id_map_from_ckpt
            )
            
            if not test_loader:
                logger.warning(f"DANN DataLoader failed for {test_set_name}. Skipping.")
                continue
            
            # Create output directory for this model-dataset combination
            model_dataset_eval_results_dir = os.path.join(
                eval_output_dir, "individual_dann_model_results",
                model_display_name_for_plots.replace(" ", "_"),
                f"dataset_{test_set_name.replace(' ', '_')}"
            )
            os.makedirs(model_dataset_eval_results_dir, exist_ok=True)
            
            # Evaluate DANN model
            evaluation_summary_dict, df_predictions_with_metadata = evaluate_dann_model_on_ood(
                model_instance, test_loader, device, model_display_name_for_plots,
                test_set_name, model_dataset_eval_results_dir
            )
            
            if evaluation_summary_dict is None or df_predictions_with_metadata is None:
                logger.error(f"DANN evaluation failed for model on {test_set_name}.")
                all_model_dataset_summaries[model_display_name_for_plots][test_set_name] = {"error": "evaluation failed"}
                continue
            
            all_model_dataset_summaries[model_display_name_for_plots][test_set_name] = evaluation_summary_dict
            
            # Store predictions for ensemble
            if 'probability' in df_predictions_with_metadata.columns and 'label' in df_predictions_with_metadata.columns:
                ensemble_df = df_predictions_with_metadata[['image_path', 'label', 'probability']].copy()
                
                if test_set_name not in all_model_predictions_per_dataset:
                    all_model_predictions_per_dataset[test_set_name] = []
                all_model_predictions_per_dataset[test_set_name].append(ensemble_df)
                
                # Collect ROC data
                fpr, tpr, _ = sk_roc_curve(df_predictions_with_metadata['label'], df_predictions_with_metadata['probability'], pos_label=1)
                roc_auc_val = sk_auc(fpr, tpr)
                all_roc_data_collected.setdefault(test_set_name, {})[model_display_name_for_plots] = {
                    'fpr': fpr, 'tpr': tpr, 'auc': roc_auc_val
                }
                logger.info(f"ROC data: AUC = {roc_auc_val:.4f}")
            
            # Optional: Run t-SNE
            if args.run_tsne:
                logger.info(f"Running t-SNE for DANN '{model_display_name_for_plots}' on OOD set '{test_set_name}'...")
                try:
                    dann_eval_utils.run_tsne_dann(
                        model_instance, test_loader, model_dataset_eval_results_dir,
                        f"{test_set_name}_{model_display_name_for_plots}", device,
                        max_samples=args.tsne_max_samples_plot
                    )
                except Exception as e:
                    logger.error(f"Error during DANN t-SNE for {test_set_name}: {e}", exc_info=True)
        
        # Clean up model
        del model_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Process ensemble predictions
    logger.info("\n--- Processing DANN Ensemble Predictions ---")
    ensemble_roc_data_per_dataset = process_dann_ensemble_ood(
        all_model_predictions_per_dataset,
        model_display_names_for_ensemble,
        eval_output_dir,
        eval_timestamp
    )
    
    # Generate comprehensive summary outputs
    logger.info("\n--- Generating DANN Comprehensive Summary Tables and Plots ---")
    
    if all_roc_data_collected:
        create_dann_ood_summary_tables(
            all_roc_data_collected,
            ensemble_roc_data_per_dataset,
            eval_output_dir,
            eval_timestamp
        )
        
        plot_combined_dann_ood_roc_with_ensemble(
            all_roc_data_collected,
            ensemble_roc_data_per_dataset,
            eval_output_dir,
            eval_timestamp
        )
    else:
        logger.warning("No DANN ROC data collected. Skipping comprehensive summary tables and enhanced ROC plots.")
    
    # Save comprehensive summary
    summary_path = os.path.join(eval_output_dir, 'all_dann_models_all_datasets_comprehensive_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_model_dataset_summaries, f, indent=4, cls=NpEncoder)
    logger.info(f"Comprehensive DANN evaluation summaries saved to: {summary_path}")
    
    logger.removeHandler(main_file_handler)
    main_file_handler.close()
    logger.info(f"\n{'='*5} DANN OOD Evaluation Script Finished. Total execution time: {datetime.now() - start_time} {'='*5}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DANN-trained Glaucoma Classification Model(s) on External OOD Test Sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--experiment_parent_dir', type=str, required=True, 
                       help="Path to parent directory of DANN model experiment folders.")
    parser.add_argument('--model_config_list', type=str, default=None, 
                       help="JSON string or path to JSON file defining DANN models to evaluate.")
    parser.add_argument('--output_dir_suffix', type=str, default=None, 
                       help="Optional suffix for the main DANN OOD evaluation output directory.")
    
    model_load_group = parser.add_argument_group("DANN Model Loading Parameters")
    model_load_group.add_argument('--num_classes', type=int, default=2)
    model_load_group.add_argument('--image_size', type=int, default=224)
    model_load_group.add_argument('--strict_load', action='store_true', default=False, 
                                 help="Use strict=True for state_dict loading.")
    model_load_group.add_argument('--load_weights_only', action=argparse.BooleanOptionalAction, default=True, 
                                 help="Use torch.load weights_only=True.")
    
    data_group = parser.add_argument_group("External Data Configuration")
    data_group.add_argument('--data_type', type=str, default='raw', choices=['raw', 'processed'], 
                           help="Target type of OOD image data ('raw' or 'processed').")
    data_group.add_argument('--base_data_root', type=str, default=r'D:\glaucoma\data', 
                           help="Absolute base root directory for all datasets.")
    data_group.add_argument('--smdg_metadata_file_raw', type=str, 
                           default=os.path.join('raw','SMDG-19','metadata - standardized.csv'))
    data_group.add_argument('--smdg_image_dir_raw', type=str, 
                           default=os.path.join('raw','SMDG-19','full-fundus','full-fundus'))
    data_group.add_argument('--chaksu_base_dir', type=str, 
                           default=os.path.join('raw','Chaksu','Train','Train','1.0_Original_Fundus_Images'))
    data_group.add_argument('--chaksu_decision_dir_raw', type=str, 
                           default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision'))
    data_group.add_argument('--chaksu_metadata_dir_raw', type=str, 
                           default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision','Majority'))
    data_group.add_argument('--acrima_image_dir_raw', type=str, 
                           default=os.path.join('raw','ACRIMA','Database','Images'))
    data_group.add_argument('--hygd_image_dir_raw', type=str, 
                           default=os.path.join('raw','HYGD','HYGD','Images'))
    data_group.add_argument('--hygd_labels_file_raw', type=str, 
                           default=os.path.join('raw','HYGD','HYGD','Labels.csv'))
    
    eval_set_group = parser.add_argument_group("OOD Set Selection")
    eval_set_group.add_argument('--eval_papilla', action=argparse.BooleanOptionalAction, default=True)
    eval_set_group.add_argument('--eval_oiaodir_test', action=argparse.BooleanOptionalAction, default=False)
    eval_set_group.add_argument('--eval_chaksu', action=argparse.BooleanOptionalAction, default=True)
    eval_set_group.add_argument('--eval_acrima', action=argparse.BooleanOptionalAction, default=True)
    eval_set_group.add_argument('--eval_hygd', action=argparse.BooleanOptionalAction, default=True)
    
    gen_eval_group = parser.add_argument_group("General Evaluation Parameters")
    gen_eval_group.add_argument('--eval_batch_size', type=int, default=32)
    gen_eval_group.add_argument('--num_workers', type=int, default=0)
    gen_eval_group.add_argument('--seed', type=int, default=42)
    
    tsne_group = parser.add_argument_group("t-SNE Visualization Parameters")
    tsne_group.add_argument('--run_tsne', action=argparse.BooleanOptionalAction, default=False)
    tsne_group.add_argument('--tsne_max_samples_plot', type=int, default=1500)
    
    args = parser.parse_args()
    args.base_data_root = os.path.abspath(args.base_data_root)
    if not os.path.isdir(args.base_data_root):
        logger.critical(f"Base data root not found: {args.base_data_root}. Exiting.")
        sys.exit(1)
    
    main(args)