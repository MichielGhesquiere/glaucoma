#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In-Distribution Test Set Evaluation Script for DANN Models.

Evaluates pre-trained DANN models from experiment directories on their
in-distribution test sets. Focuses on the primary classification task.
It can produce individual ROC plots per model and a final combined ROC plot.
t-SNE and Grad-CAM are optional and adapted for DANN models.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
import glob
from functools import reduce 

import matplotlib.pyplot as plt
# import matplotlib.cm as cm # Not directly used in simplified version
# import matplotlib.patches as patches
import numpy as np
import pandas as pd
import timm # For get_eval_transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve, accuracy_score
from sklearn.manifold import TSNE # If t-SNE is kept
import seaborn as sns # If t-SNE is kept
from timm.data import resolve_data_config # For get_eval_transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor # From get_eval_transforms
import torchvision.transforms as transforms # For InterpolationMode
import tqdm
from typing import Dict, Optional, Tuple
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.utils.helpers import NpEncoder, set_seed
    # --- DANN Adapted Imports ---
    from src.data.datasets_dann import GlaucomaSubgroupDANNDataset, safe_collate_dann
    from src.models.classification.build_model_dann import build_classifier_model_dann
    # --- End DANN Adapted Imports ---

    # --- Import Unchanged Utilities from Original Evaluation Script ---
    # Assuming 'evaluate_id_original.py' contains these and is importable
    # Or, if these are in train_classification_original.py and still relevant
    #import train_classification as orig_utils # Reusing from training script for now
    # If you have a separate evaluate_id_original.py:
    import evaluate_id as orig_utils

    find_best_checkpoint = orig_utils.find_best_checkpoint # Assuming this func is in your original script
    calculate_sensitivity_at_specificity = orig_utils.calculate_sensitivity_at_specificity
    # ECE and Calibration plots are simplified/removed for this version
    # _calculate_and_plot_metrics_from_dataframe: Will be adapted locally
    # extract_features_from_model: Will be adapted locally
    # plot_tsne_features: Will be adapted locally
    # plot_combined_roc_with_ensemble: Will be adapted locally
    # create_summary_tables: Will be adapted locally

except ImportError as e:
    print(f"Error importing DANN modules or original utility script: {e}\n"
          "Please ensure DANN modules and 'train_classification_original.py' (or your utility script) are accessible.")
    sys.exit(1)
except AttributeError as e:
    print(f"AttributeError during import from original utility script: {e}. Ensure functions exist.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

DOMAIN_COLUMN_NAME: str = 'dataset_source' # Consistent with training

def get_eval_transforms(image_size: int, model_name_for_norm: Optional[str]) -> Compose:
    logger.info(f"Defining eval transforms: size={image_size}, norm_ref='{model_name_for_norm}'")
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # Default ImageNet
    if model_name_for_norm:
        try:
            config = resolve_data_config({}, model=timm.create_model(model_name_for_norm, pretrained=False, num_classes=0))
            if config and 'mean' in config and 'std' in config:
                mean, std = config["mean"], config["std"]
                logger.info(f"Using model-specific norm for '{model_name_for_norm}': mean={mean}, std={std}")
            else: logger.warning(f"Could not resolve norm for '{model_name_for_norm}'. Using defaults.")
        except Exception as e: logger.warning(f"Timm config error for '{model_name_for_norm}': {e}. Using defaults.")
    else: logger.info("No model_name_for_norm for transforms, using ImageNet defaults.")
    return Compose([Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS), ToTensor(), Normalize(mean=mean, std=std)])


# --- DANN Adapted Evaluation Functions ---

def _calculate_and_plot_metrics_from_dataframe_dann(df, model_display_name, eval_output_dir, device=None, save_plots=True, save_data=True):
    """Calculate metrics and create plots from predictions dataframe for DANN evaluation."""
    try:
        y_true = df['true_label'].values
        y_pred = df['predicted_label'].values
        y_prob = df['predicted_prob_class_1'].values

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        try:
            auc_val = roc_auc_score(y_true, y_prob)
            auc_str = f"{auc_val:.4f}"
        except ValueError as e:
            logger.warning(f"Could not calculate AUC for {model_display_name}: {e}")
            auc_val = np.nan
            auc_str = "N/A"

        # Fixed logging statement
        logger.info(f"{model_display_name} Overall - Acc: {accuracy:.4f}, AUC: {auc_str}")

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create ROC curve data if AUC is valid
        roc_data = {}
        if not np.isnan(auc_val):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': auc_val,
                'model_name': model_display_name
            }

        # Create and save plots if requested
        if save_plots:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Confusion Matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_title(f'{model_display_name} - Confusion Matrix')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
            
            # ROC Curve
            if not np.isnan(auc_val):
                axes[1].plot(fpr, tpr, label=f'{model_display_name} (AUC = {auc_val:.4f})')
                axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                axes[1].set_xlabel('False Positive Rate')
                axes[1].set_ylabel('True Positive Rate')
                axes[1].set_title(f'{model_display_name} - ROC Curve')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'ROC Curve\nN/A', ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title(f'{model_display_name} - ROC Curve (N/A)')
            
            plt.tight_layout()
            plot_path = os.path.join(eval_output_dir, f'{model_display_name.lower().replace(" ", "_")}_dann_metrics.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Metrics plot saved to: {plot_path}")

        # Return metrics and data
        metrics = {
            'accuracy': accuracy,
            'auc': auc_val,
            'confusion_matrix': cm.tolist(),
        }

        return metrics, roc_data

    except Exception as e:
        logger.error(f"Error in metrics/plotting for {model_display_name}: {e}", exc_info=True)
        return None, None


def evaluate_on_test_set_dann(model, test_loader, eval_output_dir, model_display_name, device, save_predictions=True, is_dann_model=False):
    """Evaluate DANN model on test set with comprehensive metrics."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    all_domains = []

    logger.info(f"Evaluating {model_display_name} (DANN model: {is_dann_model})...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(test_loader, desc=f"{model_display_name} Evaluation")):
            # DEBUG: Check what's in the batch
            if batch_idx == 0:
                logger.info(f"DEBUG: Batch contains {len(batch)} elements:")
                for i, item in enumerate(batch):
                    if isinstance(item, torch.Tensor):
                        logger.info(f"  batch[{i}]: Tensor shape {item.shape}, dtype {item.dtype}")
                    elif isinstance(item, (list, tuple)):
                        logger.info(f"  batch[{i}]: {type(item).__name__} with {len(item)} items")
                        if len(item) > 0:
                            logger.info(f"    First item: {type(item[0])} - {item[0] if len(str(item[0])) < 100 else str(item[0])[:100]+'...'}")
                    else:
                        logger.info(f"  batch[{i}]: {type(item)} - {item}")
            
            # Handle 4-element batch
            if len(batch) == 4:
                images, labels, domains, extra_info = batch
                # Log what the 4th element is (only for first batch)
                if batch_idx == 0:
                    logger.info(f"4th element type: {type(extra_info)}")
                    if isinstance(extra_info, torch.Tensor):
                        logger.info(f"4th element tensor shape: {extra_info.shape}")
                    elif isinstance(extra_info, (list, tuple)) and len(extra_info) > 0:
                        logger.info(f"4th element sample: {extra_info[0]}")
            elif len(batch) == 3:
                images, labels, domains = batch
            elif len(batch) == 2:
                images, labels = batch
                domains = torch.zeros_like(labels)
            else:
                logger.warning(f"Unexpected batch format with {len(batch)} elements")
                images, labels = batch[0], batch[1]
                domains = batch[2] if len(batch) > 2 else torch.zeros_like(labels)
            
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle DANN output format vs standard model
            if is_dann_model and isinstance(outputs, tuple):
                class_logits, domain_logits = outputs
                logger.debug(f"DANN model outputs: class_logits shape {class_logits.shape}, domain_logits shape {domain_logits.shape}")
            else:
                class_logits = outputs
            
            # Get predictions and probabilities
            probs = torch.softmax(class_logits, dim=1)
            predictions = torch.argmax(class_logits, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
            all_domains.extend(domains.numpy())

    # Create results dataframe
    df_results = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_predictions,
        'predicted_prob_class_1': all_probs,
        'domain_id': all_domains
    })

    # Save predictions if requested
    if save_predictions:
        results_path = os.path.join(eval_output_dir, f'{model_display_name.lower().replace(" ", "_")}_results_epoch5_dann.csv')
        df_results.to_csv(results_path, index=False)
        logger.info(f"Predictions for {model_display_name} saved to {os.path.basename(results_path)}")

    # Calculate metrics and create plots
    metrics, roc_data = _calculate_and_plot_metrics_from_dataframe_dann(
        df_results, model_display_name, eval_output_dir
    )

    return metrics, df_results, roc_data


def run_tsne_dann(model, data_loader, output_dir, exp_name, device, max_samples=1000):
    """Run t-SNE analysis on model features for DANN evaluation."""
    logger.info(f"Running t-SNE for {exp_name}...")
    
    model.eval()
    features_list = []
    labels_list = []
    domain_ids_list = []
    
    # Collect ALL samples first, then do stratified sampling
    all_features_collected = []
    all_labels_collected = []
    all_domains_collected = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Handle different batch formats
            if len(batch) == 2:
                images, labels = batch
                domain_ids = None
            elif len(batch) == 3:
                images, labels, domain_ids = batch
            elif len(batch) == 4:
                images, labels, domain_ids, _ = batch
            else:
                logger.warning(f"Unexpected batch format with {len(batch)} elements")
                continue
            
            images = images.to(device)
            
            # Extract features from the model
            if hasattr(model, 'feature_extractor'):
                features = model.feature_extractor(images)
            else:
                if hasattr(model, 'forward_features'):
                    features = model.forward_features(images)
                    
                    if hasattr(model, 'global_pool') and model.global_pool:
                        global_pool_type = getattr(model, 'global_pool', 'avg')
                        if isinstance(global_pool_type, str):
                            if global_pool_type == 'avg' and features.dim() > 2:
                                if features.dim() == 4:
                                    features = features.mean(dim=[-2, -1])
                                elif features.dim() == 3:
                                    features = features.mean(dim=1)
                            elif global_pool_type == 'token' and features.dim() == 3:
                                features = features[:, 0]
                            elif global_pool_type == 'max' and features.dim() > 2:
                                if features.dim() == 4:
                                    features = features.amax(dim=[-2, -1])
                                elif features.dim() == 3:
                                    features = features.amax(dim=1)
                        else:
                            features = global_pool_type(features)
                    elif features.dim() > 2:
                        if features.dim() == 4:
                            features = features.mean(dim=[-2, -1])
                        elif features.dim() == 3:
                            features = features.mean(dim=1)
                            
                elif hasattr(model, 'head') and hasattr(model, 'forward'):
                    try:
                        original_head = model.head
                        model.head = torch.nn.Identity()
                        features = model(images)
                        model.head = original_head
                        
                        if features.dim() > 2:
                            if features.dim() == 4:
                                features = features.mean(dim=[-2, -1])
                            elif features.dim() == 3:
                                features = features.mean(dim=1)
                                
                    except Exception as e:
                        if 'original_head' in locals():
                            model.head = original_head
                        logger.warning(f"Could not extract features by removing head: {e}")
                        features = model(images)
                else:
                    features = model(images)
            
            if isinstance(features, tuple):
                features = features[0]
            
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            elif features.dim() == 1:
                features = features.unsqueeze(0)
            
            # Store all samples
            all_features_collected.append(features.cpu().numpy())
            all_labels_collected.append(labels.cpu().numpy())
            
            if domain_ids is not None:
                all_domains_collected.append(domain_ids.cpu().numpy())
    
    if not all_features_collected:
        logger.error("No features extracted for t-SNE")
        return
    
    # Concatenate all collected data
    all_features = np.concatenate(all_features_collected, axis=0)
    all_labels = np.concatenate(all_labels_collected, axis=0)
    all_domains = np.concatenate(all_domains_collected, axis=0) if all_domains_collected else None
    
    logger.info(f"Collected {all_features.shape[0]} total samples with {all_features.shape[1]} features")
    
    # Stratified sampling to ensure representation from all domains
    if all_domains is not None and len(all_features) > max_samples:
        logger.info("Performing stratified sampling across domains for t-SNE...")
        
        unique_domains = np.unique(all_domains)
        logger.info(f"Found {len(unique_domains)} unique domains: {unique_domains}")
        
        # Calculate samples per domain (minimum 20 per domain if possible)
        min_samples_per_domain = max(10, max_samples // (len(unique_domains) * 2))
        remaining_samples = max_samples
        
        selected_indices = []
        
        for domain in unique_domains:
            domain_mask = all_domains == domain
            domain_indices = np.where(domain_mask)[0]
            
            # Sample from this domain
            n_samples_this_domain = min(len(domain_indices), 
                                      max(min_samples_per_domain, remaining_samples // (len(unique_domains) - len(selected_indices))))
            
            if n_samples_this_domain > 0:
                if len(domain_indices) > n_samples_this_domain:
                    domain_selected = np.random.choice(domain_indices, n_samples_this_domain, replace=False)
                else:
                    domain_selected = domain_indices
                
                selected_indices.extend(domain_selected)
                remaining_samples -= len(domain_selected)
                
                logger.info(f"Domain {domain}: selected {len(domain_selected)}/{len(domain_indices)} samples")
        
        # Convert to array and shuffle
        selected_indices = np.array(selected_indices)
        np.random.shuffle(selected_indices)
        
        # Apply selection
        all_features = all_features[selected_indices]
        all_labels = all_labels[selected_indices]
        all_domains = all_domains[selected_indices]
        
        logger.info(f"Stratified sampling: selected {len(selected_indices)} samples total")
        
        # Verify domain distribution after sampling
        unique_domains_after = np.unique(all_domains)
        for domain in unique_domains_after:
            count = np.sum(all_domains == domain)
            logger.info(f"After sampling - Domain {domain}: {count} samples")
            
    elif len(all_features) > max_samples:
        # Fallback: random sampling without stratification
        indices = np.random.choice(len(all_features), max_samples, replace=False)
        all_features = all_features[indices]
        all_labels = all_labels[indices]
        if all_domains is not None:
            all_domains = all_domains[indices]
        logger.info(f"Random sampling: subsampled to {max_samples} samples for t-SNE")
    
    # Run t-SNE
    logger.info("Running t-SNE dimensionality reduction...")
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
        tsne_features = tsne.fit_transform(all_features)
    except Exception as e:
        logger.error(f"t-SNE failed: {e}")
        return
    
    # Create plots
    fig_width = 15 if all_domains is not None else 12
    plt.figure(figsize=(fig_width, 5))
    
    # Plot 1: Colored by labels
    plt.subplot(1, 3 if all_domains is not None else 2, 1)
    unique_labels = np.unique(all_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = all_labels == label
        plt.scatter(tsne_features[mask, 0], tsne_features[mask, 1], 
                   c=[colors[i]], label=f'Class {label}', alpha=0.6, s=20)
    
    plt.title('t-SNE: Colored by Class Labels')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Colored by domains (if available)
    if all_domains is not None:
        plt.subplot(1, 3, 2)
        unique_domains = np.unique(all_domains)
        domain_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_domains)))
        
        for i, domain in enumerate(unique_domains):
            mask = all_domains == domain
            plt.scatter(tsne_features[mask, 0], tsne_features[mask, 1], 
                       c=[domain_colors[i]], label=f'Domain {domain}', alpha=0.6, s=20)
        
        plt.title('t-SNE: Colored by Domain')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Combined view
        plt.subplot(1, 3, 3)
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        for i, label in enumerate(unique_labels):
            for j, domain in enumerate(unique_domains):
                mask = (all_labels == label) & (all_domains == domain)
                if np.any(mask):
                    plt.scatter(tsne_features[mask, 0], tsne_features[mask, 1], 
                               c=[colors[i]], marker=markers[j % len(markers)],
                               label=f'C{label}_D{domain}', alpha=0.6, s=20)
        
        plt.title('t-SNE: Class + Domain')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, alpha=0.3)
    else:
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], 
                             c=all_labels, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Class')
        plt.title('t-SNE: Density View')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    tsne_path = os.path.join(output_dir, f"{exp_name.lower()}_tsne.png")
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"t-SNE plot saved to: {tsne_path}")
    
    # Save the t-SNE coordinates for further analysis
    tsne_data = {
        'tsne_x': tsne_features[:, 0],
        'tsne_y': tsne_features[:, 1],
        'labels': all_labels
    }
    
    if all_domains is not None:
        tsne_data['domain_ids'] = all_domains
    
    tsne_csv_path = os.path.join(output_dir, f"{exp_name.lower()}_tsne_coordinates.csv")
    pd.DataFrame(tsne_data).to_csv(tsne_csv_path, index=False)
    logger.info(f"t-SNE coordinates saved to: {tsne_csv_path}")


def main(args: argparse.Namespace):
    start_time = datetime.now()
    start_time_str = start_time.strftime('%Y%m%d_%H%M%S')
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if not os.path.isdir(args.parent_experiment_dir):
        logger.critical(f"Parent experiment directory not found: {args.parent_experiment_dir}"); sys.exit(1)

    all_models_roc_data = {}
    all_model_prediction_dfs = [] # For ensemble
    model_display_names_for_dfs = [] # For ensemble

    # Improved directory detection logic
    def find_experiment_directories(root_dir):
        """Recursively find directories that contain 'checkpoints' and 'results' subdirectories."""
        experiment_dirs = []
        
        def check_directory(path):
            contents = os.listdir(path)
            subdirs = [d for d in contents if os.path.isdir(os.path.join(path, d))]
            
            # Check if this directory has both checkpoints and results
            if 'checkpoints' in subdirs and 'results' in subdirs:
                experiment_dirs.append(path)
                return True
            
            # If not, recursively check subdirectories
            for subdir in subdirs:
                subdir_path = os.path.join(path, subdir)
                check_directory(subdir_path)
            
            return False
        
        check_directory(root_dir)
        return experiment_dirs
    
    # Find all experiment directories
    model_dirs = find_experiment_directories(args.parent_experiment_dir)
    
    if not model_dirs:
        logger.critical(f"No experiment directories with 'checkpoints' and 'results' found in {args.parent_experiment_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(model_dirs)} experiment director{'y' if len(model_dirs)==1 else 'ies'} to process:")
    for exp_dir in model_dirs:
        logger.info(f"  - {exp_dir}")

    for experiment_path in model_dirs:
        exp_name = os.path.basename(experiment_path)
        logger.info(f"\n{'='*10} Processing DANN Experiment: {exp_name} {'='*10}")

        eval_output_dir = os.path.join(experiment_path, "results", f"id_eval_dann_{start_time_str}")
        os.makedirs(eval_output_dir, exist_ok=True)

        # Setup file logging for this specific experiment's evaluation
        log_file_eval = os.path.join(eval_output_dir, f"log_eval_dann_{exp_name}.log")
        fh_eval = logging.FileHandler(log_file_eval, mode='w')
        fh_eval.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s-[%(module)s.%(funcName)s:%(lineno)d]-%(message)s"))
        logger.addHandler(fh_eval) # Add to the script's logger

        checkpoint_dir = os.path.join(experiment_path, "checkpoints")
        orig_results_dir = os.path.join(experiment_path, "results")

        # These directories should exist based on our search criteria
        if not os.path.isdir(checkpoint_dir):
            logger.warning(f"Checkpoints directory missing for {exp_name}. Skipping.")
            logger.removeHandler(fh_eval); fh_eval.close(); continue
        
        if not os.path.isdir(orig_results_dir):
            logger.warning(f"Results directory missing for {exp_name}. Skipping.")
            logger.removeHandler(fh_eval); fh_eval.close(); continue
        
        # Try to find DANN-specific checkpoint first
        checkpoint_path = find_best_checkpoint(checkpoint_dir) # find_best_checkpoint can be reused
        if checkpoint_path and "_dann.pth" not in checkpoint_path:
            logger.warning(f"Selected checkpoint '{os.path.basename(checkpoint_path)}' might not be a DANN model. Attempting to load as DANN if config suggests.")
            # Potentially, find_best_checkpoint could be adapted to prefer "_dann.pth" files.

        if not checkpoint_path:
            logger.error(f"No suitable checkpoint for {exp_name}. Skipping."); logger.removeHandler(fh_eval); fh_eval.close(); continue
        logger.info(f"Selected checkpoint: {os.path.basename(checkpoint_path)}")
        
        epoch_str = "unknown_epoch"
        if "_epoch" in os.path.basename(checkpoint_path):
            try: epoch_str = os.path.basename(checkpoint_path).split('_epoch')[-1].split('_dann.pth')[0].split('.pth')[0]
            except: pass


        # Load training configuration (expecting DANN config)
        train_config_path = os.path.join(orig_results_dir, f"config_DANN.json") # Check for DANN specific
        if not os.path.exists(train_config_path):
             train_config_path = os.path.join(orig_results_dir, f"training_configuration_dann.json") # Older naming
        if not os.path.exists(train_config_path):
             train_config_path = os.path.join(orig_results_dir, f"training_configuration.json") # Fallback
        
        if not os.path.exists(train_config_path):
            logger.error(f"Training config missing for {exp_name} (tried DANN and generic names). Skipping."); logger.removeHandler(fh_eval); fh_eval.close(); continue

        try:
            with open(train_config_path, 'r') as f: train_args_dict = json.load(f)
            # Extract necessary args for model building and data loading
            model_arch = train_args_dict.get('model_name', args.fallback_model_name)
            img_size = train_args_dict.get('image_size', args.fallback_image_size)
            dropout = train_args_dict.get('dropout_prob', args.fallback_dropout_prob)
            num_cls = train_args_dict.get('num_classes', 2) # Should be 2 for glaucoma
            
            # DANN specific args from training config
            was_dann_trained = train_args_dict.get('use_dann', False) # Was DANN used during training?
            num_doms_train = train_args_dict.get('num_domains', 1)
            grl_lambda_train = train_args_dict.get('dann_grl_base_lambda', 1.0)
            grl_sched_train = train_args_dict.get('dann_lambda_schedule_mode', "paper")
            grl_gamma_train = train_args_dict.get('dann_gamma', 10.0)
            
            # Get domain_id_map from checkpoint if available, else None
            # This is important if the test set might have domains not seen in training's auto-generated map
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            domain_id_map_from_ckpt = checkpoint_data.get('domain_id_map', None)
            if domain_id_map_from_ckpt:
                logger.info(f"Loaded domain_id_map from checkpoint: {domain_id_map_from_ckpt}")
            num_doms_from_ckpt = checkpoint_data.get('num_domains', num_doms_train)


            model_disp_name = train_args_dict.get('experiment_tag', exp_name.split('_')[-2] if '_' in exp_name else model_arch)
            if not model_disp_name or model_disp_name == "DANN": model_disp_name = f"{model_arch}_DANN" if was_dann_trained else model_arch

        except Exception as e:
            logger.error(f"Error loading/parsing config for {exp_name}: {e}. Skipping.", exc_info=True)
            logger.removeHandler(fh_eval); fh_eval.close(); continue

        # Build DANN model for evaluation
        # Pass use_dann=True if it was DANN trained, so build_model_dann constructs the DANN shell.
        # The actual domain classification head won't be used for loss.
        model = build_classifier_model_dann(
            model_name=model_arch, num_classes=num_cls, dropout_prob=dropout,
            pretrained=False, # Always load from checkpoint for eval
            custom_weights_path=checkpoint_path, # Path to the saved DANN model
            checkpoint_key=train_args_dict.get('checkpoint_key', 'model_state_dict'), # Ensure key is correct
            use_dann=was_dann_trained, # Build as DANN model if it was trained as such
            num_domains=num_doms_from_ckpt, # Use num_domains from checkpoint/config
            dann_grl_base_lambda=grl_lambda_train, # These are mostly for structure, not active in eval
            dann_lambda_schedule_mode=grl_sched_train,
            dann_gamma=grl_gamma_train
        )
        model.to(device).eval()

        # Load Test Set Manifest
        # The manifest is from the original training run.
        test_manifest_path = os.path.join(orig_results_dir, "test_set_manifest_dann.csv")
        if not os.path.exists(test_manifest_path):
            test_manifest_path = os.path.join(orig_results_dir, "test_set_manifest.csv") # Fallback
        if not os.path.exists(test_manifest_path):
            logger.warning(f"Test manifest missing for {exp_name}. Skipping eval."); logger.removeHandler(fh_eval); fh_eval.close(); continue
        
        try:
            df_test = pd.read_csv(test_manifest_path)
            if not ({'image_path', 'types', DOMAIN_COLUMN_NAME}.issubset(df_test.columns)) or df_test.empty: # Ensure DOMAIN_COLUMN_NAME is present
                logger.error(f"Test manifest invalid for {exp_name}. Need 'image_path', 'types', '{DOMAIN_COLUMN_NAME}'. Skipping."); logger.removeHandler(fh_eval); fh_eval.close(); continue
        except Exception as e:
            logger.error(f"Error loading test manifest for {exp_name}: {e}. Skipping.", exc_info=True); logger.removeHandler(fh_eval); fh_eval.close(); continue

        eval_tf = get_eval_transforms(img_size, model_arch)
        # GlaucomaSubgroupDANNDataset needs domain_id_map. If test set might have new domains not in map from checkpoint,
        # the dataset will handle them (e.g., assign -1 or create a dynamic map if map is None).
        # It's safer to pass the map from training if available.
        test_ds = GlaucomaSubgroupDANNDataset(
            df_test, transform=eval_tf, 
            sensitive_attributes=[DOMAIN_COLUMN_NAME, 'camera', 'image_path'], # Include image_path for metadata
            domain_column=DOMAIN_COLUMN_NAME,
            domain_id_map=domain_id_map_from_ckpt, # Use map from checkpoint
            label_col='types', path_col='image_path'
        )
        if len(test_ds) == 0: 
            logger.warning(f"Test dataset for {exp_name} empty after init. Skipping."); logger.removeHandler(fh_eval); fh_eval.close(); continue
        
        test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                                 collate_fn=safe_collate_dann, persistent_workers=(args.num_workers > 0))

        metrics_summary, df_preds, roc_data = evaluate_on_test_set_dann(
            model, test_loader, eval_output_dir, exp_name, device, 
            save_predictions=True
        )
        
        if roc_data: all_models_roc_data[model_disp_name] = roc_data
        if df_preds is not None and not df_preds.empty:
            all_model_prediction_dfs.append(df_preds)
            model_display_names_for_dfs.append(model_disp_name)

        # t-SNE (optional, adapted for DANN)
        if args.run_tsne:
            logger.info(f"Running t-SNE for {model_disp_name} (DANN eval)...")
            try:
                run_tsne_dann(model, test_loader, eval_output_dir, exp_name, device)
            except Exception as e:
                logger.error(f"Error during t-SNE for {exp_name}: {e}", exc_info=True)
        

        logger.removeHandler(fh_eval); fh_eval.close()
    # --- End of loop over model_dirs ---

    logger.info(f"\n{'='*5} All DANN Evaluations Finished. Total time: {datetime.now() - start_time} {'='*5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DANN-trained models on in-distribution test sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--parent_experiment_dir', type=str, required=True, help="Path to single DANN experiment or parent of multiple.")
    parser.add_argument('--fallback_model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--fallback_image_size', type=int, default=224)
    parser.add_argument('--fallback_dropout_prob', type=float, default=0.1)
    parser.add_argument('--eval_batch_size', type=int, default=32) # Adjusted
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot_roc_per_source', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--roc_min_samples_per_source', type=int, default=30) # Lowered for potentially smaller subgroups

    viz_group = parser.add_argument_group("Visualization Options (Simplified)")
    viz_group.add_argument('--run_tsne', action=argparse.BooleanOptionalAction, default=True, help="Run t-SNE (requires adaptation for DANN features).")
    # Args for t-SNE params if enabled
    viz_group.add_argument('--tsne_perplexity', type=float, default=30.0)
    viz_group.add_argument('--tsne_n_iter', type=int, default=1000)
    viz_group.add_argument('--tsne_learning_rate', type=str, default='auto')
    viz_group.add_argument('--tsne_max_samples_plot', type=int, default=1500)
    viz_group.add_argument('--tsne_min_samples_per_source', type=int, default=50)
    # GradCAM and extensive ensemble/calibration tables are removed for this simplified version

    args = parser.parse_args()
    
    # Update BASE_DATA_DIR_CONFIG_KEY in the imported module if its functions use it
    # This is tricky if functions in orig_utils are not designed to accept it.
    # For functions like adjust_path_for_data_type, it's better if they take base_dir as param.
    # For functions that rely on a global BASE_DATA_DIR_CONFIG_KEY, we can set it here if needed.
    # For now, assuming it's not needed by the few imported utils or they handle it.
    # orig_utils.BASE_DATA_DIR_CONFIG_KEY = args.base_data_root # This would only work if base_data_root was an arg here

    main(args)