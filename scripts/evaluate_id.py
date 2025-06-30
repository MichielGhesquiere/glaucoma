#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In-Distribution Test Set Evaluation Script.

Evaluates pre-trained models from experiment directories on their
in-distribution test sets. It produces individual ROC plots per model,
a final combined ROC plot, and optionally performs t-SNE and Grad-CAM.

New: Includes functionality to create and evaluate an ensemble classifier
by averaging probabilities from all processed models, plus comprehensive
summary tables with AUC and Sensitivity at 95% Specificity.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
import glob
from functools import reduce # For merging DataFrames
from contextlib import contextmanager


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve, accuracy_score
from sklearn.manifold import TSNE
import seaborn as sns
from timm.data import resolve_data_config
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
import torchvision.transforms as transforms

# Ensure custom modules can be imported by adding the parent directory of 'src' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.utils.helpers import NpEncoder, set_seed
    from src.utils.file_utils import adjust_path_for_data_type, handle_processed_image_suffixes, find_best_checkpoint
    from src.evaluation.metrics import calculate_ece, calculate_sensitivity_at_specificity, calculate_metrics_from_predictions
    from src.evaluation.callibration import plot_calibration_curve, calculate_ece_from_dataframe
    from src.evaluation.visualization import plot_roc_curves, plot_combined_roc_with_ensemble
    from src.data.datasets import GlaucomaSubgroupDataset, safe_collate
    from src.models.classification.build_model import build_classifier_model
    from src.utils.gradcam_utils import visualize_gradcam_misclassifications #, get_default_target_layers
except ImportError as e:
    print(f"Error importing custom modules: {e}\n"
          "Please ensure that the 'src' directory is in your PYTHONPATH or "
          "that the script is run from a location where 'src' is a subdirectory.")
    sys.exit(1)

# Configure logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)], # Log to console
)
logger = logging.getLogger(__name__) # Get a logger instance for this module

@contextmanager
def file_logging_context(log_file_path: str, logger_instance):
    """Context manager for safe file logging setup and cleanup."""
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s"))
    logger_instance.addHandler(file_handler)
    
    try:
        yield file_handler
    finally:
        try:
            logger_instance.removeHandler(file_handler)
            file_handler.close()
        except (OSError, ValueError, AttributeError):
            # Silently handle cleanup issues
            pass

def get_eval_transforms(image_size: int, model_name_for_norm: str | None) -> Compose:
    """
    Creates evaluation image transformations.

    Uses mean and std from the specified timm model's data_config if available,
    otherwise defaults to ImageNet statistics.

    Args:
        image_size: The target image size (height and width).
        model_name_for_norm: The name of the timm model to derive normalization from.
                             If None, ImageNet defaults are used.

    Returns:
        A torchvision.transforms.Compose object for evaluation.
    """
    logger.info(f"Defining evaluation transforms: image_size={image_size}, normalization_ref_model='{model_name_for_norm}'")
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # Default ImageNet

    if model_name_for_norm:
        try:
            config = resolve_data_config({}, model=timm.create_model(model_name_for_norm, pretrained=False, num_classes=0))
            if config and 'mean' in config and 'std' in config:
                mean, std = config["mean"], config["std"]
                logger.info(f"Using model-specific normalization for '{model_name_for_norm}': mean={mean}, std={std}")
            else:
                logger.warning(f"Could not resolve specific mean/std for '{model_name_for_norm}'. Using ImageNet defaults.")
        except Exception as e:
            logger.warning(f"Timm config error for '{model_name_for_norm}': {e}. Using ImageNet defaults.")
    else:
        logger.info("No model_name_for_norm provided for transforms, using ImageNet defaults.")

    return Compose([
        Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS), # Added interpolation
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])


# Update the _calculate_and_plot_metrics_from_dataframe function
def _calculate_and_plot_metrics_from_dataframe(
    df_predictions: pd.DataFrame,
    dataset_name_suffix: str,
    results_dir_path: str,
    epoch_num_str: str, # Can be 'Ensemble' or epoch number
    plot_per_source_roc: bool = True,
    roc_min_samples_per_source: int = 50,
    calculate_calibration: bool = True  # Add this parameter
) -> tuple[dict | None, dict | None]:
    """
    Calculates metrics (Acc, AUC, ECE) and plots ROC from a DataFrame of predictions.

    Args:
        df_predictions: DataFrame with 'label', 'probability_class1', 'dataset_source', 'image_path'.
        dataset_name_suffix: Suffix for logs and filenames.
        results_dir_path: Directory to save results.
        epoch_num_str: Epoch identifier for filenames.
        plot_per_source_roc: If True, plot ROC per 'dataset_source'.
        roc_min_samples_per_source: Min samples for per-source ROC.
        calculate_calibration: If True, calculate ECE and generate calibration plots.

    Returns:
        Tuple: (metrics_summary, overall_roc_data_for_combined_plot).
               Returns (None, None) on error.
    """
    if df_predictions.empty:
        logger.warning(f"Prediction DataFrame for {dataset_name_suffix} is empty. Skipping metrics calculation.")
        return None, None

    # Save raw predictions (if not already saved, e.g., for ensemble)
    if "Ensemble" in dataset_name_suffix : # Save for ensemble
        results_csv_filename = f"{dataset_name_suffix.lower().replace(' ', '_')}_results.csv"
        df_predictions.to_csv(os.path.join(results_dir_path, results_csv_filename), index=False)
        logger.info(f"Ensemble prediction results saved to {results_csv_filename}")


    metrics_summary = {}
    overall_roc_data_for_combined_plot = None

    try:
        # Calculate ECE if requested
        ece_results = {}
        if calculate_calibration:
            calibration_output_dir = os.path.join(results_dir_path, "calibration_plots")
            os.makedirs(calibration_output_dir, exist_ok=True)
            ece_results = calculate_ece_from_dataframe(
                df_predictions, dataset_name_suffix, calibration_output_dir, epoch_num_str
            )
        
        # Overall metrics
        if len(np.unique(df_predictions["label"])) < 2:
            logger.warning(f"Overall {dataset_name_suffix} has only one class. AUC cannot be computed.")
            accuracy = accuracy_score(df_predictions["label"], df_predictions["probability_class1"] > 0.5) if len(df_predictions["label"]) > 0 else np.nan
            auc_value = np.nan
            fpr_overall, tpr_overall = None, None
        else:
            accuracy = accuracy_score(df_predictions["label"], df_predictions["probability_class1"] > 0.5)
            fpr_overall, tpr_overall, _ = roc_curve(df_predictions["label"], df_predictions["probability_class1"])
            auc_value = auc(fpr_overall, tpr_overall)
            overall_roc_data_for_combined_plot = {'fpr': fpr_overall, 'tpr': tpr_overall, 'auc': auc_value}

        auc_log_str = f"{auc_value:.4f}" if not np.isnan(auc_value) else "N/A"
        ece_log_str = f", ECE: {ece_results.get('overall_ece', 'N/A'):.4f}" if ece_results.get('overall_ece') is not None else ""
        logger.info(f"{dataset_name_suffix} Overall - Accuracy (0.5 thresh): {accuracy:.4f}, AUC: {auc_log_str}{ece_log_str}")
        
        metrics_summary["overall"] = {
            "accuracy": accuracy, 
            "auc": auc_value, 
            "num_samples": len(df_predictions)
        }
        
        # Add ECE to metrics summary
        if ece_results:
            metrics_summary["overall"]["ece"] = ece_results.get('overall_ece')
            if ece_results.get('per_source_ece'):
                metrics_summary["ece_per_source"] = ece_results['per_source_ece']

        # Plot ROC
        plt.figure(figsize=(12, 10))
        if fpr_overall is not None and tpr_overall is not None:
            plot_label = f'Overall {dataset_name_suffix} (AUC={auc_value:.3f})' if not np.isnan(auc_value) else f'Overall {dataset_name_suffix} (AUC=N/A)'
            plt.plot(fpr_overall, tpr_overall, color='black', lw=2.5, label=plot_label)

        if plot_per_source_roc and 'dataset_source' in df_predictions.columns:
            unique_sources = sorted([s for s in df_predictions["dataset_source"].astype(str).unique() if s.lower() not in ['nan', 'unknown_source', 'none']])
            if unique_sources:
                cmap_name = 'tab10' if len(unique_sources) <= 10 else 'tab20'
                colors_for_sources = plt.colormaps.get_cmap(cmap_name).resampled(len(unique_sources))
                for i, src_name in enumerate(unique_sources):
                    src_df = df_predictions[df_predictions["dataset_source"].astype(str) == src_name]
                    if len(src_df) < roc_min_samples_per_source or len(src_df["label"].unique()) < 2:
                        logger.info(f"Skipping ROC for source '{src_name}' (N={len(src_df)}), criteria not met.")
                        continue
                    fpr_s, tpr_s, _ = roc_curve(src_df["label"], src_df["probability_class1"])
                    auc_s = auc(fpr_s, tpr_s)
                    src_accuracy = accuracy_score(src_df["label"], src_df["probability_class1"] > 0.5)
                    metrics_summary[src_name] = {"accuracy": src_accuracy, "auc": auc_s, "num_samples": len(src_df)}
                    plt.plot(fpr_s, tpr_s, color=colors_for_sources(i), lw=1.5, linestyle='--',
                             label=f'{src_name} (AUC={auc_s:.3f}, N={len(src_df)})')

        plt.plot([0, 1], [0, 1], 'k:', lw=1)
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)"); plt.ylabel("True Positive Rate (TPR)")
        plot_title_suffix = f" (Epoch {epoch_num_str})" if "Ensemble" not in dataset_name_suffix else ""
        plt.title(f'{dataset_name_suffix} ROC Curves{plot_title_suffix}')
        plt.legend(loc="lower right", fontsize='medium'); plt.grid(True, alpha=0.7)
        roc_plot_filename_suffix = f"_epoch{epoch_num_str}" if "Ensemble" not in dataset_name_suffix else ""
        roc_plot_filename = f'{dataset_name_suffix.lower().replace(" ", "_")}_roc{roc_plot_filename_suffix}.png'
        plt.savefig(os.path.join(results_dir_path, roc_plot_filename))
        plt.close()
        logger.info(f"ROC plot for {dataset_name_suffix} saved to {roc_plot_filename}")

        metrics_json_filename_suffix = f"_epoch{epoch_num_str}" if "Ensemble" not in dataset_name_suffix else ""
        metrics_json_filename = f'{dataset_name_suffix.lower().replace(" ", "_")}_metrics{metrics_json_filename_suffix}.json'
        with open(os.path.join(results_dir_path, metrics_json_filename), 'w') as f_json:
            json.dump(metrics_summary, f_json, indent=4, cls=NpEncoder)
        logger.info(f"Metrics summary for {dataset_name_suffix} saved to {metrics_json_filename}")

        return metrics_summary, overall_roc_data_for_combined_plot
    except Exception as e:
        logger.error(f"Error in metrics/plotting for {dataset_name_suffix}: {e}", exc_info=True)
        return None, None


def evaluate_on_test_set(
    model_to_eval: nn.Module,
    loader: DataLoader,
    dataset_obj: GlaucomaSubgroupDataset,
    dataset_name_suffix: str,
    results_dir_path: str,
    epoch_num_str: str,
    device_to_use: torch.device,
    plot_per_source_roc: bool = True,
    roc_min_samples_per_source: int = 50
) -> tuple[dict | None, pd.DataFrame, dict | None]:
    """
    Evaluates a model on a test DataLoader.

    Returns:
        Tuple: (metrics_summary, df_predictions, overall_roc_data_for_combined_plot).
    """
    if not loader or not dataset_obj or len(dataset_obj) == 0:
        logger.info(f"Skipping evaluation for {dataset_name_suffix} (loader or dataset is empty).")
        return None, pd.DataFrame(), None

    logger.info(f"\n--- Evaluating on {dataset_name_suffix} ({len(dataset_obj)} samples) ---")
    model_to_eval.eval()
    all_labels, all_probabilities, all_sources_eval, all_image_paths_eval = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{dataset_name_suffix} Evaluation Progress"):
            if batch is None: continue
            inputs, labels, metadata_dict = batch[0], batch[1], batch[2] if len(batch) > 2 else {}

            img_paths_batch = metadata_dict.get('image_path', ['Unknown_Path'] * len(labels))
            img_paths_batch = [str(p) for p in (img_paths_batch if isinstance(img_paths_batch, list) else [img_paths_batch] * len(labels))]
            all_image_paths_eval.extend(img_paths_batch)

            inputs = inputs.to(device_to_use)
            outputs = model_to_eval(inputs)
            probabilities = F.softmax(outputs, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            sources_batch = metadata_dict.get('dataset_source', ['Unknown_Source'] * len(labels))
            sources_batch = [str(s) for s in (sources_batch if isinstance(sources_batch, list) else [sources_batch] * len(labels))]
            all_sources_eval.extend(sources_batch)

    if not all_labels:
        logger.warning(f"No labels collected for {dataset_name_suffix}. Skipping results processing.")
        return None, pd.DataFrame(), None

    df_predictions = pd.DataFrame({
        "image_path": all_image_paths_eval,
        "label": all_labels,
        "probability_class1": all_probabilities,
        "dataset_source": all_sources_eval
    })
    # Save raw results to CSV for the individual model
    results_csv_filename = f"{dataset_name_suffix.lower().replace(' ', '_')}_results_epoch{epoch_num_str}.csv"
    df_predictions.to_csv(os.path.join(results_dir_path, results_csv_filename), index=False)
    logger.info(f"Individual model results for {dataset_name_suffix} saved to {results_csv_filename}")

    metrics_summary, overall_roc_data = _calculate_and_plot_metrics_from_dataframe(
        df_predictions, dataset_name_suffix, results_dir_path, epoch_num_str,
        plot_per_source_roc, roc_min_samples_per_source
    )
    return metrics_summary, df_predictions, overall_roc_data


def extract_features_from_model(model: nn.Module, dataloader: DataLoader, device: torch.device,
                                model_name_from_training: str) -> tuple[np.ndarray | None, np.ndarray | None, list | None, list | None]:
    """
    Extracts features from a model using a specified dataloader.
    (Implementation largely unchanged from the original, with minor logging/robustness updates)
    """
    model.eval()
    features_list, labels_list, dataset_source_list, image_paths_list = [], [], [], []
    hook_target_layer, original_head_module, original_head_attr_name = None, None, None
    model_name_lower = model_name_from_training.lower()
    logger.info(f"Setting up feature extraction for model: {model_name_from_training}")

    # Heuristics for feature extraction layer
    if 'dinov2' in model_name_lower:
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'norm'): hook_target_layer = model.backbone.norm
        elif hasattr(model, 'norm'): hook_target_layer = model.norm
    elif 'vit' in model_name_lower:
        if hasattr(model, 'norm') and isinstance(model.norm, nn.LayerNorm): hook_target_layer = model.norm
        elif hasattr(model, 'fc_norm') and isinstance(model.fc_norm, nn.LayerNorm): hook_target_layer = model.fc_norm
        elif hasattr(model, 'head') and isinstance(model.head, (nn.Linear, nn.Sequential)):
            original_head_module, original_head_attr_name, model.head = model.head, 'head', nn.Identity()
            hook_target_layer = model
    elif any(k in model_name_lower for k in ['resnet', 'convnext', 'efficientnet']):
        if hasattr(model, 'global_pool') and not isinstance(model.global_pool, nn.Identity): hook_target_layer = model.global_pool
        elif hasattr(model, 'avgpool') and not isinstance(model.avgpool, nn.Identity): hook_target_layer = model.avgpool
        elif hasattr(model, 'get_classifier'):
            classifier = model.get_classifier()
            if isinstance(classifier, (nn.Linear, nn.Sequential)):
                original_head_module = classifier
                head_attr = model.default_cfg.get('classifier', 'fc') if hasattr(model, 'default_cfg') else 'head'
                if hasattr(model, head_attr):
                    original_head_attr_name = head_attr
                    setattr(model, original_head_attr_name, nn.Identity())
                    hook_target_layer = model
    if hook_target_layer: logger.info(f"Using layer '{hook_target_layer.__class__.__name__}' or Identity head for features.")

    if not hook_target_layer and not original_head_module:
        head_attr = model.default_cfg.get('classifier', 'head') if hasattr(model, 'default_cfg') else 'head'
        if hasattr(model, head_attr) and isinstance(getattr(model, head_attr), (nn.Linear, nn.Sequential)):
            original_head_module, original_head_attr_name = getattr(model, head_attr), head_attr
            setattr(model, head_attr, nn.Identity())
            hook_target_layer = model # Model itself is target
            logger.info(f"Fallback: Replaced '{head_attr}' with nn.Identity for {model_name_from_training}.")
        else:
            logger.error(f"Failed to set up feature extraction for {model_name_from_training}. Skipping."); return None, None, None, None

    extracted_features_hook = []
    def _hook_fn(module, input_val, output_val):
        features = output_val[0] if isinstance(output_val, tuple) else output_val
        detached = features.detach().cpu()
        extracted_features_hook.append(detached.view(detached.size(0), -1) if detached.ndim > 2 else detached)

    hook_handle = None
    if hook_target_layer is not model: hook_handle = hook_target_layer.register_forward_hook(_hook_fn)

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"Extracting Features ({model_name_from_training})"):
            if batch_data is None: continue
            inputs, labels, metadata = batch_data[0], batch_data[1], batch_data[2] if len(batch_data) > 2 else {}
            inputs = inputs.to(device)
            extracted_features_hook.clear()
            output = model(inputs)

            batch_features = torch.cat(extracted_features_hook, dim=0) if extracted_features_hook and hook_target_layer is not model else \
                             (output.detach().cpu().view(output.size(0), -1) if output.ndim > 2 else output.detach().cpu())
            if batch_features is not None:
                features_list.append(batch_features)
                labels_list.append(labels.cpu())
                ds_src = [str(s) for s in (metadata.get('dataset_source', ['N/A']*len(labels)) if isinstance(metadata.get('dataset_source'), list) else [metadata.get('dataset_source', 'N/A')]*len(labels))]
                dataset_source_list.extend(ds_src)
                img_paths = [str(p) for p in (metadata.get('image_path', ['N/A']*len(labels)) if isinstance(metadata.get('image_path'), list) else [metadata.get('image_path', 'N/A')]*len(labels))]
                image_paths_list.extend(img_paths)
            else: logger.warning("No features captured for a batch.")

    if hook_handle: hook_handle.remove()
    if original_head_module and original_head_attr_name:
        setattr(model, original_head_attr_name, original_head_module)
        logger.info(f"Restored original head '{original_head_attr_name}' for {model_name_from_training}.")

    if not features_list: logger.warning(f"No features extracted for {model_name_from_training}."); return None, None, None, None
    return torch.cat(features_list).numpy(), torch.cat(labels_list).numpy(), dataset_source_list, image_paths_list


def plot_tsne_features(features: np.ndarray, labels: np.ndarray, dataset_sources: list,
                       output_dir: str, filename_prefix: str,
                       perplexity: float = 30.0, n_iter: int = 1000, learning_rate: str | float = 'auto',
                       max_samples_plot: int = 2000, min_samples_per_source_for_tsne: int = 100): # Adjusted min_samples
    """
    Generates and saves a t-SNE plot. (Minor adjustments from original for robustness)
    """
    if features is None or len(features) == 0: logger.warning("No features for t-SNE. Skipping."); return

    df_tsne = pd.DataFrame({'label': labels, 'dataset_source': [str(s) for s in dataset_sources]})
    feature_cols = [f'feature_{i}' for i in range(features.shape[1])]
    df_features = pd.DataFrame(features, columns=feature_cols)
    df_tsne = pd.concat([df_tsne, df_features], axis=1)

    source_counts = df_tsne['dataset_source'].value_counts()
    sources_to_keep = source_counts[source_counts >= min_samples_per_source_for_tsne].index
    if not sources_to_keep.any(): logger.warning(f"No source meets min samples ({min_samples_per_source_for_tsne}) for t-SNE. Skipping."); return
    df_filtered = df_tsne[df_tsne['dataset_source'].isin(sources_to_keep)].copy()
    if df_filtered.empty: logger.warning("Empty DataFrame after t-SNE filtering. Skipping."); return

    features_f, labels_f, sources_f = df_filtered[feature_cols].values, df_filtered['label'].values, df_filtered['dataset_source'].tolist()
    logger.info(f"Running t-SNE on {len(features_f)} samples (filtered).")

    if len(features_f) > max_samples_plot:
        idx = np.random.choice(len(features_f), max_samples_plot, replace=False)
        features_p, labels_p, sources_p = features_f[idx], labels_f[idx], [sources_f[i] for i in idx]
    else:
        features_p, labels_p, sources_p = features_f, labels_f, sources_f

    actual_perp = float(perplexity)
    if len(features_p) <= actual_perp: actual_perp = max(5.0, float(len(features_p) - 1.0))
    if len(features_p) < 5: logger.warning(f"Too few samples ({len(features_p)}) for t-SNE. Skipping."); return

    lr_val = 'auto' if isinstance(learning_rate, str) and learning_rate.lower() == 'auto' else float(learning_rate)
    tsne = TSNE(n_components=2, perplexity=actual_perp, n_iter=int(n_iter), learning_rate=lr_val,
                random_state=42, init='pca', n_jobs=-1)
    try: embeddings = tsne.fit_transform(features_p)
    except Exception as e: logger.error(f"t-SNE failed: {e}", exc_info=True); return

    df_plot = pd.DataFrame({'tsne1': embeddings[:,0], 'tsne2': embeddings[:,1], 'label': labels_p, 'source': sources_p})
    plt.figure(figsize=(16, 12))
    unique_src_plot = sorted(df_plot['source'].unique())
    palette = sns.color_palette("tab20" if len(unique_src_plot) > 10 else "tab10", n_colors=len(unique_src_plot))
    markers = {0: "o", 1: "X"}

    for i, src_name in enumerate(unique_src_plot):
        src_data = df_plot[df_plot['source'] == src_name]
        for lbl_val, marker in markers.items():
            lbl_data = src_data[src_data['label'] == lbl_val]
            if not lbl_data.empty:
                plt.scatter(lbl_data['tsne1'], lbl_data['tsne2'],
                            label=f"{src_name} - {'Glaucoma' if lbl_val else 'Normal'} (N={len(lbl_data)})",
                            alpha=0.7, color=palette[i % len(palette)], marker=marker, s=60)

    plt.title(f't-SNE (Filtered Sources, N={len(features_p)})\nModel: {filename_prefix}', fontsize=16)
    plt.xlabel('t-SNE Dim 1'); plt.ylabel('t-SNE Dim 2')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Dataset - Label (Count)")
    plt.grid(True, alpha=0.5, linestyle=':'); plt.tight_layout(rect=[0, 0, 0.82, 1])
    os.makedirs(output_dir, exist_ok=True)
    tsne_path = os.path.join(output_dir, f"{filename_prefix.replace(' ', '_')}_tsne_filtered.png")
    try: plt.savefig(tsne_path); logger.info(f"t-SNE plot saved to: {tsne_path}")
    except Exception as e: logger.error(f"Failed to save t-SNE plot: {e}", exc_info=True)
    plt.close()

def process_and_evaluate_ensemble(
    list_of_model_prediction_dfs: list[pd.DataFrame],
    model_display_names_for_dfs: list[str],
    output_dir_base: str, # Base directory for saving ensemble results
    timestamp_str: str,
    args: argparse.Namespace
) -> tuple[dict | None, pd.DataFrame]:
    """
    Processes predictions from multiple models to create and evaluate an ensemble.
    
    Returns:
        Tuple: (ensemble_roc_data, ensemble_df_for_summary_tables)
    """
    if len(list_of_model_prediction_dfs) < 2:
        logger.warning("Ensemble evaluation requires at least 2 models. Skipping ensemble.")
        return None, pd.DataFrame()

    logger.info(f"\n{'='*10} Processing Ensemble Predictions ({len(list_of_model_prediction_dfs)} models) {'='*10}")

    # Prepare DataFrames for merging: set index to 'image_path' and rename prob columns
    prepared_dfs_for_ensemble = []
    for i, df_model in enumerate(list_of_model_prediction_dfs):
        model_name = model_display_names_for_dfs[i]
        # Select essential columns and rename probability column uniquely for this model
        df_renamed = df_model[['image_path', 'label', 'dataset_source', 'probability_class1']].copy()
        df_renamed.rename(columns={'probability_class1': f'prob_{model_name}'}, inplace=True)
        df_renamed.set_index('image_path', inplace=True)
        prepared_dfs_for_ensemble.append(df_renamed)

    # Merge all DataFrames using an outer join on 'image_path' (index)
    # The first df's label and dataset_source will be primary, others will get suffixes if they differ (should not for ID)
    ensemble_df = reduce(lambda left, right: pd.merge(left, right.drop(columns=['label', 'dataset_source'], errors='ignore'), 
                                                     left_index=True, right_index=True, how='outer'),
                         prepared_dfs_for_ensemble)

    # Clean up label and dataset_source columns (pick the first non-suffixed ones)
    # This assumes 'label' and 'dataset_source' are consistent for each image across manifests
    if 'label_x' in ensemble_df.columns: # From merge if not dropped
        ensemble_df['label'] = ensemble_df['label_x']
        ensemble_df.drop(columns=[col for col in ensemble_df.columns if 'label_' in col], inplace=True)
    if 'dataset_source_x' in ensemble_df.columns:
        ensemble_df['dataset_source'] = ensemble_df['dataset_source_x']
        ensemble_df.drop(columns=[col for col in ensemble_df.columns if 'dataset_source_' in col], inplace=True)

    # Identify all individual model probability columns
    prob_cols = [col for col in ensemble_df.columns if col.startswith('prob_')]

    # Calculate ensemble probability (average)
    # NaNs in individual model probs (if an image wasn't processed by a model) will be skipped by mean()
    ensemble_df['num_ensemble_models_contributed'] = ensemble_df[prob_cols].notna().sum(axis=1)
    ensemble_df['ensemble_probability_class1'] = ensemble_df[prob_cols].mean(axis=1, skipna=True)

    # Filter out rows where ensemble probability could not be computed (e.g., image missing in all models)
    # or where essential info like label is missing.
    ensemble_df.dropna(subset=['ensemble_probability_class1', 'label', 'dataset_source'], inplace=True)
    ensemble_df.reset_index(inplace=True) # Bring 'image_path' back to a column

    if ensemble_df.empty:
        logger.warning("Ensemble DataFrame is empty after merging and processing. Skipping ensemble evaluation.")
        return None, pd.DataFrame()

    logger.info(f"Ensemble DataFrame created with {len(ensemble_df)} entries.")
    ensemble_results_dir = os.path.join(output_dir_base, f"ensemble_eval_{timestamp_str}")
    os.makedirs(ensemble_results_dir, exist_ok=True)

    # Prepare DataFrame for the evaluation function
    df_ensemble_eval_input = ensemble_df[['image_path', 'label', 'ensemble_probability_class1', 'dataset_source']].copy()
    df_ensemble_eval_input.rename(columns={'ensemble_probability_class1': 'probability_class1'}, inplace=True)

    # Save the full ensemble DataFrame with individual model probabilities
    ensemble_full_csv_path = os.path.join(ensemble_results_dir, f"ensemble_all_models_predictions_{timestamp_str}.csv")
    ensemble_df.to_csv(ensemble_full_csv_path, index=False)
    logger.info(f"Full ensemble predictions (with individual model probs) saved to: {ensemble_full_csv_path}")

    # Evaluate the ensemble
    ensemble_metrics, ensemble_roc_data = _calculate_and_plot_metrics_from_dataframe(
        df_predictions=df_ensemble_eval_input,
        dataset_name_suffix="Ensemble_AvgProb",
        results_dir_path=ensemble_results_dir,
        epoch_num_str="Ensemble", #  No specific epoch for ensemble
        plot_per_source_roc=args.plot_roc_per_source,
        roc_min_samples_per_source=args.roc_min_samples_per_source
    )
    logger.info(f"Ensemble evaluation finished. Results in: {ensemble_results_dir}")
    
    return ensemble_roc_data, ensemble_df

def create_summary_tables(all_models_roc_data: dict, ensemble_roc_data: dict | None, 
                         all_individual_model_prediction_dfs: list[pd.DataFrame],
                         model_display_names_for_dfs: list[str],
                         ensemble_df: pd.DataFrame | None,
                         all_model_metrics: dict,  # Add this parameter to pass ECE data
                         output_dir: str, min_samples_for_source: int = 50):
    """
    Creates and saves comprehensive summary tables including ECE.
    """
    logger.info("Creating comprehensive summary tables with ECE...")
    
    # Table 1: Overall Performance Summary (AUC, Sensitivity, ECE)
    summary_data = []
    
    # Collect individual model data
    for model_name, roc_data in all_models_roc_data.items():
        auc_val = roc_data['auc']
        sens_95_spec = calculate_sensitivity_at_specificity(roc_data['fpr'], roc_data['tpr'], 0.95)
        
        # Get ECE from model metrics
        ece_val = None
        if model_name in all_model_metrics and 'overall' in all_model_metrics[model_name]:
            ece_val = all_model_metrics[model_name]['overall'].get('ece')
        
        summary_data.append({
            'Model': model_name,
            'Dataset': 'ID-Test',
            'AUC': auc_val,
            'Sens@95%Spec': sens_95_spec,
            'ECE': ece_val if ece_val is not None else np.nan
        })
    
    # Add ensemble data if available
    if ensemble_roc_data:
        auc_val = ensemble_roc_data['auc']
        sens_95_spec = calculate_sensitivity_at_specificity(ensemble_roc_data['fpr'], ensemble_roc_data['tpr'], 0.95)
        
        # Calculate ECE for ensemble
        ece_val = None
        if ensemble_df is not None and not ensemble_df.empty:
            y_true = ensemble_df['label'].values
            y_prob = ensemble_df['ensemble_probability_class1'].values
            ece_val, _, _, _, _ = calculate_ece(y_true, y_prob)
        
        summary_data.append({
            'Model': 'Ensemble',
            'Dataset': 'ID-Test',
            'AUC': auc_val,
            'Sens@95%Spec': sens_95_spec,
            'ECE': ece_val if ece_val is not None else np.nan
        })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        
        # Create figure for overall summary table (now with 3 tables)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        
        models = sorted([m for m in df_summary['Model'].unique() if m != 'Ensemble']) + (['Ensemble'] if ensemble_roc_data else [])
        
        # AUC Table
        auc_table_data = []
        for model in models:
            model_data = df_summary[df_summary['Model'] == model]
            if not model_data.empty:
                auc_table_data.append([model, f"{model_data['AUC'].iloc[0]:.3f}"])
            else:
                auc_table_data.append([model, "N/A"])
        
        ax1.axis('tight')
        ax1.axis('off')
        table1 = ax1.table(cellText=auc_table_data, 
                          colLabels=['Model', 'AUC'],
                          cellLoc='center',
                          loc='center')
        table1.auto_set_font_size(False)
        table1.set_fontsize(12)
        table1.scale(1, 2)
        ax1.set_title('AUC Values - In-Distribution Test Set', fontsize=14, fontweight='bold', pad=20)
        
        # Sensitivity at 95% Specificity Table
        sens_table_data = []
        for model in models:
            model_data = df_summary[df_summary['Model'] == model]
            if not model_data.empty:
                sens_table_data.append([model, f"{model_data['Sens@95%Spec'].iloc[0]:.3f}"])
            else:
                sens_table_data.append([model, "N/A"])
        
        ax2.axis('tight')
        ax2.axis('off')
        table2 = ax2.table(cellText=sens_table_data,
                          colLabels=['Model', 'Sens@95%Spec'],
                          cellLoc='center',
                          loc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(12)
        table2.scale(1, 2)
        ax2.set_title('Sensitivity at 95% Specificity - In-Distribution Test Set', fontsize=14, fontweight='bold', pad=20)
        
        # ECE Table
        ece_table_data = []
        for model in models:
            model_data = df_summary[df_summary['Model'] == model]
            if not model_data.empty and not pd.isna(model_data['ECE'].iloc[0]):
                ece_table_data.append([model, f"{model_data['ECE'].iloc[0]:.4f}"])
            else:
                ece_table_data.append([model, "N/A"])
        
        ax3.axis('tight')
        ax3.axis('off')
        table3 = ax3.table(cellText=ece_table_data,
                          colLabels=['Model', 'ECE'],
                          cellLoc='center',
                          loc='center')
        table3.auto_set_font_size(False)
        table3.set_fontsize(12)
        table3.scale(1, 2)
        ax3.set_title('Expected Calibration Error - In-Distribution Test Set', fontsize=14, fontweight='bold', pad=20)
        
        # Style tables
        for table in [table1, table2, table3]:
            # Header styling
            for i in range(2):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Ensemble row styling (if exists)
            if ensemble_roc_data and len(models) > 0:
                ensemble_row = len(models)  # Last row
                for i in range(2):
                    table[(ensemble_row, i)].set_facecolor('#FFE0B2')
                    table[(ensemble_row, i)].set_text_props(weight='bold')
        
        plt.tight_layout()
        overall_summary_path = os.path.join(output_dir, 'id_overall_performance_summary_table_with_ece.png')
        plt.savefig(overall_summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Overall performance summary table with ECE saved to: {overall_summary_path}")
    
    # Table 2: Per-Source AUC Table for ID Test Set (including ensemble)
    if all_individual_model_prediction_dfs and model_display_names_for_dfs:
        logger.info("Creating per-source AUC summary table for ID test set (including ensemble)...")
        
        source_auc_data = []
        
        # Extract per-source AUC from individual model prediction DataFrames
        for i, df_model in enumerate(all_individual_model_prediction_dfs):
            model_name = model_display_names_for_dfs[i]
            
            if 'dataset_source' in df_model.columns:
                # Calculate AUC per source
                for source_name, source_group in df_model.groupby('dataset_source'):
                    source_name = str(source_name)
                    if source_name.lower() in ['nan', 'unknown_source', 'none']:
                        continue
                        
                    if len(source_group) >= min_samples_for_source and len(source_group['label'].unique()) >= 2:
                        try:
                            fpr_s, tpr_s, _ = roc_curve(source_group['label'], source_group['probability_class1'])
                            auc_s = auc(fpr_s, tpr_s)
                            source_auc_data.append({
                                'Model': model_name,
                                'Source': source_name,
                                'AUC': auc_s,
                                'N_Samples': len(source_group)
                            })
                        except Exception as e:
                            logger.warning(f"Could not calculate AUC for {model_name} on source {source_name}: {e}")
        
        # Add ensemble per-source AUC if ensemble data is available
        if ensemble_df is not None and not ensemble_df.empty and 'dataset_source' in ensemble_df.columns:
            logger.info("Adding ensemble per-source AUC to summary table...")
            # Use ensemble_probability_class1 for ensemble AUC calculation
            ensemble_prob_col = 'ensemble_probability_class1'
            if ensemble_prob_col in ensemble_df.columns:
                for source_name, source_group in ensemble_df.groupby('dataset_source'):
                    source_name = str(source_name)
                    if source_name.lower() in ['nan', 'unknown_source', 'none']:
                        continue
                        
                    if len(source_group) >= min_samples_for_source and len(source_group['label'].unique()) >= 2:
                        try:
                            fpr_s, tpr_s, _ = roc_curve(source_group['label'], source_group[ensemble_prob_col])
                            auc_s = auc(fpr_s, tpr_s)
                            source_auc_data.append({
                                'Model': 'Ensemble',
                                'Source': source_name,
                                'AUC': auc_s,
                                'N_Samples': len(source_group)
                            })
                        except Exception as e:
                            logger.warning(f"Could not calculate ensemble AUC for source {source_name}: {e}")
        
        if source_auc_data:
            df_source = pd.DataFrame(source_auc_data)
            
            # Get unique sources and models (with Ensemble at the end if present)
            unique_sources = sorted(df_source['Source'].unique())
            individual_models = sorted([m for m in df_source['Model'].unique() if m != 'Ensemble'])
            unique_models = individual_models + (['Ensemble'] if 'Ensemble' in df_source['Model'].unique() else [])
            
            # Create table data
            table_data = []
            for model in unique_models:
                row = [model]
                for source in unique_sources:
                    model_source_data = df_source[
                        (df_source['Model'] == model) & (df_source['Source'] == source)
                    ]
                    if not model_source_data.empty:
                        auc_val = model_source_data['AUC'].iloc[0]
                        n_samples = model_source_data['N_Samples'].iloc[0]
                        row.append(f"{auc_val:.3f}\n(N={n_samples})")
                    else:
                        row.append("N/A")
                table_data.append(row)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(12, len(unique_sources) * 2), max(8, len(unique_models) * 0.8)))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=table_data,
                           colLabels=['Model'] + unique_sources,
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            # Style header
            for i in range(len(unique_sources) + 1):
                table[(0, i)].set_facecolor('#2196F3')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Style ensemble row if present
            if 'Ensemble' in unique_models:
                ensemble_row_idx = len(unique_models)  # Row index (1-based due to header)
                for i in range(len(unique_sources) + 1):
                    table[(ensemble_row_idx, i)].set_facecolor('#FFE0B2')
                    table[(ensemble_row_idx, i)].set_text_props(weight='bold')
            
            ax.set_title(f'AUC by Model and Dataset Source - In-Distribution Test Set\n(Minimum {min_samples_for_source} samples per source)', 
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            source_summary_path = os.path.join(output_dir, 'id_source_auc_summary_table.png')
            plt.savefig(source_summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Source AUC summary table saved to: {source_summary_path}")
        else:
            logger.warning("No per-source AUC data found for summary table.")

def plot_combined_roc_with_ensemble(all_models_roc_data: dict, ensemble_roc_data: dict | None, 
                                   output_dir: str, timestamp_str: str):
    """
    Enhanced ROC plotting function that includes ensemble curves.
    """
    logger.info("\n--- Generating Combined ROC Plot with Ensemble ---")
    plt.figure(figsize=(13, 11))
    
    # Plot individual models
    num_models = len(all_models_roc_data)
    colors_cmap = plt.cm.get_cmap('tab10' if num_models <= 10 else 'viridis', num_models if num_models > 0 else 1)
    sorted_roc_data = sorted(all_models_roc_data.items(), key=lambda item: item[1]['auc'] if not np.isnan(item[1]['auc']) else -1, reverse=True)

    for i, (model_disp_name, roc_data) in enumerate(sorted_roc_data):
        if roc_data and roc_data['fpr'] is not None and roc_data['tpr'] is not None:
            auc_val = roc_data['auc']
            label_str = f'{model_disp_name} (AUC={auc_val:.3f})' if not np.isnan(auc_val) else f'{model_disp_name} (AUC=N/A)'
            plt.plot(roc_data['fpr'], roc_data['tpr'], lw=2.5, label=label_str, color=colors_cmap(i), alpha=0.8)

    # Plot ensemble if available
    if ensemble_roc_data:
        plt.plot(ensemble_roc_data['fpr'], ensemble_roc_data['tpr'], 
                color='black', lw=4, linestyle='--',
                label=f'Ensemble (AUC={ensemble_roc_data["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k:', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('Combined ROC Curves - In-Distribution Test Sets (Individual Models + Ensemble)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize='large', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    combined_roc_filename = f"all_models_id_test_combined_roc_with_ensemble_{timestamp_str}.png"
    combined_roc_filepath = os.path.join(output_dir, combined_roc_filename)
    try:
        plt.savefig(combined_roc_filepath, bbox_inches='tight', dpi=150)
        logger.info(f"Combined ROC plot with ensemble saved to: {combined_roc_filepath}")
    except Exception as e: 
        logger.error(f"Failed to save combined ROC plot: {e}", exc_info=True)
    plt.close()


def main(args: argparse.Namespace):
    """
    Main function to orchestrate the evaluation of models and ensemble.
    """
    start_time = datetime.now()
    start_time_str = start_time.strftime('%Y%m%d_%H%M%S')
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    input_dir = args.parent_experiment_dir
    if not os.path.isdir(input_dir):
        logger.critical(f"Parent experiment directory not found: {input_dir}"); sys.exit(1)

    all_models_roc_data = {}
    all_model_metrics = {}  # Add this to store all metrics including ECE
    all_individual_model_prediction_dfs = []
    model_display_names_for_ensemble_dfs = []

    is_single_model_dir = any(os.path.isdir(os.path.join(input_dir, d)) and d in ['checkpoints', 'results'] for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir,d)))
    model_dirs_to_process = [input_dir] if is_single_model_dir else \
                            [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    if not model_dirs_to_process: logger.critical(f"No model experiment directories found in {input_dir}."); sys.exit(1)
    logger.info(f"Found {len(model_dirs_to_process)} model directories to process.")

    for experiment_run_path in model_dirs_to_process:
        experiment_name_from_path = os.path.basename(experiment_run_path)
        logger.info(f"\n{'='*10} Processing Experiment: {experiment_name_from_path} {'='*10}")

        eval_output_base_dir = os.path.join(experiment_run_path, "results", f"id_eval_{start_time_str}")
        os.makedirs(eval_output_base_dir, exist_ok=True)

        log_file_path = os.path.join(eval_output_base_dir, f"log_eval_{experiment_name_from_path}.log")
        
        # Use context manager for safe logging
        with file_logging_context(log_file_path, logger):
            # Check for checkpoint directory
            checkpoint_dir = os.path.join(experiment_run_path, "checkpoints")
            original_training_results_dir = os.path.join(experiment_run_path, "results")

            if not os.path.isdir(checkpoint_dir):
                logger.warning(f"No 'checkpoints' dir for {experiment_name_from_path}. Skipping.")
                continue
                
            checkpoint_path = find_best_checkpoint(checkpoint_dir)
            if not checkpoint_path:
                logger.error(f"No checkpoint for {experiment_name_from_path}. Skipping.")
                continue
                
            logger.info(f"Selected checkpoint: {os.path.basename(checkpoint_path)}")
            epoch_str = "unknown"
            if "_epoch" in os.path.basename(checkpoint_path):
                try: epoch_str = os.path.basename(checkpoint_path).split('_epoch')[-1].split('.pth')[0]
                except IndexError: logger.warning(f"Could not parse epoch from: {os.path.basename(checkpoint_path)}")

            training_config_path = os.path.join(original_training_results_dir, "training_configuration.json")
            print(f"DEBUG: Looking for config at: {training_config_path}")
            print(f"DEBUG: Config file exists: {os.path.exists(training_config_path)}")
            if os.path.exists(training_config_path):
                print(f"DEBUG: File size: {os.path.getsize(training_config_path)} bytes")
                with open(training_config_path, 'r') as f:
                    content = f.read()
                    print(f"DEBUG: First 100 chars: {content[:100]}")
        
            if not os.path.exists(training_config_path):
                logger.error(f"Config 'training_configuration.json' missing for {experiment_name_from_path}. Skipping."); continue

            try:
                with open(training_config_path, 'r', encoding='utf-8-sig') as f: 
                    train_args_dict = json.load(f)
                model_arch_name = train_args_dict.get('model_name', args.fallback_model_name)
                image_size = train_args_dict.get('image_size', args.fallback_image_size)
                dropout_prob = train_args_dict.get('dropout_prob', args.fallback_dropout_prob)
                num_classes = train_args_dict.get('num_classes', 2)
                experiment_tag = train_args_dict.get('experiment_tag', '')
                model_display_name = experiment_tag if experiment_tag else experiment_name_from_path.split('_')[-2] if '_' in experiment_name_from_path else model_arch_name
                if not model_display_name: model_display_name = model_arch_name
                original_training_args_namespace = argparse.Namespace(**train_args_dict)
            except Exception as e:
                logger.error(f"Error loading/parsing config for {experiment_name_from_path}: {e}. Skipping.", exc_info=True); continue

            is_custom_sequential_vit_needed, is_cnn_sequential_head_needed = False, False
            model_arch_lower = model_arch_name.lower()
            if 'vit' in model_arch_lower and any(k in model_display_name.lower() or k in experiment_tag.lower() for k in ['vfm_custom', 'vfm-custom']):
                is_custom_sequential_vit_needed = True
            elif ('resnet' in model_arch_lower or 'efficientnet' in model_arch_lower) and "resnet50_timm" in model_display_name.lower() and 'resnet50' == model_arch_lower: # Example specific condition
                is_cnn_sequential_head_needed = True
            logger.info(f"Model {model_display_name} (arch: {model_arch_name}), custom_seq_vit: {is_custom_sequential_vit_needed}, cnn_seq_head: {is_cnn_sequential_head_needed}")

            model_for_evaluation = build_classifier_model(
                model_name=model_arch_name, 
                num_classes=num_classes,
                dropout_prob=dropout_prob,
                pretrained=False, 
                custom_weights_path=checkpoint_path,
                checkpoint_key='model',
                is_custom_sequential_head_vit=is_custom_sequential_vit_needed,
                is_cnn_with_sequential_head=is_cnn_sequential_head_needed
            )

            model_for_evaluation.to(device).eval()


            test_set_manifest_path = os.path.join(original_training_results_dir, "test_set_manifest.csv")
            if not os.path.exists(test_set_manifest_path):
                logger.warning(f"Manifest 'test_set_manifest.csv' missing for {experiment_name_from_path}. Skipping."); continue
            try:
                df_test_manifest = pd.read_csv(test_set_manifest_path)
                if not {'image_path', 'types'}.issubset(df_test_manifest.columns) or df_test_manifest.empty:
                    logger.error(f"Manifest empty/missing required columns for {experiment_name_from_path}. Skipping."); continue
                
                # NEW: Adjust image paths based on data_type
                logger.info(f"Adjusting image paths for data_type: {args.data_type}")
                original_paths = df_test_manifest['image_path'].tolist()
                
                if args.data_type == 'processed':
                    # Convert to processed paths and handle suffixes
                    adjusted_paths = []
                    for orig_path in original_paths:
                        processed_path = adjust_path_for_data_type(orig_path, 'processed')
                        final_path = handle_processed_image_suffixes(processed_path)
                        adjusted_paths.append(final_path)
                    
                    df_test_manifest['image_path'] = adjusted_paths
                    
                    # Check how many images actually exist
                    existing_count = sum(1 for path in adjusted_paths if os.path.exists(path))
                    logger.info(f"Found {existing_count}/{len(adjusted_paths)} processed images for {experiment_name_from_path}")
                    
                    if existing_count == 0:
                        logger.error(f"No processed images found for {experiment_name_from_path}. Skipping.")
                        continue
                        
                elif args.data_type == 'raw':
                    # Ensure we're using raw paths
                    adjusted_paths = [adjust_path_for_data_type(path, 'raw') for path in original_paths]
                    df_test_manifest['image_path'] = adjusted_paths
                    
                    # Check existence
                    existing_count = sum(1 for path in adjusted_paths if os.path.exists(path))
                    logger.info(f"Found {existing_count}/{len(adjusted_paths)} raw images for {experiment_name_from_path}")
                    
            except Exception as e:
                logger.error(f"Error loading/processing manifest for {experiment_name_from_path}: {e}. Skipping.", exc_info=True); continue

            eval_transforms = get_eval_transforms(image_size, model_arch_name)
            attrs_to_load = [attr for attr in ['dataset_source', 'image_path'] if attr in df_test_manifest.columns]
            if 'image_path' in df_test_manifest.columns and 'image_path' not in attrs_to_load: attrs_to_load.append('image_path')

            test_dataset = GlaucomaSubgroupDataset(df_test_manifest, eval_transforms, attrs_to_load, False)
            if len(test_dataset) == 0:
                logger.warning(f"Test dataset for {experiment_name_from_path} empty. Skipping."); continue

            test_loader = DataLoader(
                test_dataset, 
                batch_size=args.eval_batch_size, 
                shuffle=False, 
                num_workers=args.num_workers, 
                pin_memory=torch.cuda.is_available(), 
                collate_fn=safe_collate, 
                persistent_workers=(args.num_workers > 0)
            )
            metrics_summary, df_model_predictions, overall_roc_data = evaluate_on_test_set(
                model_for_evaluation, test_loader, test_dataset,
                f"ID-Test_{model_display_name}", eval_output_base_dir,
                epoch_str, device, args.plot_roc_per_source, args.roc_min_samples_per_source
            )
            
            if overall_roc_data: 
                all_models_roc_data[model_display_name] = overall_roc_data
            
            if metrics_summary:
                all_model_metrics[model_display_name] = metrics_summary  # Store all metrics
                
            if df_model_predictions is not None and not df_model_predictions.empty:
                all_individual_model_prediction_dfs.append(df_model_predictions)
                model_display_names_for_ensemble_dfs.append(model_display_name) # Store name matching the df

            if args.run_tsne:
                logger.info(f"Running t-SNE for {model_display_name}...")
                features, labels, sources, _ = extract_features_from_model(model_for_evaluation, test_loader, device, model_arch_name)
                if features is not None:
                    plot_tsne_features(features, labels, sources, os.path.join(eval_output_base_dir, "tsne_visualizations"),
                                       f"id_test_{model_display_name}_epoch{epoch_str}", args.tsne_perplexity,
                                       args.tsne_n_iter, args.tsne_learning_rate, args.tsne_max_samples_plot,
                                       args.tsne_min_samples_per_source)
                else: logger.warning(f"Feature extraction failed for {model_display_name}, skipping t-SNE.")

            if args.run_gradcam:
                logger.info(f"Running Grad-CAM for {model_display_name}...")
                visualize_gradcam_misclassifications(model_for_evaluation, test_loader, device,
                                                     os.path.join(eval_output_base_dir, "gradcam_visualizations"),
                                                     original_training_args_namespace, args.num_gradcam_samples,
                                                     ["Normal", "Glaucoma"], True)

    # --- End of loop over model_dirs_to_process ---

    # Process and evaluate ensemble if multiple models were processed
    ensemble_roc_data = None
    if len(all_individual_model_prediction_dfs) >= 1: # Changed to >=1 to allow ensemble processing logic to decide if <2
        ensemble_output_base_dir = args.parent_experiment_dir # Save ensemble results in the top-level dir
        ensemble_roc_data, ensemble_df = process_and_evaluate_ensemble(
            all_individual_model_prediction_dfs,
            model_display_names_for_ensemble_dfs,
            ensemble_output_base_dir,
            start_time_str,
            args
        )
    else:
        logger.info("Not enough model predictions collected to form an ensemble.")

    # Generate comprehensive summary tables and plots
    if all_models_roc_data:
        save_dir_combined_outputs = args.parent_experiment_dir if not is_single_model_dir else os.path.dirname(original_training_results_dir)
        
        logger.info("\n--- Creating Comprehensive Summary Tables with ECE ---")
        create_summary_tables(
            all_models_roc_data=all_models_roc_data,
            ensemble_roc_data=ensemble_roc_data,
            all_individual_model_prediction_dfs=all_individual_model_prediction_dfs,
            model_display_names_for_dfs=model_display_names_for_ensemble_dfs,
            ensemble_df=ensemble_df if 'ensemble_df' in locals() else None,
            all_model_metrics=all_model_metrics,  # Pass ECE data
            output_dir=save_dir_combined_outputs,
            min_samples_for_source=args.roc_min_samples_per_source
        )
        
        logger.info("\n--- Generating Combined ROC Plot with Ensemble ---")
        # Enhanced ROC plot that includes ensemble curve
        plot_combined_roc_with_ensemble(
            all_models_roc_data=all_models_roc_data,
            ensemble_roc_data=ensemble_roc_data,
            output_dir=save_dir_combined_outputs,
            timestamp_str=start_time_str
        )
    else:
        logger.warning("No ROC data from individual models. Skipping combined plots and summary tables.")

    logger.info(f"\n{'='*5} All Evaluations Finished. Total execution time: {datetime.now() - start_time} {'='*5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate models on in-distribution test sets and optionally create an ensemble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--parent_experiment_dir', type=str, required=True, help="Path to single experiment or parent of multiple experiment runs.")
    
    # NEW: Add data_type argument
    parser.add_argument('--data_type', type=str, choices=['raw', 'processed'], default='raw',
                       help="Type of data to use for evaluation: 'raw' or 'processed'")
    
    parser.add_argument('--fallback_model_name', type=str, default='vit_base_patch16_224', help="Fallback model name.")
    parser.add_argument('--fallback_image_size', type=int, default=224, help="Fallback image size.")
    parser.add_argument('--fallback_dropout_prob', type=float, default=0.1, help="Fallback dropout probability.")
    parser.add_argument('--eval_batch_size', type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader workers (0 for main process).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--load_weights_only', action=argparse.BooleanOptionalAction, default=True, help="Use torch.load with weights_only=True.")
    parser.add_argument('--plot_roc_per_source', action=argparse.BooleanOptionalAction, default=True, help="Plot ROC per dataset source.")
    parser.add_argument('--roc_min_samples_per_source', type=int, default=50, help="Min samples from a source for its ROC curve.")

    viz_group = parser.add_argument_group("Visualization Options")
    viz_group.add_argument('--run_gradcam', action=argparse.BooleanOptionalAction, default=False, help="Run Grad-CAM.")
    viz_group.add_argument('--num_gradcam_samples', type=int, default=5, help="Num Grad-CAM samples per category.")
    viz_group.add_argument('--run_tsne', action=argparse.BooleanOptionalAction, default=True, help="Run t-SNE.")
    viz_group.add_argument('--tsne_perplexity', type=float, default=30.0, help="t-SNE: perplexity.")
    viz_group.add_argument('--tsne_n_iter', type=int, default=1000, help="t-SNE: iterations.")
    viz_group.add_argument('--tsne_learning_rate', type=str, default='auto', help="t-SNE: learning rate ('auto' or float).")
    viz_group.add_argument('--tsne_max_samples_plot', type=int, default=2000, help="t-SNE: Max samples to plot.")
    viz_group.add_argument('--tsne_min_samples_per_source', type=int, default=100, help="t-SNE: Min samples per source for plot.")

    calib_group = parser.add_argument_group("Calibration Options")
    calib_group.add_argument('--calculate_calibration', action=argparse.BooleanOptionalAction, default=True, 
                            help="Calculate Expected Calibration Error (ECE) and generate calibration plots.")
    calib_group.add_argument('--ece_n_bins', type=int, default=10, 
                            help="Number of bins for ECE calculation.")

    args = parser.parse_args()
    main(args)