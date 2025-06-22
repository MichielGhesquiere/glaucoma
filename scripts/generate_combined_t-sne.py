#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined t-SNE Plot Generation Script with Dataset Sampling.

This script loads a specified model checkpoint, extracts features from its
In-Distribution (ID) test set and specified Out-of-Distribution (OOD) datasets.
It samples a maximum number of points per dataset source (stratified by label if possible)
before running t-SNE. It then generates two t-SNE plots:
1.  Combined ID (as one group) & OOD datasets, colored by these major groupings.
2.  Combined detailed ID sources & OOD datasets, colored by specific dataset sources.
Markers are uniform, and the legend does not distinguish glaucoma class.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import seaborn as sns
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

# Ensure custom modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.utils.helpers import NpEncoder, set_seed
    from src.data.datasets import GlaucomaSubgroupDataset, safe_collate
    from src.models.classification.build_model import build_classifier_model
    from src.data.utils import adjust_path_for_data_type
    from src.data.external_loader import load_external_test_data
except ImportError as e:
    print(f"Error importing custom modules: {e}\nEnsure 'src' is accessible.")
    sys.exit(1)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    log_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s"
    )
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

RAW_DIR_NAME_CONST: str = "raw"
PROCESSED_DIR_NAME_CONST: str = "processed"

def get_eval_transforms(image_size: int, model_name_for_norm: str | None) -> Compose:
    logger.debug(f"Defining evaluation transforms: image_size={image_size}, normalization_ref_model='{model_name_for_norm}'")
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if model_name_for_norm:
        try:
            config = resolve_data_config({}, model=timm.create_model(model_name_for_norm, pretrained=False, num_classes=0))
            if config and 'mean' in config and 'std' in config:
                mean, std = config["mean"], config["std"]
                logger.debug(f"Using model-specific normalization for '{model_name_for_norm}': mean={mean}, std={std}")
            else:
                logger.warning(f"Could not resolve specific mean/std for '{model_name_for_norm}'. Using ImageNet defaults.")
        except Exception as e:
            logger.warning(f"Timm config error for '{model_name_for_norm}': {e}. Using ImageNet defaults.")
    else:
        logger.debug("No model_name_for_norm provided for transforms, using ImageNet defaults.")
    return Compose([Resize((image_size, image_size)), ToTensor(), Normalize(mean=mean, std=std)])

def extract_features_from_model(model: nn.Module, dataloader: DataLoader, device: torch.device,
                                model_name_from_training: str,
                                dataset_source_metadata_key: str = 'dataset_source' # Metadata key for the source
                               ) -> tuple[np.ndarray | None, np.ndarray | None, list | None, list | None]:
    model.eval()
    features_list, labels_list, dataset_source_list, image_paths_list = [], [], [], []
    hook_target_layer, original_head_module, original_head_attr_name = None, None, None
    model_name_lower = model_name_from_training.lower()

    # Simplified feature extraction strategy
    if 'dinov2' in model_name_lower and hasattr(model, 'norm'): hook_target_layer = model.norm
    elif 'vit' in model_name_lower and hasattr(model, 'norm') and isinstance(model.norm, nn.LayerNorm): hook_target_layer = model.norm
    elif hasattr(model, 'head') and isinstance(model.head, (nn.Linear, nn.Sequential)):
        original_head_module, original_head_attr_name, model.head, hook_target_layer = model.head, 'head', nn.Identity(), model
    elif hasattr(model, 'fc') and isinstance(model.fc, (nn.Linear, nn.Sequential)):
        original_head_module, original_head_attr_name, model.fc, hook_target_layer = model.fc, 'fc', nn.Identity(), model
    elif hasattr(model, 'classifier') and isinstance(model.classifier, (nn.Linear, nn.Sequential)):
        original_head_module, original_head_attr_name, model.classifier, hook_target_layer = model.classifier, 'classifier', nn.Identity(), model
    elif hasattr(model, 'head') and hasattr(model.head, 'fc') and isinstance(model.head.fc, (nn.Linear, nn.Sequential)):
        original_head_module, model.head.fc, hook_target_layer = model.head.fc, nn.Identity(), model.head # Hook before Identity

    if hook_target_layer is None and not original_head_module:
        logger.error(f"Feature extraction strategy undetermined for {model_name_from_training}."); return None, None, None, None
    logger.debug(f"Feature extraction: {'Replace head' if original_head_module else 'Hook layer'}: {original_head_attr_name or hook_target_layer.__class__.__name__}")

    extracted_features_hook = []
    def _hook_fn(module, input_val, output_val):
        features = output_val[0] if isinstance(output_val, tuple) else output_val
        detached = features.detach().cpu()
        extracted_features_hook.append(detached.view(detached.size(0), -1) if detached.ndim > 2 else detached)

    hook_handle = None
    if hook_target_layer is not model and hook_target_layer is not None:
        hook_handle = hook_target_layer.register_forward_hook(_hook_fn)

    dataset_name_log = getattr(getattr(dataloader.dataset, 'dataset', dataloader.dataset), 'dataset_name', 'Unknown Dataset')

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"Extracting Features ({dataset_name_log})"):
            if batch_data is None: continue
            inputs, labels, metadata = batch_data[0], batch_data[1], batch_data[2] if len(batch_data) > 2 else {}
            inputs = inputs.to(device)
            extracted_features_hook.clear()
            output = model(inputs)

            batch_f = output.detach().cpu() if hook_target_layer is model else (torch.cat(extracted_features_hook, dim=0) if extracted_features_hook else None)
            if batch_f is None: logger.warning("No features for batch."); continue
            if batch_f.ndim > 2: batch_f = batch_f.view(batch_f.size(0), -1)
            
            features_list.append(batch_f)
            labels_list.append(labels.cpu()) # Store labels, even if not used in this plot's legend
            
            sources_val = metadata.get(dataset_source_metadata_key, [dataset_name_log] * len(labels))
            sources = [str(s) for s in (sources_val if isinstance(sources_val, list) else [str(sources_val)] * len(labels))]
            dataset_source_list.extend(sources)
            
            paths_val = metadata.get('image_path', ['Unknown_Path'] * len(labels))
            paths = [str(p) for p in (paths_val if isinstance(paths_val, list) else [str(paths_val)] * len(labels))]
            image_paths_list.extend(paths)

    if hook_handle: hook_handle.remove()
    if original_head_module and original_head_attr_name:
        setattr(model, original_head_attr_name, original_head_module)
        logger.debug(f"Restored head '{original_head_attr_name}'.")

    if not features_list: return None, None, None, None
    return torch.cat(features_list).numpy(), torch.cat(labels_list).numpy(), dataset_source_list, image_paths_list

def plot_tsne_visualization(
    features_data: np.ndarray,
    sources_data: list, # Labels are passed but not used in legend for this version
    output_dir_path: str,
    plot_title_prefix: str,
    plot_filename_suffix: str,
    perplexity_val: float,
    n_iter_val: int,
    learning_rate_val: str | float,
    max_total_samples_for_tsne: int,
    min_samples_per_src_for_plot: int
):
    logger.info(f"Generating t-SNE plot: {plot_title_prefix} ({plot_filename_suffix})")
    
    df_for_tsne_processing = pd.DataFrame({'dataset_source': sources_data})
    feature_cols_names = [f'feature_{i}' for i in range(features_data.shape[1])]
    df_features_src = pd.DataFrame(features_data, columns=feature_cols_names)
    df_for_tsne_processing = pd.concat([df_for_tsne_processing, df_features_src], axis=1)

    source_value_counts = df_for_tsne_processing['dataset_source'].value_counts()
    sources_to_include_in_legend = source_value_counts[source_value_counts >= min_samples_per_src_for_plot].index.tolist()
    
    df_tsne_plot_subset = df_for_tsne_processing # By default, use all data for t-SNE calculation
    if not sources_to_include_in_legend:
        logger.warning(f"For {plot_filename_suffix}, no source meets min samples ({min_samples_per_src_for_plot}) for distinct legend entry.")

    if df_tsne_plot_subset.empty:
        logger.warning(f"No data for t-SNE plot '{plot_filename_suffix}'. Skipping.")
        return

    features_for_tsne_run = df_tsne_plot_subset[feature_cols_names].values
    sources_for_tsne_run = df_tsne_plot_subset['dataset_source'].tolist()
    
    num_samples_current = len(features_for_tsne_run)
    if num_samples_current > max_total_samples_for_tsne:
        logger.info(f"Subsampling {max_total_samples_for_tsne} from {num_samples_current} total samples for t-SNE run: {plot_filename_suffix}")
        # Random subsampling of the combined data for t-SNE calculation
        chosen_indices = np.random.choice(num_samples_current, max_total_samples_for_tsne, replace=False)
        features_final = features_for_tsne_run[chosen_indices]
        sources_final = [sources_for_tsne_run[i] for i in chosen_indices]
    else:
        features_final = features_for_tsne_run
        sources_final = sources_for_tsne_run
    
    num_final_for_tsne = len(features_final)
    if num_final_for_tsne < 5:
        logger.warning(f"Too few samples ({num_final_for_tsne}) for t-SNE '{plot_filename_suffix}'. Skipping.")
        return

    actual_perplexity_val = float(perplexity_val)
    if num_final_for_tsne <= actual_perplexity_val:
        actual_perplexity_val = max(5.0, float(num_final_for_tsne - 1.0))
        logger.warning(f"Perplexity adjusted to {actual_perplexity_val} for '{plot_filename_suffix}'.")

    lr_tsne = 'auto' if isinstance(learning_rate_val, str) and learning_rate_val.lower() == 'auto' else float(learning_rate_val)

    logger.info(f"Running t-SNE on {num_final_for_tsne} samples for '{plot_filename_suffix}'...")
    tsne_model = TSNE(n_components=2, perplexity=actual_perplexity_val, n_iter=int(n_iter_val),
                      learning_rate=lr_tsne, random_state=42, init='pca', n_jobs=-1)
    try:
        tsne_embeddings = tsne_model.fit_transform(features_final)
    except Exception as e:
        logger.error(f"t-SNE failed for '{plot_filename_suffix}': {e}", exc_info=True); return

    df_tsne_embedded = pd.DataFrame({'tsne-1': tsne_embeddings[:,0], 'tsne-2': tsne_embeddings[:,1],
                                     'dataset_source': sources_final}) # Only sources needed for legend

    plt.figure(figsize=(18, 14))
    unique_sources_in_plot = sorted(df_tsne_embedded['dataset_source'].unique())
    palette_colors = sns.color_palette("husl", n_colors=len(unique_sources_in_plot)) \
        if len(unique_sources_in_plot) > 20 else sns.color_palette("tab20", n_colors=max(10, len(unique_sources_in_plot)))

    for idx, src_name_plot in enumerate(unique_sources_in_plot):
        data_subset_source = df_tsne_embedded[df_tsne_embedded['dataset_source'] == src_name_plot]
        plt.scatter(data_subset_source['tsne-1'], data_subset_source['tsne-2'],
                    label=f"{src_name_plot} (N={len(data_subset_source)})",
                    alpha=0.7, color=palette_colors[idx % len(palette_colors)], marker='o', s=70)

    plt.title(f'{plot_title_prefix}\n(N={num_final_for_tsne}, Perplexity={actual_perplexity_val:.1f})', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='medium', title="Dataset Source (Count)")
    plt.grid(True, alpha=0.4, linestyle=':')
    plt.tight_layout(rect=[0, 0, 0.80, 1])

    final_plot_filename = f"tsne_{plot_filename_suffix}.png"
    final_plot_path = os.path.join(output_dir_path, final_plot_filename)
    try:
        plt.savefig(final_plot_path, dpi=150); logger.info(f"t-SNE plot saved: {final_plot_path}")
    except Exception as e:
        logger.error(f"Failed to save t-SNE plot '{final_plot_filename}': {e}", exc_info=True)
    plt.close()

def robust_stratified_sample(df_group, n_samples, label_col='types', random_state=None):
    """
    Performs stratified sampling if possible, otherwise random sampling.
    Ensures that the number of samples per class does not exceed available samples.
    """
    if n_samples >= len(df_group):
        return df_group # Not enough samples to reduce, or just enough

    unique_labels = df_group[label_col].unique()
    if len(unique_labels) > 1 and all(df_group[label_col].value_counts() >= 1): # Can attempt stratification
        # Calculate proportional samples per stratum, ensuring we don't ask for more than available
        proportions = df_group[label_col].value_counts(normalize=True)
        samples_per_stratum = (proportions * n_samples).round().astype(int)
        
        # Adjust if sum is slightly off due to rounding, or if any stratum asks for more than available
        # This part can get complex; for simplicity, we'll cap by available and then adjust total if needed
        sampled_dfs = []
        for lbl, count in samples_per_stratum.items():
            stratum_df = df_group[df_group[label_col] == lbl]
            actual_samples_for_stratum = min(count, len(stratum_df))
            if actual_samples_for_stratum > 0 :
                 sampled_dfs.append(stratum_df.sample(n=actual_samples_for_stratum, random_state=random_state))
        
        if not sampled_dfs: # e.g. if n_samples was too small and all strata rounded to 0
            return df_group.sample(n=min(n_samples, len(df_group)), random_state=random_state)

        final_sampled_df = pd.concat(sampled_dfs)
        
        # If after this, we still have more than n_samples (due to min counts per stratum), do a final random sample
        if len(final_sampled_df) > n_samples:
            final_sampled_df = final_sampled_df.sample(n=n_samples, random_state=random_state)
        elif len(final_sampled_df) < n_samples and len(final_sampled_df) < len(df_group): # If we sampled too few but more are available
            # Try to top up with random samples from the remainder (could slightly skew stratification)
            remaining_to_sample = n_samples - len(final_sampled_df)
            if remaining_to_sample > 0:
                remainder_df = df_group.drop(final_sampled_df.index)
                if len(remainder_df) >= remaining_to_sample:
                    final_sampled_df = pd.concat([final_sampled_df, remainder_df.sample(n=remaining_to_sample, random_state=random_state)])
                else: # sample all remaining
                    final_sampled_df = pd.concat([final_sampled_df, remainder_df])

        return final_sampled_df
    else: # Fallback to random sampling
        return df_group.sample(n=min(n_samples, len(df_group)), random_state=random_state)


def main(args: argparse.Namespace):
    start_time = datetime.now()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"--- Combined t-SNE Generation Script (Sampled per Dataset) ---"); logger.info(f"Device: {device}")
    if not os.path.exists(args.checkpoint_path): logger.critical(f"Checkpoint not found: {args.checkpoint_path}"); sys.exit(1)

    model_name_from_path = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint_path), "..", "results", f"tsne_analysis_{model_name_from_path}")
    os.makedirs(output_dir, exist_ok=True); logger.info(f"Output directory: {output_dir}")

    experiment_dir = os.path.dirname(os.path.dirname(args.checkpoint_path))
    training_config_path = os.path.join(experiment_dir, "results", "training_configuration.json")

    if not os.path.exists(training_config_path) and not (args.model_arch_name and args.dropout_prob is not None and args.image_size is not None):
        logger.critical(f"Training config {training_config_path} not found and fallback params not fully specified. Exiting."); sys.exit(1)
    
    train_args_source = "JSON config"
    if os.path.exists(training_config_path):
        with open(training_config_path, 'r') as f: train_args_json = json.load(f)
        model_arch_name, dropout_prob, image_size, num_classes, experiment_tag = (
            train_args_json.get('model_name', args.model_arch_name), train_args_json.get('dropout_prob', args.dropout_prob),
            train_args_json.get('image_size', args.image_size), train_args_json.get('num_classes', args.num_classes),
            train_args_json.get('experiment_tag', args.experiment_tag_if_no_config) )
    else:
        train_args_source = "Command-line fallbacks"
        model_arch_name, dropout_prob, image_size, num_classes, experiment_tag = (
            args.model_arch_name, args.dropout_prob, args.image_size, args.num_classes, args.experiment_tag_if_no_config)
    logger.info(f"Model params (from {train_args_source}): Arch={model_arch_name}, Dropout={dropout_prob}, ImgSize={image_size}, NumClasses={num_classes}, ExpTag='{experiment_tag}'")

    is_custom_sequential_vit, is_cnn_with_sequential_head = False, False
    model_arch_lower = model_arch_name.lower()
    if 'vit' in model_arch_lower and any(kw in experiment_tag.lower() for kw in ['vfm_custom', 'vfm-custom']): is_custom_sequential_vit = True
    elif ('resnet' in model_arch_lower and "resnet50_timm" == experiment_tag.lower()) or \
         ('efficientnet' in model_arch_lower and "some_effnet_seq_tag" == experiment_tag.lower()): is_cnn_with_sequential_head = True # Example for EffNet
    logger.info(f"Build flags: is_custom_sequential_vit={is_custom_sequential_vit}, is_cnn_with_sequential_head={is_cnn_with_sequential_head}")

    model = build_classifier_model(
        model_name=model_arch_name, num_classes=num_classes, dropout_prob=dropout_prob, pretrained=False,
        is_custom_sequential_head_vit=is_custom_sequential_vit, is_cnn_with_sequential_head=is_cnn_with_sequential_head
    )
    try:
        ckpt_data = torch.load(args.checkpoint_path, map_location=device, weights_only=args.load_weights_only)
        sd = ckpt_data.get('model_state_dict', ckpt_data.get('state_dict', ckpt_data.get('model', ckpt_data)))
        if sd is None: raise ValueError("state_dict not found in checkpoint.")
        if all(k.startswith('module.') for k in sd.keys()): sd = {k[len('module.'):]: v for k, v in sd.items()}
        ls = model.load_state_dict(sd, strict=False); logger.info(f"Model weights loaded. Missing: {ls.missing_keys}, Unexpected: {ls.unexpected_keys}")
        model.to(device).eval()
    except Exception as e: logger.critical(f"Error loading model weights: {e}. Exiting.", exc_info=True); sys.exit(1)

    # Store (DataFrame, source_name_for_plot1, source_name_for_plot2)
    # source_name_for_plot1 is the grouped name (e.g., "ID_TestSet")
    # source_name_for_plot2 is the detailed name (e.g., "AIROGS", "PAPILLA")
    list_of_dataset_dfs_for_extraction = []
    eval_transforms = get_eval_transforms(image_size, model_arch_name)

    # 1. Load and Prepare In-Distribution (ID) Test Set
    id_test_manifest_path = os.path.join(experiment_dir, "results", "test_set_manifest.csv")
    if os.path.exists(id_test_manifest_path):
        logger.info(f"Loading ID test set from: {id_test_manifest_path}")
        df_id_full = pd.read_csv(id_test_manifest_path)
        if not df_id_full.empty and {'image_path', 'types'}.issubset(df_id_full.columns):
            if 'dataset_source' not in df_id_full.columns: df_id_full['dataset_source'] = "ID_UnknownSource"
            
            # Add 'types' column for stratification and ensure it's int for GlaucomaSubgroupDataset
            df_id_full['types'] = pd.to_numeric(df_id_full['types'], errors='coerce').fillna(-1).astype(int) # -1 for unlabelable
            df_id_full = df_id_full[df_id_full['types'] != -1] # Remove unlabelable

            # Prepare data for Plot 1 (ID grouped)
            df_id_grouped_plot1_sampled = robust_stratified_sample(df_id_full, args.max_samples_per_source, label_col='types', random_state=args.seed)
            df_id_grouped_plot1_sampled = df_id_grouped_plot1_sampled.copy() # Avoid SettingWithCopyWarning
            df_id_grouped_plot1_sampled.loc[:, 'source_name_for_plot1'] = "ID_TestSet"
            df_id_grouped_plot1_sampled.loc[:, 'source_name_for_plot2'] = df_id_grouped_plot1_sampled['dataset_source'] # Keep original for plot 2
            if not df_id_grouped_plot1_sampled.empty:
                 list_of_dataset_dfs_for_extraction.append(df_id_grouped_plot1_sampled)

        else: logger.warning(f"ID manifest {id_test_manifest_path} empty/bad. Skipping ID for t-SNE.")
    else: logger.warning(f"ID manifest not found. Skipping ID for t-SNE.")

    # 2. Load and Prepare Out-of-Distribution (OOD) Test Sets
    chaksu_base_relative = args.chaksu_base_dir_relative
    if args.ood_data_type == 'processed':
        chaksu_base_relative = adjust_path_for_data_type(args.chaksu_base_dir_relative, 'processed', args.base_data_root, RAW_DIR_NAME_CONST, PROCESSED_DIR_NAME_CONST)
    
    ood_datasets_map = load_external_test_data(
        args.smdg_metadata_file_relative, args.smdg_image_dir_relative, chaksu_base_relative,
        args.chaksu_decision_dir_relative, args.chaksu_metadata_dir_relative, args.ood_data_type,
        args.base_data_root, RAW_DIR_NAME_CONST, PROCESSED_DIR_NAME_CONST,
        args.eval_papilla, args.eval_oiaodir_test, args.eval_chaksu
    )

    for ood_name, df_ood_full in ood_datasets_map.items():
        if df_ood_full.empty: continue
        if 'label' in df_ood_full.columns and 'types' not in df_ood_full.columns: df_ood_full.rename(columns={'label':'types'}, inplace=True)
        if 'types' not in df_ood_full.columns: logger.warning(f"OOD '{ood_name}' missing 'types'. Skipping."); continue
        
        df_ood_full['types'] = pd.to_numeric(df_ood_full['types'], errors='coerce').fillna(-1).astype(int)
        df_ood_full = df_ood_full[df_ood_full['types'] != -1]
        if df_ood_full.empty: logger.warning(f"OOD '{ood_name}' empty after removing invalid types. Skipping."); continue

        df_ood_sampled = robust_stratified_sample(df_ood_full, args.max_samples_per_source, label_col='types', random_state=args.seed)
        df_ood_sampled = df_ood_sampled.copy()
        df_ood_sampled.loc[:, 'source_name_for_plot1'] = ood_name # OOD names are the same for plot 1
        df_ood_sampled.loc[:, 'source_name_for_plot2'] = ood_name # And for plot 2
        if not df_ood_sampled.empty:
            list_of_dataset_dfs_for_extraction.append(df_ood_sampled)

    if not list_of_dataset_dfs_for_extraction:
        logger.critical("No data available after sampling ID and OOD. Cannot generate t-SNE. Exiting."); sys.exit(1)

    # --- Combine all sampled DFs and extract features ONCE ---
    logger.info("\n--- Extracting features from combined sampled datasets ---")

    # Create combined DataFrames for feature extraction for each plot type
    combined_df_for_plot1 = pd.concat([df.rename(columns={'source_name_for_plot1': 'current_source_for_extraction'}) 
                                       for df in list_of_dataset_dfs_for_extraction if not df.empty], ignore_index=True)
    combined_df_for_plot2 = pd.concat([df.rename(columns={'source_name_for_plot2': 'current_source_for_extraction'})
                                       for df in list_of_dataset_dfs_for_extraction if not df.empty], ignore_index=True)

    all_plot1_features, all_plot1_labels, all_plot1_sources, _ = None, None, None, None
    if not combined_df_for_plot1.empty:
        if 'types' not in combined_df_for_plot1.columns: logger.error("Plot 1 DF missing 'types'"); sys.exit(1)
        dataset_p1 = GlaucomaSubgroupDataset(combined_df_for_plot1, eval_transforms, ['current_source_for_extraction', 'image_path', 'types'], False)
        dataset_p1.dataset_name = "Combined_For_Plot1"
        loader_p1 = DataLoader(dataset_p1, args.batch_size, num_workers=args.num_workers, collate_fn=safe_collate)
        all_plot1_features, _, all_plot1_sources, _ = extract_features_from_model(model, loader_p1, device, model_arch_name, dataset_source_metadata_key='current_source_for_extraction')
        if all_plot1_features is not None:
             plot_tsne_visualization(
                all_plot1_features, all_plot1_sources,
                output_dir, f"Model_{model_name_from_path}", f"IDgrouped_vs_OOD_max{args.max_samples_per_source}perSrc_total{args.tsne_max_total_samples}",
                args.tsne_perplexity, args.tsne_n_iter, args.tsne_learning_rate,
                args.tsne_max_total_samples, args.tsne_min_samples_per_source_plot
            )
    else: logger.warning("No data for Plot 1 (ID Grouped vs OOD).")


    all_plot2_features, all_plot2_labels, all_plot2_sources, _ = None, None, None, None
    if not combined_df_for_plot2.empty:
        if 'types' not in combined_df_for_plot2.columns: logger.error("Plot 2 DF missing 'types'"); sys.exit(1)
        dataset_p2 = GlaucomaSubgroupDataset(combined_df_for_plot2, eval_transforms, ['current_source_for_extraction', 'image_path', 'types'], False)
        dataset_p2.dataset_name = "Combined_For_Plot2"
        loader_p2 = DataLoader(dataset_p2, args.batch_size, num_workers=args.num_workers, collate_fn=safe_collate)
        all_plot2_features, _, all_plot2_sources, _ = extract_features_from_model(model, loader_p2, device, model_arch_name, dataset_source_metadata_key='current_source_for_extraction')
        if all_plot2_features is not None:
            plot_tsne_visualization(
                all_plot2_features, all_plot2_sources,
                output_dir, f"Model_{model_name_from_path}", f"IDdetailed_vs_OOD_max{args.max_samples_per_source}perSrc_total{args.tsne_max_total_samples}",
                args.tsne_perplexity, args.tsne_n_iter, args.tsne_learning_rate,
                args.tsne_max_total_samples, args.tsne_min_samples_per_source_plot
            )
    else: logger.warning("No data for Plot 2 (Detailed ID Sources vs OOD).")


    logger.info(f"--- Combined t-SNE Generation Finished. Total time: {datetime.now() - start_time} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate combined t-SNE plots (sampled per dataset) for a model checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--model_arch_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--experiment_tag_if_no_config', type=str, default='')

    data_group = parser.add_argument_group("Data Loading Configuration")
    data_group.add_argument('--base_data_root', type=str, default=r'D:\glaucoma\data')
    data_group.add_argument('--ood_data_type', type=str, default='raw', choices=['raw', 'processed'])
    data_group.add_argument('--smdg_metadata_file_relative', type=str, default=os.path.join('raw','SMDG-19','metadata - standardized.csv'))
    data_group.add_argument('--smdg_image_dir_relative', type=str, default=os.path.join('raw','SMDG-19','full-fundus','full-fundus'))
    data_group.add_argument('--chaksu_base_dir_relative', type=str, default=os.path.join('raw','Chaksu','Train','Train','1.0_Original_Fundus_Images'))
    data_group.add_argument('--chaksu_decision_dir_relative', type=str, default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision'))
    data_group.add_argument('--chaksu_metadata_dir_relative', type=str, default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision','Majority'))
    
    eval_set_group = parser.add_argument_group("OOD Set Selection for t-SNE")
    eval_set_group.add_argument('--eval_papilla', action=argparse.BooleanOptionalAction, default=True)
    eval_set_group.add_argument('--eval_oiaodir_test', action=argparse.BooleanOptionalAction, default=False)
    eval_set_group.add_argument('--eval_chaksu', action=argparse.BooleanOptionalAction, default=True)

    tsne_group = parser.add_argument_group("t-SNE Parameters")
    tsne_group.add_argument('--batch_size', type=int, default=32)
    tsne_group.add_argument('--num_workers', type=int, default=0)
    tsne_group.add_argument('--seed', type=int, default=42)
    tsne_group.add_argument('--load_weights_only', action=argparse.BooleanOptionalAction, default=True)
    tsne_group.add_argument('--output_dir', type=str, default=None)
    tsne_group.add_argument('--tsne_perplexity', type=float, default=30.0)
    tsne_group.add_argument('--tsne_n_iter', type=int, default=1000)
    tsne_group.add_argument('--tsne_learning_rate', type=str, default='auto')
    tsne_group.add_argument('--max_samples_per_source', type=int, default=2000, help="Max samples to draw from each individual dataset source (ID or OOD) before combining for t-SNE.")
    tsne_group.add_argument('--tsne_max_total_samples', type=int, default=4000, help="Overall max samples for the final t-SNE calculation (applied after per-source sampling).")
    tsne_group.add_argument('--tsne_min_samples_per_source_plot', type=int, default=10, help="Min samples a source must have to be included in t-SNE plot legend with distinct color.")

    args = parser.parse_args()
    args.base_data_root = os.path.abspath(args.base_data_root)
    if not os.path.isdir(args.base_data_root):
        logger.critical(f"Base data root not found: {args.base_data_root}. Exiting."); sys.exit(1)
    main(args)