# src/evaluation/bias_analysis.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier # Alternative probe
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import json
import warnings

# --- Fix: Import NpEncoder ---
try:
     # Adjust relative path if needed based on your structure
     from ..utils.helpers import NpEncoder
except ImportError:
     # Fallback if running as a script or structure differs
     try:
          from src.utils.helpers import NpEncoder
     except ImportError:
          print("Warning: NpEncoder not found. JSON saving might fail for numpy types.")
          # Define a dummy encoder if needed for script execution
          class NpEncoder(json.JSONEncoder):
              def default(self, obj):
                  if isinstance(obj, np.integer): return int(obj)
                  if isinstance(obj, np.floating): return float(obj)
                  if isinstance(obj, np.ndarray): return obj.tolist()
                  return super(NpEncoder, self).default(obj)


logger = logging.getLogger(__name__)
# Set a default style if needed
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
     plt.style.use('ggplot') # Fallback style


class FeatureExtractorHook:
    """Helper class to capture features using a forward hook."""
    def __init__(self):
        self.features = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            self.features = output[0].detach().clone()
        elif isinstance(output, torch.Tensor):
            self.features = output.detach().clone()
        else:
            logger.warning(f"Hook captured unexpected output type: {type(output)}")
            self.features = None

    def get_features(self):
        return self.features

    def clear(self):
        self.features = None


def extract_features(model, loader, device, feature_layer_hook=None, use_amp=True):
    """
    Extracts features and associated labels/metadata using hooks.
    Assumes loader yields batches of (images, labels, attributes_dict).
    Returns: features_array, labels_array, valid_metadata, indices_array
    """
    if not isinstance(model, nn.Module):
        logger.error("Invalid model provided to extract_features.")
        return None, None, None, None
    if not isinstance(loader, torch.utils.data.DataLoader):
        logger.error("Invalid loader provided to extract_features.")
        return None, None, None, None

    model.eval()
    features_list = []
    labels_list = []
    indices_list = []  # <-- Track indices of successfully processed samples

    sensitive_attributes_names = []
    if hasattr(loader, 'dataset') and hasattr(loader.dataset, 'sensitive_attributes'):
        sensitive_attributes_names = loader.dataset.sensitive_attributes or []
    collected_attributes = {attr: [] for attr in sensitive_attributes_names}

    target_layer = None
    hook_handle = None
    hook_helper = FeatureExtractorHook()

    # --- Hook Setup Logic ---
    if feature_layer_hook:
        module_found = False
        for name, mod in model.named_modules():
            if name == feature_layer_hook:
                target_layer = mod
                module_found = True
                logger.info(f"Found specified hook layer: '{feature_layer_hook}'")
                break
        if not module_found:
            logger.warning(f"Specified hook name '{feature_layer_hook}' not found. Attempting auto-detection.")
            feature_layer_hook = None # Allow auto-detection

    if target_layer is None:
        logger.info("Attempting automatic feature layer detection...")
        preferred_hooks = ['norm', 'head.norm', 'avgpool', 'global_pool']
        model_name_lower = model.__class__.__name__.lower()
        for hook_name in preferred_hooks:
            module_found = False
            for name, mod in model.named_modules():
                if name == hook_name or name == f"model.{hook_name}":
                    target_layer = mod
                    feature_layer_hook = name
                    logger.info(f"Auto-detected feature layer: '{name}'")
                    module_found = True
                    break
            if module_found:
                break
        if target_layer is None:
            logger.error("Could not identify a suitable feature extraction layer automatically.")
            return None, None, None, None

    # --- Register Hook ---
    try:
        hook_handle = target_layer.register_forward_hook(hook_helper.hook_fn)
        logger.info(f"Registering forward hook on layer: {feature_layer_hook}")
    except Exception as e:
        logger.error(f"Failed to register hook on '{feature_layer_hook}': {e}")
        return None, None, None, None

    # --- Feature Extraction Loop ---
    device_type = device.type
    amp_enabled = use_amp and (device_type == 'cuda')
    amp_dtype = torch.float16 if device_type == 'cuda' else None

    logger.info("Starting feature extraction loop...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(loader, desc="Extracting Features")):
            if batch_data is None:
                continue
            if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 3:
                logger.error(f"Batch {batch_idx}: Unexpected format. Expected 3 items, got {len(batch_data)}. Skipping.")
                continue
            inputs, labels, attributes_batch = batch_data
            if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                continue

            inputs = inputs.to(device)
            current_batch_size = len(labels)

            try:
                hook_helper.clear()
                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                    _ = model(inputs)
                batch_features_tensor = hook_helper.get_features()

                if batch_features_tensor is not None and batch_features_tensor.shape[0] == current_batch_size:
                    batch_features_np = batch_features_tensor.flatten(start_dim=1).cpu().numpy()
                    features_list.append(batch_features_np)
                    labels_list.append(labels.cpu().numpy())

                    # --- Collect indices ---
                    if isinstance(attributes_batch, dict) and 'index' in attributes_batch:
                        indices = attributes_batch['index']
                        if isinstance(indices, torch.Tensor):
                            indices_list.append(indices.cpu().numpy())
                        elif isinstance(indices, (list, np.ndarray)):
                            indices_list.append(np.array(indices))
                        else:
                            logger.warning(f"Batch {batch_idx}: Unexpected index type. Appending NaNs.")
                            indices_list.append(np.full(current_batch_size, np.nan))
                    else:
                        # fallback: running index (not robust if batches are skipped)
                        indices_list.append(np.full(current_batch_size, np.nan))

                    if isinstance(attributes_batch, dict):
                        for attr_name in sensitive_attributes_names:
                            if attr_name in attributes_batch:
                                attr_values = attributes_batch[attr_name]
                                if isinstance(attr_values, torch.Tensor):
                                    collected_attributes[attr_name].append(attr_values.cpu().numpy())
                                elif isinstance(attr_values, (list, tuple)) and len(attr_values) == current_batch_size:
                                    collected_attributes[attr_name].append(np.array(attr_values, dtype=object))
                                else:
                                    logger.warning(f"Batch {batch_idx}: Unexpected attribute type/length for '{attr_name}'. Appending NaNs.")
                                    collected_attributes[attr_name].append(np.full(current_batch_size, np.nan))
                            else:
                                collected_attributes[attr_name].append(np.full(current_batch_size, np.nan))
                    else:
                        for attr_name in sensitive_attributes_names:
                            collected_attributes[attr_name].append(np.full(current_batch_size, np.nan))
                else:
                    logger.warning(f"Hook did not capture valid/matching activation for batch {batch_idx}. Skipping.")
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}", exc_info=True)
                continue

    hook_handle.remove()
    logger.info("Feature extraction loop finished.")

    if not features_list or not labels_list:
        logger.error("No features or labels were successfully extracted.")
        return None, None, None, None

    try:
        features_array = np.concatenate(features_list, axis=0)
        labels_array = np.concatenate(labels_list, axis=0)
        indices_array = np.concatenate(indices_list, axis=0) if indices_list else None
        final_metadata = {}
        final_n = len(labels_array)

        for attr_name, list_of_batches in collected_attributes.items():
            if not list_of_batches:
                final_metadata[attr_name] = np.full(final_n, np.nan)
                continue
            try:
                concatenated_attr = np.concatenate(list_of_batches, axis=0)
                if len(concatenated_attr) == final_n:
                    final_metadata[attr_name] = concatenated_attr
                else:
                    logger.error(f"Length mismatch for attribute '{attr_name}' ({len(concatenated_attr)} vs {final_n}). Filling with NaNs.")
                    final_metadata[attr_name] = np.full(final_n, np.nan)
            except ValueError as concat_err:
                logger.error(f"Could not concatenate attribute '{attr_name}': {concat_err}. Filling with NaNs.")
                final_metadata[attr_name] = np.full(final_n, np.nan)

        if features_array.shape[0] != final_n:
            logger.error(f"FATAL: Final feature count mismatch ({features_array.shape[0]} vs {final_n}).")
            return None, None, None, None

        valid_metadata = {}
        for key, arr in final_metadata.items():
            if len(arr) == final_n:
                valid_metadata[key] = arr
                logger.info(f"Final extracted metadata '{key}' shape: {arr.shape}")

        logger.info(f"Final extracted features shape: {features_array.shape}")
        logger.info(f"Final extracted labels shape: {labels_array.shape}")
        if indices_array is not None:
            logger.info(f"Final extracted indices shape: {indices_array.shape}")

        return features_array, labels_array, valid_metadata, indices_array

    except Exception as e:
        logger.error(f"Error consolidating extracted data: {e}", exc_info=True)
        return None, None, None, None


def train_attribute_probes(features, labels_df, attributes_to_predict,
                           results_dir, experiment_name="probe",
                           test_size=0.3, random_state=42,
                           probe_type='logistic'):
    """
    Trains simple classifiers (probes) to predict sensitive attributes from features.
    """
    logger.info(f"--- Training Sensitive Attribute Probes ({probe_type}) ---")
    probe_results = {}
    os.makedirs(results_dir, exist_ok=True)

    if features is None or labels_df is None or not isinstance(labels_df, pd.DataFrame) or features.shape[0] != len(labels_df):
         logger.error("Invalid input for probe training.")
         return {}

    for attribute in attributes_to_predict:
        if attribute not in labels_df.columns:
            logger.warning(f"Attribute '{attribute}' not found in labels_df. Skipping probe.")
            continue

        y_series = labels_df[attribute].copy()
        valid_idx = y_series.notna()
        if valid_idx.sum() < 10:
            logger.warning(f"Attribute '{attribute}' has < 10 non-NaN values ({valid_idx.sum()}). Skipping probe.")
            continue

        X_valid = features[valid_idx]
        y_valid = y_series[valid_idx]

        unique_labels = y_valid.unique()
        if len(unique_labels) != 2:
            # --- Log the actual unique values causing the failure ---
            logger.warning(f"Attribute '{attribute}' is not binary after NaN removal (unique values: {unique_labels}). Skipping probe.")
            continue

        try:
             y_valid_numeric = pd.factorize(y_valid)[0]
             if len(np.unique(y_valid_numeric)) != 2: continue # Skip if factorization fails
        except Exception as factorize_e:
             logger.warning(f"Could not factorize labels for '{attribute}': {factorize_e}. Skipping.")
             continue

        logger.info(f"Training probe for attribute: '{attribute}' (N={len(y_valid_numeric)})")

        try:
             X_train, X_test, y_train, y_test = train_test_split(
                 X_valid, y_valid_numeric, test_size=test_size, random_state=random_state, stratify=y_valid_numeric
             )
        except ValueError as e:
             logger.warning(f"Stratified split failed for '{attribute}' ({e}). Trying non-stratified split.")
             try:
                  X_train, X_test, y_train, y_test = train_test_split(
                      X_valid, y_valid_numeric, test_size=test_size, random_state=random_state
                  )
             except Exception as split_e:
                  logger.error(f"Data splitting failed for '{attribute}': {split_e}. Skipping.")
                  continue

        # Build and Train Probe
        if probe_type == 'logistic':
            probe_model = Pipeline([('scaler', StandardScaler()), ('logistic', LogisticRegression(solver='liblinear', random_state=random_state, class_weight='balanced', max_iter=1000))])
        elif probe_type == 'mlp':
             probe_model = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(64,), max_iter=100, random_state=random_state, early_stopping=True, n_iter_no_change=5))])
        else: continue

        try:
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore")
                 probe_model.fit(X_train, y_train)

            auc_score = None
            if hasattr(probe_model, "predict_proba"):
                 y_pred_proba_all = probe_model.predict_proba(X_test)
                 if y_pred_proba_all.shape[1] >= 2:
                      y_pred_proba = y_pred_proba_all[:, 1]
                      auc_score = roc_auc_score(y_test, y_pred_proba)
                 else: auc_score = 0.5
            else: logger.warning(f"Probe for '{attribute}' lacks predict_proba.")

            y_pred_binary = probe_model.predict(X_test)
            acc_score = accuracy_score(y_test, y_pred_binary)

            # --- Fix: Correct f-string formatting ---
            auc_str = f"{auc_score:.4f}" if auc_score is not None else "N/A"
            logger.info(f"  Probe Result '{attribute}': Test AUC = {auc_str}, Test Acc = {acc_score:.4f}")
            probe_results[attribute] = {'auc': auc_score, 'accuracy': acc_score}

        except Exception as e:
            logger.error(f"Error training or evaluating probe for '{attribute}': {e}", exc_info=True)
            probe_results[attribute] = {'auc': None, 'accuracy': None, 'error': str(e)}

    # Save probe results
    probe_results_path = os.path.join(results_dir, f'{experiment_name}_probe_results.json')
    try:
        # --- Fix: Use NpEncoder ---
        with open(probe_results_path, 'w') as f:
            json.dump(probe_results, f, indent=4, cls=NpEncoder) # Pass encoder class
        logger.info(f"Probe results saved to {probe_results_path}")
    except Exception as e:
        # Log error without breaking if NpEncoder is missing
        logger.error(f"Failed to save probe results (JSON serialization error): {e}")

    return probe_results


def analyze_feature_space_pca(features, labels_df, attributes_to_analyze,
                              n_components=4, results_dir=None, experiment_name="pca_viz"):
    """
    Performs PCA on features and visualizes distributions across subgroups.
    (Using the corrected version from the previous response)
    """
    logger.info(f"--- Analyzing Feature Space using PCA (Top {n_components} Components) ---")

    if features is None or labels_df is None or not isinstance(labels_df, pd.DataFrame) or features.shape[0] != len(labels_df):
         logger.error("Invalid input for PCA analysis: Features or labels_df mismatch or invalid type.")
         return
    if n_components <= 0 or n_components > features.shape[1]:
         logger.error(f"Invalid n_components for PCA: {n_components}.")
         return

    # --- Standardize Features ---
    scaler = StandardScaler()
    if np.any(~np.isfinite(features)):
         logger.warning("Non-finite values found in features before PCA. Imputing with mean.")
         col_means = np.nanmean(features, axis=0)
         inds = np.where(np.isnan(features))
         features[inds] = np.take(col_means, inds[1])
         # Handle potential infs after NaN imputation
         features = np.clip(features, np.finfo(features.dtype).min, np.finfo(features.dtype).max)

    try:
        features_scaled = scaler.fit_transform(features)
    except ValueError as scale_e:
         logger.error(f"StandardScaler failed: {scale_e}. Check for zero variance columns.")
         return

    # --- Apply PCA ---
    logger.info("Fitting PCA...")
    pca = PCA(n_components=n_components)
    try:
        features_pca = pca.fit_transform(features_scaled)
    except Exception as e:
        logger.error(f"PCA fitting failed: {e}", exc_info=True)
        return

    explained_variance_ratio = pca.explained_variance_ratio_
    logger.info(f"PCA Explained Variance Ratio (Top {n_components}): {explained_variance_ratio}")
    logger.info(f"Total Variance Explained: {explained_variance_ratio.sum():.4f}")

    pca_cols = [f'PCA Mode {i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(features_pca, columns=pca_cols, index=labels_df.index)
    # Ensure index alignment before concat
    if not labels_df.index.equals(df_pca.index):
         logger.warning("Index mismatch between labels_df and df_pca. Resetting index.")
         labels_df = labels_df.reset_index(drop=True)
         df_pca = df_pca.reset_index(drop=True)
    df_plot = pd.concat([labels_df, df_pca], axis=1)


    # --- Generate Plots ---
    num_attributes = len(attributes_to_analyze)
    logger.info("Generating PCA distribution plots...")
    fig, axes = plt.subplots(n_components, num_attributes,
                             figsize=(num_attributes * 5, n_components * 4),
                             squeeze=False, sharex=True)

    for i in range(n_components):
        pca_col = pca_cols[i]
        for j, attribute in enumerate(attributes_to_analyze):
            ax = axes[i, j]
            if attribute not in df_plot.columns:
                 logger.warning(f"Attribute '{attribute}' not found for PCA plot. Skipping.")
                 ax.axis('off'); ax.set_title(f"{attribute}\n(Not Found)")
                 continue

            df_plot_attr = df_plot.dropna(subset=[attribute, pca_col])
            if df_plot_attr.empty:
                 logger.warning(f"No valid data for PCA plot: {pca_col} vs {attribute}")
                 ax.axis('off'); ax.set_title(f"{attribute}\n(No Data)")
                 continue

            try:
                hue_order = None
                # Convert attribute to category for potentially better handling by seaborn
                if not pd.api.types.is_categorical_dtype(df_plot_attr[attribute]):
                     df_plot_attr[attribute] = df_plot_attr[attribute].astype('category')
                hue_order = df_plot_attr[attribute].cat.categories.tolist()

                sns.kdeplot(data=df_plot_attr, x=pca_col, hue=attribute, hue_order=hue_order,
                            fill=True, common_norm=False, ax=ax, palette='viridis',
                            alpha=0.6, linewidth=1.5, legend=True)

                title_text = attribute.replace('_', ' ').title()
                ax.set_title(title_text)
                ax.set_xlabel(pca_col if i == n_components - 1 else "")
                ax.set_ylabel("Density" if j == 0 else "")
                ax.grid(True, alpha=0.3)
                current_legend = ax.get_legend()
                if current_legend:
                     current_legend.set_title(title_text)
                     plt.setp(current_legend.get_texts(), fontsize='small')

            except Exception as plot_e:
                 logger.error(f"Error plotting KDE for {pca_col} vs {attribute}: {plot_e}", exc_info=True)
                 ax.axis('off'); ax.set_title(f"{attribute}\n(Plot Error)")

    plt.suptitle(f'Feature Space PCA Distributions ({experiment_name})', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if results_dir:
        pca_plot_path = os.path.join(results_dir, f'{experiment_name}_pca_distributions.png')
        try:
            os.makedirs(os.path.dirname(pca_plot_path), exist_ok=True)
            plt.savefig(pca_plot_path, bbox_inches='tight')
            logger.info(f"PCA distribution plot saved to {pca_plot_path}")
        except Exception as e:
            logger.error(f"Failed to save PCA plot: {e}")
    plt.show()
    plt.close(fig)


def extract_features_for_tsne(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                              device: torch.device, model_name_from_training: str,
                              attributes_to_collect: list[str] = None):
    """
    Extracts features from a model for t-SNE visualization.
    Handles different model architectures for locating the feature extraction layer.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for the data.
        device: Device to run the model on ('cuda' or 'cpu').
        model_name_from_training: Name of the model architecture (e.g., 'resnet50', 'vit_base_patch16_224').
                                   Used to determine the feature extraction layer.
        attributes_to_collect: List of metadata attribute keys to collect from the dataloader's output.
                               'image_path' is always collected.

    Returns:
        A tuple (final_features, final_labels, df_meta_for_tsne):
        - final_features (np.ndarray): Extracted features.
        - final_labels (np.ndarray): Corresponding labels.
        - df_meta_for_tsne (pd.DataFrame or None): DataFrame of collected metadata, or None if issues.
    """
    model.eval()
    features_list, labels_list = [], []
    
    # Ensure 'image_path' is always requested if attributes_to_collect is provided
    final_attributes_to_collect = list(set((attributes_to_collect or []) + ['image_path']))
    metadata_collected = {attr: [] for attr in final_attributes_to_collect}

    hook_target_layer = None
    original_head_state = None # To store original classifier head if replaced

    # --- Determine hook target layer based on model name ---
    # This logic is specific and might need adjustment for new model types
    model_name_lower = model_name_from_training.lower()
    if 'dinov2' in model_name_lower:
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'norm'):
            hook_target_layer = model.backbone.norm
        elif hasattr(model, 'norm'): # e.g. timm's ViT models often have a final norm
            hook_target_layer = model.norm
        else: logger.warning(f"Could not find 'norm' layer for DINOv2-like model: {model_name_from_training}")
    elif 'resnet' in model_name_lower or 'convnext' in model_name_lower or 'efficientnet' in model_name_lower:
        # Try common pooling layers or head replacement
        if hasattr(model, 'global_pool') and model.global_pool is not None and not isinstance(model.global_pool, nn.Identity):
            hook_target_layer = model.global_pool
        elif hasattr(model, 'avgpool') and model.avgpool is not None and not isinstance(model.avgpool, nn.Identity):
            hook_target_layer = model.avgpool
        elif hasattr(model, 'get_classifier') and isinstance(model.get_classifier(), nn.Linear): # Timm models often have this
            logger.info(f"Attempting to replace classifier head for {model_name_from_training} to get features.")
            original_head_module = model.get_classifier()
            original_head_state = {
                'state_dict': original_head_module.state_dict(),
                'out_features': original_head_module.out_features,
                'in_features': original_head_module.in_features
            }
            # Replace head with an Identity layer to get features before the original head
            model.reset_classifier(0, '') # Global pool name '' implies using existing or default
            hook_target_layer = model # Features are now the direct output of the modified model
            logger.info(f"Temporarily replaced head of {model_name_from_training} with Identity for feature extraction.")
        else:
            logger.warning(f"Cannot find a standard feature layer (global_pool, avgpool) or resettable classifier for {model_name_from_training}.")
    # Add more model families if needed (e.g., ViT typically has a 'norm' layer before head)
    elif 'vit' in model_name_lower: # General ViT, check for 'norm' layer
        if hasattr(model, 'norm') and model.norm is not None: # Common in timm ViTs
             hook_target_layer = model.norm
        elif hasattr(model, 'fc_norm') and model.fc_norm is not None: # Another ViT variant
             hook_target_layer = model.fc_norm
        else: logger.warning(f"Could not find 'norm' or 'fc_norm' layer for ViT model: {model_name_from_training}")


    if hook_target_layer is None and original_head_state is None: # if head wasn't replaced, hook_target_layer must be set
        logger.error(f"Could not determine feature extraction layer for '{model_name_from_training}'. "
                     "Features for t-SNE cannot be extracted. Check model architecture and hook logic.")
        return None, None, None

    batch_hook_features = []
    def _hook_fn(module, input_tensor, output_tensor):
        # Input can be a tuple for some layers, output is usually the tensor
        features = output_tensor.detach().cpu()
        if features.ndim > 2: # Flatten if not already flat (e.g., from conv layers)
            features = features.view(features.size(0), -1)
        batch_hook_features.append(features)
    
    handle = None
    if hook_target_layer is not model: # Only register hook if not using replaced head method
        if hook_target_layer is not None:
            handle = hook_target_layer.register_forward_hook(_hook_fn)
        else: # Should not happen if original_head_state is None
            logger.error("Hook target layer is None and head was not replaced. Cannot extract features.")
            return None, None, None


    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"Extracting t-SNE Features ({model_name_from_training})"):
            if batch_data is None: 
                logger.warning("Encountered None batch_data, skipping.")
                continue
            
            # Assuming dataloader yields: inputs, labels, metadata_dict
            inputs, labels, metadata_batch_dict = batch_data[0], batch_data[1], batch_data[2]
            
            current_batch_size = len(labels)
            inputs = inputs.to(device)
            batch_hook_features.clear() # Clear for each batch
            
            # --- Forward pass ---
            current_model_output = model(inputs) 
            
            # --- Collect features ---
            if original_head_state is not None: # Features are direct output if head was replaced
                batch_f = current_model_output.detach().cpu()
                if batch_f.ndim > 2: batch_f = batch_f.view(batch_f.size(0), -1)
                features_list.append(batch_f)
            elif batch_hook_features: # Features from hook
                # Concatenate if hook was called multiple times (should not happen with typical global pool hooks)
                collected_feats_for_batch = torch.cat(batch_hook_features, dim=0)
                features_list.append(collected_feats_for_batch)
            else: 
                logger.warning("Hook did not capture features for a batch, or head replacement method failed. Skipping feature collection for this batch.")
                continue # Skip labels and metadata for this batch too if features are missing

            labels_list.append(labels.cpu())
            
            # Collect metadata
            for attr_key in final_attributes_to_collect:
                # .get from the dict of lists, providing default list of Nones
                values_for_attr_in_batch = metadata_batch_dict.get(attr_key, [None] * current_batch_size)
                
                if len(values_for_attr_in_batch) != current_batch_size:
                    logger.warning(f"Metadata for '{attr_key}' has length {len(values_for_attr_in_batch)}, "
                                   f"but batch size is {current_batch_size}. Using Nones for this batch for this attribute.")
                    metadata_collected[attr_key].extend([None] * current_batch_size)
                else:
                    metadata_collected[attr_key].extend(values_for_attr_in_batch)
    
    if handle: handle.remove() # Remove the hook

    # Restore original head if it was replaced
    if original_head_state is not None:
        try:
            # Determine global_pool type if needed by reset_classifier
            gp_type = ''
            if hasattr(model, 'global_pool') and hasattr(model.global_pool, 'pool_type'):
                gp_type = model.global_pool.pool_type
            
            model.reset_classifier(num_classes=original_head_state['out_features'], global_pool=gp_type)
            # model.get_classifier().in_features = original_head_state['in_features'] # May not be needed if reset_classifier handles it
            model.get_classifier().load_state_dict(original_head_state['state_dict'])
            logger.info(f"Successfully restored original classifier head for {model_name_from_training}.")
        except Exception as e_res:
            logger.error(f"Error restoring original classifier head for {model_name_from_training}: {e_res}", exc_info=True)

    if not features_list or not labels_list:
        logger.warning("No features or labels were collected. Cannot proceed with t-SNE data preparation.")
        return None, None, None
    
    final_features = torch.cat(features_list).numpy()
    final_labels = torch.cat(labels_list).numpy()
    
    # Verify metadata list lengths before creating DataFrame
    expected_len = len(final_labels)
    valid_metadata_for_df = {}
    all_metadata_keys = list(metadata_collected.keys()) # Iterate over a copy of keys

    for key in all_metadata_keys:
        val_list = metadata_collected[key]
        if len(val_list) == expected_len:
            valid_metadata_for_df[key] = val_list
        else:
            logger.error(f"Length mismatch for metadata key '{key}': expected {expected_len}, got {len(val_list)}. "
                         "This attribute will be dropped for the t-SNE metadata DataFrame.")
            # Optionally, fill with Nones to maintain column, but safer to drop if lengths are inconsistent.
            # valid_metadata_for_df[key] = [None] * expected_len # If you want to keep the column with Nones

    if not valid_metadata_for_df:
        logger.error("No valid metadata collected with consistent lengths. Cannot create t-SNE metadata DataFrame.")
        # Still return features and labels if they are consistent
        return final_features, final_labels, None 
    
    try:
        df_meta_for_tsne = pd.DataFrame(valid_metadata_for_df)
    except ValueError as e:
        logger.error(f"Pandas DataFrame creation failed for metadata: {e}. Check list lengths in metadata_collected.")
        logger.error(f"Collected metadata lengths: {[ (k, len(v)) for k,v in valid_metadata_for_df.items() ]}")
        return final_features, final_labels, None # Return None for metadata_df on error
        
    return final_features, final_labels, df_meta_for_tsne