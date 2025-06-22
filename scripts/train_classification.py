#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Glaucoma Classification Model Training Script.

This script handles the end-to-end process of training a deep learning model
for glaucoma classification. It includes:
1.  Data loading from multiple sources (SMDG-19, CHAKSU, AIROGS).
2.  Path adjustments for raw vs. processed data.
3.  Data preprocessing and splitting into training, validation, and test sets.
    (PAPILLA/CHAKSU are separate held-out sets).
4.  Definition of image transformations.
5.  Creation of PyTorch DataLoaders.
6.  Model building (leveraging timm library and custom weights).
7.  Training loop with features like learning rate scheduling, early stopping,
    and mixed-precision training.
8.  Checkpointing and saving of training history and results.
9.  Resuming training from checkpoints.
10. Final evaluation on validation and test sets.
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from timm.data import resolve_data_config, Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    RandAugment,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm import tqdm

# Ensure custom modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.utils.helpers import NpEncoder, set_seed
    from src.data.datasets import GlaucomaSubgroupDataset, safe_collate
    from src.models.classification.build_model import build_classifier_model
    from src.training.engine import train_model
    from src.utils.gradcam_utils import visualize_gradcam_misclassifications, get_default_target_layers
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print(
        "Please ensure 'src' directory is in PYTHONPATH or accessible from the script's location."
    )
    sys.exit(1)

# --- Global Configuration & Constants ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PAPILLA_PREFIX: str = "PAPILA"
# BASE_DATA_DIR_CONFIG_KEY will be set from args in main script execution block
BASE_DATA_DIR_CONFIG_KEY: str = ""  # Placeholder
RAW_DIR_NAME: str = "raw"
PROCESSED_DIR_NAME: str = "processed"


# --- Utility Functions ---
def adjust_path_for_data_type(current_path: str, data_type: str) -> str:
    """
    Adjusts a file path from 'raw' data directory to 'processed' data directory.

    If `data_type` is 'processed', this function attempts to transform
    `current_path` by replacing segments related to `RAW_DIR_NAME` with
    `PROCESSED_DIR_NAME`.

    Args:
        current_path: The original file or directory path.
        data_type: The type of data ('raw' or 'processed'). If 'raw',
                   the path is returned unchanged.

    Returns:
        The adjusted path if `data_type` is 'processed' and the path
        contained 'raw' segments; otherwise, the original path.
    """
    if data_type != "processed":
        return current_path

    norm_current_path = os.path.normpath(current_path)
    norm_base_data_dir = os.path.normpath(BASE_DATA_DIR_CONFIG_KEY)
    norm_base_raw_path = os.path.join(norm_base_data_dir, RAW_DIR_NAME)
    norm_base_processed_path = os.path.join(norm_base_data_dir, PROCESSED_DIR_NAME)

    # Case 1: Path starts with the base raw data directory
    if norm_current_path.lower().startswith(norm_base_raw_path.lower()):
        relative_part = os.path.relpath(norm_current_path, norm_base_raw_path)
        new_path = os.path.join(norm_base_processed_path, relative_part)
        logger.debug(
            f"Path adjustment (base replace): '{current_path}' -> '{new_path}'"
        )
        return new_path

    # Case 2: 'raw' segment is somewhere in the path
    # Construct OS-specific path segments
    raw_path_segment = os.path.join(os.sep, RAW_DIR_NAME, os.sep)
    processed_path_segment = os.path.join(os.sep, PROCESSED_DIR_NAME, os.sep)
    lower_norm_current_path = norm_current_path.lower()
    lower_raw_path_segment = raw_path_segment.lower()

    idx = lower_norm_current_path.find(lower_raw_path_segment)
    if idx != -1:
        # Replace the 'raw' segment with 'processed' segment
        new_path = (
            norm_current_path[:idx]
            + processed_path_segment
            + norm_current_path[idx + len(raw_path_segment) :]
        )
        logger.debug(
            f"Path adjustment (segment replace): '{current_path}' -> '{new_path}'"
        )
        return new_path

    logger.debug(f"Path '{current_path}' not adjusted for 'processed'.")
    return current_path

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    
    Label smoothing replaces hard targets (0, 1) with soft targets:
    - True class gets probability (1 - smoothing + smoothing/num_classes)
    - Other classes get probability (smoothing/num_classes)
    
    For binary classification:
    - True class: (1 - smoothing + smoothing/2) = (1 - smoothing/2)
    - False class: smoothing/2
    """
    def __init__(self, smoothing: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions (logits) of shape [batch_size, num_classes]
            target: Ground truth labels of shape [batch_size] with class indices
        """
        log_probs = F.log_softmax(pred, dim=1)
        
        # Create smooth target distribution
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / self.num_classes)
        smooth_targets.scatter_(1, target.unsqueeze(1), self.confidence + self.smoothing / self.num_classes)
        
        # Compute KL divergence
        loss = -torch.sum(smooth_targets * log_probs, dim=1)
        return loss.mean()

class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for model calibration.
    
    Temperature scaling learns a single scalar parameter T that rescales the logits:
    p_i = exp(z_i/T) / sum_j(exp(z_j/T))
    
    This helps calibrate the model's confidence to match its accuracy.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initialize temperature
        
    def forward(self, x):
        """Apply temperature scaling to model logits."""
        logits = self.model(x)
        return logits / self.temperature
    
    def calibrate(self, cal_loader, device, max_iter=50, lr=0.01):
        """
        Calibrate temperature using a calibration dataset.
        
        Args:
            cal_loader: DataLoader for calibration data
            device: Device to run calibration on
            max_iter: Maximum optimization iterations
            lr: Learning rate for temperature optimization
            
        Returns:
            Optimized temperature value
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        
        # Collect all logits and labels first to avoid memory issues during optimization
        logger.info("Collecting calibration data...")
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in cal_loader:
                if batch_data is None:
                    continue
                
                try:
                    if len(batch_data) == 3:
                        inputs, labels, _ = batch_data
                    elif len(batch_data) == 2:
                        inputs, labels = batch_data
                    else:
                        continue
                        
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Get model logits without temperature scaling
                    logits = self.model(inputs)
                    all_logits.append(logits.detach())
                    all_labels.append(labels.detach())
                    
                except Exception as e:
                    logger.warning(f"Error collecting calibration batch: {e}")
                    continue
        
        if not all_logits:
            logger.warning("No calibration data collected")
            return 1.0
        
        # Concatenate all data
        try:
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            logger.info(f"Collected {len(all_logits)} calibration samples")
        except Exception as e:
            logger.error(f"Error concatenating calibration data: {e}")
            return 1.0
        
        # Clear CUDA cache before optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Only optimize temperature parameter
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            
            # Apply temperature scaling to pre-collected logits
            scaled_logits = all_logits / self.temperature
            loss = nll_criterion(scaled_logits, all_labels.long())
            
            loss.backward()
            return loss
        
        initial_temp = self.temperature.item()
        logger.info(f"Starting temperature calibration with initial T={initial_temp:.4f}")
        
        try:
            optimizer.step(eval_loss)
            final_temp = self.temperature.item()
            logger.info(f"Temperature calibration completed. Final T={final_temp:.4f}")
            
            # Clamp temperature to reasonable range
            with torch.no_grad():
                self.temperature.clamp_(0.1, 10.0)
                
            return self.temperature.item()
            
        except Exception as e:
            logger.error(f"Temperature calibration failed: {e}")
            # Reset to reasonable default
            try:
                with torch.no_grad():
                    self.temperature.data = torch.ones(1, device=self.temperature.device)
            except:
                # If even this fails, create new parameter on CPU
                self.temperature = nn.Parameter(torch.ones(1))
            return 1.0
        finally:
            # Clear cache after calibration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def split_validation_for_calibration(val_df: pd.DataFrame, cal_ratio: float = 0.3, 
                                   seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split validation set into validation and calibration sets.
    
    Args:
        val_df: Original validation DataFrame
        cal_ratio: Fraction to use for calibration (0.0-1.0)
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (new_val_df, cal_df)
    """
    if val_df.empty or cal_ratio <= 0.0:
        return val_df.copy(), pd.DataFrame(columns=val_df.columns)
    
    if cal_ratio >= 1.0:
        return pd.DataFrame(columns=val_df.columns), val_df.copy()
    
    try:
        # Stratify by label if possible
        stratify_col = val_df['types'] if 'types' in val_df.columns else None
        if stratify_col is not None:
            unique_labels = stratify_col.nunique()
            min_count = stratify_col.value_counts().min()
            if unique_labels > 1 and min_count >= 2:
                new_val_df, cal_df = train_test_split(
                    val_df, test_size=cal_ratio, random_state=seed, 
                    stratify=stratify_col
                )
            else:
                new_val_df, cal_df = train_test_split(
                    val_df, test_size=cal_ratio, random_state=seed
                )
        else:
            new_val_df, cal_df = train_test_split(
                val_df, test_size=cal_ratio, random_state=seed
            )
            
        logger.info(f"Split validation set: {len(new_val_df)} for validation, {len(cal_df)} for calibration")
        return new_val_df, cal_df
        
    except Exception as e:
        logger.warning(f"Could not split validation set for calibration: {e}")
        return val_df.copy(), pd.DataFrame(columns=val_df.columns)

def extract_base_chaksu(name: str) -> str:
    """
    Extracts a base identifier from a CHAKSU dataset filename.

    Removes extension and any content after a hyphen '-' or period '.'
    (other than the extension).

    Args:
        name: The filename string.

    Returns:
        The extracted base name.
    """
    name = str(name)  # Ensure it's a string
    name = os.path.splitext(name)[0]  # Remove extension
    if "-" in name:
        name = name.split("-")[0]
    if "." in name:  # Handle cases like "image.part.jpg" becoming "image"
        name = name.split(".")[0]
    return name


# --- Data Loading Functions ---
def load_chaksu_data(config: argparse.Namespace) -> pd.DataFrame:
    """
    Loads and preprocesses data from the CHAKSU dataset.

    Args:
        config: An argparse.Namespace object containing configuration parameters,
                including paths to CHAKSU data and decision files.

    Returns:
        A pandas DataFrame containing CHAKSU samples with image paths,
        labels, and metadata. Returns an empty DataFrame if loading fails.
    """
    logger.info(f"Loading CHAKSU data (data_type: {config.data_type})...")
    chaksu_rows = []
    camera_types = ["Bosch", "Forus", "Remidio"]
    camera_csv_map = {
        "Bosch": "Glaucoma_Decision_Comparison_Bosch_majority.csv",
        "Forus": "Glaucoma_Decision_Comparison_Forus_majority.csv",
        "Remidio": "Glaucoma_Decision_Comparison_Remidio_majority.csv",
    }

    def label_to_numeric(label: str) -> int | None:
        """Converts string label to numeric (0 for Normal, 1 for Glaucoma)."""
        label_upper = str(label).strip().upper()
        if "NORMAL" in label_upper:
            return 0
        if "GLAUCOMA" in label_upper:
            return 1
        return None

    for camera in camera_types:
        img_dir_for_camera = os.path.join(config.chaksu_base_dir, camera)
        decision_csv_path = os.path.join(
            config.chaksu_decision_dir, camera_csv_map[camera]
        )
        metadata_csv_path = os.path.join(
            config.chaksu_metadata_dir, f"{camera}.csv"
        )

        if not os.path.exists(decision_csv_path):
            logger.warning(
                f"Chaksu decision CSV for {camera} not found: {decision_csv_path}. Skipping."
            )
            continue
        if not os.path.exists(img_dir_for_camera):
            logger.warning(
                f"Chaksu image directory for {camera} not found: {img_dir_for_camera}. Skipping."
            )
            continue

        try:
            label_df = pd.read_csv(decision_csv_path)
            if "Images" not in label_df.columns:
                logger.error(
                    f"'Images' column missing in {decision_csv_path}. Skipping {camera}."
                )
                continue

            label_df["MajorityLabel"] = label_df["Majority Decision"].apply(
                label_to_numeric
            )
            label_df["base"] = label_df["Images"].apply(
                lambda x: extract_base_chaksu(os.path.basename(str(x)))
            )
            label_df = label_df.set_index("base")

            meta_df = pd.DataFrame()
            if os.path.exists(metadata_csv_path):
                meta_df = pd.read_csv(metadata_csv_path)
                if not meta_df.empty and "Images" in meta_df.columns:
                    meta_df["ImageBase"] = meta_df["Images"].apply(
                        lambda x: extract_base_chaksu(os.path.basename(str(x)))
                        if isinstance(x, str)
                        else x
                    )
                    meta_df = meta_df.set_index("ImageBase")

            img_files = []
            for ext in ["*.JPG", "*.jpg", "*.png", "*.PNG"]:
                img_files.extend(glob.glob(os.path.join(img_dir_for_camera, ext)))

            found_labels_for_camera = 0
            for img_path in img_files:
                img_name = os.path.basename(img_path)
                img_base = extract_base_chaksu(img_name)

                if img_base not in label_df.index:
                    continue

                row_label_info = label_df.loc[img_base]
                label_val = row_label_info["MajorityLabel"]
                if label_val is None:
                    continue

                meta_values = {}
                if not meta_df.empty and img_base in meta_df.index:
                    meta_values = meta_df.loc[img_base].to_dict()
                    meta_values.pop("Images", None)  # Remove redundant 'Images' column

                chaksu_rows.append(
                    {
                        "names": img_name,
                        "types": label_val,
                        "image_path": img_path,
                        "file_exists": True, # Assumed true as glob found it
                        "dataset": f"CHAKSU-{camera}",
                        "camera": camera,
                        **meta_values,
                    }
                )
                found_labels_for_camera += 1
            logger.info(f"Found {found_labels_for_camera} labeled images for CHAKSU {camera}.")

        except Exception as e:
            logger.error(f"Error processing CHAKSU {camera}: {e}", exc_info=True)

    df_chaksu = pd.DataFrame(chaksu_rows)
    logger.info(f"Loaded {len(df_chaksu)} CHAKSU samples in total.")
    return df_chaksu


def load_airogs_data(config: argparse.Namespace) -> pd.DataFrame:
    """
    Loads and samples data from the AIROGS dataset.

    Supports caching of the sampled manifest to speed up subsequent loads.

    Args:
        config: An argparse.Namespace object with configuration parameters,
                including AIROGS paths, sampling numbers, and cache settings.

    Returns:
        A pandas DataFrame with AIROGS samples. Returns an empty DataFrame
        on failure or if no samples are found/specified.
    """
    logger.info(f"Loading AIROGS data (data_type: {config.data_type})...")
    
    airogs_img_base_dir = config.airogs_image_dir # This path is already adjusted by main's pre-processing
    cache_dir = os.path.join(os.path.dirname(config.airogs_label_file), "manifest_cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_filename_parts = [
        "airogs_sampled",
        f"dtype_{config.data_type}",
        f"{config.airogs_num_rg_samples}RG",
        f"{config.airogs_num_nrg_samples}NRG",
        f"seed{config.seed}.csv",
    ]
    cached_manifest_path = os.path.join(cache_dir, "_".join(cache_filename_parts))

    if config.use_airogs_cache and os.path.exists(cached_manifest_path):
        try:
            logger.info(f"Loading AIROGS data from cached manifest: {cached_manifest_path}")
            df_airogs_cached = pd.read_csv(cached_manifest_path)
            # Basic check for required columns
            if not {'image_path', 'types'}.issubset(df_airogs_cached.columns):
                logger.warning("Cached AIROGS manifest missing required columns. Re-processing.")
            elif df_airogs_cached.empty:
                logger.warning("Cached AIROGS manifest was empty. Re-processing.")
            else:
                # Optional: Verify path existence (can be slow)
                # if getattr(config, 'verify_airogs_cache_paths', False):
                #     logger.info("Verifying cached AIROGS paths...")
                #     exists_flags = [os.path.exists(p) for p in tqdm(df_airogs_cached['image_path'], desc="Verify cached AIROGS")]
                #     if (~pd.Series(exists_flags)).sum() > 0:
                #         logger.warning(f"{(~pd.Series(exists_flags)).sum()} cached AIROGS paths no longer exist. Manifest might be stale.")
                #     df_airogs_cached = df_airogs_cached[exists_flags]

                logger.info(f"Loaded {len(df_airogs_cached)} AIROGS samples from cache.")
                return df_airogs_cached
        except Exception as e:
            logger.warning(f"Failed to load AIROGS from cache {cached_manifest_path}: {e}. Re-processing.")

    # --- If cache not used or failed, proceed with full loading ---
    if not config.airogs_label_file or not airogs_img_base_dir:
        logger.warning("AIROGS label file or image directory not specified. Skipping AIROGS.")
        return pd.DataFrame()
    if not os.path.exists(config.airogs_label_file):
        logger.error(f"AIROGS label file not found: {config.airogs_label_file}")
        return pd.DataFrame()
    if not os.path.exists(airogs_img_base_dir):
        logger.error(f"AIROGS image base dir not found: {airogs_img_base_dir}")
        return pd.DataFrame()

    try:
        label_df = pd.read_csv(config.airogs_label_file)
        if not {"challenge_id", "class"}.issubset(label_df.columns):
            logger.error("AIROGS CSV missing 'challenge_id' or 'class' columns.")
            return pd.DataFrame()

        label_map = {"NRG": 0, "RG": 1}
        label_df["types"] = label_df["class"].map(label_map)
        label_df = label_df.dropna(subset=["types"])
        label_df["types"] = label_df["types"].astype(int)
        label_df["names"] = label_df["challenge_id"] + ".jpg" # Assuming .jpg, adjust if other extensions exist

        logger.info("Constructing AIROGS image paths and checking existence...")
        label_df["image_path"] = label_df["names"].apply(lambda name: os.path.join(airogs_img_base_dir, name))
        
        exists_flags = [os.path.exists(path) for path in tqdm(label_df['image_path'], desc="Verifying AIROGS files")]
        label_df['file_exists'] = exists_flags
        
        missing_files_count = (~label_df['file_exists']).sum()
        if missing_files_count > 0:
            logger.warning(f"{missing_files_count} AIROGS image files specified in CSV are missing. These entries will be excluded.")
            label_df = label_df[label_df['file_exists']]
        
        if label_df.empty:
            logger.warning("No valid AIROGS samples after file existence check.")
            return pd.DataFrame()

        df_rg = label_df[label_df["types"] == 1]
        df_nrg = label_df[label_df["types"] == 0]

        num_rg_to_sample = min(len(df_rg), config.airogs_num_rg_samples)
        num_nrg_to_sample = min(len(df_nrg), config.airogs_num_nrg_samples)

        logger.info(f"AIROGS: Found {len(df_rg)} RG, {len(df_nrg)} NRG samples with existing files.")
        logger.info(f"Sampling {num_rg_to_sample} RG and {num_nrg_to_sample} NRG samples.")

        if num_rg_to_sample == 0 and num_nrg_to_sample == 0:
            logger.warning("Not enough AIROGS samples to meet sampling criteria, or no samples requested.")
            return pd.DataFrame()

        sampled_rg_df = df_rg.sample(n=num_rg_to_sample, random_state=config.seed) if num_rg_to_sample > 0 else pd.DataFrame()
        sampled_nrg_df = df_nrg.sample(n=num_nrg_to_sample, random_state=config.seed) if num_nrg_to_sample > 0 else pd.DataFrame()
        
        df_airogs_sampled = pd.concat([sampled_rg_df, sampled_nrg_df], ignore_index=True)

        # Prepare final DataFrame structure
        df_airogs_final = pd.DataFrame()
        if not df_airogs_sampled.empty:
            df_airogs_final["names"] = df_airogs_sampled["names"]
            df_airogs_final["types"] = df_airogs_sampled["types"]
            df_airogs_final["image_path"] = df_airogs_sampled["image_path"]
            df_airogs_final["file_exists"] = True # All paths here are verified
            df_airogs_final["dataset"] = "AIROGS" # Main dataset identifier
            df_airogs_final["dataset_source"] = "AIROGS" # For finer-grained source tracking
        
        if config.use_airogs_cache and not df_airogs_final.empty:
            try:
                logger.info(f"Saving AIROGS sampled data to cache: {cached_manifest_path}")
                df_airogs_final.to_csv(cached_manifest_path, index=False)
            except Exception as e:
                logger.warning(f"Failed to save AIROGS to cache {cached_manifest_path}: {e}")
        
        logger.info(f"Loaded and sampled {len(df_airogs_final)} AIROGS images ({num_rg_to_sample} RG, {num_nrg_to_sample} NRG).")
        return df_airogs_final

    except Exception as e:
        logger.error(f"Error loading AIROGS data: {e}", exc_info=True)
        return pd.DataFrame()


def save_data_samples_for_verification(df_all: pd.DataFrame, config: argparse.Namespace):
    """
    Saves a few sample images from each identified dataset source for verification.

    Args:
        df_all: DataFrame containing all loaded data.
        config: Configuration object with output paths and sampling parameters.
    """
    logger.info("Saving data loading samples for verification...")
    sample_save_base_dir = os.path.join(config.output_dir, 'data_samples') # Save within experiment's output_dir
    sample_save_dir_typed = os.path.join(sample_save_base_dir, config.data_type)
    os.makedirs(sample_save_dir_typed, exist_ok=True)

    num_samples_per_source = config.num_data_samples_per_source
    
    # Create a temporary copy for assigning dataset_source for sampling purposes
    df_temp_sampling = df_all.copy()
    
    if 'dataset_source' not in df_temp_sampling.columns:
        df_temp_sampling['dataset_source'] = None # Initialize if not present
    
    # Ensure 'names' column exists if needed for assign_dataset_source
    if 'names' not in df_temp_sampling.columns and 'image_path' in df_temp_sampling.columns:
        df_temp_sampling['names'] = df_temp_sampling['image_path'].apply(os.path.basename)

    # Consolidate dataset_source for sampling
    # Use 'dataset' field if 'dataset_source' is NaN (e.g. for CHAKSU-Camera, AIROGS)
    df_temp_sampling['dataset_source'] = df_temp_sampling['dataset_source'].fillna(df_temp_sampling['dataset'])
    
    # For SMDG items that might not have a prefix-based source, assign using prefix logic
    # Also, fill any remaining NaNs to avoid issues with groupby
    unassigned_mask_sampling = df_temp_sampling['dataset_source'].isna() | \
                               df_temp_sampling['dataset_source'].isin(['SMDG_Unknown', 'SMDG-19'])
    if 'names' in df_temp_sampling.columns and unassigned_mask_sampling.any():
         df_temp_sampling.loc[unassigned_mask_sampling, 'dataset_source_for_sampling'] = \
             assign_dataset_source(df_temp_sampling.loc[unassigned_mask_sampling, 'names'])
    else:
        df_temp_sampling['dataset_source_for_sampling'] = df_temp_sampling['dataset_source']

    # Ensure Papilla is correctly identified if missed
    is_papilla_mask_sampling = df_temp_sampling['names'].str.lower().str.startswith(PAPILLA_PREFIX.lower(), na=False)
    df_temp_sampling.loc[is_papilla_mask_sampling, 'dataset_source_for_sampling'] = PAPILLA_PREFIX

    df_temp_sampling['dataset_source_for_sampling'] = df_temp_sampling['dataset_source_for_sampling'].fillna('Pool_For_Sampling_Check')

    sampled_counts = {}
    for source_name, source_df in df_temp_sampling.groupby('dataset_source_for_sampling'):
        if source_df.empty:
            continue
        
        samples_to_take = min(len(source_df), num_samples_per_source)
        if samples_to_take == 0:
            continue

        source_samples_df = source_df.sample(n=samples_to_take, random_state=config.seed)
        
        # Sanitize source_name for directory creation
        safe_source_name = str(source_name).replace(os.sep, "_").replace(":", "_").replace("/", "_")
        source_specific_dir = os.path.join(sample_save_dir_typed, safe_source_name)
        os.makedirs(source_specific_dir, exist_ok=True)
        
        saved_for_this_source = 0
        for _, row_sample in source_samples_df.iterrows():
            src_path = row_sample['image_path']
            if not os.path.exists(src_path):
                logger.warning(f"Sample image missing during save: {src_path} for source {source_name}")
                continue
            
            dst_filename = os.path.basename(src_path)
            base_s, ext_s = os.path.splitext(dst_filename)
            
            # Handle potential duplicate filenames from different original subfolders
            temp_dst_path = os.path.join(source_specific_dir, dst_filename)
            copy_idx = 0
            while os.path.exists(temp_dst_path):
                copy_idx += 1
                temp_dst_path = os.path.join(source_specific_dir, f"{base_s}_copy{copy_idx}{ext_s}")
            
            try:
                shutil.copy2(src_path, temp_dst_path)
                saved_for_this_source += 1
            except Exception as e_copy:
                logger.error(f"Error copying sample {src_path} to {temp_dst_path}: {e_copy}")
        
        if saved_for_this_source > 0:
            logger.info(f"Saved {saved_for_this_source} samples for source '{source_name}' to '{source_specific_dir}'.")
        sampled_counts[str(source_name)] = saved_for_this_source
        
    logger.info(f"Sample saving summary: {sampled_counts}")


def assign_dataset_source(name_series: pd.Series) -> pd.Series:
    """
    Assigns a 'dataset_source' label based on filename prefixes.

    Used primarily for SMDG sub-datasets.

    Args:
        name_series: A pandas Series of image filenames.

    Returns:
        A pandas Series with 'dataset_source' labels.
        Defaults to 'SMDG_Unknown' if no prefix matches.
    """
    labels = pd.Series("SMDG_Unknown", index=name_series.index, dtype=str)
    if name_series.empty:
        return labels

    name_series_lower = name_series.str.lower().fillna("")
    
    # Order matters if prefixes overlap; more specific should come first if necessary.
    # Current map seems fine.
    prefix_to_dataset_map = {
        "beh": "BEH", "crfo": "CRFO", "dr-hagis": "DR-HAGIS", 
        "drishti-gs1-test": "DRISHTI-GS1", "drishti-gs1-train": "DRISHTI-GS1", 
        "fives": "FIVES", "g1020": "G1020", "hrf": "HRF", 
        "jsiec": "JSIEC", "les-av": "LES-AV", 
        "oia-odir-test": "OIA-ODIR-test", "oia-odir-train": "OIA-ODIR-train", 
        "origa": "ORIGA", "papila": PAPILLA_PREFIX, # Papila also handled separately later
        "refuge1": "REFUGE1", "sjchoi86": "sjchoi86-HRF"
    }

    for prefix, dataset_name in prefix_to_dataset_map.items():
        mask = name_series_lower.str.startswith(prefix)
        labels.loc[mask] = dataset_name
    return labels


def load_and_split_data(config: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads data from all configured sources, combines them, assigns dataset sources,
    handles EyePACS exclusion, optionally saves samples, and splits into
    training, validation, and test sets.

    - PAPILLA and CHAKSU are treated as separate, additional held-out test sets.
    - OIA-ODIR-test is explicitly assigned to the script's test set.
    - OIA-ODIR-train is split (e.g., 80/20) between the script's training and validation sets.
    - The remaining pool is split approximately 60% train, 20% validation, 20% test.

    Args:
        config: Configuration object with data paths, usage flags, and split parameters.

    Returns:
        A tuple (train_df, val_df, test_df) of pandas DataFrames.
    """
    logger.info(f"--- Data Loading & Splitting (data_type: {config.data_type}) for Train/Val/Test ---")

    # --- Load SMDG-19 Data ---
    df_smdg_all = pd.DataFrame()
    if os.path.exists(config.smdg_metadata_file):
        logger.info(f"Loading SMDG-19 metadata: {config.smdg_metadata_file}, Image dir: {config.smdg_image_dir}")
        try:
            df_smdg_all = pd.read_csv(config.smdg_metadata_file)
            if "names" not in df_smdg_all.columns:
                raise ValueError("'names' column missing in SMDG metadata.")
            
            df_smdg_all["image_path"] = df_smdg_all["names"].apply(
                lambda name: os.path.join(config.smdg_image_dir, f"{name}.png")
            )
            df_smdg_all["file_exists"] = df_smdg_all["image_path"].apply(os.path.exists)
            
            missing_smdg_count = (~df_smdg_all["file_exists"]).sum()
            if missing_smdg_count > 0:
                logger.warning(f"{missing_smdg_count} SMDG files listed in metadata not found. Excluding them.")
            df_smdg_all = df_smdg_all[df_smdg_all["file_exists"]]
            
            df_smdg_all["dataset"] = "SMDG-19"
            df_smdg_all = df_smdg_all.dropna(subset=["types"])
            df_smdg_all = df_smdg_all[df_smdg_all["types"].isin([0, 1])]
            df_smdg_all["types"] = df_smdg_all["types"].astype(int)
            logger.info(f"Loaded {len(df_smdg_all)} SMDG-19 samples with existing files and valid labels.")
        except Exception as e:
            logger.error(f"Failed to load SMDG-19: {e}", exc_info=True)
            df_smdg_all = pd.DataFrame() 
    else:
        logger.warning(f"SMDG-19 metadata file not found: {config.smdg_metadata_file}. Skipping SMDG-19.")

    # --- Load CHAKSU and AIROGS Data ---
    df_chaksu = load_chaksu_data(config) if config.use_chaksu else pd.DataFrame()
    df_airogs = load_airogs_data(config) if config.use_airogs else pd.DataFrame()

    # --- Combine All Data Sources --- # <<< THIS BLOCK WAS MISSING/COMMENTED
    all_dfs_to_concat = [df for df in [df_smdg_all, df_chaksu, df_airogs] if not df.empty]
    if not all_dfs_to_concat:
        logger.critical("No data loaded from any source. Exiting.")
        sys.exit(1)
    
    df_all = pd.DataFrame() # Initialize df_all
    if len(all_dfs_to_concat) > 1:
        # Determine all unique columns across all DataFrames to concatenate
        all_cols = set()
        for df_part in all_dfs_to_concat:
            all_cols.update(df_part.columns)
        
        standardized_dfs = []
        for df_part in all_dfs_to_concat:
            temp_df = df_part.copy()
            # Add missing columns filled with NaN to ensure consistent structure
            for col_to_add in all_cols.difference(temp_df.columns):
                temp_df[col_to_add] = np.nan
            standardized_dfs.append(temp_df[list(all_cols)]) # Enforce consistent column order
        df_all = pd.concat(standardized_dfs, ignore_index=True)
    elif all_dfs_to_concat: # Only one non-empty DataFrame
        df_all = all_dfs_to_concat[0].copy()
    # --- END OF RESTORED BLOCK ---

    if df_all.empty: # Should not happen if all_dfs_to_concat was not empty, but good check
        logger.critical("df_all is empty after attempting to combine sources. Exiting.")
        sys.exit(1)

    df_all["types"] = pd.to_numeric(df_all["types"], errors="coerce").astype("Int64")
    df_all.dropna(subset=["types", "image_path"], inplace=True) # Crucial: ensure these exist and are valid
    logger.info(f"Total combined samples before further processing: {len(df_all)}")

    # --- Dataset Source Assignment ---
    if 'names' not in df_all.columns and 'image_path' in df_all.columns:
        df_all['names'] = df_all['image_path'].apply(os.path.basename)
    elif 'names' not in df_all.columns: # Should not happen if image_path exists, but defensive
        logger.warning("Missing 'names' column, prefix-based dataset_source assignment might be incomplete.")
        # Create an empty 'names' column to prevent errors later if it's expected
        df_all['names'] = pd.Series([""] * len(df_all), index=df_all.index, dtype=str)


    if "dataset_source" not in df_all.columns: # Initialize if it wasn't set by individual loaders
        df_all["dataset_source"] = None
    # Use 'dataset' (e.g. 'CHAKSU-Bosch', 'AIROGS' from their loaders) if 'dataset_source' is still NaN
    df_all["dataset_source"] = df_all["dataset_source"].fillna(df_all["dataset"])

    # Apply prefix-based assignment for SMDG sub-datasets or any remaining unassigned items
    smdg_or_unknown_mask = df_all["dataset_source"].isna() | \
                           df_all["dataset_source"].isin(["SMDG-19", "SMDG_Unknown"]) # SMDG-19 is the 'dataset' field for SMDG
    if smdg_or_unknown_mask.any() and 'names' in df_all.columns:
        df_all.loc[smdg_or_unknown_mask, "dataset_source"] = \
            assign_dataset_source(df_all.loc[smdg_or_unknown_mask, "names"])
    
    # Explicitly set PAPILLA source (overrides any prefix assignment if names start with 'papila')
    is_papilla_mask = df_all["names"].str.lower().str.startswith(PAPILLA_PREFIX.lower(), na=False)
    df_all.loc[is_papilla_mask, "dataset_source"] = PAPILLA_PREFIX
    
    # --- Drop EyePACS ---
    # Ensure EyePACS is identified if not already tagged by assign_dataset_source
    unassigned_eyepacs_mask = (df_all['dataset_source'].isna() | df_all['dataset_source'].isin(['SMDG_Unknown'])) & \
                              (df_all['names'].str.lower().str.startswith('eyepacs', na=False))
    df_all.loc[unassigned_eyepacs_mask, 'dataset_source'] = 'EyePACS' # Tag it
    
    eyepacs_count_before_drop = (df_all["dataset_source"] == "EyePACS").sum()
    if eyepacs_count_before_drop > 0:
        df_all = df_all[df_all["dataset_source"] != "EyePACS"].copy()
        logger.info(f"Dropped {eyepacs_count_before_drop} EyePACS samples. Remaining samples: {len(df_all)}")
    else:
        logger.info("No EyePACS samples identified to drop (or they were not tagged as 'EyePACS').")

    if df_all.empty:
        logger.critical("No data remaining after EyePACS drop (or initial loading was empty). Exiting.")
        sys.exit(1)

    # --- Optional: Save Data Samples for Verification ---
    if config.save_data_samples:
        save_data_samples_for_verification(df_all, config)

    # --- Define Additional Held-out Test Sets (PAPILLA, CHAKSU) ---
    # These are external to the main train/val/test split of the remaining pool
    is_papilla_final_mask = df_all["dataset_source"] == PAPILLA_PREFIX
    is_chaksu_final_mask = df_all["dataset_source"].str.startswith("CHAKSU", na=False) # e.g., CHAKSU-Bosch
    
    df_additional_held_out_test = df_all[is_papilla_final_mask | is_chaksu_final_mask].copy()
    if not df_additional_held_out_test.empty:
        logger.info(f"Identified {len(df_additional_held_out_test)} samples from PAPILLA/CHAKSU as additional held-out test sets.")
        # Optional: Save manifest for these additional held-out sets
        # add_held_out_path = os.path.join(config.output_dir, "results", "additional_held_out_test_manifest.csv")
        # df_additional_held_out_test.to_csv(add_held_out_path, index=False)
        # logger.info(f"Additional held-out manifest saved to {add_held_out_path}")

    # Pool for our main train/val/test split, EXCLUDING the above major held-out sets
    remaining_pool = df_all[~(is_papilla_final_mask | is_chaksu_final_mask)].copy()
    logger.info(f"Pool for Train/Val/Test split (after PAPILLA/CHAKSU exclusion & EyePACS drop): {len(remaining_pool)}")

    # Initialize DataFrames for train/val/test
    cols_for_empty_df = remaining_pool.columns if not remaining_pool.empty else df_all.columns
    train_df = pd.DataFrame(columns=cols_for_empty_df)
    val_df = pd.DataFrame(columns=cols_for_empty_df)
    test_df = pd.DataFrame(columns=cols_for_empty_df)

    if remaining_pool.empty:
        logger.warning("Pool for train/val/test splitting is empty. All script-defined sets (train/val/test) will be empty.")
        # We might still have df_additional_held_out_test for later evaluation.
        return train_df, val_df, test_df # Return empty DFs

    # Refine 'dataset_source' for the `remaining_pool` before splitting it
    # This ensures AIROGS and other SMDG sources are correctly labeled if they weren't already by prefix or initial load
    mask_needs_refinement = remaining_pool['dataset_source'].isna() | \
                            remaining_pool['dataset_source'].isin(['SMDG_Unknown', 'SMDG-19', None]) # Explicitly include None
    if mask_needs_refinement.any() and 'names' in remaining_pool.columns:
        remaining_pool.loc[mask_needs_refinement, 'dataset_source'] = \
            assign_dataset_source(remaining_pool.loc[mask_needs_refinement, 'names'])
    
    airogs_in_pool_mask_refinement = remaining_pool['dataset'] == 'AIROGS' # 'dataset' field from AIROGS loader
    remaining_pool.loc[airogs_in_pool_mask_refinement, 'dataset_source'] = 'AIROGS' # Ensure AIROGS is AIROGS
    
    remaining_pool['dataset_source'].fillna('Pool_Final_Unknown_For_Split', inplace=True)
    logger.info(f"Source distribution in remaining_pool before OIA-ODIR specific split:\n{remaining_pool['dataset_source'].value_counts(dropna=False)}")


    # --- Splitting Logic with new OIA-ODIR handling ---
    train_df_parts = []
    val_df_parts = []
    test_df_parts = []

    # 1. Explicitly assign OIA-ODIR-test to the script's TEST set
    is_oiaodir_test_for_script_test_mask = remaining_pool['dataset_source'] == 'OIA-ODIR-test'
    oiaodir_test_for_script_test_df = remaining_pool[is_oiaodir_test_for_script_test_mask].copy()
    if not oiaodir_test_for_script_test_df.empty:
        logger.info(f"Assigned {len(oiaodir_test_for_script_test_df)} OIA-ODIR-test samples directly to SCRIPT'S TEST set.")
        test_df_parts.append(oiaodir_test_for_script_test_df)
    
    current_processing_pool = remaining_pool[~is_oiaodir_test_for_script_test_mask].copy()

    # 2. Split OIA-ODIR-train into script's TRAIN and VALIDATION sets
    is_oiaodir_train_mask = current_processing_pool['dataset_source'] == 'OIA-ODIR-train'
    oiaodir_train_pool_df = current_processing_pool[is_oiaodir_train_mask].copy()
    
    current_processing_pool = current_processing_pool[~is_oiaodir_train_mask].copy()

    if not oiaodir_train_pool_df.empty:
        logger.info(f"OIA-ODIR-train pool (size: {len(oiaodir_train_pool_df)}) will be split into script's Train/Val (e.g., 80/20).")
        if len(oiaodir_train_pool_df) >= 2:
            oiaodir_train_pool_df['stratify_key_oia'] = oiaodir_train_pool_df['types'].astype(str)
            oia_counts = oiaodir_train_pool_df['stratify_key_oia'].value_counts()
            min_strat_oia = getattr(config, "min_samples_per_group_for_stratify", 2)
            
            can_stratify_oia = (oiaodir_train_pool_df['stratify_key_oia'].nunique() > 1 and
                                oia_counts.min() >= min_strat_oia)
            strat_col_oia = oiaodir_train_pool_df['stratify_key_oia'] if can_stratify_oia else None
            if not can_stratify_oia:
                logger.warning("Cannot stratify OIA-ODIR-train split. Proceeding unstratified.")

            oia_train_split_df, oia_val_split_df = train_test_split(
                oiaodir_train_pool_df, test_size=0.20, random_state=config.seed + 42, stratify=strat_col_oia
            )
            train_df_parts.append(oia_train_split_df.drop(columns=['stratify_key_oia'], errors='ignore'))
            val_df_parts.append(oia_val_split_df.drop(columns=['stratify_key_oia'], errors='ignore'))
            logger.info(f"Split OIA-ODIR-train -> Script Train: {len(oia_train_split_df)}, Script Val: {len(oia_val_split_df)}")
        else:
            logger.warning(f"OIA-ODIR-train pool (size: {len(oiaodir_train_pool_df)}) too small to split. Assigning all to script's training.")
            train_df_parts.append(oiaodir_train_pool_df.drop(columns=['stratify_key_oia'], errors='ignore') if 'stratify_key_oia' in oiaodir_train_pool_df.columns else oiaodir_train_pool_df)

    # 3. Handle other general 'test' datasets from the remaining `current_processing_pool`
    explicit_general_test_df = pd.DataFrame(columns=current_processing_pool.columns)
    pool_for_random_split = current_processing_pool.copy()

    if not current_processing_pool.empty and 'names' in current_processing_pool.columns:
        is_general_test_mask = (current_processing_pool['names'].str.contains('test', case=False, na=False)) & \
                               (~current_processing_pool['dataset_source'].isin(['OIA-ODIR-train', 'OIA-ODIR-test']))
        explicit_general_test_df = current_processing_pool[is_general_test_mask].copy()
        pool_for_random_split = current_processing_pool[~is_general_test_mask].copy()
    
    if not explicit_general_test_df.empty:
        logger.info(f"Assigned {len(explicit_general_test_df)} other general 'test' samples to SCRIPT'S TEST set.")
        test_df_parts.append(explicit_general_test_df)
    
    logger.info(f"'pool_for_random_split' (size: {len(pool_for_random_split)}) will be split into Train/Val/Test (approx 60/20/20).")

    # 4. Perform 60/20/20 split on the final 'pool_for_random_split'
    # ... (The rest of the splitting logic for pool_for_random_split as in your previous correct version) ...
    if len(pool_for_random_split) >= 3: 
        pool_for_random_split['stratify_key_pool'] = pool_for_random_split['dataset_source'].astype(str) + '_' + pool_for_random_split['types'].astype(str)
        pool_counts = pool_for_random_split['stratify_key_pool'].value_counts()
        min_strat_pool = getattr(config, "min_samples_per_group_for_stratify", 2) 
        
        can_stratify_pool = (pool_for_random_split['stratify_key_pool'].nunique() > 1 and 
                             pool_counts.min() >= min_strat_pool)
        
        strat_col_pool_split = pool_for_random_split['stratify_key_pool'] if can_stratify_pool else None
        if not can_stratify_pool and len(pool_for_random_split) > 0 :
            logger.warning(f"Cannot stratify 3-way split for 'pool_for_random_split'. Proceeding unstratified.")
        try:
            val_test_size_ratio = 0.40 
            
            if len(pool_for_random_split) * val_test_size_ratio < 1 and len(pool_for_random_split) > 1:
                num_samples_for_val_test = 1
                val_test_size_ratio = num_samples_for_val_test / len(pool_for_random_split)
                if len(pool_for_random_split) - num_samples_for_val_test < 1:
                     train_pool_split_main = pool_for_random_split.copy()
                     val_test_pool_main = pd.DataFrame(columns=pool_for_random_split.columns)
                else:
                    train_pool_split_main, val_test_pool_main = train_test_split(
                        pool_for_random_split, test_size=val_test_size_ratio, random_state=config.seed, stratify=strat_col_pool_split)
            elif len(pool_for_random_split) <=1:
                train_pool_split_main = pool_for_random_split.copy()
                val_test_pool_main = pd.DataFrame(columns=pool_for_random_split.columns)
            else: 
                train_pool_split_main, val_test_pool_main = train_test_split(
                    pool_for_random_split, test_size=val_test_size_ratio, random_state=config.seed, stratify=strat_col_pool_split
                )

            val_pool_split_main = pd.DataFrame(columns=pool_for_random_split.columns)
            test_pool_split_main = pd.DataFrame(columns=pool_for_random_split.columns)

            if len(val_test_pool_main) >= 2:
                strat_col_vt_main_split = val_test_pool_main['stratify_key_pool'] if strat_col_pool_split is not None and 'stratify_key_pool' in val_test_pool_main.columns else None
                if strat_col_vt_main_split is not None:
                    vt_main_counts = val_test_pool_main['stratify_key_pool'].value_counts()
                    if not (val_test_pool_main['stratify_key_pool'].nunique() > 1 and vt_main_counts.min() >= min_strat_pool):
                        logger.warning(f"Cannot stratify 50/50 val/test split for main pool. Unstratified.")
                        strat_col_vt_main_split = None
                
                val_pool_split_main, test_pool_split_main = train_test_split(
                    val_test_pool_main, test_size=0.50, random_state=config.seed + 1, stratify=strat_col_vt_main_split
                )
            elif not val_test_pool_main.empty: 
                val_pool_split_main = val_test_pool_main.copy() 
            
            train_df_parts.append(train_pool_split_main.drop(columns=['stratify_key_pool'], errors='ignore'))
            val_df_parts.append(val_pool_split_main.drop(columns=['stratify_key_pool'], errors='ignore'))
            test_df_parts.append(test_pool_split_main.drop(columns=['stratify_key_pool'], errors='ignore'))
            logger.info(f"Split 'pool_for_random_split' -> Script Train: {len(train_pool_split_main)}, Script Val: {len(val_pool_split_main)}, Script Test: {len(test_pool_split_main)}")

        except ValueError as e:
            logger.error(f"Error during 3-way split of 'pool_for_random_split': {e}. Assigning all to training.")
            train_df_parts.append(pool_for_random_split.drop(columns=['stratify_key_pool'], errors='ignore'))
    
    elif not pool_for_random_split.empty:
        logger.warning(f"'pool_for_random_split' (size: {len(pool_for_random_split)}) too few for 3-way split. Assigning all to script's training.")
        train_df_parts.append(pool_for_random_split.drop(columns=['stratify_key_pool'], errors='ignore') if 'stratify_key_pool' in pool_for_random_split.columns else pool_for_random_split)


    # Concatenate all parts for final train_df, val_df, and test_df
    train_df = pd.concat([p for p in train_df_parts if p is not None and not p.empty], ignore_index=True) if train_df_parts else pd.DataFrame(columns=cols_for_empty_df)
    val_df   = pd.concat([p for p in val_df_parts if p is not None and not p.empty], ignore_index=True) if val_df_parts else pd.DataFrame(columns=cols_for_empty_df)
    test_df  = pd.concat([p for p in test_df_parts if p is not None and not p.empty], ignore_index=True) if test_df_parts else pd.DataFrame(columns=cols_for_empty_df)
    
    # Final cleanup of any stratify_key columns
    for df_clean in [train_df, val_df, test_df]:
        if df_clean is not None: 
            for col_key in ['stratify_key_oia', 'stratify_key_pool']:
                if col_key in df_clean.columns:
                    df_clean.drop(columns=[col_key], errors='ignore', inplace=True)
    
    logger.info(f"Final Training set samples: {len(train_df)}")
    if not train_df.empty: logger.info(f"Final Training set source distribution:\n{train_df['dataset_source'].value_counts(dropna=False)}")
    
    logger.info(f"Final Validation set samples: {len(val_df)}")
    if not val_df.empty: logger.info(f"Final Validation set source distribution:\n{val_df['dataset_source'].value_counts(dropna=False)}")

    logger.info(f"Final Test set samples: {len(test_df)}")
    if not test_df.empty: logger.info(f"Final Test set source distribution:\n{test_df['dataset_source'].value_counts(dropna=False)}")

    if not df_additional_held_out_test.empty:
        logger.info(f"Additionally, {len(df_additional_held_out_test)} PAPILLA/CHAKSU samples are available as separate held-out tests.")

    return train_df, val_df, test_df

def get_transforms(image_size: int, model_name: str, use_data_augmentation: bool = True) -> tuple[Compose, Compose]:
    """
    Get training and evaluation image transformations.

    Uses mean and std from the specified timm model's data_config if available,
    otherwise defaults to ImageNet statistics.

    Args:
        image_size: The target image size (height and width).
        model_name: The name of the timm model to derive normalization from.
        use_data_augmentation: Whether to apply data augmentation to training set.

    Returns:
        A tuple (train_transforms, eval_transforms).
    """
    logger.info(f"Defining image transforms for image_size={image_size}, model={model_name}, data_aug={use_data_augmentation}")
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ImageNet defaults

    try:
        dummy_model = timm.create_model(model_name, pretrained=False)
        data_config = resolve_data_config({}, model=dummy_model)
        if data_config and 'mean' in data_config and 'std' in data_config:
            mean, std = data_config["mean"], data_config["std"]
            logger.info(f"Using model-specific normalization: mean={mean}, std={std}")
        else:
            logger.warning(f"Could not resolve mean/std from timm model '{model_name}'. Using ImageNet defaults.")
    except Exception as e:
        logger.warning(
            f"Error getting data_config from timm model '{model_name}': {e}. "
            "Using ImageNet default normalization."
        )

    if use_data_augmentation:
        train_transforms = Compose([
            Resize((image_size + 32, image_size + 32)),
            RandAugment(num_ops=2, magnitude=9, fill=0),
            RandomCrop(image_size),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        logger.info("Using data augmentation for training transforms")
    else:
        train_transforms = Compose([
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        logger.info("Using basic transforms (no augmentation) for training")

    eval_transforms = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    logger.info(f"Train Transforms: {train_transforms}")
    logger.info(f"Eval Transforms: {eval_transforms}")
    return train_transforms, eval_transforms


def create_dataloaders(
    config: argparse.Namespace,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame, 
    train_transforms: Compose,
    eval_transforms: Compose,
) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None,
           GlaucomaSubgroupDataset | None, GlaucomaSubgroupDataset | None, GlaucomaSubgroupDataset | None]:
    """
    Creates PyTorch DataLoaders for training, validation, and test sets.

    Args:
        config: Configuration object.
        train_df: DataFrame for the training set.
        val_df: DataFrame for the validation set.
        test_df: DataFrame for the test set.
        train_transforms: Transformations for training data.
        eval_transforms: Transformations for validation and test data.

    Returns:
        A tuple (train_loader, val_loader, test_loader, 
                   train_dataset, val_dataset, test_dataset).
    """
    logger.info("--- Creating DataLoaders (Train/Val/Test) ---")
    train_loader, val_loader, test_loader = None, None, None
    train_dataset, val_dataset, test_dataset = None, None, None

    num_workers = config.num_workers
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    if not train_df.empty:
        train_dataset = GlaucomaSubgroupDataset(train_df, transform=train_transforms)
        if len(train_dataset) > 0:
            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory, collate_fn=safe_collate,
                persistent_workers=persistent_workers,
                drop_last=config.drop_last_batch if hasattr(config, 'drop_last_batch') else True,
            )
            logger.info(f"Train DataLoader created with {len(train_dataset)} samples.")
        else: logger.warning("Train dataset is empty after initialization. Train loader will be None.")
    else: logger.warning("Training DataFrame is empty. No Train DataLoader will be created.")

    if not val_df.empty:
        sensitive_attrs_val = ['dataset_source']
        if 'camera' in val_df.columns: sensitive_attrs_val.append('camera')
        val_dataset = GlaucomaSubgroupDataset(val_df, transform=eval_transforms, sensitive_attributes=sensitive_attrs_val)
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset, batch_size=config.eval_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory, collate_fn=safe_collate,
                persistent_workers=persistent_workers,
            )
            logger.info(f"Validation DataLoader created with {len(val_dataset)} samples.")
        else: logger.warning("Validation dataset is empty after initialization. Val loader will be None.")
    else: logger.warning("Validation DataFrame is empty. No Val DataLoader will be created.")

    if not test_df.empty:
        sensitive_attrs_test = ['dataset_source']
        if 'camera' in test_df.columns: sensitive_attrs_test.append('camera')
        test_dataset = GlaucomaSubgroupDataset(test_df, transform=eval_transforms, sensitive_attributes=sensitive_attrs_test)
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset, batch_size=config.eval_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory, collate_fn=safe_collate,
                persistent_workers=persistent_workers,
            )
            logger.info(f"Test DataLoader created with {len(test_dataset)} samples.")
        else: logger.warning("Test dataset is empty after initialization. Test loader will be None.")
    else: logger.warning("Test DataFrame is empty. No Test DataLoader will be created.")
        
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def freeze_backbone(model, model_name):
    """Freeze backbone parameters for initial fine-tuning epochs."""
    frozen_params = 0
    
    for name, param in model.named_parameters():
        # Don't freeze the classification head
        if not any(head_attr in name for head_attr in ['head.', 'fc.', 'classifier.']):
            param.requires_grad = False
            frozen_params += param.numel()
    
    logger.info(f"Frozen {frozen_params} backbone parameters")
    return frozen_params

def unfreeze_backbone(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Unfrozen all parameters")

def create_optimizer_with_differential_lr(model, base_lr, backbone_lr_multiplier, weight_decay):
    """Create optimizer with different learning rates for backbone and head."""
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if any(head_attr in name for head_attr in ['head.', 'fc.', 'classifier.']):
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': base_lr * backbone_lr_multiplier, 'name': 'backbone'},
        {'params': head_params, 'lr': base_lr, 'name': 'head'}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    logger.info(f"Created optimizer with backbone LR: {base_lr * backbone_lr_multiplier:.2e}, head LR: {base_lr:.2e}")
    return optimizer

def save_mixup_samples_before_training(train_loader, mixup_fn, save_dir, num_samples=5, device='cpu'):
    """
    Save a few sample images after applying Mixup/CutMix transformations.
    
    Args:
        train_loader: Training DataLoader
        mixup_fn: Mixup function from timm
        save_dir: Directory to save sample images
        num_samples: Number of samples to save
        device: Device to run on
    """
    import torchvision.transforms.functional as TF
    
    try:
        # Get one batch from the training loader
        batch_iter = iter(train_loader)
        batch_data = next(batch_iter)
        
        if batch_data is None or len(batch_data) < 2:
            logger.warning("Could not get valid batch for Mixup sampling")
            return
        
        if len(batch_data) == 3:
            inputs, labels, _ = batch_data
        else:
            inputs, labels = batch_data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Apply Mixup/CutMix
        mixed_inputs, mixed_labels = mixup_fn(inputs, labels)
        
        # Save the first few samples
        samples_to_save = min(num_samples, mixed_inputs.size(0))
        
        for i in range(samples_to_save):
            # Convert tensor to PIL Image
            # Denormalize first (assuming ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            
            img_tensor = mixed_inputs[i] * std + mean
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            # Convert to PIL and save
            pil_img = TF.to_pil_image(img_tensor.cpu())
            
            # Create filename with label info
            if hasattr(mixed_labels, 'shape') and len(mixed_labels.shape) > 1:
                # Soft labels from mixup
                label_info = f"soft_{mixed_labels[i, 0]:.2f}_{mixed_labels[i, 1]:.2f}"
            else:
                # Hard labels
                label_info = f"label_{mixed_labels[i].item()}"
            
            filename = f"mixup_sample_{i+1}_{label_info}.png"
            save_path = os.path.join(save_dir, filename)
            pil_img.save(save_path)
        
        logger.info(f"Saved {samples_to_save} Mixup/CutMix sample images to {save_dir}")
        
    except Exception as e:
        logger.error(f"Error saving Mixup samples: {e}")


# --- Main Training Orchestration ---
def main(args: argparse.Namespace):
    """
    Main function to orchestrate the model training process.

    This function sets up the environment, loads data, initializes the model,
    and manages the training loop, including features like Mixup/CutMix,
    two-phase training for regression-to-classification, and temperature scaling.
    """
    start_time = datetime.now()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Output Directory and Experiment Name Setup ---
    # This block determines the output directory and a unique experiment name
    # based on whether training is being resumed or started fresh.
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            logger.error(f"Resume checkpoint not found: {args.resume_from_checkpoint}. Exiting.")
            sys.exit(1)
        # model_run_dir_resume is the parent of the 'checkpoints' dir where the .pth file is
        model_run_dir_resume = os.path.dirname(os.path.dirname(args.resume_from_checkpoint))
        timestamp_suffix = start_time.strftime("%Y%m%d_%H%M%S")
        # Create a new subdirectory for the resumed run to avoid overwriting
        output_dir = os.path.join(
            model_run_dir_resume,
            f"resumed_training_{args.num_epochs_to_add}epochs_{timestamp_suffix}"
        )
        experiment_name = os.path.basename(model_run_dir_resume) # Retain original experiment name for context
        logger.info(f"Resuming training for experiment: {experiment_name}. New outputs in: {output_dir}")
    else:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        model_short_name = args.model_name.replace("_patch16_224", "").replace("_base", "").replace("dinov2_", "d2_")
        mixup_tag = "mixup" if args.use_mixup else None # Add tag if Mixup is enabled
        exp_parts = [
            model_short_name,
            "proc" if args.data_type == "processed" else "raw",
            mixup_tag,
            args.experiment_tag if args.experiment_tag else None,
            timestamp
        ]
        experiment_name = "_".join(filter(None, exp_parts))
        output_dir = os.path.join(args.base_output_dir, experiment_name)

    args.output_dir = output_dir  # Store in args for accessibility
    args.experiment_name = experiment_name # Store full experiment name

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    args.output_dir_results = results_dir # For functions needing the results path

    # Setup directory for saving Mixup/CutMix sample images, if enabled
    mixup_sample_dir = None
    if args.use_mixup and args.save_mixup_samples:
        mixup_sample_dir = os.path.join(results_dir, "mixup_samples") # Save under this run's results
        os.makedirs(mixup_sample_dir, exist_ok=True)
        logger.info(f"Mixup/CutMix samples will be saved to: {mixup_sample_dir}")

    # --- Logging Setup ---
    # Configure file handler for logging to a file specific to this run
    log_filename = f'training_log_{start_time.strftime("%Y%m%d_%H%M%S")}.log'
    log_filepath = os.path.join(output_dir, log_filename)
    # Remove existing file handlers for this path to prevent duplicate logs if main is re-run
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_filepath:
            logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"--- Experiment: {args.experiment_name} ---")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Full script arguments: {vars(args)}")
    with open(os.path.join(results_dir, "training_configuration.json"), "w") as f:
        json.dump(vars(args), f, indent=4, cls=NpEncoder)

    # --- Data Loading and Splitting ---
    train_df, val_df, test_df = load_and_split_data(args)
    if train_df.empty:
        logger.critical("Training DataFrame is empty. Cannot proceed. Exiting.")
        sys.exit(1)
    if val_df.empty: logger.warning("Validation DataFrame is empty. Validation-dependent features might be affected.")
    if test_df.empty: logger.warning("Test DataFrame is empty.")

    # --- Transformations and DataLoaders ---
    train_transforms, eval_transforms = get_transforms(args.image_size, args.model_name, args.use_data_augmentation)
    train_loader, val_loader, test_loader, \
    train_dataset, val_dataset, test_dataset = create_dataloaders(
        args, train_df, val_df, test_df, train_transforms, eval_transforms
    )
    if train_loader is None: # Should be caught by train_df.empty check, but good safeguard
        logger.critical("Train DataLoader is None. Cannot proceed. Exiting.")
        sys.exit(1)

    # --- Save Data Manifests (lists of files in train/val/test sets) ---
    cols_to_save_manifest = ['image_path', 'types', 'dataset', 'dataset_source', 'names']
    for df_data, df_name_str in [(train_df, "training"), (val_df, "validation"), (test_df, "test")]:
        if not df_data.empty:
            manifest_path = os.path.join(results_dir, f'{df_name_str}_set_manifest.csv')
            # Ensure all desired columns are present, add others if they exist
            cols_present = [col for col in cols_to_save_manifest if col in df_data.columns]
            other_cols = [
                col for col in df_data.columns
                if col not in cols_present and col not in ['file_exists', 'stratify_key_oia', 'stratify_key_pool']
            ]
            cols_final_manifest = cols_present + other_cols
            try:
                df_to_save = df_data[cols_final_manifest].copy()
                df_to_save.to_csv(manifest_path, index=False)
                logger.info(f"{df_name_str.capitalize()} set manifest saved to: {manifest_path} (Cols: {list(df_to_save.columns)})")
            except KeyError as e_key:
                logger.error(f"KeyError saving {df_name_str} manifest: {e_key}. Available columns: {list(df_data.columns)}")
            except Exception as e_save:
                logger.error(f"Failed to save {df_name_str} set manifest: {e_save}")

    # --- Model Building ---
    logger.info(f"Building model: {args.model_name}")
    weights_path_to_load = args.resume_from_checkpoint or args.custom_weights_path
    use_timm_pretrained = (not weights_path_to_load) and args.use_timm_pretrained_if_no_custom
    model = build_classifier_model(
        model_name=args.model_name, num_classes=args.num_classes, dropout_prob=args.dropout_prob,
        pretrained=use_timm_pretrained, custom_weights_path=weights_path_to_load,
        checkpoint_key=args.checkpoint_key,
        is_regression_to_classification=args.is_regression_to_classification
    )
    model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model architecture:\n{model}")
    # Add after: model = build_classifier_model(...)
    logger.info("=== MODEL ARCHITECTURE DEBUG ===")
    logger.info(f"Model type: {type(model)}")
    if hasattr(model, 'drop'):
        logger.info(f"Model has dropout layer: {model.drop}")
        logger.info(f"Dropout probability: {model.drop.p}")
    else:
        logger.info("Model has no dropout layer")

    # Print the final few layers
    if hasattr(model, 'global_pool') and hasattr(model, 'fc'):
        logger.info(f"Global pool: {model.global_pool}")
        if hasattr(model, 'drop'):
            logger.info(f"Dropout: {model.drop}")
        logger.info(f"Final FC: {model.fc}")
    logger.info("=== END MODEL DEBUG ===")
    logger.info(f"Model built. Trainable parameters: {trainable_params}")

    # --- Mixup Function Initialization ---
    mixup_fn = None
    if args.use_mixup:
        mixup_args = {
            'mixup_alpha': args.mixup_alpha,
            'cutmix_alpha': args.cutmix_alpha,
            'cutmix_minmax': None,  # Use timm's default
            'prob': args.mixup_prob,
            'switch_prob': args.mixup_switch_prob,
            'mode': args.mixup_mode,
            'label_smoothing': args.label_smoothing_factor if args.use_label_smoothing else 0.0,
            'num_classes': args.num_classes
        }
        mixup_fn = Mixup(**mixup_args)
        logger.info(f"Using Mixup/CutMix with args: {mixup_args}")

    # --- Criterion Selection ---
    # Selects the appropriate loss function based on whether Mixup or label smoothing is used.
    if args.use_mixup:
        # SoftTargetCrossEntropy is designed for soft labels produced by Mixup
        # and can also handle label smoothing via the mixup_fn.
        criterion = SoftTargetCrossEntropy()
        logger.info("Using SoftTargetCrossEntropy (suitable for Mixup/CutMix).")
        if args.use_label_smoothing:
            logger.info(f"Label smoothing factor ({args.label_smoothing_factor}) incorporated into Mixup function.")
    elif args.use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=args.label_smoothing_factor,
            num_classes=args.num_classes
        )
        logger.info(f"Using Label Smoothing Cross Entropy with smoothing factor: {args.label_smoothing_factor}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using standard Cross Entropy Loss.")
    
    # --- Optimizer and Scheduler Setup ---
    start_epoch = 0  # Default for new training, updated if resuming
    if args.is_regression_to_classification and args.backbone_lr_multiplier != 1.0:
        optimizer = create_optimizer_with_differential_lr(
            model, args.learning_rate, args.backbone_lr_multiplier, args.weight_decay
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    scheduler_patience = max(1, args.early_stopping_patience // 2) if args.early_stopping_patience > 0 else 5
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=scheduler_patience, verbose=True)
    
    # --- AMP and Resume Setup ---
    use_amp = args.use_amp and torch.cuda.is_available()
    resumed_history_data = None # To store history loaded from checkpoint

    # --- Save Mixup Samples Before Training (if enabled) ---
    if args.use_mixup and args.save_mixup_samples and mixup_fn is not None and train_loader is not None:
        logger.info("Saving sample Mixup/CutMix images before training starts...")
        save_mixup_samples_before_training(
            train_loader, mixup_fn, mixup_sample_dir, 
            num_samples=args.num_mixup_samples_to_save, device=device
        )

    if args.resume_from_checkpoint:
        logger.info(f"Loading states from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        # Model state is loaded by build_classifier_model if resume_from_checkpoint is used as custom_weights_path.
        # If it's a training checkpoint, it might also contain optimizer, scheduler, epoch, history.
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer state loaded from checkpoint.")
        if "scheduler_state_dict" in checkpoint and scheduler: # Check if scheduler exists
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Scheduler state loaded from checkpoint.")
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1 # Start from the next epoch
            logger.info(f"Resuming training from epoch {start_epoch}")
        if "history" in checkpoint:
            resumed_history_data = checkpoint["history"]
            logger.info("Resuming with history data from checkpoint.")
        # Note: AMP scaler state would typically be handled within the training engine if saved in checkpoint.

    logger.info(f"Optimizer: {type(optimizer).__name__}, Initial LR: {args.learning_rate}, Weight Decay: {args.weight_decay}")
    logger.info(f"AMP: {use_amp}, Gradient Accumulation Steps: {args.grad_accum_steps}")
    if args.is_regression_to_classification:
        logger.info(f"Regression-to-classification mode: Freeze backbone epochs: {args.freeze_backbone_epochs}, Backbone LR multiplier: {args.backbone_lr_multiplier}")

    epochs_to_run_total = args.num_epochs # Total epochs for a fresh run
    if args.resume_from_checkpoint:
        # If resuming, calculate total epochs based on where we left off and how many more to add
        epochs_to_run_total = start_epoch + args.num_epochs_to_add
    logger.info(f"Training scheduled from epoch {start_epoch} up to epoch {epochs_to_run_total - 1} (0-indexed).")

    # --- Training Call ---
    # Consolidate arguments for train_model
    common_train_args = {
        "model": model, "train_loader": train_loader, "val_loader": val_loader,
        "criterion": criterion, "optimizer": optimizer, "scheduler": scheduler,
        "device": device, "checkpoint_dir": checkpoint_dir,
        "early_stopping_patience": args.early_stopping_patience if args.early_stopping_patience > 0 else None,
        "use_amp": use_amp, "gradient_accumulation_steps": args.grad_accum_steps,
        "mixup_fn": mixup_fn,
        "mixup_sample_dir": mixup_sample_dir,
        "num_mixup_samples_to_save": args.num_mixup_samples_to_save,
        "resume_history": resumed_history_data # Pass history loaded from checkpoint
    }

    history = {} # Initialize history dictionary to store metrics

    # Handle two-phase training (frozen backbone then unfrozen) if specified
    if args.is_regression_to_classification and args.freeze_backbone_epochs > 0:
        if start_epoch < args.freeze_backbone_epochs:
            logger.info(f"Two-phase training: Backbone will be unfrozen after epoch {args.freeze_backbone_epochs - 1}.")
            # Phase 1: Train with frozen backbone
            frozen_epochs_target = min(args.freeze_backbone_epochs, epochs_to_run_total)
            logger.info(f"Phase 1: Training with frozen backbone for {frozen_epochs_target - start_epoch} epochs (up to epoch {frozen_epochs_target - 1}).")
            
            if args.freeze_backbone_epochs > start_epoch: # Ensure freezing only if not past that stage
                freeze_backbone(model, args.model_name)

            model, history_phase1 = train_model(
                **common_train_args,
                num_epochs=frozen_epochs_target, # Train up to this epoch number
                experiment_name=f"{args.experiment_name}_phase1_frozen",
                start_epoch=start_epoch
            )
            history = history_phase1 # Start with phase 1 history
            
            # Phase 2: Unfreeze and continue training, if more epochs are scheduled
            if frozen_epochs_target < epochs_to_run_total:
                logger.info(f"Phase 2: Unfreezing backbone and continuing training from epoch {frozen_epochs_target}...")
                unfreeze_backbone(model)
                
                # Recreate optimizer for Phase 2 if differential LR was used, to ensure correct LR for unfrozen backbone
                if args.backbone_lr_multiplier != 1.0: # Check if differential LR was used initially
                    # For Phase 2, typically the backbone also uses the main learning rate
                    optimizer = create_optimizer_with_differential_lr(
                        model, args.learning_rate, 1.0, args.weight_decay # Use 1.0 multiplier for backbone now
                    )
                    logger.info("Recreated optimizer with full learning rate for backbone for Phase 2.")
                else: # Or ensure all parameters are included if optimizer wasn't differential
                    optimizer = optim.AdamW(model.parameters(), lr=optimizer.param_groups[0]['lr'], weight_decay=args.weight_decay)
                
                # Update common_train_args with new optimizer and re-initialize scheduler
                common_train_args["optimizer"] = optimizer
                scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=scheduler_patience, verbose=True)
                common_train_args["scheduler"] = scheduler
                
                common_train_args["resume_history"] = history
                
                model, history_phase2 = train_model(
                    **common_train_args,
                    num_epochs=epochs_to_run_total, # Train up to the final total number of epochs
                    experiment_name=f"{args.experiment_name}_phase2_unfrozen",
                    start_epoch=frozen_epochs_target # Start from where phase 1 left off
                )
                # Combine histories from both phases
                for key in history_phase2: # history_phase2 contains metrics for epochs of phase 2
                    history[key] = history.get(key, []) + history_phase2[key]
            # If frozen_epochs_target == epochs_to_run_total, history is already history_phase1
        else: # start_epoch is already past the backbone freezing point
            logger.info("Start epoch is past specified backbone unfreezing point. Training with unfrozen backbone.")
            unfreeze_backbone(model) # Ensure backbone is unfrozen if resuming at this stage
            model, history = train_model(
                **common_train_args,
                num_epochs=epochs_to_run_total,
                experiment_name=args.experiment_name,
                start_epoch=start_epoch
            )
    else: # Standard training (no special backbone freezing logic here)
        model, history = train_model(
            **common_train_args,
            num_epochs=epochs_to_run_total,
            experiment_name=args.experiment_name,
            start_epoch=start_epoch
        )

    # --- Post-Training Operations ---
    # Determine the last epoch actually trained and recorded
    last_epoch_trained = start_epoch - 1 # Default if no training happened or no history
    if history:
        num_recorded_epochs = 0
        for key in ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr']:
            if key in history and history[key] is not None:
                num_recorded_epochs = max(num_recorded_epochs, len(history[key]))
        
        # The last_epoch_trained should reflect the 0-indexed number of the final completed epoch.
        # If history contains N entries, and training started at start_epoch S (0-indexed),
        # the epochs trained are S, S+1, ..., S+N-1.
        # So, the last completed epoch is S + N - 1.
        if num_recorded_epochs > 0:
            # This calculation depends on how 'start_epoch' relates to the 'history' object.
            # If 'history' is purely for the *last* call to train_model, then it's:
            # last_completed_epoch_of_this_run = its_start_epoch + len(its_history_train_loss) - 1
            # The overall 'history' object should be cumulative.
            # The 'epoch' saved in checkpoints by engine.py is 0-indexed for *completed* epoch.
            # If engine.py's returned history has N items, and its start_epoch was X,
            # then last_epoch_trained = X + N - 1.
            # The 'history' object here should be the fully combined one.
            # Its length directly indicates the number of total epochs for which metrics were recorded.
            # The 'start_epoch' here is the *initial* start_epoch of the entire process.
            # If history has combined entries from epoch 0 up to M-1, its length is M.
            # So last_epoch_trained = M - 1.
             last_epoch_trained = len(history.get('train_loss', [])) -1 if history.get('train_loss') else epochs_to_run_total -1

    logger.info(f"Training concluded. Last epoch effectively trained/logged: {last_epoch_trained} (0-indexed).")

    # --- Temperature Scaling (if enabled and validation data available) ---
    temperature_value = 1.0 # Default temperature
    # Ensure val_df is not empty for calibration, as val_loader might exist but be based on an empty df
    if args.use_temperature_scaling and val_loader is not None and not val_df.empty:
        logger.info("=== Starting Temperature Scaling Calibration ===")
        # Use the original val_df as loaded and split by load_and_split_data
        new_val_df_for_eval, cal_df_for_temp = split_validation_for_calibration(
            val_df, args.calibration_split_ratio, args.seed
        )
        
        if not cal_df_for_temp.empty:
            cal_dataset = GlaucomaSubgroupDataset(cal_df_for_temp, transform=eval_transforms)
            if len(cal_dataset) > 0:
                cal_loader = DataLoader(
                    cal_dataset, batch_size=args.eval_batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=safe_collate
                )
                # `model` is the model from the final state of train_model (potentially best weights loaded by engine)
                temp_model_scaler = TemperatureScaling(model) 
                temp_model_scaler.to(device)
                temperature_value = temp_model_scaler.calibrate(
                    cal_loader, device, 
                    max_iter=args.temperature_max_iter, lr=args.temperature_lr
                )
                logger.info(f"Temperature scaling calibration completed. Optimal T = {temperature_value:.4f}")
                
                # The original 'model' weights are not changed. 'temperature_value' is stored.
                # If you need to use the temperature-scaled model for immediate evaluation here,
                # you would call `temp_model_scaler(inputs)` instead of `model(inputs)`.

                # Save a checkpoint that includes this learned temperature alongside model weights
                calibrated_info_filename = f"{args.experiment_name}_epoch{last_epoch_trained}_calibrated_T{temperature_value:.4f}.pth"
                calibrated_info_path = os.path.join(checkpoint_dir, calibrated_info_filename)
                torch.save({
                    "epoch": last_epoch_trained,
                    "model_state_dict": model.state_dict(), # Original model weights
                    "temperature": temperature_value,      # Learned temperature
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "args": vars(args),
                    "history": history # Save history up to this point
                }, calibrated_info_path)
                logger.info(f"Model weights with calibration temperature info saved to: {calibrated_info_path}")
            else:
                logger.warning("Calibration dataset was empty after GlaucomaSubgroupDataset initialization. Skipping temperature scaling.")
        else:
            logger.warning("No calibration data available after splitting validation set. Skipping temperature scaling.")
    elif args.use_temperature_scaling:
        logger.warning("Temperature scaling requested, but no validation data/loader available or val_df was empty.")

    # --- Save Final Model and History ---
    # The `model` variable should hold the weights from the best epoch if `train_model` loaded them.
    final_model_filename = f"{args.experiment_name}_final_model_epoch{last_epoch_trained}.pth"
    final_model_path = os.path.join(checkpoint_dir, final_model_filename)
    torch.save({
        "epoch": last_epoch_trained,
        "model_state_dict": model.state_dict(),
        "temperature": temperature_value, # Store the determined temperature (1.0 if not calibrated)
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "args": vars(args),
        "history": history # Save the complete, potentially combined, history
    }, final_model_path)
    logger.info(f"Final model state (epoch {last_epoch_trained}) saved to: {final_model_path}")

    if history:
        # Add final temperature scaling info to the history object before saving JSON
        history['temperature_scaling_info'] = {
            'enabled': args.use_temperature_scaling,
            'final_temperature_value': temperature_value,
            'calibration_split_ratio': args.calibration_split_ratio if args.use_temperature_scaling and (not val_df.empty if 'val_df' in locals() else False) else None
        }
        history_filename = f"training_history_epoch{last_epoch_trained}.json" # Reflects last trained epoch
        history_filepath = os.path.join(results_dir, history_filename)
        with open(history_filepath, "w") as f:
            json.dump(history, f, indent=4, cls=NpEncoder)
        logger.info(f"Final training history saved to: {history_filepath}")
    
    end_time = datetime.now()
    logger.info(f"--- Experiment Finished: {args.experiment_name} ---")
    if args.use_temperature_scaling:
        logger.info(f"Final applied/recorded temperature: {temperature_value:.4f}")
    logger.info(f"Total execution time: {end_time - start_time}")

# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Glaucoma Classification Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data Arguments ---
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument('--data_type', type=str, default='raw', choices=['raw', 'processed'],
                            help='Image data type to use (raw or processed paths).')
    data_group.add_argument('--base_data_root', type=str, default=r'D:\glaucoma\data',
                            help='Root directory for all datasets (e.g., contains raw/processed folders).')
    data_group.add_argument('--min_samples_per_group_for_stratify', type=int, default=2,
                            help='Min samples per group (source_label) needed for stratified splitting.')

    
    smdg_group = parser.add_argument_group("SMDG-19 Data")
    smdg_group.add_argument('--smdg_metadata_file', type=str,
                            default=r'D:\glaucoma\data\raw\SMDG-19\metadata - standardized.csv',
                            help='Path to SMDG-19 metadata CSV file.')
    smdg_group.add_argument('--smdg_image_dir', type=str,
                            default=r'D:\glaucoma\data\raw\SMDG-19\full-fundus\full-fundus',
                            help='Path to SMDG-19 full-fundus image directory.')

    chaksu_group = parser.add_argument_group("CHAKSU Data")
    chaksu_group.add_argument('--use_chaksu', action='store_true', default=True, help="Enable CHAKSU dataset.")
    chaksu_group.add_argument('--no_chaksu', action='store_false', dest='use_chaksu', help="Disable CHAKSU dataset.")
    chaksu_group.add_argument('--chaksu_base_dir', type=str,
                              default=r'D:\glaucoma\data\raw\Chaksu\Train\Train\1.0_Original_Fundus_Images',
                              help='Base directory for CHAKSU images (contains camera type subfolders).')
    chaksu_group.add_argument('--chaksu_decision_dir', type=str,
                              default=r'D:\glaucoma\data\raw\Chaksu\Train\Train\6.0_Glaucoma_Decision',
                              help='Directory containing CHAKSU decision CSV files.')
    chaksu_group.add_argument('--chaksu_metadata_dir', type=str,
                              default=r'D:\glaucoma\data\raw\Chaksu\Train\Train\6.0_Glaucoma_Decision\Majority',
                              help='Directory containing CHAKSU metadata CSV files (per camera).')

    airogs_group = parser.add_argument_group("AIROGS Data")
    airogs_group.add_argument('--use_airogs', action='store_true', default=True, help="Enable AIROGS dataset.")
    airogs_group.add_argument('--no_airogs', action='store_false', dest='use_airogs', help="Disable AIROGS dataset.")
    airogs_group.add_argument('--airogs_label_file', type=str,
                              default=r'D:\glaucoma\data\raw\AIROGS\train_labels.csv',
                              help='Path to AIROGS training labels CSV file.')
    airogs_group.add_argument('--airogs_image_dir', type=str,
                              default=r'D:\glaucoma\data\raw\AIROGS\img',
                              help='Path to AIROGS image directory.')
    airogs_group.add_argument('--airogs_num_rg_samples', type=int, default=1000,
                              help='Number of RG (Referable Glaucoma) samples to use from AIROGS.')
    airogs_group.add_argument('--airogs_num_nrg_samples', type=int, default=9000,
                              help='Number of NRG (No Referable Glaucoma) samples to use from AIROGS.')
    airogs_group.add_argument('--use_airogs_cache', action='store_true', default=False,
                              help="Use cached AIROGS manifest if available, generate if not.")
    airogs_group.add_argument('--no_airogs_cache', action='store_false', dest='use_airogs_cache')


    # --- Model Arguments ---
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                             help='Name of the model architecture from timm library.')
    model_group.add_argument('--num_classes', type=int, default=2, help='Number of output classes.')
    model_group.add_argument('--image_size', type=int, default=224, help='Input image size (square).')
    model_group.add_argument('--dropout_prob', type=float, default=0.1, help='Dropout probability for classifier head.')
    model_group.add_argument('--custom_weights_path', type=str, default=None,
                             help='Path to custom pre-trained weights (.pth or .pt file).')
    model_group.add_argument('--checkpoint_key', type=str, default='teacher',
                             help='Key in checkpoint file to load model state_dict (e.g., "model", "state_dict", "teacher").')
    model_group.add_argument('--use_timm_pretrained_if_no_custom', action='store_true', default=False,
                             help='If no custom_weights_path or resume_checkpoint, use timm ImageNet pretrained weights.')
    model_group.add_argument('--is_regression_to_classification', action='store_true', default=False,
                             help='Flag indicating the custom weights are from a regression model to be adapted for classification.')
    model_group.add_argument('--freeze_backbone_epochs', type=int, default=0,
                             help='Number of initial epochs to freeze backbone when fine-tuning from regression.')
    model_group.add_argument('--backbone_lr_multiplier', type=float, default=1.0,
                             help='Learning rate multiplier for backbone when fine-tuning (e.g., 0.1 for 10x smaller LR).')
    
    # --- Training Arguments ---
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument('--num_epochs', type=int, default=25, help='Total number of epochs to train for a new run.')
    train_group.add_argument('--batch_size', type=int, default=8, help='Training batch size.')
    train_group.add_argument('--eval_batch_size', type=int, default=64, help='Evaluation batch size.')
    train_group.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate for AdamW.')
    train_group.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for AdamW.')
    train_group.add_argument('--early_stopping_patience', type=int, default=0,
                             help='Patience for early stopping based on validation loss. 0 to disable.')
    train_group.add_argument('--grad_accum_steps', type=int, default=4,
                             help='Number of gradient accumulation steps.')
    train_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    train_group.add_argument('--num_workers', type=int, default=0,
                             help='Number of worker processes for DataLoader. 0 for main process.')
    train_group.add_argument('--use_amp', action='store_true', default=True, help="Enable Automatic Mixed Precision training.")
    train_group.add_argument('--no_amp', action='store_false', dest='use_amp', help="Disable Automatic Mixed Precision training.")

    # Add to train_group in the argument parser
    train_group.add_argument('--use_label_smoothing', action='store_true', default=False,
                             help='Enable label smoothing for cross entropy loss.')
    train_group.add_argument('--label_smoothing_factor', type=float, default=0.1,
                             help='Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing).')
    
    train_group.add_argument('--use_temperature_scaling', action='store_true', default=False,
                         help='Enable temperature scaling for model calibration.')
    train_group.add_argument('--calibration_split_ratio', type=float, default=0.3,
                             help='Fraction of validation set to use for temperature calibration (0.0-1.0).')
    train_group.add_argument('--temperature_lr', type=float, default=0.01,
                             help='Learning rate for temperature parameter optimization.')
    train_group.add_argument('--temperature_max_iter', type=int, default=50,
                             help='Maximum iterations for temperature optimization.')

    # === NEW MIXUP ARGUMENTS ===
    train_group.add_argument('--use_mixup', action='store_true', default=False,
                             help='Enable Mixup and CutMix data augmentation.')
    train_group.add_argument('--mixup_alpha', type=float, default=0.8,
                             help='Mixup alpha, mixup enabled if > 0.')
    train_group.add_argument('--cutmix_alpha', type=float, default=1.0,
                             help='Cutmix alpha, cutmix enabled if > 0.')
    train_group.add_argument('--mixup_prob', type=float, default=1.0,
                             help='Probability of performing mixup or cutmix when either/both is enabled.')
    train_group.add_argument('--mixup_switch_prob', type=float, default=0.5,
                             help='Probability of switching to cutmix when both mixup and cutmix enabled.')
    train_group.add_argument('--mixup_mode', type=str, default='batch',
                             help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem".')
    train_group.add_argument('--save_mixup_samples', action='store_true', default=True,
                             help='Save a few sample images after Mixup/CutMix is applied.')
    train_group.add_argument('--num_mixup_samples_to_save', type=int, default=5,
                             help='Number of Mixup/CutMix augmented samples to save per epoch (if enabled).')
    train_group.add_argument('--drop_last_batch', action='store_true', default=True,
                         help='Drop the last incomplete batch during training to ensure even batch sizes for Mixup.')

    train_group.add_argument('--use_data_augmentation', action='store_true', default=True,
                             help='Enable data augmentation (RandAugment, crops, flips).')
    train_group.add_argument('--no_data_augmentation', action='store_false', dest='use_data_augmentation',
                             help='Disable data augmentation - use only resize and normalize.')    

    # --- Experiment Management Arguments ---
    exp_group = parser.add_argument_group("Experiment Management")
    exp_group.add_argument('--base_output_dir', type=str, default='experiments/training_runs_glaucoma',
                           help='Base directory to save experiment outputs.')
    exp_group.add_argument('--experiment_tag', type=str, default='',
                           help='Custom tag to append to the experiment name.')
    exp_group.add_argument('--resume_from_checkpoint', type=str, default=None,
                           help='Path to a checkpoint .pth file to resume training from.')
    exp_group.add_argument('--num_epochs_to_add', type=int, default=25,
                           help='Number of additional epochs to train when resuming.')

    # --- Data Verification Arguments ---
    verify_group = parser.add_argument_group("Data Verification")
    verify_group.add_argument('--save_data_samples', action='store_true', default=False,
                              help="Save a few sample images from each data source for verification.")
    verify_group.add_argument('--num_data_samples_per_source', type=int, default=3,
                              help="Number of image samples to save per identified data source.")
    
    # --- Visualization Arguments (New Group) ---
    vis_group = parser.add_argument_group("Visualization Parameters")
    vis_group.add_argument('--run_gradcam_visualizations', action='store_true', default=False,
                           help="Enable Grad-CAM visualizations after training.")
    vis_group.add_argument('--num_gradcam_samples', type=int, default=3,
                           help="Number of samples per category for Grad-CAM visualizations.")

    args = parser.parse_args()

    BASE_DATA_DIR_CONFIG_KEY = args.base_data_root 

    if args.data_type == 'processed':
        logger.info(f"Data type '{args.data_type}'. Adjusting image paths...")
        args.smdg_image_dir = adjust_path_for_data_type(args.smdg_image_dir, args.data_type)
        args.chaksu_base_dir = adjust_path_for_data_type(args.chaksu_base_dir, args.data_type)
        args.airogs_image_dir = adjust_path_for_data_type(args.airogs_image_dir, args.data_type)
        logger.info(f"  Adjusted SMDG Img Dir: {args.smdg_image_dir}")
        logger.info(f"  Adjusted Chaksu Base Dir: {args.chaksu_base_dir}")
        logger.info(f"  Adjusted AIROGS Img Dir: {args.airogs_image_dir}")
    else:
        logger.info(f"Data type '{args.data_type}'. Using raw paths.")

    if args.resume_from_checkpoint and not os.path.exists(args.resume_from_checkpoint): 
        logger.error(f"Resume ckpt {args.resume_from_checkpoint} missing."); sys.exit(1)
    if not args.resume_from_checkpoint and args.custom_weights_path and not os.path.exists(args.custom_weights_path): 
        logger.warning(f"Custom weights {args.custom_weights_path} missing.")
    if args.resume_from_checkpoint: 
        logger.info(f"Resuming. --num_epochs ({args.num_epochs}) ignored. Training for {args.num_epochs_to_add} more epochs.")

    main(args)