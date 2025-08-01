#!/usr/bin/env python3
"""
Glaucoma Classification Model Training Script.

End-to-end training pipeline for glaucoma classification with support for:
- Multiple data sources (SMDG-19, CHAKSU, AIROGS)
- Fine-tuning strategies (linear probing, gradual unfreezing, full fine-tuning)
- Advanced augmentation (Mixup/CutMix) and regularization techniques
- Model calibration with temperature scaling
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
    from src.data.loaders import extract_base_chaksu, assign_dataset_source, load_chaksu_data, load_airogs_data, split_validation_for_calibration
    from src.data.transforms import get_transforms
    from src.data.utils import create_dataloaders
    from src.models.classification.build_model import build_classifier_model
    from src.models.losses import LabelSmoothingCrossEntropy, TemperatureScaling
    from src.training.engine import train_model
    from src.training.fine_tuning import (
        freeze_backbone, unfreeze_backbone, create_optimizer_for_strategy,
        setup_fine_tuning_strategy, should_update_gradual_strategy,
        create_optimizer_with_differential_lr
    )
    from src.training.utils import save_mixup_samples_before_training, save_data_samples_for_verification
    from src.utils.gradcam_utils import visualize_gradcam_misclassifications, get_default_target_layers
    from src.utils.fine_tune_tools import (
        freeze_up_to, param_groups_llrd, get_gradual_unfreeze_patterns,
        count_trainable_parameters, print_parameter_status
    )
    from src.utils.paths import adjust_path_for_data_type, set_base_data_dir
    from src.utils.config_loader import load_config
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print(
        "Please ensure 'src' directory is in PYTHONPATH or accessible from the script's location."
    )
    sys.exit(1)

# Global Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PAPILLA_PREFIX = "PAPILA"
BASE_DATA_DIR_CONFIG_KEY = ""  # Set from args in main
RAW_DIR_NAME = "raw"
PROCESSED_DIR_NAME = "processed"


def load_and_split_data(config: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data from all sources, combine, assign dataset sources, and split into train/val/test sets."""
    logger.info(f"--- Data Loading & Splitting (data_type: {config.data_type}) for Train/Val/Test ---")

    # Load SMDG-19 Data
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

    # Load CHAKSU and AIROGS Data
    df_chaksu = load_chaksu_data(config) if config.use_chaksu else pd.DataFrame()
    df_airogs = load_airogs_data(config) if config.use_airogs else pd.DataFrame()

    # Combine All Data Sources
    all_dfs_to_concat = [df for df in [df_smdg_all, df_chaksu, df_airogs] if not df.empty]
    if not all_dfs_to_concat:
        logger.critical("No data loaded from any source. Exiting.")
        sys.exit(1)
    
    df_all = pd.DataFrame()
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
            standardized_dfs.append(temp_df[list(all_cols)])
        df_all = pd.concat(standardized_dfs, ignore_index=True)
    elif all_dfs_to_concat:
        df_all = all_dfs_to_concat[0].copy()

    if df_all.empty:
        logger.critical("df_all is empty after attempting to combine sources. Exiting.")
        sys.exit(1)

    df_all["types"] = pd.to_numeric(df_all["types"], errors="coerce").astype("Int64")
    df_all.dropna(subset=["types", "image_path"], inplace=True)
    logger.info(f"Total combined samples before further processing: {len(df_all)}")

    # Dataset Source Assignment
    if 'names' not in df_all.columns and 'image_path' in df_all.columns:
        df_all['names'] = df_all['image_path'].apply(os.path.basename)
    elif 'names' not in df_all.columns:
        logger.warning("Missing 'names' column, prefix-based dataset_source assignment might be incomplete.")
        df_all['names'] = pd.Series([""] * len(df_all), index=df_all.index, dtype=str)

    if "dataset_source" not in df_all.columns:
        df_all["dataset_source"] = None
    df_all["dataset_source"] = df_all["dataset_source"].fillna(df_all["dataset"])

    # Apply prefix-based assignment for SMDG sub-datasets
    smdg_or_unknown_mask = df_all["dataset_source"].isna() | \
                           df_all["dataset_source"].isin(["SMDG-19", "SMDG_Unknown"])
    if smdg_or_unknown_mask.any() and 'names' in df_all.columns:
        df_all.loc[smdg_or_unknown_mask, "dataset_source"] = \
            assign_dataset_source(df_all.loc[smdg_or_unknown_mask, "names"])
    
    # Explicitly set PAPILLA source
    is_papilla_mask = df_all["names"].str.lower().str.startswith(PAPILLA_PREFIX.lower(), na=False)
    df_all.loc[is_papilla_mask, "dataset_source"] = PAPILLA_PREFIX
    
    # Drop EyePACS samples
    unassigned_eyepacs_mask = (df_all['dataset_source'].isna() | df_all['dataset_source'].isin(['SMDG_Unknown'])) & \
                              (df_all['names'].str.lower().str.startswith('eyepacs', na=False))
    df_all.loc[unassigned_eyepacs_mask, 'dataset_source'] = 'EyePACS'
    
    eyepacs_count_before_drop = (df_all["dataset_source"] == "EyePACS").sum()
    if eyepacs_count_before_drop > 0:
        df_all = df_all[df_all["dataset_source"] != "EyePACS"].copy()
        logger.info(f"Dropped {eyepacs_count_before_drop} EyePACS samples. Remaining samples: {len(df_all)}")
    else:
        logger.info("No EyePACS samples identified to drop.")

    if df_all.empty:
        logger.critical("No data remaining after EyePACS drop. Exiting.")
        sys.exit(1)

    # Optional: Save Data Samples for Verification
    if config.save_data_samples:
        save_data_samples_for_verification(df_all, config)

    # Define Additional Held-out Test Sets (PAPILLA, CHAKSU)
    is_papilla_final_mask = df_all["dataset_source"] == PAPILLA_PREFIX
    is_chaksu_final_mask = df_all["dataset_source"].str.startswith("CHAKSU", na=False)
    
    df_additional_held_out_test = df_all[is_papilla_final_mask | is_chaksu_final_mask].copy()
    if not df_additional_held_out_test.empty:
        logger.info(f"Identified {len(df_additional_held_out_test)} samples from PAPILLA/CHAKSU as additional held-out test sets.")

    # Pool for main train/val/test split, EXCLUDING the above held-out sets
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

# --- Main Training Orchestration ---
def main(args: argparse.Namespace):
    """Main training function."""
    start_time = datetime.now()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup output directory and experiment name
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            logger.error(f"Resume checkpoint not found: {args.resume_from_checkpoint}. Exiting.")
            sys.exit(1)
        model_run_dir_resume = os.path.dirname(os.path.dirname(args.resume_from_checkpoint))
        timestamp_suffix = start_time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            model_run_dir_resume,
            f"resumed_training_{args.num_epochs_to_add}epochs_{timestamp_suffix}"
        )
        experiment_name = os.path.basename(model_run_dir_resume)
        logger.info(f"Resuming training for experiment: {experiment_name}. New outputs in: {output_dir}")
    else:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        model_short_name = args.model_name.replace("_patch16_224", "").replace("_base", "").replace("dinov2_", "d2_")
        mixup_tag = "mixup" if args.use_mixup else None
        exp_parts = [
            model_short_name,
            "proc" if args.data_type == "processed" else "raw",
            mixup_tag,
            args.experiment_tag if args.experiment_tag else None,
            timestamp
        ]
        experiment_name = "_".join(filter(None, exp_parts))
        output_dir = os.path.join(args.base_output_dir, experiment_name)

    args.output_dir = output_dir
    args.experiment_name = experiment_name

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    args.output_dir_results = results_dir

    # Setup Mixup sample directory
    mixup_sample_dir = None
    if args.use_mixup and args.save_mixup_samples:
        mixup_sample_dir = os.path.join(results_dir, "mixup_samples")
        os.makedirs(mixup_sample_dir, exist_ok=True)
        logger.info(f"Mixup/CutMix samples will be saved to: {mixup_sample_dir}")

    # Setup logging
    log_filename = f'training_log_{start_time.strftime("%Y%m%d_%H%M%S")}.log'
    log_filepath = os.path.join(output_dir, log_filename)
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

    # Data Loading and Splitting
    train_df, val_df, test_df = load_and_split_data(args)
    if train_df.empty:
        logger.critical("Training DataFrame is empty. Cannot proceed. Exiting.")
        sys.exit(1)
    if val_df.empty: 
        logger.warning("Validation DataFrame is empty. Validation-dependent features might be affected.")
    if test_df.empty: 
        logger.warning("Test DataFrame is empty.")

    # --- Transformations and DataLoaders ---
    train_transforms, eval_transforms = get_transforms(args.image_size, args.model_name, args.use_data_augmentation)
    train_loader, val_loader, test_loader, \
    train_dataset, val_dataset, test_dataset = create_dataloaders(
        args, train_df, val_df, test_df, train_transforms, eval_transforms
    )
    if train_loader is None: # Should be caught by train_df.empty check, but good safeguard
        logger.critical("Train DataLoader is None. Cannot proceed. Exiting.")
        sys.exit(1)

    # Save Data Manifests
    cols_to_save_manifest = ['image_path', 'types', 'dataset', 'dataset_source', 'names']
    for df_data, df_name_str in [(train_df, "training"), (val_df, "validation"), (test_df, "test")]:
        if not df_data.empty:
            manifest_path = os.path.join(results_dir, f'{df_name_str}_set_manifest.csv')
            cols_present = [col for col in cols_to_save_manifest if col in df_data.columns]
            other_cols = [
                col for col in df_data.columns
                if col not in cols_present and col not in ['file_exists', 'stratify_key_oia', 'stratify_key_pool']
            ]
            cols_final_manifest = cols_present + other_cols
            try:
                df_to_save = df_data[cols_final_manifest].copy()
                df_to_save.to_csv(manifest_path, index=False)
                logger.info(f"{df_name_str.capitalize()} set manifest saved to: {manifest_path}")
            except KeyError as e_key:
                logger.error(f"KeyError saving {df_name_str} manifest: {e_key}. Available columns: {list(df_data.columns)}")
            except Exception as e_save:
                logger.error(f"Failed to save {df_name_str} set manifest: {e_save}")

    # Model Building
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
    logger.info(f"Model built. Trainable parameters: {trainable_params}")

    # Mixup Function Initialization
    mixup_fn = None
    if args.use_mixup:
        mixup_args = {
            'mixup_alpha': args.mixup_alpha,
            'cutmix_alpha': args.cutmix_alpha,
            'cutmix_minmax': None,
            'prob': args.mixup_prob,
            'switch_prob': args.mixup_switch_prob,
            'mode': args.mixup_mode,
            'label_smoothing': args.label_smoothing_factor if args.use_label_smoothing else 0.0,
            'num_classes': args.num_classes
        }
        mixup_fn = Mixup(**mixup_args)
        logger.info(f"Using Mixup/CutMix with args: {mixup_args}")

    # Criterion Selection
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
    
    # --- Initialize Training State ---
    start_epoch = 0  # Default for new training, updated if resuming
    use_amp = args.use_amp and torch.cuda.is_available()
    resumed_history_data = None # To store history loaded from checkpoint
    
    # --- Optimizer and Scheduler Setup with Fine-tuning Strategy ---
    logger.info(f"=== Setting up Fine-tuning Strategy: {args.ft_strategy} ===")
    
    # Set up initial fine-tuning strategy
    optimizer, strategy_info = setup_fine_tuning_strategy(model, args, current_epoch=start_epoch)
    
    # Store strategy state for gradual unfreezing
    gradual_state = {
        'current_phase': 0,
        'last_improvement_epoch': start_epoch,
        'best_val_loss': float('inf')
    }
    
    scheduler_patience = max(1, args.early_stopping_patience // 2) if args.early_stopping_patience > 0 else 5
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=scheduler_patience, verbose=True)

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

    # --- Training with Fine-tuning Strategy ---
    logger.info(f"=== Starting Training with {args.ft_strategy.upper()} Strategy ===")
    
    # Consolidate arguments for train_model
    common_train_args = {
        "train_loader": train_loader, "val_loader": val_loader,
        "criterion": criterion, "optimizer": optimizer, "scheduler": scheduler,
        "device": device, "checkpoint_dir": checkpoint_dir,
        "early_stopping_patience": args.early_stopping_patience if args.early_stopping_patience > 0 else None,
        "use_amp": use_amp, "gradient_accumulation_steps": args.grad_accum_steps,
        "mixup_fn": mixup_fn,
        "mixup_sample_dir": mixup_sample_dir,
        "num_mixup_samples_to_save": args.num_mixup_samples_to_save,
        "resume_history": resumed_history_data
    }
    
    history = {}  # Initialize history dictionary
    
    # Handle different fine-tuning strategies
    if args.ft_strategy == 'linear':
        # Linear probing: train for limited epochs with frozen backbone
        effective_epochs = min(args.linear_probe_epochs, epochs_to_run_total)
        logger.info(f"Linear probing: Training for {effective_epochs} epochs with frozen backbone")
        
        model, history = train_model(
            model=model,
            **common_train_args,
            num_epochs=effective_epochs,
            experiment_name=f"{args.experiment_name}_linear_probe",
            start_epoch=start_epoch
        )
        
    elif args.ft_strategy == 'gradual':
        # Gradual unfreezing: use simplified approach for now
        logger.info(f"Gradual unfreezing: Training with phase transitions every {args.gradual_patience} epochs")
        
        # For now, use a simple approach: train in phases
        current_epoch = start_epoch
        patterns = get_gradual_unfreeze_patterns(args.model_name)
        
        # Phase 0: Head only
        if current_epoch < args.gradual_patience:
            phase_epochs = min(args.gradual_patience, epochs_to_run_total)
            freeze_up_to(model, patterns['phase_0'])
            optimizer, _ = setup_fine_tuning_strategy(model, args, current_epoch)
            common_train_args['optimizer'] = optimizer
            
            logger.info(f"Gradual Phase 0: Training head-only for epochs {current_epoch} to {phase_epochs-1}")
            model, history_p0 = train_model(
                model=model,
                **common_train_args,
                num_epochs=phase_epochs,
                experiment_name=f"{args.experiment_name}_gradual_p0",
                start_epoch=current_epoch
            )
            history = history_p0
            current_epoch = phase_epochs
        
        # Phase 1: Head + top layers
        if current_epoch < 2 * args.gradual_patience and current_epoch < epochs_to_run_total:
            phase_epochs = min(2 * args.gradual_patience, epochs_to_run_total)
            freeze_up_to(model, patterns['phase_1'])
            optimizer, _ = setup_fine_tuning_strategy(model, args, current_epoch)
            common_train_args['optimizer'] = optimizer
            common_train_args['resume_history'] = history
            
            logger.info(f"Gradual Phase 1: Training head+top layers for epochs {current_epoch} to {phase_epochs-1}")
            model, history_p1 = train_model(
                model=model,
                **common_train_args,
                num_epochs=phase_epochs,
                experiment_name=f"{args.experiment_name}_gradual_p1",
                start_epoch=current_epoch
            )
            # Combine histories
            for key in history_p1:
                history[key] = history.get(key, []) + history_p1[key]
            current_epoch = phase_epochs
        
        # Phase 2: Full model
        if current_epoch < epochs_to_run_total:
            freeze_up_to(model, patterns['phase_2'])  # Unfreeze everything
            optimizer, _ = setup_fine_tuning_strategy(model, args, current_epoch)
            common_train_args['optimizer'] = optimizer
            common_train_args['resume_history'] = history
            
            logger.info(f"Gradual Phase 2: Training full model for epochs {current_epoch} to {epochs_to_run_total-1}")
            model, history_p2 = train_model(
                model=model,
                **common_train_args,
                num_epochs=epochs_to_run_total,
                experiment_name=f"{args.experiment_name}_gradual_p2",
                start_epoch=current_epoch
            )
            # Combine histories
            for key in history_p2:
                history[key] = history.get(key, []) + history_p2[key]
        
    elif args.ft_strategy == 'full':
        # Full fine-tuning with LLRD
        logger.info(f"Full fine-tuning: Training all layers with LLRD (decay={args.llrd_decay})")
        
        model, history = train_model(
            model=model,
            **common_train_args,
            num_epochs=epochs_to_run_total,
            experiment_name=f"{args.experiment_name}_full_llrd",
            start_epoch=start_epoch
        )
    
    else:
        logger.error(f"Unknown fine-tuning strategy: {args.ft_strategy}")
        sys.exit(1)

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
    logger.info(f"Fine-tuning strategy used: {args.ft_strategy}")
    if args.ft_strategy == 'full':
        logger.info(f"LLRD decay factor: {args.llrd_decay}")
    elif args.ft_strategy == 'gradual':
        logger.info(f"Gradual unfreezing patience: {args.gradual_patience} epochs")
    elif args.ft_strategy == 'linear':
        logger.info(f"Linear probe epochs: {args.linear_probe_epochs}")
    if args.use_temperature_scaling:
        logger.info(f"Final applied/recorded temperature: {temperature_value:.4f}")
    logger.info(f"Total execution time: {end_time - start_time}")

# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Glaucoma Classification Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a YAML configuration file with default arguments.')

    # Data Configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument('--data_type', type=str, default='raw', choices=['raw', 'processed'],
                            help='Image data type to use (raw or processed paths).')
    data_group.add_argument('--base_data_root', type=str, default=r'D:\glaucoma\data',
                            help='Root directory for all datasets.')
    data_group.add_argument('--min_samples_per_group_for_stratify', type=int, default=2,
                            help='Min samples per group needed for stratified splitting.')

    # SMDG-19 Data
    smdg_group = parser.add_argument_group("SMDG-19 Data")
    smdg_group.add_argument('--smdg_metadata_file', type=str,
                            default=r'D:\glaucoma\data\raw\SMDG-19\metadata - standardized.csv',
                            help='Path to SMDG-19 metadata CSV file.')
    smdg_group.add_argument('--smdg_image_dir', type=str,
                            default=r'D:\glaucoma\data\raw\SMDG-19\full-fundus\full-fundus',
                            help='Path to SMDG-19 full-fundus image directory.')

    # CHAKSU Data
    chaksu_group = parser.add_argument_group("CHAKSU Data")
    chaksu_group.add_argument('--use_chaksu', action='store_true', default=True, help="Enable CHAKSU dataset.")
    chaksu_group.add_argument('--no_chaksu', action='store_false', dest='use_chaksu', help="Disable CHAKSU dataset.")
    chaksu_group.add_argument('--chaksu_base_dir', type=str,
                              default=r'D:\glaucoma\data\raw\Chaksu\Train\Train\1.0_Original_Fundus_Images',
                              help='Base directory for CHAKSU images.')
    chaksu_group.add_argument('--chaksu_decision_dir', type=str,
                              default=r'D:\glaucoma\data\raw\Chaksu\Train\Train\6.0_Glaucoma_Decision',
                              help='Directory containing CHAKSU decision CSV files.')
    chaksu_group.add_argument('--chaksu_metadata_dir', type=str,
                              default=r'D:\glaucoma\data\raw\Chaksu\Train\Train\6.0_Glaucoma_Decision\Majority',
                              help='Directory containing CHAKSU metadata CSV files.')

    # AIROGS Data
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
                              help='Number of RG samples to use from AIROGS.')
    airogs_group.add_argument('--airogs_num_nrg_samples', type=int, default=9000,
                              help='Number of NRG samples to use from AIROGS.')
    airogs_group.add_argument('--use_airogs_cache', action='store_true', default=False,
                              help="Use cached AIROGS manifest if available.")

    # Model Configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                             help='Name of the model architecture from timm library.')
    model_group.add_argument('--num_classes', type=int, default=2, help='Number of output classes.')
    model_group.add_argument('--image_size', type=int, default=224, help='Input image size (square).')
    model_group.add_argument('--dropout_prob', type=float, default=0.1, help='Dropout probability for classifier head.')
    model_group.add_argument('--custom_weights_path', type=str, default=None,
                             help='Path to custom pre-trained weights.')
    model_group.add_argument('--checkpoint_key', type=str, default='teacher',
                             help='Key in checkpoint file to load model state_dict.')
    model_group.add_argument('--use_timm_pretrained_if_no_custom', action='store_true', default=False,
                             help='Use timm ImageNet pretrained weights if no custom weights provided.')
    model_group.add_argument('--is_regression_to_classification', action='store_true', default=False,
                             help='Flag indicating custom weights are from a regression model.')
    model_group.add_argument('--freeze_backbone_epochs', type=int, default=0,
                             help='Number of initial epochs to freeze backbone.')
    model_group.add_argument('--backbone_lr_multiplier', type=float, default=1.0,
                             help='Learning rate multiplier for backbone when fine-tuning.')

    # Training Parameters
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument('--num_epochs', type=int, default=25, help='Total number of epochs to train.')
    train_group.add_argument('--batch_size', type=int, default=8, help='Training batch size.')
    train_group.add_argument('--eval_batch_size', type=int, default=64, help='Evaluation batch size.')
    train_group.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate for AdamW.')
    train_group.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for AdamW.')
    train_group.add_argument('--early_stopping_patience', type=int, default=0,
                             help='Patience for early stopping. 0 to disable.')
    train_group.add_argument('--grad_accum_steps', type=int, default=4,
                             help='Number of gradient accumulation steps.')
    train_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    train_group.add_argument('--num_workers', type=int, default=0,
                             help='Number of worker processes for DataLoader.')
    train_group.add_argument('--use_amp', action='store_true', default=True, help="Enable Automatic Mixed Precision.")
    train_group.add_argument('--no_amp', action='store_false', dest='use_amp', help="Disable AMP.")

    # Label smoothing and temperature scaling
    train_group.add_argument('--use_label_smoothing', action='store_true', default=False,
                             help='Enable label smoothing for cross entropy loss.')
    train_group.add_argument('--label_smoothing_factor', type=float, default=0.1,
                             help='Label smoothing factor.')
    train_group.add_argument('--use_temperature_scaling', action='store_true', default=False,
                             help='Enable temperature scaling for model calibration.')
    train_group.add_argument('--calibration_split_ratio', type=float, default=0.3,
                             help='Fraction of validation set to use for temperature calibration.')
    train_group.add_argument('--temperature_lr', type=float, default=0.01,
                             help='Learning rate for temperature parameter optimization.')
    train_group.add_argument('--temperature_max_iter', type=int, default=50,
                             help='Maximum iterations for temperature optimization.')

    # Mixup/CutMix Arguments
    train_group.add_argument('--use_mixup', action='store_true', default=False,
                             help='Enable Mixup and CutMix data augmentation.')
    train_group.add_argument('--mixup_alpha', type=float, default=0.8,
                             help='Mixup alpha, mixup enabled if > 0.')
    train_group.add_argument('--cutmix_alpha', type=float, default=1.0,
                             help='Cutmix alpha, cutmix enabled if > 0.')
    train_group.add_argument('--mixup_prob', type=float, default=1.0,
                             help='Probability of performing mixup or cutmix.')
    train_group.add_argument('--mixup_switch_prob', type=float, default=0.5,
                             help='Probability of switching to cutmix when both mixup and cutmix enabled.')
    train_group.add_argument('--mixup_mode', type=str, default='batch',
                             help='How to apply mixup/cutmix params.')
    train_group.add_argument('--save_mixup_samples', action='store_true', default=True,
                             help='Save sample images after Mixup/CutMix is applied.')
    train_group.add_argument('--num_mixup_samples_to_save', type=int, default=5,
                             help='Number of Mixup/CutMix augmented samples to save per epoch.')
    train_group.add_argument('--drop_last_batch', action='store_true', default=True,
                             help='Drop the last incomplete batch during training.')

    train_group.add_argument('--use_data_augmentation', action='store_true', default=True,
                             help='Enable data augmentation.')
    train_group.add_argument('--no_data_augmentation', action='store_false', dest='use_data_augmentation',
                             help='Disable data augmentation.')

    # Experiment Management
    exp_group = parser.add_argument_group("Experiment Management")
    exp_group.add_argument('--base_output_dir', type=str, default='experiments/training_runs_glaucoma',
                           help='Base directory to save experiment outputs.')
    exp_group.add_argument('--experiment_tag', type=str, default='',
                           help='Custom tag to append to the experiment name.')
    exp_group.add_argument('--resume_from_checkpoint', type=str, default=None,
                           help='Path to a checkpoint .pth file to resume training from.')
    exp_group.add_argument('--num_epochs_to_add', type=int, default=25,
                           help='Number of additional epochs to train when resuming.')

    # Data Verification
    verify_group = parser.add_argument_group("Data Verification")
    verify_group.add_argument('--save_data_samples', action='store_true', default=False,
                              help="Save sample images from each data source for verification.")
    verify_group.add_argument('--num_data_samples_per_source', type=int, default=3,
                              help="Number of image samples to save per data source.")

    # Visualization
    vis_group = parser.add_argument_group("Visualization Parameters")
    vis_group.add_argument('--run_gradcam_visualizations', action='store_true', default=False,
                           help="Enable Grad-CAM visualizations after training.")
    vis_group.add_argument('--num_gradcam_samples', type=int, default=3,
                           help="Number of samples per category for Grad-CAM visualizations.")

    # Fine-tuning Strategy
    tune_group = parser.add_argument_group("Fine-tuning Strategy")
    tune_group.add_argument('--ft_strategy', type=str, default='full',
                           choices=['linear', 'gradual', 'full'],
                           help='Fine-tuning strategy: linear=head only, gradual=ULMFit style, full=all layers with LLRD')
    tune_group.add_argument('--llrd_decay', type=float, default=0.9,
                           help='Layer-wise learning rate decay factor when using --ft_strategy full')
    tune_group.add_argument('--gradual_patience', type=int, default=3,
                           help='Epochs to wait before unfreezing next layer group in gradual strategy')
    tune_group.add_argument('--linear_probe_epochs', type=int, default=10,
                           help='Maximum epochs for linear probing (only used with --ft_strategy linear)')

    args, _ = parser.parse_known_args()
    if args.config:
        config_values = load_config(args.config)
        if isinstance(config_values, dict):
            parser.set_defaults(**config_values)
    args = parser.parse_args()

    BASE_DATA_DIR_CONFIG_KEY = args.base_data_root
    set_base_data_dir(args.base_data_root)

    # Adjust paths for processed data if needed
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

    # Validation checks
    if args.resume_from_checkpoint and not os.path.exists(args.resume_from_checkpoint):
        logger.error(f"Resume checkpoint {args.resume_from_checkpoint} missing.")
        sys.exit(1)
    if not args.resume_from_checkpoint and args.custom_weights_path and not os.path.exists(args.custom_weights_path):
        logger.warning(f"Custom weights {args.custom_weights_path} missing.")
    if args.resume_from_checkpoint:
        logger.info(f"Resuming. Training for {args.num_epochs_to_add} more epochs.")

    main(args)