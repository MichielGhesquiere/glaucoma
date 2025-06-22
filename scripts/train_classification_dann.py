#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Glaucoma Classification Model Training Script with DANN (Domain Adversarial Neural Networks).

This script adapts the original classification training pipeline for DANN.
It reuses utility functions from 'train_classification_original.py' where possible.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple, List # For type hinting

import numpy as np # For NpEncoder and other utilities
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
# torchvision.transforms are imported within get_transforms from original script

# Ensure custom modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.utils.helpers import NpEncoder, set_seed
    # --- DANN Adapted Imports ---
    from src.data.datasets_dann import GlaucomaSubgroupDANNDataset, safe_collate_dann
    from src.models.classification.build_model_dann import build_classifier_model_dann
    from src.training.engine_dann import train_model_dann
    # --- End DANN Adapted Imports ---

    # --- Import Unchanged Utilities from Original Script ---
    # Ensure 'train_classification_original.py' is in the same directory or accessible via PYTHONPATH
    # If it's in the same directory:
    import train_classification as orig_train
    # If it's elsewhere and you added to sys.path, you might import it directly:
    # from my_project_scripts import train_classification_original as orig_train
    
    # Assign imported functions to local names for clarity and direct use
    adjust_path_for_data_type = orig_train.adjust_path_for_data_type
    LabelSmoothingCrossEntropy = orig_train.LabelSmoothingCrossEntropy
    # extract_base_chaksu = orig_train.extract_base_chaksu # Used within load_chaksu_data
    # load_chaksu_data = orig_train.load_chaksu_data       # Used within load_and_split_data
    # load_airogs_data = orig_train.load_airogs_data       # Used within load_and_split_data
    # save_data_samples_for_verification = orig_train.save_data_samples_for_verification # Used within load_and_split_data
    # assign_dataset_source = orig_train.assign_dataset_source # Used within load_and_split_data
    load_and_split_data = orig_train.load_and_split_data # This is a major component
    get_transforms = orig_train.get_transforms
    freeze_backbone = orig_train.freeze_backbone
    unfreeze_backbone = orig_train.unfreeze_backbone
    create_optimizer_with_differential_lr = orig_train.create_optimizer_with_differential_lr
    # TemperatureScaling and GradCAM related functions are omitted as requested

except ImportError as e:
    print(f"Error importing DANN modules or 'train_classification_original.py': {e}")
    print("Please ensure 'src.data.datasets_dann', 'src.models.classification.build_model_dann', "
          "'src.training.engine_dann', 'src.utils.helpers' are accessible, and that "
          "'train_classification_original.py' can be imported (e.g., in the same directory or PYTHONPATH).")
    sys.exit(1)
except AttributeError as e:
    print(f"AttributeError during import from 'train_classification_original.py': {e}")
    print("Ensure all expected functions are defined in 'train_classification_original.py'.")
    sys.exit(1)


# --- Global Configuration & Constants ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# BASE_DATA_DIR_CONFIG_KEY will be set from args in main script execution block
# These globals are used by functions imported from train_classification_original
# so they need to be defined here as well, or those functions need to be adapted
# to take them as arguments if they can't access globals across modules easily.
# For simplicity, we define them here. The `adjust_path_for_data_type` from original
# script might rely on these being module-level globals in its own file.
# If `orig_train.BASE_DATA_DIR_CONFIG_KEY` is used by its functions, we must set it.
orig_train.BASE_DATA_DIR_CONFIG_KEY = "" # Placeholder
orig_train.RAW_DIR_NAME = "raw"
orig_train.PROCESSED_DIR_NAME = "processed"
orig_train.PAPILLA_PREFIX = "PAPILA" # Ensure consistency

DOMAIN_COLUMN_NAME: str = 'dataset_source' # Column for DANN domains

# --- DANN Adapted create_dataloaders (moved from previous response) ---
def create_dataloaders_dann(
    config: argparse.Namespace,
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    train_transforms: torch.nn.Module, eval_transforms: torch.nn.Module, # Corrected type hint
    domain_id_map: Optional[Dict[str, int]], num_domains: int
) -> tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader],
           Optional[GlaucomaSubgroupDANNDataset], Optional[GlaucomaSubgroupDANNDataset], Optional[GlaucomaSubgroupDANNDataset]]:
    logger.info("--- Creating DataLoaders (DANN Mode) ---")
    train_loader, val_loader, test_loader = None, None, None
    train_dataset, val_dataset, test_dataset = None, None, None
    num_workers, pin_memory = config.num_workers, torch.cuda.is_available()
    persistent_workers = num_workers > 0
    common_args = {'domain_column': DOMAIN_COLUMN_NAME, 'domain_id_map': domain_id_map, 
                   'label_col': 'types', 'path_col': 'image_path'} # Assuming 'types' is label column

    if not train_df.empty:
        # For DANN, primary sensitive attribute is the domain itself for the dataset class.
        # Other attributes can be passed if needed for external analysis, but dataset focuses on domain.
        train_dataset = GlaucomaSubgroupDANNDataset(
            train_df, transform=train_transforms, 
            sensitive_attributes=[DOMAIN_COLUMN_NAME], # Pass domain column for potential metadata access
            **common_args
        )
        if len(train_dataset) > 0:
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers,
                                      pin_memory=pin_memory, collate_fn=safe_collate_dann, persistent_workers=persistent_workers)
            logger.info(f"Train DANN DataLoader: {len(train_dataset)} samples, {train_dataset.num_domains} domains from dataset.")
    if not val_df.empty:
        # For validation, you might want other sensitive attributes for subgroup performance.
        sensitive_attrs_val = [DOMAIN_COLUMN_NAME] 
        if 'camera' in val_df.columns: sensitive_attrs_val.append('camera')

        val_dataset = GlaucomaSubgroupDANNDataset(
            val_df, transform=eval_transforms, 
            sensitive_attributes=sensitive_attrs_val, 
            **common_args
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=num_workers,
                                    pin_memory=pin_memory, collate_fn=safe_collate_dann, persistent_workers=persistent_workers)
            logger.info(f"Validation DANN DataLoader: {len(val_dataset)} samples.")
    if not test_df.empty:
        sensitive_attrs_test = [DOMAIN_COLUMN_NAME]
        if 'camera' in test_df.columns: sensitive_attrs_test.append('camera')

        test_dataset = GlaucomaSubgroupDANNDataset(
            test_df, transform=eval_transforms, 
            sensitive_attributes=sensitive_attrs_test, 
            **common_args
        )
        if len(test_dataset) > 0:
            test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=num_workers,
                                     pin_memory=pin_memory, collate_fn=safe_collate_dann, persistent_workers=persistent_workers)
            logger.info(f"Test DANN DataLoader: {len(test_dataset)} samples.")
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


# --- Main Training Orchestration ---
def main(args: argparse.Namespace):
    start_time = datetime.now()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Output Directory and Experiment Naming ---
    dann_tag_exp = "DANN" if args.use_dann else "NoDANN"
    if args.resume_from_checkpoint:
        model_run_dir_resume = os.path.dirname(os.path.dirname(args.resume_from_checkpoint))
        timestamp_suffix = start_time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(model_run_dir_resume, f"resumed_{dann_tag_exp}_{args.num_epochs_to_add}ep_{timestamp_suffix}")
        experiment_name = os.path.basename(model_run_dir_resume)
        logger.info(f"Resuming {dann_tag_exp} training. Original: {experiment_name}. Outputs: {output_dir}")
    else:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        model_short_name = args.model_name.split('/')[-1].replace("_patch16_224", "").replace("_base", "").replace("dinov2_", "d2_")
        exp_parts = [model_short_name, dann_tag_exp, "proc" if args.data_type == "processed" else "raw",
                     args.experiment_tag if args.experiment_tag else None, timestamp]
        experiment_name = "_".join(filter(None, exp_parts))
        output_dir = os.path.join(args.base_output_dir, experiment_name)
    
    args.output_dir, args.experiment_name = output_dir, experiment_name
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(checkpoint_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True)

    # --- File Handler for Logging ---
    log_filename = f'training_log_{dann_tag_exp}_{"resumed_" if args.resume_from_checkpoint else ""}{start_time.strftime("%Y%m%d_%H%M%S")}.log'
    log_filepath = os.path.join(output_dir, log_filename)
    root_logger = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_filepath for h in root_logger.handlers):
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
    
    logger.info(f"--- Experiment ({dann_tag_exp}): {experiment_name} ---")
    logger.info(f"Output dir: {output_dir}. Args: {vars(args)}")
    with open(os.path.join(results_dir, f"config_{dann_tag_exp}.json"), "w") as f: json.dump(vars(args), f, indent=4, cls=NpEncoder)

    # --- Data Loading and Splitting (using imported function) ---
    train_df, val_df, test_df = load_and_split_data(args)
    if train_df.empty: logger.critical("Train DataFrame empty. Exit."); sys.exit(1)

    # --- DANN: Determine num_domains and create domain_id_map ---
    domain_id_map: Optional[Dict[str, int]] = None
    num_domains = 1
    if args.use_dann:
        if DOMAIN_COLUMN_NAME not in train_df.columns:
            logger.error(f"DANN Error: Column '{DOMAIN_COLUMN_NAME}' not in train_df. DANN disabled.")
            args.use_dann = False
        else:
            all_dfs_domain = [df for df in [train_df, val_df, test_df] if not df.empty and DOMAIN_COLUMN_NAME in df.columns]
            if all_dfs_domain:
                unique_doms = sorted(list(pd.concat([df[DOMAIN_COLUMN_NAME] for df in all_dfs_domain]).astype(str).unique()))
                domain_id_map = {name: i for i, name in enumerate(unique_doms)}
                num_domains = len(domain_id_map)
                logger.info(f"DANN: Using domain column '{DOMAIN_COLUMN_NAME}'. Map ({num_domains} domains): {domain_id_map}")
                if num_domains <= 1: logger.warning("DANN: Only 1 domain. DANN disabled."); args.use_dann = False
            else: logger.warning("DANN: No data for domain map. DANN disabled."); args.use_dann = False
    args.num_domains = num_domains

    # --- Transformations and DataLoaders ---
    train_tf, eval_tf = get_transforms(args.image_size, args.model_name) # Using imported get_transforms
    train_loader, val_loader, _, train_dataset, _, _ = create_dataloaders_dann( # Using DANN-specific create_dataloaders
        args, train_df, val_df, test_df, train_tf, eval_tf, domain_id_map, num_domains
    )
    if train_loader is None: logger.critical("Train DataLoader None. Exit."); sys.exit(1)
    # Update num_domains from dataset if it calculated differently (e.g. after internal filtering)
    if args.use_dann and train_dataset and hasattr(train_dataset, 'num_domains') and train_dataset.num_domains != num_domains:
        logger.info(f"Updating num_domains from {num_domains} to {train_dataset.num_domains} based on train_dataset actual domains.")
        args.num_domains = train_dataset.num_domains
        if args.num_domains <=1: # Re-check if DANN should be disabled
             logger.warning("DANN: num_domains became <=1 after dataset processing. Disabling DANN.")
             args.use_dann = False


    # --- Save Data Manifests ---
    cols_manifest = ['image_path', 'types', DOMAIN_COLUMN_NAME, 'dataset', 'names']
    for df_man, name_man in [(train_df, "train"), (val_df, "validation"), (test_df, "test")]:
        if not df_man.empty:
            path_man = os.path.join(results_dir, f'{name_man}_manifest_{dann_tag_exp}.csv')
            final_cols = [c for c in cols_manifest if c in df_man.columns] + \
                         [c for c in df_man.columns if c not in cols_manifest and c not in ['file_exists', 'stratify_key', 'domain_id']]
            try: 
                # Ensure all columns in final_cols actually exist in df_man before trying to select
                actual_final_cols = [c for c in final_cols if c in df_man.columns]
                df_man[actual_final_cols].to_csv(path_man, index=False)
                logger.info(f"Saved {name_man} manifest: {path_man} (Cols: {actual_final_cols})")
            except Exception as e: logger.error(f"Error saving {name_man} manifest: {e}", exc_info=True)

    # --- Model Building ---
    model = build_classifier_model_dann( # Using DANN-specific builder
        model_name=args.model_name, num_classes=args.num_classes, dropout_prob=args.dropout_prob,
        pretrained=(not (args.resume_from_checkpoint or args.custom_weights_path) and args.use_timm_pretrained_if_no_custom),
        custom_weights_path=(args.resume_from_checkpoint or args.custom_weights_path),
        checkpoint_key=args.checkpoint_key,
        is_regression_to_classification=args.is_regression_to_classification,
        # DANN specific args for build_model_dann
        use_dann=args.use_dann,
        num_domains=args.num_domains, # Use the potentially updated num_domains
        dann_grl_base_lambda=args.dann_grl_base_lambda,
        dann_lambda_schedule_mode=args.dann_lambda_schedule_mode,
        dann_gamma=args.dann_gamma
    )
    model.to(device)
    logger.info(f"Model built. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- Loss, Optimizer, Scheduler ---
    crit_label = LabelSmoothingCrossEntropy(args.label_smoothing_factor, args.num_classes) if args.use_label_smoothing else nn.CrossEntropyLoss()
    crit_domain = nn.CrossEntropyLoss() if args.use_dann and args.num_domains > 1 else None
    logger.info(f"Label Criterion: {crit_label}")
    if crit_domain: logger.info(f"Domain Criterion: {crit_domain}")
    
    if args.is_regression_to_classification and args.backbone_lr_multiplier != 1.0 :
        optimizer = create_optimizer_with_differential_lr(model, args.learning_rate, args.backbone_lr_multiplier, args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=max(1, args.early_stopping_patience // 2 if args.early_stopping_patience > 0 else 5), verbose=True)
    use_amp_actual = args.use_amp and torch.cuda.is_available()
    
    # --- Resume Optimizer/Scheduler/Epoch ---
    start_epoch = 0
    resumed_history = None
    if args.resume_from_checkpoint:
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        if "optimizer_state_dict" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state_dict"]); logger.info("Optimizer state loaded.")
        if "scheduler_state_dict" in ckpt and scheduler: scheduler.load_state_dict(ckpt["scheduler_state_dict"]); logger.info("Scheduler state loaded.")
        if "epoch" in ckpt: start_epoch = ckpt["epoch"] + 1 # Start from next epoch
        if "history" in ckpt: resumed_history = ckpt["history"] # Load past history
        # Load domain_id_map and num_domains from checkpoint if resuming and DANN is active
        if args.use_dann and "domain_id_map" in ckpt and "num_domains" in ckpt:
            domain_id_map = ckpt["domain_id_map"]
            args.num_domains = ckpt["num_domains"] # Override with checkpoint's num_domains
            logger.info(f"Resumed DANN with domain_id_map and num_domains={args.num_domains} from checkpoint.")

        logger.info(f"Resuming training from epoch {start_epoch}.")

    epochs_total = args.num_epochs if not args.resume_from_checkpoint else start_epoch + args.num_epochs_to_add
    logger.info(f"Training {dann_tag_exp} from epoch {start_epoch} up to epoch {epochs_total - 1}.")

    # --- Training ---
    history = {}
    model_component_to_freeze = model.feature_extractor if args.use_dann and hasattr(model, 'feature_extractor') else model

    if args.is_regression_to_classification and args.freeze_backbone_epochs > 0:
        if start_epoch < args.freeze_backbone_epochs:
            logger.info(f"Phase 1 (Frozen): Regr2Class {dann_tag_exp} training for {args.freeze_backbone_epochs - start_epoch} epochs.")
            freeze_backbone(model_component_to_freeze, args.model_name) # Use imported freeze_backbone

            model, history_p1 = train_model_dann( # Using DANN-specific engine
                model, train_loader, val_loader, crit_label, optimizer, scheduler,
                min(args.freeze_backbone_epochs, epochs_total), device, checkpoint_dir, f"{experiment_name}_frozen",
                args.early_stopping_patience if args.early_stopping_patience > 0 else None, use_amp_actual, args.grad_accum_steps,
                start_epoch, resumed_history, args.use_dann, crit_domain, args.dann_tradeoff_lambda)
            history = history_p1
            start_epoch = min(args.freeze_backbone_epochs, epochs_total)
            
            if start_epoch < epochs_total:
                logger.info(f"Phase 2 (Unfrozen): Regr2Class {dann_tag_exp} training from epoch {start_epoch}.")
                unfreeze_backbone(model_component_to_freeze, args.model_name) # Use imported unfreeze_backbone
                if args.backbone_lr_multiplier != 1.0:
                     optimizer = create_optimizer_with_differential_lr(model, args.learning_rate, args.backbone_lr_multiplier, args.weight_decay)
                
                model, history_p2 = train_model_dann(
                    model, train_loader, val_loader, crit_label, optimizer, scheduler,
                    epochs_total, device, checkpoint_dir, f"{experiment_name}_unfrozen",
                    args.early_stopping_patience if args.early_stopping_patience > 0 else None, use_amp_actual, args.grad_accum_steps,
                    start_epoch, history, args.use_dann, crit_domain, args.dann_tradeoff_lambda)
                for k in history_p2: history[k] = history.get(k, []) + history_p2[k]
        else:
            logger.info("Regr2Class: Start epoch past freezing. Training unfrozen.")
            unfreeze_backbone(model_component_to_freeze, args.model_name)
            model, history = train_model_dann(
                model, train_loader, val_loader, crit_label, optimizer, scheduler,
                epochs_total, device, checkpoint_dir, experiment_name,
                args.early_stopping_patience if args.early_stopping_patience > 0 else None, use_amp_actual, args.grad_accum_steps,
                start_epoch, resumed_history, args.use_dann, crit_domain, args.dann_tradeoff_lambda)
    else:
        model, history = train_model_dann(
            model, train_loader, val_loader, crit_label, optimizer, scheduler,
            epochs_total, device, checkpoint_dir, experiment_name,
            args.early_stopping_patience if args.early_stopping_patience > 0 else None, use_amp_actual, args.grad_accum_steps,
            start_epoch, resumed_history, args.use_dann, crit_domain, args.dann_tradeoff_lambda)

    # --- Post-Training ---
    len_history_current_session = len(history.get('val_loss', history.get('train_loss', [])))
    num_epochs_in_resumed_history = len(resumed_history.get('val_loss', [])) if resumed_history else 0
    last_epoch_trained = num_epochs_in_resumed_history + len_history_current_session - 1 if len_history_current_session > 0 else epochs_total - 1
    
    # --- Save Final Model & History ---
    final_model_path = os.path.join(checkpoint_dir, f"{experiment_name}_final_epoch{last_epoch_trained}_{dann_tag_exp}.pth")
    torch.save({
        "epoch": last_epoch_trained, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "args": vars(args), "domain_id_map": domain_id_map, "num_domains": args.num_domains
    }, final_model_path)
    logger.info(f"Final model (epoch {last_epoch_trained}) saved: {final_model_path}")

    final_history_to_save = {}
    if resumed_history:
        for k in set(resumed_history.keys()) | set(history.keys()):
            final_history_to_save[k] = resumed_history.get(k, []) + history.get(k, [])
    else: final_history_to_save = history
    
    history_filepath = os.path.join(results_dir, f"history_epoch{last_epoch_trained}_{dann_tag_exp}.json")
    with open(history_filepath, "w") as f: json.dump(final_history_to_save, f, indent=4, cls=NpEncoder)
    logger.info(f"Full training history saved: {history_filepath}")

    end_time = datetime.now()
    logger.info(f"--- Experiment ({dann_tag_exp}) Finished: {args.experiment_name}. Total time: {end_time - start_time} ---")


# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Train Glaucoma Classification Model with DANN option.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- Data Arguments ---
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument('--data_type', type=str, default='raw', choices=['raw', 'processed'], help='Image data type.')
    data_group.add_argument('--base_data_root', type=str, default=r'./data', help='Root directory for datasets.') # Generic default
    data_group.add_argument('--min_samples_per_group_for_stratify', type=int, default=2, help='Min samples for stratified split.')
    
    smdg_group = parser.add_argument_group("SMDG-19 Data")
    smdg_group.add_argument('--smdg_metadata_file', type=str, default=r'./data/raw/SMDG-19/metadata - standardized.csv')
    smdg_group.add_argument('--smdg_image_dir', type=str, default=r'./data/raw/SMDG-19/full-fundus/full-fundus')
    
    chaksu_group = parser.add_argument_group("CHAKSU Data")
    chaksu_group.add_argument('--use_chaksu', action='store_true', default=True)
    chaksu_group.add_argument('--no_chaksu', action='store_false', dest='use_chaksu') # Allow disabling
    chaksu_group.add_argument('--chaksu_base_dir', type=str, default=r'./data/raw/Chaksu/Train/Train/1.0_Original_Fundus_Images')
    chaksu_group.add_argument('--chaksu_decision_dir', type=str, default=r'./data/raw/Chaksu/Train/Train/6.0_Glaucoma_Decision')
    chaksu_group.add_argument('--chaksu_metadata_dir', type=str, default=r'./data/raw/Chaksu/Train/Train/6.0_Glaucoma_Decision/Majority')
    
    airogs_group = parser.add_argument_group("AIROGS Data")
    airogs_group.add_argument('--use_airogs', action='store_true', default=True)
    airogs_group.add_argument('--no_airogs', action='store_false', dest='use_airogs') # Allow disabling
    airogs_group.add_argument('--airogs_label_file', type=str, default=r'./data/raw/AIROGS/train_labels.csv')
    airogs_group.add_argument('--airogs_image_dir', type=str, default=r'./data/raw/AIROGS/img')
    airogs_group.add_argument('--airogs_num_rg_samples', type=int, default=1000)
    airogs_group.add_argument('--airogs_num_nrg_samples', type=int, default=9000)
    airogs_group.add_argument('--use_airogs_cache', action='store_true', default=False)
    
    # --- Model Arguments ---
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    model_group.add_argument('--num_classes', type=int, default=2)
    model_group.add_argument('--image_size', type=int, default=224)
    model_group.add_argument('--dropout_prob', type=float, default=0.1)
    model_group.add_argument('--custom_weights_path', type=str, default=None)
    model_group.add_argument('--checkpoint_key', type=str, default='teacher')
    model_group.add_argument('--use_timm_pretrained_if_no_custom', action='store_true', default=True)
    model_group.add_argument('--is_regression_to_classification', action='store_true', default=False)
    model_group.add_argument('--freeze_backbone_epochs', type=int, default=0)
    model_group.add_argument('--backbone_lr_multiplier', type=float, default=1.0)

    # --- Training Arguments ---
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument('--num_epochs', type=int, default=25)
    train_group.add_argument('--batch_size', type=int, default=8)
    train_group.add_argument('--eval_batch_size', type=int, default=32)
    train_group.add_argument('--learning_rate', type=float, default=1e-4)
    train_group.add_argument('--weight_decay', type=float, default=0.05)
    train_group.add_argument('--early_stopping_patience', type=int, default=0)
    train_group.add_argument('--grad_accum_steps', type=int, default=4)
    train_group.add_argument('--seed', type=int, default=42)
    train_group.add_argument('--num_workers', type=int, default=2)
    train_group.add_argument('--use_amp', action='store_true', default=True)
    train_group.add_argument('--no_amp', action='store_false', dest='use_amp')
    train_group.add_argument('--use_label_smoothing', action='store_true', default=False)
    train_group.add_argument('--label_smoothing_factor', type=float, default=0.1)
    
    # --- DANN Specific Arguments ---
    dann_group = parser.add_argument_group("DANN Configuration")
    dann_group.add_argument('--use_dann', action='store_true', default=False, help="Enable DANN.")
    dann_group.add_argument('--dann_grl_base_lambda', type=float, default=1.0, help="Base lambda for GRL.")
    dann_group.add_argument('--dann_lambda_schedule_mode', type=str, default="paper", choices=["paper", "fixed"], help="GRL lambda schedule.")
    dann_group.add_argument('--dann_gamma', type=float, default=10.0, help="Gamma for 'paper' GRL lambda schedule.")
    dann_group.add_argument('--dann_tradeoff_lambda', type=float, default=0.1, help="Trade-off (alpha) for domain loss.")

    # --- Experiment Management ---
    exp_group = parser.add_argument_group("Experiment Management")
    exp_group.add_argument('--base_output_dir', type=str, default='experiments/glaucoma_dann_runs', help='Base output directory.')
    exp_group.add_argument('--experiment_tag', type=str, default='', help='Custom tag for experiment name.')
    exp_group.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume.')
    exp_group.add_argument('--num_epochs_to_add', type=int, default=25, help='Additional epochs when resuming.')

    # --- Data Verification ---
    verify_group = parser.add_argument_group("Data Verification")
    verify_group.add_argument('--save_data_samples', action='store_true', default=False, help="Save sample images.")
    verify_group.add_argument('--num_data_samples_per_source', type=int, default=2, help="Num samples to save per source.")
    
    args = parser.parse_args()

    # --- Path Adjustments and Pre-run Checks ---
    # Update global variable in the imported module if its functions rely on it
    orig_train.BASE_DATA_DIR_CONFIG_KEY = args.base_data_root
    if args.data_type == 'processed':
        logger.info(f"Data type '{args.data_type}'. Adjusting DANN image paths...")
        args.smdg_image_dir = adjust_path_for_data_type(args.smdg_image_dir, args.data_type)
        args.chaksu_base_dir = adjust_path_for_data_type(args.chaksu_base_dir, args.data_type)
        args.airogs_image_dir = adjust_path_for_data_type(args.airogs_image_dir, args.data_type)

    if args.resume_from_checkpoint and not os.path.exists(args.resume_from_checkpoint):
        logger.error(f"Resume DANN ckpt {args.resume_from_checkpoint} missing."); sys.exit(1)
    if not args.resume_from_checkpoint and args.custom_weights_path and not os.path.exists(args.custom_weights_path):
        logger.warning(f"Custom weights {args.custom_weights_path} missing. Model might train from scratch or timm pretrained.")

    main(args)