"""
Refactored Multi-Source Domain Fine-Tuning Script for Glaucoma Classification.

This script implements leave-one-dataset-out domain adaptation with modular components.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.cuda.amp import GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from torch.utils.data import DataLoader

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Set backend to prevent GUI issues
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Training plots will not be saved.")

# Enable Flash Attention optimization
# Configure Flash Attention more carefully
try:
    torch.backends.cuda.enable_flash_sdp(True)
    flash_enabled = torch.backends.cuda.flash_sdp_enabled() if hasattr(torch.backends.cuda, 'flash_sdp_enabled') else False
    print(f"Flash Attention enabled: {flash_enabled}")
except Exception as e:
    print(f"Flash Attention configuration failed: {e}")
    print("Continuing without Flash Attention optimization")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modular components
from src.data.external_loader import load_external_test_data
from src.data.loaders import load_chaksu_data, load_airogs_data, assign_dataset_source
from src.data.utils import create_external_test_dataloader, get_eval_transforms, safe_collate
from src.models.classification.build_model import build_classifier_model
from src.training.engine import train_model
from src.utils.experiment_management import ExperimentCheckpoint, ExperimentLogger, ResultsManager
from src.evaluation.metrics import calculate_ece, calculate_sensitivity_at_specificity
from src.data.transforms import get_transforms
from src.utils.helpers import set_seed

# Optionally suppress common FutureWarnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.*compile.*')


class MultiSourceExperiment:
    """Main class for running multi-source domain adaptation experiments."""
    
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(__name__)
        
        # Setup experiment management
        self.checkpoint = ExperimentCheckpoint(
            args.checkpoint_dir, 
            f"multisource_{args.experiment_tag}"
        )
        self.experiment_logger = ExperimentLogger(
            args.log_dir, 
            f"multisource_{args.experiment_tag}",
            args.detailed_logging
        )
        self.results_manager = ResultsManager(args.output_dir)
        
        # Load checkpoint if resuming
        if args.resume_from_checkpoint:
            self.checkpoint.load_checkpoint()
            
        # Initialize results list
        self.results = self.checkpoint.get_results()
        
        # Setup interrupt handler for graceful shutdown
        signal.signal(signal.SIGINT, self._interrupt_handler)
        
    def _interrupt_handler(self, signum, frame):
        """Handle Ctrl+C interruption gracefully."""
        self.logger.info("Interrupt received. Saving progress and exiting...")
        self.checkpoint.save_checkpoint()
        self.results_manager.save_results(self.results, "interrupted_results.json")
        self.experiment_logger.cleanup()
        sys.exit(0)
        
    def load_datasets(self):
        """Load all datasets for the experiment using proven external_loader approach."""
        self.logger.info("Loading datasets...")
        
        # Use the proven external_loader approach with correct argument names
        external_datasets = load_external_test_data(
            smdg_metadata_file_raw=self.args.smdg_metadata_file_raw,
            smdg_image_dir_raw=self.args.smdg_image_dir_raw,
            chaksu_base_dir_eval=self.args.chaksu_base_dir,
            chaksu_decision_dir_raw=self.args.chaksu_decision_dir_raw,
            chaksu_metadata_dir_raw=self.args.chaksu_metadata_dir_raw,
            data_type=self.args.data_type,
            base_data_root=self.args.base_data_root,
            raw_dir_name='raw',
            processed_dir_name='processed',
            eval_papilla=self.args.use_papila,
            eval_oiaodir_test=False,  # Not used in multisource training
            eval_chaksu=self.args.use_chaksu,
            eval_acrima=self.args.use_acrima,
            eval_hygd=self.args.use_hygd,
            acrima_image_dir_raw=self.args.acrima_image_dir_raw,
            hygd_image_dir_raw=self.args.hygd_image_dir_raw,
            hygd_labels_file_raw=self.args.hygd_labels_file_raw
        )
        
        # Add SMDG-19 manually if needed (not covered by external loader)
        datasets = {}
        
        # Load SMDG-19 if requested and split into subdatasets
        if getattr(self.args, 'use_smdg', True):
            smdg_metadata_file = os.path.join(self.args.base_data_root, self.args.smdg_metadata_file_raw)
            smdg_image_dir = os.path.join(self.args.base_data_root, self.args.smdg_image_dir_raw)
            
            if os.path.exists(smdg_metadata_file):
                try:
                    df_smdg_all = pd.read_csv(smdg_metadata_file)
                    
                    # Filter out PAPILA to avoid overlap with external datasets
                    # Keep both OIA-ODIR-train and OIA-ODIR-test to combine them later
                    smdg_main = df_smdg_all[
                        ~df_smdg_all['names'].str.startswith('PAPILA', na=False)
                    ].copy()
                    
                    if not smdg_main.empty:
                        # Add image paths and filter existing files
                        smdg_main["image_path"] = smdg_main["names"].apply(
                            lambda name: os.path.join(smdg_image_dir, f"{name}.png")
                        )
                        smdg_main["file_exists"] = smdg_main["image_path"].apply(os.path.exists)
                        smdg_main = smdg_main[smdg_main["file_exists"]]
                        
                        # Clean data
                        smdg_main = smdg_main.dropna(subset=["types"])
                        smdg_main = smdg_main[smdg_main["types"].isin([0, 1])]
                        smdg_main["types"] = smdg_main["types"].astype(int)
                        smdg_main["label"] = smdg_main["types"]  # Add compatibility
                        
                        # Assign dataset sources to identify subdatasets (using proven logic from dataset_analysis.py)
                        smdg_main["dataset_source"] = assign_dataset_source(smdg_main["names"])
                        
                        self.logger.info(f"Loaded SMDG-19 total samples: {len(smdg_main)}")
                        
                        # Split into subdatasets and filter by sample count
                        smdg_subdatasets = smdg_main.groupby('dataset_source')
                        min_samples_per_subdataset = 100
                        
                        # Log all subdatasets before filtering
                        all_subdatasets = list(smdg_subdatasets.groups.keys())
                        self.logger.info(f"SMDG-19 subdatasets found: {all_subdatasets}")
                        
                        # Filter subdatasets by sample count and drop "unknown" samples
                        # Also combine OIA-ODIR train and test into single dataset
                        valid_subdatasets = []
                        unknown_count = 0
                        oia_odir_combined = []
                        
                        for subdataset_name, subdataset_df in smdg_subdatasets:
                            subdataset_size = len(subdataset_df)
                            
                            if subdataset_name == "SMDG_Unknown":
                                unknown_count = subdataset_size
                                self.logger.info(f"Dropping SMDG_Unknown subdataset: {unknown_count} samples")
                                continue
                            
                            # Handle OIA-ODIR train and test - combine them
                            if subdataset_name in ["OIA-ODIR-train", "OIA-ODIR-test"]:
                                oia_odir_combined.append(subdataset_df.copy())
                                self.logger.info(f"Adding {subdataset_name} to combined OIA-ODIR: {subdataset_size} samples")
                                continue
                                
                            if subdataset_size >= min_samples_per_subdataset:
                                datasets[f'SMDG-{subdataset_name}'] = subdataset_df.copy()
                                valid_subdatasets.append(subdataset_name)
                                self.logger.info(f"Added SMDG-{subdataset_name}: {subdataset_size} samples")
                            else:
                                self.logger.info(f"Skipping SMDG-{subdataset_name}: only {subdataset_size} samples (< {min_samples_per_subdataset})")
                        
                        # Combine OIA-ODIR train and test if we have them
                        if oia_odir_combined:
                            combined_oia_odir = pd.concat(oia_odir_combined, ignore_index=True)
                            combined_size = len(combined_oia_odir)
                            if combined_size >= min_samples_per_subdataset:
                                datasets['SMDG-OIA-ODIR'] = combined_oia_odir
                                valid_subdatasets.append('OIA-ODIR')
                                self.logger.info(f"Added SMDG-OIA-ODIR (combined train+test): {combined_size} samples")
                            else:
                                self.logger.info(f"Skipping combined SMDG-OIA-ODIR: only {combined_size} samples (< {min_samples_per_subdataset})")
                        
                        self.logger.info(f"SMDG-19 processing summary:")
                        self.logger.info(f"  - Total samples: {len(smdg_main)}")
                        self.logger.info(f"  - Unknown samples dropped: {unknown_count}")
                        self.logger.info(f"  - Valid subdatasets (>={min_samples_per_subdataset} samples): {len(valid_subdatasets)}")
                        self.logger.info(f"  - Subdatasets included: {valid_subdatasets}")
                        
                except Exception as e:
                    self.logger.error(f"Error loading SMDG-19: {e}")
                    traceback.print_exc()
        
        # Add AIROGS using proven loader approach with correct sample sizes
        if self.args.use_airogs:
            try:
                # Create a config-like object for AIROGS loader
                class AirogsConfig:
                    def __init__(self, args):
                        self.airogs_label_file = args.airogs_label_file
                        self.airogs_image_dir = args.airogs_image_dir
                        self.airogs_num_rg_samples = args.airogs_num_rg_samples
                        self.airogs_num_nrg_samples = args.airogs_num_nrg_samples
                        self.use_airogs_cache = args.use_airogs_cache
                        self.seed = args.seed
                        self.data_type = args.data_type
                
                airogs_config = AirogsConfig(self.args)
                df_airogs = load_airogs_data(airogs_config)
                
                if not df_airogs.empty:
                    df_airogs["label"] = df_airogs["types"]  # Add compatibility
                    datasets['AIROGS'] = df_airogs
                    self.logger.info(f"Loaded AIROGS: {len(df_airogs)} samples")
            except Exception as e:
                self.logger.error(f"Error loading AIROGS: {e}")
        
        # Add external datasets from external_loader
        for dataset_name, dataset_df in external_datasets.items():
            if not dataset_df.empty and dataset_name not in datasets:
                # Ensure label column exists for compatibility
                if 'label' not in dataset_df.columns and 'types' in dataset_df.columns:
                    dataset_df['label'] = dataset_df['types']
                elif 'types' not in dataset_df.columns and 'label' in dataset_df.columns:
                    dataset_df['types'] = dataset_df['label']
                
                # Filter out invalid labels (e.g., -1 in PAPILLA dataset)
                if 'label' in dataset_df.columns:
                    original_count = len(dataset_df)
                    dataset_df = dataset_df[dataset_df['label'].isin([0, 1])].copy()
                    filtered_count = len(dataset_df)
                    if filtered_count < original_count:
                        self.logger.info(f"Filtered {dataset_name}: removed {original_count - filtered_count} samples with invalid labels, kept {filtered_count} samples")
                
                if not dataset_df.empty:
                    datasets[dataset_name] = dataset_df
                    self.logger.info(f"Added external dataset {dataset_name}: {len(dataset_df)} samples")
                else:
                    self.logger.warning(f"Dataset {dataset_name} became empty after filtering invalid labels")
        
        # Validate we have at least some datasets
        if not datasets:
            raise ValueError("No datasets loaded successfully")
            
        # Simple validation - check each dataset has required columns
        required_columns = ['image_path', 'label', 'dataset_source']
        for name, df in datasets.items():
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Dataset {name} missing columns: {missing_cols}")
        
        # Log dataset statistics (simplified)
        for name, df in datasets.items():
            self.logger.info(f"Dataset {name}: {len(df)} samples")
        
        # Apply dataset filtering
        datasets = self.filter_datasets(datasets)
        
        return datasets
        
    def get_model_configurations(self):
        """Get list of model configurations to test."""
        models = [
            {
                'name': 'VFM',
                'model_name': 'vit_base_patch16_224',
                'custom_weights_path': self.args.vfm_weights_path,
                'description': 'Vision Foundation Model (VFM)'
            }
        ]
        
        # Add additional models if specified
        if self.args.additional_models:
            for model_spec in self.args.additional_models:
                if ':' in model_spec:
                    name, model_name = model_spec.split(':', 1)
                    models.append({
                        'name': name,
                        'model_name': model_name,
                        'custom_weights_path': None,
                        'description': f'Additional model: {name}'
                    })
                    
        return models
        
    def run_single_experiment(self, model_config, test_dataset_name, train_datasets, test_dataset):
        """Run a single experiment (one model, one test dataset)."""
        experiment_key = f"{model_config['name']}_{test_dataset_name}"
        
        # Check if already completed
        if self.checkpoint.is_completed(experiment_key):
            self.logger.info(f"Skipping completed experiment: {experiment_key}")
            return
            
        self.logger.info(f"Starting experiment: {experiment_key}")
        start_time = time.time()
        
        try:
            # Prepare data
            train_df = self.combine_train_datasets(train_datasets)
            test_df = test_dataset
            
            if train_df.empty or test_df.empty:
                raise ValueError("Empty training or test data")
                
            # Split training data into train/val
            # from sklearn.model_selection import train_test_split
            train_split_df, val_split_df = train_test_split(
                train_df, test_size=0.2, random_state=self.args.seed, 
                stratify=train_df['label']
            )
            
            # Get transforms
            train_transforms, eval_transforms = get_transforms(
                self.args.image_size, 
                model_config['model_name'], 
                self.args.use_data_augmentation
            )
            
            # Create datasets and loaders
            from src.data.datasets import GlaucomaSubgroupDataset
            
            train_dataset = GlaucomaSubgroupDataset(train_split_df, transform=train_transforms)
            val_dataset = GlaucomaSubgroupDataset(val_split_df, transform=eval_transforms)
            test_dataset = GlaucomaSubgroupDataset(test_df, transform=eval_transforms)
            
            # Use optimized DataLoader settings for faster training
            train_loader = DataLoader(
                train_dataset, batch_size=self.args.batch_size, 
                shuffle=True, num_workers=self.args.num_workers,
                collate_fn=safe_collate, pin_memory=True, persistent_workers=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.args.eval_batch_size, 
                shuffle=False, num_workers=self.args.num_workers,
                collate_fn=safe_collate, pin_memory=True, persistent_workers=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.args.eval_batch_size, 
                shuffle=False, num_workers=self.args.num_workers,
                collate_fn=safe_collate, pin_memory=True, persistent_workers=True
            )
            
            # Build model architecture without loading any weights first
            # We'll load weights in the correct order below
            model = build_classifier_model(
                model_name=model_config['model_name'],
                num_classes=self.args.num_classes,
                custom_weights_path=None  # Don't load base weights yet
            )
            
            # Check if a fine-tuned checkpoint for this specific experiment exists
            # Ensure output directory exists
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            model_save_path = os.path.join(
                self.args.output_dir,
                f"{model_config['name']}_{test_dataset_name}_best.pth"
            )
            
            skip_training = False
            fine_tuned_loaded = False
            
            # FIRST: Try to load fine-tuned weights (highest priority)
            if os.path.exists(model_save_path) and not self.args.force_retrain:
                try:
                    self.logger.info(f"Found existing fine-tuned model weights at {model_save_path}")
                    self.logger.info(f"Absolute path: {os.path.abspath(model_save_path)}")
                    
                    # Load the entire checkpoint first, without weights_only=True for inspection
                    checkpoint = torch.load(model_save_path, map_location='cpu', weights_only=True)
                    
                    # The state dict might be the checkpoint itself or nested inside a 'model' key
                    state_dict = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                    
                    # Validate that the state_dict has the expected keys
                    expected_keys = ['patch_embed.proj.weight', 'blocks.0.norm1.weight', 'head.weight', 'head.bias']
                    missing_expected = [key for key in expected_keys if key not in state_dict]
                    
                    if missing_expected:
                        self.logger.warning(f"Fine-tuned checkpoint missing critical keys: {missing_expected}")
                        self.logger.warning("This checkpoint may be incomplete. Will retrain from base model.")
                        skip_training = False
                    else:
                        # Load the state dict into the model with strict=False to handle any minor incompatibilities
                        load_result = model.load_state_dict(state_dict, strict=False)
                        
                        if load_result.missing_keys:
                            self.logger.warning(f"Missing keys when loading fine-tuned weights: {load_result.missing_keys}")
                        if load_result.unexpected_keys:
                            self.logger.info(f"Unexpected keys in fine-tuned weights: {load_result.unexpected_keys}")
                        
                        # Check if critical classification head is loaded
                        if 'head.weight' in load_result.missing_keys or 'head.bias' in load_result.missing_keys:
                            self.logger.warning("Classification head not properly loaded from fine-tuned weights. Will retrain from base model.")
                            skip_training = False
                        else:
                            self.logger.info(f"Successfully loaded fine-tuned weights for {experiment_key}")
                            # Log some key model parameters to verify loading
                            if hasattr(model, 'head') and hasattr(model.head, 'weight'):
                                head_weight_sum = model.head.weight.data.sum().item()
                                head_bias_sum = model.head.bias.data.sum().item() if model.head.bias is not None else 0
                                self.logger.info(f"  Head weight sum: {head_weight_sum:.6f}, bias sum: {head_bias_sum:.6f}")
                                
                                # Debug: Check if model parameters have reasonable values after loading
                                head_weight_stats = {
                                    'mean': model.head.weight.data.mean().item(),
                                    'std': model.head.weight.data.std().item(),
                                    'min': model.head.weight.data.min().item(),
                                    'max': model.head.weight.data.max().item()
                                }
                                self.logger.info(f"Model head weight statistics: {head_weight_stats}")
                                
                                # Check if weights look like they've been trained (not random initialization)
                                if abs(head_weight_stats['mean']) < 0.01 and head_weight_stats['std'] < 0.01:
                                    self.logger.warning("Head weights appear to be near-zero initialization - possible loading issue!")
                                elif head_weight_stats['std'] > 1.0:
                                    self.logger.warning("Head weights have very high variance - possible loading issue!")
                                else:
                                    self.logger.info("Head weights appear to be properly initialized/loaded")
                            
                            skip_training = True
                            fine_tuned_loaded = True
                            
                except Exception as e:
                    self.logger.warning(f"Failed to load fine-tuned weights from {model_save_path}: {e}. Will try base weights.")
                    self.logger.debug(traceback.format_exc())
                    skip_training = False
                    fine_tuned_loaded = False
            elif os.path.exists(model_save_path) and self.args.force_retrain:
                self.logger.info(f"Found existing weights at {model_save_path} but force_retrain=True. Will use base weights and retrain.")
                skip_training = False
                fine_tuned_loaded = False
            else:
                self.logger.info(f"No existing fine-tuned weights found at {model_save_path}. Will use base weights and train.")
                skip_training = False
                fine_tuned_loaded = False

            # SECOND: If no fine-tuned weights loaded, load VFM base weights
            if not fine_tuned_loaded and model_config['custom_weights_path']:
                try:
                    self.logger.info(f"Loading VFM base weights since no fine-tuned weights were loaded...")
                    
                    # Load VFM base weights using the model builder's logic
                    from src.models.classification.build_model import load_custom_pretrained_weights
                    load_custom_pretrained_weights(
                        model, 
                        model_config['custom_weights_path'], 
                        checkpoint_key='model',
                        model_name_for_adapter=model_config['model_name'], 
                        strict=False
                    )
                    self.logger.info(f"Successfully loaded VFM base weights from {model_config['custom_weights_path']}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load VFM base weights: {e}. Using random initialization.")
                    self.logger.debug(traceback.format_exc())

            # Apply MixStyle if requested
            if self.args.use_mixstyle:
                from src.training.domain_adaptation import MixStyle
                self.logger.info("Applying MixStyle to the model.")
                
                # For Vision Transformers, we need to apply MixStyle differently
                # Check if this is a Vision Transformer
                if hasattr(model, 'patch_embed') and hasattr(model, 'blocks'):
                    # This is a Vision Transformer - apply MixStyle after patch embedding
                    self.logger.info("Detected Vision Transformer - applying MixStyle after patch embedding")
                    
                    # Store original forward method
                    original_forward = model.forward
                    mixstyle_module = MixStyle(p=0.3, alpha=0.2)  # Less aggressive for ViTs
                    
                    def vit_forward_with_mixstyle(x):
                        # Get patch embeddings
                        x = model.patch_embed(x)
                        
                        # Handle class token if it exists
                        cls_token = None
                        if hasattr(model, 'cls_token') and model.cls_token is not None:
                            cls_token = model.cls_token.expand(x.shape[0], -1, -1)
                            x = torch.cat((cls_token, x), dim=1)
                        
                        # Add position embeddings BEFORE applying MixStyle
                        if hasattr(model, 'pos_embed') and model.pos_embed is not None:
                            x = x + model.pos_embed
                        
                        # Apply dropout after position embeddings
                        if hasattr(model, 'pos_drop'):
                            x = model.pos_drop(x)
                        
                        # Apply MixStyle to patch embeddings (but preserve class token)
                        if len(x.shape) == 3:  # [B, N, C] format
                            B, N, C = x.shape
                            
                            # Check if we have a class token (first token)
                            if hasattr(model, 'cls_token') and model.cls_token is not None:
                                # Separate class token from patch tokens
                                cls_tokens = x[:, 0:1, :]  # [B, 1, C]
                                patch_tokens = x[:, 1:, :]  # [B, N-1, C]
                                
                                # Convert patch tokens to spatial format for MixStyle
                                patch_count = patch_tokens.shape[1]
                                H = W = int(patch_count ** 0.5)
                                
                                if H * W == patch_count:  # Ensure it's a perfect square
                                    patch_tokens_spatial = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
                                    
                                    # Apply MixStyle to patch tokens only
                                    mixed_patches = mixstyle_module(patch_tokens_spatial)
                                    
                                    # Convert back to sequence format
                                    mixed_patches_seq = mixed_patches.permute(0, 2, 3, 1).reshape(B, H*W, C)
                                    
                                    # Recombine with class token
                                    x = torch.cat([cls_tokens, mixed_patches_seq], dim=1)
                            else:
                                # No class token - apply MixStyle to all tokens
                                patch_count = x.shape[1]
                                H = W = int(patch_count ** 0.5)
                                
                                if H * W == patch_count:
                                    x_spatial = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
                                    mixed_x = mixstyle_module(x_spatial)
                                    x = mixed_x.permute(0, 2, 3, 1).reshape(B, patch_count, C)
                        
                        # Forward through transformer blocks
                        if hasattr(model, 'blocks'):
                            for block in model.blocks:
                                x = block(x)
                        
                        # Apply final norm
                        if hasattr(model, 'norm'):
                            x = model.norm(x)
                        
                        # Forward through head
                        if hasattr(model, 'head'):
                            if hasattr(model, 'global_pool') and model.global_pool == 'avg':
                                x = x[:, 1:].mean(dim=1) if x.shape[1] > 1 else x.mean(dim=1)
                            else:
                                x = x[:, 0]  # Use cls token
                            x = model.head(x)
                        
                        return x
                    
                    # Replace forward method
                    model.forward = vit_forward_with_mixstyle
                    self.logger.info("MixStyle applied to Vision Transformer after patch embedding")
                    
                elif hasattr(model, 'features'): # For models like DenseNet/ResNet
                    model.features = nn.Sequential(
                        model.features,
                        MixStyle(p=0.5, alpha=0.1)
                    )
                    print("[MixStyle] MixStyle module applied to model.features.")
                    self.logger.info("MixStyle module applied to model.features.")
                elif hasattr(model, 'backbone') and isinstance(model.backbone, torch.nn.Sequential):
                    model.backbone = torch.nn.Sequential(
                        model.backbone,
                        MixStyle(p=0.5, alpha=0.1)
                    )
                    print("[MixStyle] MixStyle module applied to model.backbone.")
                    self.logger.info("MixStyle module applied to model.backbone.")
                else:
                    # Fallback: Apply with reduced aggressiveness
                    self.logger.warning("Could not identify optimal MixStyle placement. Applying conservatively.")
                    original_forward = model.forward
                    mixstyle_module = MixStyle(p=0.2, alpha=0.05)  # Very conservative
                    
                    def conservative_forward(x):
                        # Apply MixStyle to input if it has spatial dimensions
                        if len(x.shape) == 4:  # [B, C, H, W]
                            x = mixstyle_module(x)
                        return original_forward(x)
                    
                    model.forward = conservative_forward
                    print("[MixStyle] MixStyle module applied conservatively to model input.")
                    self.logger.info("MixStyle module applied conservatively to model input.")
            
            # Setup device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Apply performance optimizations
            if self.args.channels_last and device.type == 'cuda':
                model = model.to(memory_format=torch.channels_last)
                self.logger.info("Using channels_last memory format")
            
            # Compile model for faster inference (PyTorch 2.0+)
            if self.args.compile_model and hasattr(torch, 'compile'):
                try:
                    # Configure torch._dynamo to suppress errors and fall back to eager mode
                    if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
                        torch._dynamo.config.suppress_errors = True
                    
                    model = torch.compile(model, mode='reduce-overhead')
                    self.logger.info("Model compiled with torch.compile for faster inference")
                except Exception as e:
                    self.logger.warning(f"Failed to compile model: {e}. Falling back to eager mode.")
                    # Disable compile_model for subsequent experiments
                    self.args.compile_model = False
            
            # Setup training components
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=self.args.learning_rate, 
                weight_decay=self.args.weight_decay
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5
            )
            
            # Enable mixed precision training for faster training
            scaler = GradScaler() if (device.type == 'cuda' and self.args.use_amp) else None
            use_amp = device.type == 'cuda' and scaler is not None and self.args.use_amp
            
            if use_amp:
                self.logger.info("Using Automatic Mixed Precision (AMP) for faster training")
            
            # Train model (skip if pre-trained weights were loaded)
            trained = False  # Track whether we actually trained
            train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
            
            if skip_training:
                self.logger.info(f"Skipping training for {experiment_key} - using pre-trained weights")
                print(f"\n⚡ Using pre-trained weights: {model_config['name']} → Test on {test_dataset_name}")
                print(f"📊 Training samples: {len(train_split_df)}, Validation: {len(val_split_df)}, Test: {len(test_df)}")
                
                # Set dummy history values since we didn't train
                history = {
                    'final_train_loss': 0.0,
                    'final_val_loss': 0.0,
                    'final_train_acc': 0.0,
                    'final_val_acc': 0.0
                }
            else:
                trained = True  # Mark that we're training
                self.logger.info(f"Training {model_config['name']} on {list(train_datasets.keys())} for testing on {test_dataset_name}")
                print(f"\n🚀 Starting training: {model_config['name']} → Test on {test_dataset_name}")
                print(f"📊 Training samples: {len(train_split_df)}, Validation: {len(val_split_df)}, Test: {len(test_df)}")
                
                # Implement actual training loop with progress bars
                model.train()
                best_val_loss = float('inf')
                epochs_no_improve = 0
                early_stop_patience = self.args.early_stop_patience  # Use argument
                
                # Install tqdm if not available
                try:
                    from tqdm import tqdm
                except ImportError:
                    print("Installing tqdm for progress bars...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
                    from tqdm import tqdm
                
                for epoch in range(self.args.num_epochs):
                    epoch_start = time.time()
                    print(f"\n📈 Epoch {epoch+1}/{self.args.num_epochs}")
                    
                    # Training phase with progress bar
                    model.train()
                    train_loss = 0.0
                    train_correct = 0
                    train_total = 0
                    
                    # Create progress bar for training with enhanced metrics display
                    train_pbar = tqdm(train_loader, desc="Training", 
                                    leave=False, ncols=120, 
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
                    
                    for batch_idx, batch_data in enumerate(train_pbar):
                        if len(batch_data) == 3:
                            images, targets, _ = batch_data
                        else:
                            images, targets = batch_data
                            
                        images, targets = images.to(device), targets.to(device).long()
                        
                        # Apply channels_last format if enabled
                        if self.args.channels_last and device.type == 'cuda':
                            images = images.to(memory_format=torch.channels_last)
                        
                        optimizer.zero_grad()
                        
                        # Use mixed precision if available
                        if use_amp:
                            with torch.amp.autocast('cuda'):
                                outputs = model(images)
                                loss = criterion(outputs, targets)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(images)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
                        
                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += targets.size(0)
                        train_correct += (predicted == targets).sum().item()
                        
                        # Update progress bar with detailed metrics
                        current_loss = train_loss / (batch_idx + 1)
                        current_acc = 100.0 * train_correct / train_total
                        current_lr = optimizer.param_groups[0]['lr']
                        
                        train_pbar.set_postfix({
                            'Loss': f'{current_loss:.4f}',
                            'Acc': f'{current_acc:.1f}%',
                            'LR': f'{current_lr:.2e}',
                            'Batch': f'{batch_idx+1}/{len(train_loader)}'
                        })
                    
                    train_pbar.close()
                    
                    # Validation phase with progress bar
                    model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    val_pbar = tqdm(val_loader, desc="Validation", 
                                  leave=False, ncols=120,
                                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
                    
                    with torch.no_grad():
                        for batch_idx, batch_data in enumerate(val_pbar):
                            if len(batch_data) == 3:
                                images, targets, _ = batch_data
                            else:
                                images, targets = batch_data
                                
                            images, targets = images.to(device), targets.to(device).long()
                            
                            # Apply channels_last format if enabled
                            if self.args.channels_last and device.type == 'cuda':
                                images = images.to(memory_format=torch.channels_last)
                                
                            outputs = model(images)
                            loss = criterion(outputs, targets)
                            
                            val_loss += loss.item()
                            _, predicted = torch.max(outputs.data, 1)
                            val_total += targets.size(0)
                            val_correct += (predicted == targets).sum().item()
                            
                            # Update validation progress bar with detailed metrics
                            current_val_loss = val_loss / (batch_idx + 1)
                            current_val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
                            val_pbar.set_postfix({
                                'Loss': f'{current_val_loss:.4f}',
                                'Acc': f'{current_val_acc:.1f}%',
                                'Samples': f'{val_total}',
                                'Batch': f'{batch_idx+1}/{len(val_loader)}'
                            })
                    
                    val_pbar.close()
                    
                    # Calculate metrics
                    avg_train_loss = train_loss / len(train_loader)
                    avg_val_loss = val_loss / len(val_loader)
                    train_acc = 100.0 * train_correct / train_total
                    val_acc = 100.0 * val_correct / val_total
                    epoch_time = time.time() - epoch_start
                    
                    # Learning rate scheduling
                    scheduler.step(avg_val_loss)
                    
                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                        # Save model weights
                        torch.save(model.state_dict(), model_save_path)
                        self.logger.info(f"Saved best model weights to {model_save_path}")
                    else:
                        epochs_no_improve += 1
                    
                    # Early stopping
                    if epochs_no_improve >= early_stop_patience:
                        self.logger.info(f"Early stopping triggered after {epoch+1} epochs (no improvement for {early_stop_patience} epochs).")
                        break
                    
                    # Log epoch results
                    self.logger.info(f"Epoch {epoch+1}/{self.args.num_epochs}: "
                                   f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                                   f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                                   f"Time: {epoch_time:.1f}s")
                    
                    # Store metrics for plotting
                    train_losses.append(avg_train_loss)
                    val_losses.append(avg_val_loss)
                    train_accuracies.append(train_acc)
                    val_accuracies.append(val_acc)
                
                history = {
                    'final_train_loss': avg_train_loss,
                    'final_val_loss': avg_val_loss,
                    'final_train_acc': train_acc,
                    'final_val_acc': val_acc
                }
            
            # Evaluate model
            metrics = self._evaluate_model(model, test_loader, device, test_dataset_name)
            
            # Save training plots only if we actually trained
            if trained and len(train_losses) > 0:
                self._save_training_plots(
                    train_losses, val_losses, train_accuracies, val_accuracies, 
                    model_save_path, model_config['name'], test_dataset_name
                )
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Create result record
            result = {
                'model_name': model_config['name'],
                'test_dataset': test_dataset_name,
                'train_datasets': list(train_datasets.keys()),
                'training_time': training_time,
                **metrics
            }
            
            # Log and save result
            self.experiment_logger.log_experiment_result(experiment_key, result)
            self.checkpoint.mark_completed(experiment_key, result)
            self.results.append(result)
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment_key} failed: {e}", exc_info=True)
            
    def _evaluate_model(self, model, test_loader, device, dataset_name):
        """Evaluate model and return metrics."""
        # from sklearn.metrics import roc_auc_score, accuracy_score
        
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                # Handle the 3-tuple return from GlaucomaSubgroupDataset
                if len(batch_data) == 3:
                    images, targets, _ = batch_data  # Ignore attributes for evaluation
                else:
                    images, targets = batch_data
                    
                images, targets = images.to(device), targets.to(device).long()
                
                # Apply channels_last format if enabled
                if self.args.channels_last and device.type == 'cuda':
                    images = images.to(memory_format=torch.channels_last)
                
                # Apply Test-Time Augmentation if enabled
                if self.args.use_tta:
                    outputs = self._apply_tta(model, images)
                else:
                    outputs = model(images)
                
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Prob of positive class
                all_targets.extend(targets.cpu().numpy())
        
        # Add debugging information
        self.logger.info(f"Evaluation on {dataset_name}:")
        self.logger.info(f"  - Total samples: {len(all_targets)}")
        self.logger.info(f"  - Class distribution: {np.bincount(all_targets)}")
        self.logger.info(f"  - Prediction distribution: {np.bincount(all_predictions)}")
        self.logger.info(f"  - Probability range: [{np.min(all_probabilities):.4f}, {np.max(all_probabilities):.4f}]")
        self.logger.info(f"  - Mean probability: {np.mean(all_probabilities):.4f}")
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        auc = roc_auc_score(all_targets, all_probabilities)
        ece, _, _, _, _ = calculate_ece(np.array(all_targets), np.array(all_probabilities))
        
        # Calculate sensitivity at 95% specificity
        fpr, tpr, _ = roc_curve(all_targets, all_probabilities)
        sens_at_95_spec = calculate_sensitivity_at_specificity(fpr, tpr, target_spec=0.95)
        
        # Log final metrics
        self.logger.info(f"COMPLETED: {dataset_name}")
        self.logger.info(f"  AUC: {auc:.4f}")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Sensitivity@95%Spec: {sens_at_95_spec:.4f}")
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'ece': ece,
            'sensitivity_at_95_specificity': sens_at_95_spec
        }
        
    def _save_training_plots(self, train_losses, val_losses, train_accuracies, val_accuracies, 
                           save_path, model_name, test_dataset_name):
        """Save training and validation loss/accuracy plots."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("matplotlib not available. Skipping training plots.")
            return
            
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            epochs = range(1, len(train_losses) + 1)
            
            # Plot loss
            ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_title(f'Training and Validation Loss\n{model_name} → {test_dataset_name}')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot accuracy
            ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
            ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
            ax2.set_title(f'Training and Validation Accuracy\n{model_name} → {test_dataset_name}')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Create plot filename
            plot_filename = save_path.replace('.pth', '_learning_curves.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Training plots saved to {plot_filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save training plots: {e}")
    
    def run_all_experiments(self):
        """Run all experiments in the multi-source domain adaptation setup."""
        # Load datasets
        datasets = self.load_datasets()
        
        # Filter datasets based on command line arguments
        datasets = self.filter_datasets(datasets)
        
        if self.args.dry_run:
            self.logger.info("DRY RUN MODE: Datasets loaded and filtered successfully. Would run the following experiments:")
            
            # Get model configurations
            models = self.get_model_configurations()
            
            # Prepare leave-one-out splits
            splits = self.prepare_leave_one_out_splits(datasets)
            
            # Log what would be run
            for test_dataset_name, train_datasets in splits:
                for model_config in models:
                    self.logger.info(f"  - Test on {test_dataset_name}, train on {list(train_datasets.keys())} using {model_config['name']}")
            
            self.logger.info(f"Total planned experiments: {len(splits) * len(models)}")
            return
        
        # Get model configurations
        models = self.get_model_configurations()
        
        # Prepare leave-one-out splits
        splits = self.prepare_leave_one_out_splits(datasets)
        
        # Log experiment configuration
        config = {
            'datasets': list(datasets.keys()),
            'models': [m['name'] for m in models],
            'num_experiments': len(splits) * len(models),
            'args': vars(self.args)
        }
        self.experiment_logger.log_experiment_start(config)
        
        # Run experiments
        total_experiments = len(splits) * len(models)
        completed = 0
        
        for test_dataset_name, train_datasets in splits:
            test_dataset = datasets[test_dataset_name]
            for model_config in models:
                self.run_single_experiment(model_config, test_dataset_name, train_datasets, test_dataset)
                completed += 1
                
                progress_pct = (completed / total_experiments) * 100
                self.logger.info(f"Progress: {completed}/{total_experiments} ({progress_pct:.1f}%)")
                
        # Generate final results
        self.finalize_results()
        
    def finalize_results(self):
        """Generate final results and cleanup."""
        self.logger.info("Finalizing experiment results...")
        
        # Save all results
        self.results_manager.save_results(self.results)
        
        # Create summary tables
        summary_df = self.results_manager.create_summary_table(self.results)
        comparison_df = self.results_manager.generate_model_comparison(self.results)
        
        # Print final summary
        self.results_manager.print_final_summary(self.results)
        
        # Cleanup
        self.experiment_logger.cleanup()
        self.logger.info("Experiment completed successfully!")
    
    def prepare_leave_one_out_splits(self, datasets):
        """Prepare leave-one-dataset-out splits with filtering for test datasets only."""
        splits = []
        
        # Get datasets that can be used for testing
        if self.args.test_datasets == ['ALL']:
            test_candidates = list(datasets.keys())
        else:
            test_candidates = [d for d in self.args.test_datasets if d in datasets]
            # Log available datasets if requested ones are not found
            missing_test_datasets = [d for d in self.args.test_datasets if d not in datasets]
            if missing_test_datasets:
                available_datasets = list(datasets.keys())
                self.logger.warning(f"Requested test datasets not found: {missing_test_datasets}")
                self.logger.warning(f"Available datasets: {available_datasets}")
        
        self.logger.info(f"Test candidates: {test_candidates}")
        self.logger.info(f"Training will use all available datasets in leave-one-out fashion")
        
        for test_dataset_name in test_candidates:
            # For leave-one-out, training datasets are ALL datasets except the test dataset
            train_datasets = {k: v for k, v in datasets.items() if k != test_dataset_name}
            
            # Only add split if we have training data
            if train_datasets:
                splits.append((test_dataset_name, train_datasets))
                self.logger.info(f"Split: Test on '{test_dataset_name}', train on {list(train_datasets.keys())}")
            else:
                self.logger.warning(f"Skipping test on '{test_dataset_name}' - no training datasets available")
        
        if not splits:
            raise ValueError("No valid train/test splits generated. Check your dataset filtering arguments.")
        
        return splits
    
    def combine_train_datasets(self, train_datasets):
        """Combine multiple training datasets into a single DataFrame."""
        
        if not train_datasets:
            return pd.DataFrame()
        
        # Get all dataframes and standardize columns
        dfs_to_concat = []
        all_columns = set()
        
        # First pass: collect all unique columns
        for df in train_datasets.values():
            all_columns.update(df.columns)
        
        # Second pass: standardize each DataFrame to have all columns
        for name, df in train_datasets.items():
            df_copy = df.copy()
            # Add missing columns with NaN
            for col in all_columns:
                if col not in df_copy.columns:
                    df_copy[col] = pd.NA
            dfs_to_concat.append(df_copy)
        
        if dfs_to_concat:
            combined_df = pd.concat(dfs_to_concat, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

    def _validate_loaded_model(self, model, val_loader, device):
        """Simple validation of loaded model to ensure it's working correctly."""
        model.eval()
        correct = 0
        total = 0
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    images, targets, _ = batch_data
                else:
                    images, targets = batch_data
                    
                images, targets = images.to(device), targets.to(device).long()
                
                if self.args.channels_last and device.type == 'cuda':
                    images = images.to(memory_format=torch.channels_last)
                    
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                total += targets.size(0)
                correct += (predictions == targets).sum().item()
                
                all_probs.extend(probabilities[:, 1].cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Stop after a few batches for quick validation
                if total >= 200:  # Quick validation on ~200 samples
                    break
        
        if total > 0:
            accuracy = correct / total
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5
                return accuracy, auc
            except:
                return accuracy, 0.5
        return 0.0, 0.5
    
    def _apply_tta(self, model, images):
        """Apply Test-Time Augmentation for more robust predictions."""
        import torchvision.transforms.functional as TF
        
        batch_size = images.size(0)
        tta_predictions = []
        
        # Original image
        outputs = model(images)
        tta_predictions.append(torch.softmax(outputs, dim=1))
        
        # Horizontal flip
        flipped_images = TF.hflip(images)
        flipped_outputs = model(flipped_images)
        tta_predictions.append(torch.softmax(flipped_outputs, dim=1))
        
        # Rotation augmentations (small rotations that preserve medical relevance)
        for angle in [-5, 5]:  # Small rotations to avoid losing medical information
            rotated_images = TF.rotate(images, angle)
            rotated_outputs = model(rotated_images)
            tta_predictions.append(torch.softmax(rotated_outputs, dim=1))
        
        # Brightness adjustments (common in medical imaging)
        for brightness_factor in [0.9, 1.1]:
            bright_images = TF.adjust_brightness(images, brightness_factor)
            bright_outputs = model(bright_images)
            tta_predictions.append(torch.softmax(bright_outputs, dim=1))
        
        # Contrast adjustments
        for contrast_factor in [0.9, 1.1]:
            contrast_images = TF.adjust_contrast(images, contrast_factor)
            contrast_outputs = model(contrast_images)
            tta_predictions.append(torch.softmax(contrast_outputs, dim=1))
        
        # Average all predictions
        avg_probabilities = torch.stack(tta_predictions).mean(dim=0)
        
        # Convert back to logits for consistency with rest of code
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        avg_probabilities = torch.clamp(avg_probabilities, epsilon, 1 - epsilon)
        avg_logits = torch.log(avg_probabilities / (1 - avg_probabilities + epsilon))
        
        self.logger.info(f"Applied TTA with {len(tta_predictions)} augmentations")
        return avg_logits
        
    def filter_datasets(self, datasets):
        """Filter datasets based on command line arguments."""
        original_count = len(datasets)
        self.logger.info(f"Original datasets loaded: {list(datasets.keys())}")
        
        # First, exclude any explicitly excluded datasets
        if self.args.exclude_datasets:
            for exclude_name in self.args.exclude_datasets:
                if exclude_name in datasets:
                    del datasets[exclude_name]
                    self.logger.info(f"Excluded dataset: {exclude_name}")
        
        filtered_count = len(datasets)
        self.logger.info(f"Datasets after filtering: {list(datasets.keys())} ({filtered_count}/{original_count})")
        
        if not datasets:
            raise ValueError("No datasets remaining after filtering. Check your dataset names.")
        
        return datasets
        
def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Multi-Source Domain Fine-Tuning for Glaucoma Classification"
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data_type', type=str, default='raw', 
                           choices=['raw', 'processed'],
                           help="Target type of image data ('raw' or 'processed')")
    data_group.add_argument('--base_data_root', type=str, default=r'D:\glaucoma\data',
                           help="Absolute base root directory for all datasets")
    
    # NEW: Dataset filtering arguments
    data_group.add_argument('--test_datasets', nargs='*', default=['ALL'],
                           help='Specific datasets to test on. Use "ALL" for all datasets, or specify dataset names like "SMDG-FIVES" "SMDG-PAPILLA" "AIROGS". Training always uses all available datasets in leave-one-out fashion.')
    data_group.add_argument('--exclude_datasets', nargs='*', default=[],
                           help='Datasets to exclude from experiments entirely (both training and testing)')
    
    # SMDG/PAPILLA dataset
    data_group.add_argument('--smdg_metadata_file_raw', type=str, 
                           default=os.path.join('raw','SMDG-19','metadata - standardized.csv'))
    data_group.add_argument('--smdg_image_dir_raw', type=str, 
                           default=os.path.join('raw','SMDG-19','full-fundus','full-fundus'))
    
    # CHAKSU dataset
    data_group.add_argument('--chaksu_base_dir', type=str, 
                           default=os.path.join('raw','Chaksu','Train','Train','1.0_Original_Fundus_Images'))
    data_group.add_argument('--chaksu_decision_dir_raw', type=str, 
                           default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision'))
    data_group.add_argument('--chaksu_metadata_dir_raw', type=str, 
                           default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision','Majority'))
    
    # AIROGS dataset configuration
    data_group.add_argument('--airogs_label_file', type=str, 
                           default=r'D:\glaucoma\data\raw\AIROGS\train_labels.csv',
                           help="Path to AIROGS training labels CSV file")
    data_group.add_argument('--airogs_image_dir', type=str, 
                           default=r'D:\glaucoma\data\raw\AIROGS\img',
                           help="Path to AIROGS image directory")
    data_group.add_argument('--airogs_num_rg_samples', type=int, default=3000,
                           help="Number of RG (glaucoma) samples to load from AIROGS")
    data_group.add_argument('--airogs_num_nrg_samples', type=int, default=3000,
                           help="Number of NRG (normal) samples to load from AIROGS")
    data_group.add_argument('--use_airogs_cache', action='store_true', default=True,
                           help="Use cached AIROGS manifest if available")
    
    # Additional external datasets
    data_group.add_argument('--acrima_image_dir_raw', type=str, 
                           default=os.path.join('raw','ACRIMA','Database','Images'))
    data_group.add_argument('--hygd_image_dir_raw', type=str, 
                           default=os.path.join('raw','HYGD','HYGD','Images'))
    data_group.add_argument('--hygd_labels_file_raw', type=str, 
                           default=os.path.join('raw','HYGD','HYGD','Labels.csv'))
    data_group.add_argument('--papila_base_dir', type=str,
                           default='data/raw/PAPILA')
    data_group.add_argument('--grape_base_dir', type=str,
                           default='data/raw/GRAPE')
    data_group.add_argument('--keh_base_dir', type=str,
                           default='data/raw/KEH')
    data_group.add_argument('--grisk_base_dir', type=str,
                           default='data/raw/griskFundus')
    
    # Dataset selection flags
    data_group.add_argument('--use_smdg', action='store_true', default=True)
    data_group.add_argument('--use_chaksu', action='store_true', default=True)
    data_group.add_argument('--use_airogs', action='store_true', default=True)
    data_group.add_argument('--use_acrima', action='store_true', default=True)
    data_group.add_argument('--use_hygd', action='store_true', default=True)
    data_group.add_argument('--use_papila', action='store_true', default=True)
    data_group.add_argument('--use_grape', action='store_true', default=False)
    data_group.add_argument('--use_keh', action='store_true', default=False)
    data_group.add_argument('--use_grisk', action='store_true', default=False)
    
    # Evaluation flags for external datasets
    data_group.add_argument('--eval_papilla', action='store_true', default=True)
    data_group.add_argument('--eval_chaksu', action='store_true', default=True) 
    data_group.add_argument('--eval_acrima', action='store_true', default=True)
    data_group.add_argument('--eval_hygd', action='store_true', default=True)
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--vfm_weights_path', type=str,
                            default=os.path.join(os.path.dirname(__file__), '..', 'models', 'VFM_Fundus_weights.pth'))
    model_group.add_argument('--additional_models', nargs='*', default=[],
                            help='Additional models in format "name:model_name"')
    model_group.add_argument('--num_classes', type=int, default=2)
    model_group.add_argument('--image_size', type=int, default=224)
    
    # Training arguments
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--num_epochs', type=int, default=10)
    train_group.add_argument('--batch_size', type=int, default=64)
    train_group.add_argument('--eval_batch_size', type=int, default=64)
    train_group.add_argument('--learning_rate', type=float, default=1e-05)
    train_group.add_argument('--weight_decay', type=float, default=0.01)
    train_group.add_argument('--use_data_augmentation', action='store_true', default=True)
    train_group.add_argument('--use_amp', action='store_true', default=True,
                           help='Use Automatic Mixed Precision for faster training')
    train_group.add_argument('--compile_model', action='store_true', default=False,
                           help='Use torch.compile for faster inference (PyTorch 2.0+, requires Triton)')
    train_group.add_argument('--channels_last', action='store_true', default=False,
                           help='Use channels_last memory format for potential speedup')
    train_group.add_argument('--early_stop_patience', type=int, default=2,
                           help='Number of epochs with no improvement after which training will be stopped early')
    
    # Domain adaptation arguments
    domain_group = parser.add_argument_group('Domain Adaptation')
    domain_group.add_argument('--use_domain_adversarial', action='store_true', default=False)
    domain_group.add_argument('--use_mixstyle', action='store_true', default=False)
    domain_group.add_argument('--use_swa', action='store_true', default=False)
    domain_group.add_argument('--use_tta', action='store_true', default=False)
    
    # System arguments
    sys_group = parser.add_argument_group('System Configuration')
    sys_group.add_argument('--seed', type=int, default=42)
    sys_group.add_argument('--num_workers', type=int, default=4)
    sys_group.add_argument('--device', type=str, default='auto')
    
    # Experiment management
    exp_group = parser.add_argument_group('Experiment Management')
    exp_group.add_argument('--experiment_tag', type=str, default='default')
    exp_group.add_argument('--output_dir', type=str, default='experiments/multisource')
    exp_group.add_argument('--checkpoint_dir', type=str, default='checkpoints/multisource')
    exp_group.add_argument('--log_dir', type=str, default='logs/multisource')
    exp_group.add_argument('--resume_from_checkpoint', action='store_true', default=False)
    exp_group.add_argument('--detailed_logging', action='store_true', default=False)
    exp_group.add_argument('--dry_run', action='store_true', default=False,
                          help='Only load and validate datasets without running experiments')
    exp_group.add_argument('--force_retrain', action='store_true', default=False,
                          help='Force retraining even if pre-trained model weights exist')
    exp_group.add_argument('--validate_loaded_models', action='store_true', default=True,
                          help='Validate performance of loaded fine-tuned models and retrain if performance is too low')
    exp_group.add_argument('--min_validation_auc', type=float, default=0.75,
                          help='Minimum validation AUC required for loaded models. If lower, model will be retrained')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create and run experiment
        experiment = MultiSourceExperiment(args)
        experiment.run_all_experiments()
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("Experiment interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
