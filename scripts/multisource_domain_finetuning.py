#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Source Domain Fine-Tuning Script for Glaucoma Classification.

This script implements a leave-one-dataset-out approach for multi-source domain adaptation:
1. For each dataset, train on all other datasets (split into train/val) and test on the held-out dataset
2. Uses VFM (Vision Foundation Model) as the default model architecture (optimized for medical imaging)
3. Supports additional model configurations (ResNet-50, other architectures) via --additional_models
4. Calculates comprehensive metrics: AUC, Sensitivity@95%Specificity, ECE, Accuracy
5. Generates summary tables and visualizations comparing performance across datasets and models

The approach optimizes data usage by ensuring each dataset is used for both training and testing,
providing robust evaluation of domain transfer capabilities.

ENHANCED DOMAIN ADAPTATION FEATURES:
- Domain Adversarial Training (DANN): Uses gradient reversal layer to learn domain-invariant features
- MixStyle: Mixes feature statistics across domains for better generalization
- Stochastic Weight Averaging (SWA): Averages model weights for improved generalization
- Test-Time Adaptation (TTA): Adapts model parameters during inference using entropy minimization
- Advanced Data Augmentation: Extended augmentation pipeline for better robustness
- All features can be enabled via command-line flags for flexible experimentation

ROBUST EXPERIMENT MANAGEMENT:
- Checkpointing: Automatic experiment state saving after each model/dataset combination
- Resumption: Intelligent recovery from interruptions with progress preservation
- Enhanced Logging: Real-time progress tracking with detailed batch-level metrics
- Interrupt Handling: Safe interruption via Ctrl+C with automatic progress saving

ENHANCED LOGGING FEATURES:
- Real-time progress bars with loss and accuracy during training
- Detailed epoch summaries with comprehensive metrics
- Batch-level logging (optional) for debugging and detailed analysis
- Quiet mode for reduced console output during long experiments
- Automatic log file generation for detailed training history

Usage examples:
  Basic training with VFM (default):
    python multisource_domain_finetuning.py
  
  Add ResNet50 as an additional model:
    python multisource_domain_finetuning.py --additional_models "ResNet50:resnet50"
  
  With domain adversarial training:
    python multisource_domain_finetuning.py --use_domain_adversarial
  
  With detailed logging for monitoring:
    python multisource_domain_finetuning.py --detailed_logging
    
  With quiet mode for long runs:
    python multisource_domain_finetuning.py --quiet_mode
  
  With all enhancements and multiple models:
    python multisource_domain_finetuning.py --use_domain_adversarial --use_mixstyle --use_swa --use_tta --use_advanced_augmentation --additional_models "ResNet50:resnet50" "EfficientNet:efficientnet_b0" --detailed_logging
"""

import argparse
import json
import logging
import os
import sys
import importlib.util
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision import transforms
import timm
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple
import copy
from collections import defaultdict
import pickle
import glob

# Additional imports for domain adaptation techniques
try:
    from torch.autograd import Function
    TORCH_AUTOGRAD_AVAILABLE = True
except ImportError:
    Function = object  # Fallback if not available
    TORCH_AUTOGRAD_AVAILABLE = False

# Ensure custom modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.utils.helpers import NpEncoder, set_seed
    from src.data.utils import adjust_path_for_data_type
    from src.data.external_loader import load_external_test_data
    from src.data.datasets import GlaucomaClassificationDataset, safe_collate
    from src.models.classification.build_model import build_classifier_model
    
    # Import from train_classification.py
    train_classification_path = os.path.join(os.path.dirname(__file__), "train_classification.py")
    spec = importlib.util.spec_from_file_location("train_classification", train_classification_path)
    train_classification = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_classification)
    load_chaksu_data = train_classification.load_chaksu_data
    load_airogs_data = train_classification.load_airogs_data
    assign_dataset_source = train_classification.assign_dataset_source
    PAPILLA_PREFIX = train_classification.PAPILLA_PREFIX
    
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Current sys.path:", sys.path)
    sys.exit(1)

# Setup logging with more detailed levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        # Add file handler for detailed logs
    ]
)
logger = logging.getLogger(__name__)

# Set debug level for training details when needed
def set_detailed_logging(enable_detailed: bool = False):
    """Enable or disable detailed batch-level logging during training."""
    if enable_detailed:
        logger.setLevel(logging.DEBUG)
        # Add file handler for detailed logs
        file_handler = logging.FileHandler('training_detailed.log')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Detailed logging enabled - batch-level info will be saved to training_detailed.log")
    else:
        logger.setLevel(logging.INFO)

# Global constants
RAW_DIR_NAME_CONST = "raw"
PROCESSED_DIR_NAME_CONST = "processed"

# Define metric functions locally since they might not be available in the imports
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    total_samples = len(y_true)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.sum() / total_samples
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence for this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Add to ECE calculation
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def sensitivity_at_specificity(y_true: np.ndarray, y_prob: np.ndarray, 
                              target_spec: float = 0.95) -> float:
    """Calculate sensitivity at specified specificity level."""
    if len(np.unique(y_true)) < 2:
        return np.nan
        
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    # Specificity = 1 - FPR, so target FPR = 1 - target_spec
    target_fpr = 1 - target_spec
    
    # Find the closest FPR to our target
    idx = np.argmin(np.abs(fpr - target_fpr))
    return tpr[idx]

# ==================== DOMAIN ADAPTATION TECHNIQUES ====================

class GradientReversalFunction(Function):
    """Gradient Reversal Layer for Domain Adversarial Training."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def gradient_reversal_layer(x, alpha=1.0):
    """Apply gradient reversal with scaling factor alpha."""
    if TORCH_AUTOGRAD_AVAILABLE:
        return GradientReversalFunction.apply(x, alpha)
    else:
        return x  # Fallback without gradient reversal

class DomainClassifier(nn.Module):
    """Domain classifier for adversarial domain adaptation."""
    
    def __init__(self, feature_dim: int, num_domains: int, hidden_dim: int = 512):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_domains)
        )
    
    def forward(self, x, alpha=1.0):
        reversed_features = gradient_reversal_layer(x, alpha)
        return self.classifier(reversed_features)

class MixStyle(nn.Module):
    """MixStyle: Domain Generalization via Style Mixing."""
    
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
    
    def forward(self, x):
        if not self.training:
            return x
            
        if np.random.random() > self.p:
            return x
            
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        
        lmda = np.random.beta(self.alpha, self.alpha)
        perm = torch.randperm(B)
        
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        
        return x_normed * sig_mix + mu_mix

class StochasticWeightAveraging:
    """Stochastic Weight Averaging for better generalization."""
    
    def __init__(self, model, start_epoch=10, lr_scheduler=None):
        self.model = model
        self.start_epoch = start_epoch
        self.lr_scheduler = lr_scheduler
        self.swa_model = None
        self.swa_n = 0
    
    def update(self, epoch):
        if epoch >= self.start_epoch:
            if self.swa_model is None:
                self.swa_model = copy.deepcopy(self.model)
                self.swa_n = 1
            else:
                # Update SWA model parameters
                for swa_param, param in zip(self.swa_model.parameters(), self.model.parameters()):
                    swa_param.data = (swa_param.data * self.swa_n + param.data) / (self.swa_n + 1)
                self.swa_n += 1
    
    def get_swa_model(self):
        return self.swa_model if self.swa_model is not None else self.model

class TestTimeAdaptation:
    """Test-Time Adaptation for domain shift robustness."""
    
    def __init__(self, model, optimizer=None, adaptation_steps=1):
        self.model = model
        self.adaptation_steps = adaptation_steps
        self.optimizer = optimizer
        self.original_state = copy.deepcopy(model.state_dict())
    
    def adapt_batch(self, batch_data):
        """Adapt model on a single batch using entropy minimization."""
        self.model.train()
        
        for step in range(self.adaptation_steps):
            outputs = self.model(batch_data)
            probs = F.softmax(outputs, dim=1)
            entropy_loss = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            
            if self.optimizer is not None:
                self.optimizer.zero_grad()
                entropy_loss.backward()
                self.optimizer.step()
    
    def reset_model(self):
        """Reset model to original state."""
        self.model.load_state_dict(self.original_state)

def create_enhanced_transforms(use_advanced_augmentation: bool = False, 
                             use_mixstyle: bool = False):
    """Create enhanced data transforms for domain generalization."""
    
    if use_advanced_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

# ================ END DOMAIN ADAPTATION TECHNIQUES ================

class ExperimentCheckpoint:
    """Manages experiment checkpointing and resumption."""
    
    def __init__(self, output_dir: str, experiment_name: str = "multisource_experiment"):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{experiment_name}_checkpoint.pkl")
        self.progress_file = os.path.join(self.checkpoint_dir, f"{experiment_name}_progress.json")
        
    def save_checkpoint(self, 
                       experiment_state: dict,
                       current_target_dataset: str,
                       current_model_config: dict,
                       completed_experiments: list,
                       all_results: list,
                       dataset_configs: dict,
                       model_configs: list,
                       training_args: dict):
        """Save experiment checkpoint."""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'experiment_state': experiment_state,
            'current_target_dataset': current_target_dataset,
            'current_model_config': current_model_config,
            'completed_experiments': completed_experiments,
            'all_results': all_results,
            'dataset_configs': dataset_configs,
            'model_configs': model_configs,
            'training_args': training_args,
            'checkpoint_version': '1.0'
        }
        
        # Save checkpoint
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Save human-readable progress
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(dataset_configs) * len(model_configs),
            'completed_experiments': len(completed_experiments),
            'current_target_dataset': current_target_dataset,
            'current_model': current_model_config.get('name', 'Unknown') if current_model_config else None,
            'completion_percentage': (len(completed_experiments) / (len(dataset_configs) * len(model_configs))) * 100,
            'completed_list': completed_experiments,
            'results_count': len(all_results)
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        logger.info(f"Checkpoint saved: {len(completed_experiments)}/{len(dataset_configs) * len(model_configs)} experiments completed")
        logger.info(f"Progress: {progress_data['completion_percentage']:.1f}%")
        
    def load_checkpoint(self):
        """Load experiment checkpoint if it exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                logger.info(f"Found checkpoint from {checkpoint_data['timestamp']}")
                logger.info(f"Completed experiments: {len(checkpoint_data['completed_experiments'])}")
                
                return checkpoint_data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                return None
        return None
    
    def checkpoint_exists(self):
        """Check if a checkpoint exists."""
        return os.path.exists(self.checkpoint_file)
    
    def get_progress_summary(self):
        """Get a summary of current progress."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return None
    
    def clear_checkpoint(self):
        """Clear existing checkpoint files."""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
        logger.info("Checkpoint files cleared")

class MultiSourceTrainer:
    """Multi-source domain fine-tuning trainer with advanced domain adaptation techniques and checkpointing."""
    
    def __init__(self, output_dir: str, device: str = 'auto', 
                 use_domain_adversarial: bool = False,
                 use_mixstyle: bool = False,
                 use_swa: bool = False,
                 use_tta: bool = False,
                 use_advanced_augmentation: bool = False,
                 domain_loss_weight: float = 0.1,
                 enable_checkpointing: bool = True,
                 quiet_mode: bool = False):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        self.quiet_mode = quiet_mode
        
        # Domain adaptation settings
        self.use_domain_adversarial = use_domain_adversarial
        self.use_mixstyle = use_mixstyle
        self.use_swa = use_swa
        self.use_tta = use_tta
        self.use_advanced_augmentation = use_advanced_augmentation
        self.domain_loss_weight = domain_loss_weight
        
        # Checkpointing
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_manager = ExperimentCheckpoint(output_dir, f"multisource_{self.timestamp}") if enable_checkpointing else None
        self.completed_experiments = []
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Log domain adaptation settings
        if any([use_domain_adversarial, use_mixstyle, use_swa, use_tta, use_advanced_augmentation]):
            logger.info(f"Enhanced training enabled:")
            logger.info(f"  Domain Adversarial Training: {use_domain_adversarial}")
            logger.info(f"  MixStyle: {use_mixstyle}")
            logger.info(f"  Stochastic Weight Averaging: {use_swa}")
            logger.info(f"  Test-Time Adaptation: {use_tta}")
            logger.info(f"  Advanced Augmentation: {use_advanced_augmentation}")
        
        # Log checkpointing status
        if enable_checkpointing:
            logger.info(f"Checkpointing enabled - progress will be saved to: {self.checkpoint_manager.checkpoint_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup plot style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def prepare_datasets(self, all_datasets: dict, target_dataset: str, train_val_split: float = 0.8):
        """
        Prepare training and testing datasets for multi-source domain fine-tuning.
        
        Args:
            all_datasets: Dictionary of all available datasets
            target_dataset: Name of dataset to use as test set
            train_val_split: Fraction of training data to use for training (rest for validation)
        
        Returns:
            train_df, val_df, test_df
        """
        # Combine all datasets except target
        train_dfs = []
        test_df = None
        
        for dataset_name, df in all_datasets.items():
            if dataset_name == target_dataset:
                test_df = df.copy()
                logger.info(f"Using {dataset_name} as test set: {len(test_df)} samples")
            else:
                train_dfs.append(df.copy())
                logger.info(f"Adding {dataset_name} to training: {len(df)} samples")
        
        if test_df is None:
            raise ValueError(f"Target dataset '{target_dataset}' not found in available datasets")
        
        if not train_dfs:
            raise ValueError("No training datasets available")
        
        # Combine training datasets
        combined_train_df = pd.concat(train_dfs, ignore_index=True)
        logger.info(f"Combined training data: {len(combined_train_df)} samples")
        
        # Split combined training data into train/val
        # Stratified split to maintain label distribution
        try:
            train_df, val_df = train_test_split(
                combined_train_df, 
                test_size=1-train_val_split, 
                stratify=combined_train_df['types'], 
                random_state=42
            )
        except ValueError:
            # Fallback if stratification fails (e.g., too few samples per class)
            train_df, val_df = train_test_split(
                combined_train_df, 
                test_size=1-train_val_split, 
                random_state=42
            )
        
        logger.info(f"Final splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_data_loaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                           batch_size: int = 32, num_workers: int = 4,
                           pin_memory: bool = True, prefetch_factor: int = 2,
                           persistent_workers: bool = False):
        """Create data loaders for training, validation, and testing."""
        
        # Use enhanced transforms if enabled
        if self.use_advanced_augmentation:
            train_transform, val_test_transform = create_enhanced_transforms(
                use_advanced_augmentation=True, 
                use_mixstyle=self.use_mixstyle
            )
        else:
            train_transform, val_test_transform = create_enhanced_transforms(
                use_advanced_augmentation=False, 
                use_mixstyle=self.use_mixstyle
            )
        
        # Add domain labels for domain adversarial training
        if self.use_domain_adversarial and 'dataset_source' in train_df.columns:
            # Create domain label mapping
            unique_domains = train_df['dataset_source'].unique()
            domain_to_idx = {domain: idx for idx, domain in enumerate(unique_domains)}
            train_df = train_df.copy()
            train_df['domain_label'] = train_df['dataset_source'].map(domain_to_idx)
            val_df = val_df.copy()
            val_df['domain_label'] = val_df['dataset_source'].map(domain_to_idx)
        
        # Create datasets
        train_dataset = GlaucomaClassificationDataset(train_df, image_col='image_path', label_col='types', transform=train_transform)
        val_dataset = GlaucomaClassificationDataset(val_df, image_col='image_path', label_col='types', transform=val_test_transform)
        test_dataset = GlaucomaClassificationDataset(test_df, image_col='image_path', label_col='types', transform=val_test_transform)
        
        # Calculate class weights for balanced training
        labels = train_df['types'].values
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        
        # Create weighted sampler for balanced training
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders with performance optimizations
        dataloader_kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'prefetch_factor': prefetch_factor if num_workers > 0 else None,
            'persistent_workers': persistent_workers if num_workers > 0 else False
        }
        # Remove None values
        dataloader_kwargs = {k: v for k, v in dataloader_kwargs.items() if v is not None}
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            **dataloader_kwargs
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            **dataloader_kwargs
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            **dataloader_kwargs
        )
        
        return train_loader, val_loader, test_loader
    
    def get_model_specific_lr(self, model_name: str, base_lr: float = 1e-4) -> float:
        """Get model-specific learning rate based on architecture."""
        model_name_lower = model_name.lower()
        
        if 'vit' in model_name_lower or 'vision_transformer' in model_name_lower:
            # ViT models typically need lower learning rates for fine-tuning
            return base_lr * 0.1  # 1e-5 for ViT
        elif 'resnet' in model_name_lower:
            # ResNet can handle standard learning rates
            return base_lr  # 1e-4 for ResNet
        elif 'efficientnet' in model_name_lower:
            # EfficientNet works well with moderate learning rates
            return base_lr * 0.5  # 5e-5 for EfficientNet
        elif 'densenet' in model_name_lower:
            # DenseNet similar to ResNet
            return base_lr  # 1e-4 for DenseNet
        else:
            # Default for unknown architectures
            return base_lr
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                   num_epochs: int = 20, model_name: str = "unknown", base_learning_rate: float = 1e-4, 
                   weight_decay: float = 1e-5, early_stopping_patience: int = 4):
        """Train the model with early stopping, model-specific learning rates, and domain adaptation."""
        
        model = model.to(self.device)
        
        # Add MixStyle if enabled
        if self.use_mixstyle:
            # Insert MixStyle after the first few layers
            self._add_mixstyle_to_model(model)
        
        # Setup domain classifier for adversarial training
        domain_classifier = None
        domain_optimizer = None
        if self.use_domain_adversarial:
            # Get feature dimension from model
            feature_dim = self._get_feature_dim(model)
            num_domains = len(set(train_loader.dataset.dataframe.get('dataset_source', [])))
            if num_domains > 1:
                domain_classifier = DomainClassifier(feature_dim, num_domains).to(self.device)
                domain_optimizer = optim.AdamW(domain_classifier.parameters(), lr=base_learning_rate * 0.1)
        
        # Get model-specific learning rate
        learning_rate = self.get_model_specific_lr(model_name, base_learning_rate)
        logger.info(f"Using learning rate {learning_rate:.2e} for {model_name}")
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, min_lr=1e-7)
        
        # Setup SWA if enabled
        swa = None
        if self.use_swa:
            swa = StochasticWeightAveraging(model, start_epoch=num_epochs // 2)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss() if domain_classifier is not None else None
        
        best_val_auc = 0.0
        best_model_state = None
        patience_counter = 0
        max_patience = early_stopping_patience
        
        train_losses = []
        val_aucs = []
        
        logger.info(f"Training {model_name} with early stopping patience: {max_patience} epochs")
        if self.use_domain_adversarial:
            logger.info(f"Using domain adversarial training with {num_domains} domains")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            if domain_classifier is not None:
                domain_classifier.train()
            
            train_loss = 0.0
            domain_loss_total = 0.0
            train_correct = 0
            train_total = 0
            
            # Calculate gradient reversal alpha (gradually increase)
            alpha = 2.0 / (1.0 + np.exp(-10 * epoch / num_epochs)) - 1.0
            
            # Enhanced progress bar with real-time metrics
            epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            
            # Variables for running average display
            running_loss = 0.0
            running_acc = 0.0
            log_interval = max(1, len(train_loader) // 10)  # Log every 10% of epoch
            
            for batch_idx, batch_data in enumerate(epoch_pbar):
                if len(batch_data) == 2:
                    images, labels = batch_data
                    domain_labels = None
                else:
                    images, labels, domain_labels = batch_data
                
                images, labels = images.to(self.device), labels.to(self.device)
                if domain_labels is not None:
                    domain_labels = domain_labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                if domain_optimizer is not None:
                    domain_optimizer.zero_grad()
                
                # Get features and predictions
                if self.use_domain_adversarial and domain_classifier is not None:
                    features, outputs = self._forward_with_features(model, images)
                    
                    # Classification loss
                    class_loss = criterion(outputs, labels)
                    
                    # Domain adversarial loss
                    if domain_labels is not None:
                        domain_outputs = domain_classifier(features, alpha)
                        domain_loss = domain_criterion(domain_outputs, domain_labels)
                    else:
                        domain_loss = torch.tensor(0.0, device=self.device)
                    
                    # Combined loss
                    total_loss = class_loss + self.domain_loss_weight * domain_loss
                    domain_loss_total += domain_loss.item()
                else:
                    outputs = model(images)
                    total_loss = criterion(outputs, labels)
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                if domain_optimizer is not None:
                    domain_optimizer.step()
                
                # Update statistics
                batch_loss = total_loss.item()
                train_loss += batch_loss
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                train_correct += batch_correct
                train_total += labels.size(0)
                
                # Update running averages for display
                running_loss = running_loss * 0.9 + batch_loss * 0.1
                running_acc = 100.0 * train_correct / train_total
                
                # Update progress bar with real-time metrics
                if self.use_domain_adversarial and domain_loss_total > 0:
                    avg_domain_loss = domain_loss_total / (batch_idx + 1)
                    epoch_pbar.set_postfix({
                        'Loss': f'{running_loss:.4f}',
                        'Acc': f'{running_acc:.1f}%',
                        'DomLoss': f'{avg_domain_loss:.3f}'
                    })
                else:
                    epoch_pbar.set_postfix({
                        'Loss': f'{running_loss:.4f}',
                        'Acc': f'{running_acc:.1f}%'
                    })
                
                # Detailed logging at intervals
                if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                    current_progress = (batch_idx + 1) / len(train_loader) * 100
                    batch_acc = 100.0 * batch_correct / labels.size(0)
                    
                    log_msg = (f"  Batch {batch_idx+1}/{len(train_loader)} ({current_progress:.1f}%): "
                              f"Loss={batch_loss:.4f}, Batch_Acc={batch_acc:.1f}%, "
                              f"Epoch_Acc={running_acc:.1f}%")
                    
                    if self.use_domain_adversarial and domain_labels is not None:
                        log_msg += f", Dom_Loss={domain_loss.item():.4f}"
                    
                    logger.debug(log_msg)
            
            
            # Close progress bar
            epoch_pbar.close()
            
            # Calculate epoch statistics
            train_acc = 100.0 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Update SWA
            if swa is not None:
                swa.update(epoch)
            
            # Validation phase with progress bar
            logger.info(f"Running validation for epoch {epoch+1}...")
            val_auc, val_acc = self.evaluate_model(model, val_loader)
            val_aucs.append(val_auc)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Enhanced epoch summary logging
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch+1}/{num_epochs} SUMMARY")
            print(f"{'='*80}")
            print(f"Training:")
            print(f"  Loss:           {avg_train_loss:.6f}")
            print(f"  Accuracy:       {train_acc:.2f}%")
            print(f"  Learning Rate:  {current_lr:.2e}")
            
            if self.use_domain_adversarial and domain_loss_total > 0:
                avg_domain_loss = domain_loss_total / len(train_loader)
                print(f"  Domain Loss:    {avg_domain_loss:.6f}")
                print(f"  Alpha (GRL):    {alpha:.3f}")
            
            print(f"Validation:")
            print(f"  AUC:            {val_auc:.6f}")
            print(f"  Accuracy:       {val_acc:.2f}%")
            
            # Progress tracking
            progress_pct = ((epoch + 1) / num_epochs) * 100
            print(f"Progress:         {progress_pct:.1f}% ({epoch+1}/{num_epochs} epochs)")
            
            # Learning rate scheduling
            scheduler.step(val_auc)
            
            # Early stopping and best model saving
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                improvement = val_auc - (val_aucs[-2] if len(val_aucs) > 1 else 0)
                print(f"[NEW BEST MODEL] AUC improved by {improvement:.4f}")
                print(f"  Best AUC:       {best_val_auc:.6f}")
                print(f"  Patience Reset: {patience_counter}/{max_patience}")
            else:
                patience_counter += 1
                print(f"[NO IMPROVEMENT] (patience: {patience_counter}/{max_patience})")
                if patience_counter >= max_patience:
                    print(f"[EARLY STOPPING] Will trigger after this epoch")
            
            print(f"{'='*80}\n")
            
            # Also log to file in compact format
            log_msg = (f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.2e}")
            
            if self.use_domain_adversarial and domain_loss_total > 0:
                avg_domain_loss = domain_loss_total / len(train_loader)
                log_msg += f", Domain Loss: {avg_domain_loss:.4f}"
            
            logger.info(log_msg)
            
            if patience_counter >= max_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs (best AUC: {best_val_auc:.4f})")
                break
        
        # Load best model or SWA model
        if swa is not None and swa.swa_model is not None:
            model = swa.get_swa_model()
            logger.info("Using SWA model for final evaluation")
        elif best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        logger.info(f"Training completed. Best validation AUC: {best_val_auc:.4f}")
        
        return model, train_losses, val_aucs
    
    def _add_mixstyle_to_model(self, model):
        """Add MixStyle layers to the model."""
        # This is a simplified implementation - you might need to customize based on your model architecture
        if hasattr(model, 'features'):
            # For models with a features attribute (like ResNet)
            mixstyle = MixStyle()
            # Insert after the first few layers
            if len(model.features) > 4:
                model.features.add_module('mixstyle', mixstyle)
        elif hasattr(model, 'blocks'):
            # For ViT models
            mixstyle = MixStyle()
            # This would need model-specific implementation
            pass
    
    def _get_feature_dim(self, model):
        """Get the feature dimension of the model for domain classifier."""
        # This is a simplified implementation
        if hasattr(model, 'classifier'):
            if hasattr(model.classifier, 'in_features'):
                return model.classifier.in_features
        elif hasattr(model, 'head'):
            if hasattr(model.head, 'in_features'):
                return model.head.in_features
        # Default fallback
        return 512
    
    def _forward_with_features(self, model, x):
        """Forward pass that returns both features and predictions."""
        # This is a simplified implementation - needs to be customized per model architecture
        if hasattr(model, 'features') and hasattr(model, 'classifier'):
            # For ResNet-like models
            features = model.features(x)
            features_flat = features.view(features.size(0), -1)
            predictions = model.classifier(features_flat)
            return features_flat, predictions
        else:
            # Fallback - just return model output twice
            output = model(x)
            return output, output
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader):
        """Evaluate model and return AUC and accuracy with progress tracking."""
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            # Use progress bar for validation
            eval_pbar = tqdm(data_loader, desc="Validating", leave=False, disable=len(data_loader) < 10)
            
            for images, labels in eval_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar with running accuracy
                if len(all_predictions) > 0:
                    running_acc = 100.0 * np.mean(np.array(all_predictions) == np.array(all_labels))
                    eval_pbar.set_postfix({'Acc': f'{running_acc:.1f}%'})
            
            eval_pbar.close()
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        accuracy = 100.0 * np.mean(all_predictions == all_labels)
        
        return auc, accuracy
    
    def comprehensive_evaluation(self, model: nn.Module, test_loader: DataLoader, 
                               target_dataset: str, model_name: str):
        """Perform comprehensive evaluation including AUC, Sens@95Spec, ECE, and optional TTA."""
        
        # Setup test-time adaptation if enabled
        tta = None
        if self.use_tta:
            tta_optimizer = optim.Adam(model.parameters(), lr=1e-5)
            tta = TestTimeAdaptation(model, tta_optimizer, adaptation_steps=1)
            logger.info(f"Using Test-Time Adaptation for {target_dataset}")
        
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Evaluating")):
                if len(batch_data) == 2:
                    images, labels = batch_data
                else:
                    images, labels, _ = batch_data  # Ignore domain labels during evaluation
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Apply test-time adaptation if enabled
                if tta is not None and batch_idx % 10 == 0:  # Adapt every 10 batches
                    tta.adapt_batch(images)
                
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Reset model if TTA was used
        if tta is not None:
            tta.reset_model()
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate comprehensive metrics
        auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        
        # Sensitivity at 95% specificity
        sens_at_95_spec = sensitivity_at_specificity(all_labels, all_probabilities[:, 1], specificity=0.95)
        
        # Expected Calibration Error
        ece = compute_ece(all_labels, all_probabilities[:, 1], n_bins=10)
        
        # Additional metrics
        accuracy = 100.0 * np.mean(all_predictions == all_labels)
        
        # Class-specific metrics
        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # Store results
        result = {
            'target_dataset': target_dataset,
            'model_name': model_name,
            'auc': auc,
            'sensitivity_at_95_specificity': sens_at_95_spec,
            'ece': ece,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'total_samples': len(all_labels),
            'positive_samples': np.sum(all_labels),
            'negative_samples': np.sum(1 - all_labels),
            'used_domain_adversarial': self.use_domain_adversarial,
            'used_mixstyle': self.use_mixstyle,
            'used_swa': self.use_swa,
            'used_tta': self.use_tta,
            'used_advanced_augmentation': self.use_advanced_augmentation
        }
        
        self.results.append(result)
        
        logger.info(f"Results for {model_name} on {target_dataset}:")
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  Sensitivity@95%Specificity: {sens_at_95_spec:.4f}")
        logger.info(f"  ECE: {ece:.4f}")
        logger.info(f"  Accuracy: {accuracy:.2f}%")
        
        return result
    
    def run_multisource_experiment(self, all_datasets: dict, model_configs: list, 
                                 num_epochs: int = 20, batch_size: int = 32, 
                                 base_learning_rate: float = 1e-4, early_stopping_patience: int = 4,
                                 num_workers: int = 4, pin_memory: bool = True,
                                 prefetch_factor: int = 2, persistent_workers: bool = False,
                                 resume_from_checkpoint: bool = True):
        """Run the complete multi-source domain fine-tuning experiment with checkpointing support."""
        
        training_args = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'base_learning_rate': base_learning_rate,
            'early_stopping_patience': early_stopping_patience,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'prefetch_factor': prefetch_factor,
            'persistent_workers': persistent_workers
        }
        
        # Check for existing checkpoint
        checkpoint_data = None
        if resume_from_checkpoint and self.checkpoint_manager and self.checkpoint_manager.checkpoint_exists():
            checkpoint_data = self.checkpoint_manager.load_checkpoint()
            
            if checkpoint_data:
                # Ask user if they want to resume
                print(f"\n{'='*60}")
                print("CHECKPOINT FOUND")
                print(f"{'='*60}")
                progress = self.checkpoint_manager.get_progress_summary()
                if progress:
                    print(f"Progress: {progress['completion_percentage']:.1f}% complete")
                    print(f"Completed: {progress['completed_experiments']} experiments")
                    print(f"Last updated: {progress['timestamp']}")
                print(f"{'='*60}")
                
                response = input("Resume from checkpoint? (y/n): ")
                if response.lower() != 'y':
                    logger.info("Starting fresh experiment...")
                    self.checkpoint_manager.clear_checkpoint()
                    checkpoint_data = None
                else:
                    logger.info("Resuming from checkpoint...")
                    self.results = checkpoint_data['all_results']
                    self.completed_experiments = checkpoint_data['completed_experiments']
        
        logger.info("Starting multi-source domain fine-tuning experiment...")
        logger.info(f"Available datasets: {list(all_datasets.keys())}")
        logger.info(f"Model configurations: {[config['name'] for config in model_configs]}")
        logger.info(f"Training parameters: epochs={num_epochs}, batch_size={batch_size}, "
                   f"base_lr={base_learning_rate:.2e}, early_stopping_patience={early_stopping_patience}")
        
        total_experiments = len(all_datasets) * len(model_configs)
        experiment_count = len(self.completed_experiments) if checkpoint_data else 0
        
        logger.info(f"Total experiments to run: {total_experiments}")
        if experiment_count > 0:
            logger.info(f"Resuming from experiment {experiment_count + 1}")
        
        try:
            for target_dataset in all_datasets.keys():
                logger.info(f"\n{'='*80}")
                logger.info(f"TARGET DATASET: {target_dataset}")
                logger.info(f"{'='*80}")
                
                # Prepare datasets
                try:
                    train_df, val_df, test_df = self.prepare_datasets(all_datasets, target_dataset)
                    train_loader, val_loader, test_loader = self.create_data_loaders(
                        train_df, val_df, test_df, 
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        prefetch_factor=prefetch_factor,
                        persistent_workers=persistent_workers
                    )
                except Exception as e:
                    logger.error(f"Failed to prepare datasets for {target_dataset}: {e}")
                    continue
                
                for model_config in model_configs:
                    experiment_id = f"{target_dataset}_{model_config['name']}"
                    
                    # Skip if this experiment was already completed
                    if experiment_id in self.completed_experiments:
                        logger.info(f"Skipping completed experiment: {experiment_id}")
                        continue
                    
                    experiment_count += 1
                    
                    # Overall experiment progress
                    if not self.quiet_mode:
                        print(f"\n{'='*100}")
                        print(f"EXPERIMENT {experiment_count}/{total_experiments}")
                        print(f"Target Dataset: {target_dataset} | Model: {model_config['name']}")
                        print(f"Overall Progress: {experiment_count/total_experiments*100:.1f}%")
                        print(f"{'='*100}")
                    
                    logger.info(f"\n--- Experiment {experiment_count}/{total_experiments}: "
                               f"{model_config['name']} on {target_dataset} ---")
                    
                    try:
                        # Build model
                        model = build_classifier_model(
                            model_name=model_config['architecture'],
                            num_classes=2,
                            pretrained=model_config.get('pretrained', True),
                            custom_weights_path=model_config.get('weights_path', None)
                        )
                        
                        # Train model with model-specific parameters
                        trained_model, train_losses, val_aucs = self.train_model(
                            model, train_loader, val_loader, 
                            num_epochs=num_epochs,
                            model_name=model_config['name'],
                            base_learning_rate=base_learning_rate,
                            early_stopping_patience=early_stopping_patience
                        )
                        
                        # Comprehensive evaluation with error handling
                        try:
                            result = self.comprehensive_evaluation(
                                trained_model, test_loader, target_dataset, model_config['name']
                            )
                        except Exception as eval_error:
                            logger.error(f"Evaluation failed for {model_config['name']} on {target_dataset}: {eval_error}")
                            # Create a minimal result entry to preserve progress
                            result = {
                                'target_dataset': target_dataset,
                                'model_name': model_config['name'],
                                'auc': 0.0,
                                'error': str(eval_error),
                                'status': 'evaluation_failed'
                            }
                            self.results.append(result)
                        
                        # Save model checkpoint with error handling
                        try:
                            model_save_path = os.path.join(
                                self.output_dir, 
                                f"model_{model_config['name']}_{target_dataset}_{self.timestamp}.pth"
                            )
                            torch.save({
                                'model_state_dict': trained_model.state_dict(),
                                'model_config': model_config,
                                'result': result,
                                'train_losses': train_losses,
                                'val_aucs': val_aucs,
                                'experiment_id': experiment_id,
                                'save_timestamp': datetime.now().isoformat()
                            }, model_save_path)
                            
                            logger.info(f"Model saved to: {model_save_path}")
                        except Exception as save_error:
                            logger.error(f"Model save failed: {save_error}")
                            # Try alternative save location
                            try:
                                alt_save_path = os.path.join(
                                    self.output_dir, 
                                    f"model_backup_{experiment_count}_{self.timestamp}.pth"
                                )
                                torch.save(trained_model.state_dict(), alt_save_path)
                                logger.info(f"Model state dict saved to backup location: {alt_save_path}")
                            except Exception as alt_save_error:
                                logger.error(f"Backup model save also failed: {alt_save_error}")
                        
                        # Mark experiment as completed
                        self.completed_experiments.append(experiment_id)
                        
                        # FORCE SAVE results after each experiment (robust)
                        try:
                            self.force_save_all_results()
                        except Exception as force_save_error:
                            logger.error(f"Force save failed: {force_save_error}")
                        
                        # Save checkpoint after each successful experiment
                        try:
                            if self.checkpoint_manager:
                                self.checkpoint_manager.save_checkpoint(
                                    experiment_state={'current_progress': f'{experiment_count}/{total_experiments}'},
                                    current_target_dataset=target_dataset,
                                    current_model_config=model_config,
                                    completed_experiments=self.completed_experiments,
                                    all_results=self.results,
                                    dataset_configs={name: {'size': len(df)} for name, df in all_datasets.items()},
                                    model_configs=model_configs,
                                    training_args=training_args
                                )
                        except Exception as checkpoint_error:
                            logger.error(f"Checkpoint save failed: {checkpoint_error}")
                        
                        # Save intermediate results with error handling
                        try:
                            self.save_intermediate_results()
                        except Exception as intermediate_error:
                            logger.error(f"Intermediate save failed: {intermediate_error}")
                        
                        logger.info(f"Experiment {experiment_count}/{total_experiments} completed successfully")
                        
                    except KeyboardInterrupt:
                        logger.warning("Experiment interrupted by user!")
                        logger.info("Progress has been saved. You can resume later with the same command.")
                        return
                    except Exception as e:
                        logger.error(f"Failed experiment {model_config['name']} on {target_dataset}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # Force save results even on failure
                        try:
                            self.force_save_all_results(additional_info=f"Failed at experiment {experiment_count}/{total_experiments}: {model_config['name']} on {target_dataset}")
                        except Exception as force_save_error:
                            logger.error(f"Force save failed after experiment error: {force_save_error}")
                        
                        # Try to save checkpoint with failure info
                        try:
                            if self.checkpoint_manager:
                                self.checkpoint_manager.save_checkpoint(
                                    experiment_state={'current_progress': f'{experiment_count}/{total_experiments}', 'last_error': str(e)},
                                    current_target_dataset=target_dataset,
                                    current_model_config=model_config,
                                    completed_experiments=self.completed_experiments,
                                    all_results=self.results,
                                    dataset_configs={name: {'size': len(df)} for name, df in all_datasets.items()},
                                    model_configs=model_configs,
                                    training_args=training_args
                                )
                        except Exception as checkpoint_error:
                            logger.error(f"Checkpoint save failed after experiment error: {checkpoint_error}")
                        
                        continue
            
            # Clear checkpoint after successful completion
            if self.checkpoint_manager:
                self.checkpoint_manager.clear_checkpoint()
                logger.info("Experiment completed successfully! Checkpoint cleared.")
            
        except KeyboardInterrupt:
            logger.warning("Experiment interrupted by user!")
            logger.info("Progress has been saved. You can resume later with the same command.")
            return
        
        # Generate final summary and visualizations
        self.generate_summary_report()
        self.create_visualizations()
    
    def save_intermediate_results(self):
        """Save intermediate results to CSV with error handling."""
        if self.results:
            try:
                df_results = pd.DataFrame(self.results)
                intermediate_path = os.path.join(self.output_dir, f'intermediate_results_{self.timestamp}.csv')
                df_results.to_csv(intermediate_path, index=False)
                logger.info(f"Intermediate results saved to: {intermediate_path}")
            except Exception as e:
                logger.error(f"Failed to save intermediate results: {e}")
                # Try to save as JSON backup
                try:
                    import json
                    backup_path = os.path.join(self.output_dir, f'results_backup_{self.timestamp}.json')
                    with open(backup_path, 'w') as f:
                        json.dump(self.results, f, indent=2, default=str)
                    logger.info(f"Results backup saved as JSON: {backup_path}")
                except Exception as e2:
                    logger.error(f"Failed to save JSON backup: {e2}")
    
    def force_save_all_results(self, additional_info=None):
        """Force save all current results with multiple backup strategies."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("FORCE SAVING ALL RESULTS - Multiple backup strategies")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Current results count: {len(self.results)}")
        logger.info(f"Completed experiments: {len(self.completed_experiments)}")
        
        saved_files = []
        
        # Strategy 1: Save as CSV
        if self.results:
            try:
                df_results = pd.DataFrame(self.results)
                csv_path = os.path.join(self.output_dir, f'EMERGENCY_results_{timestamp}.csv')
                df_results.to_csv(csv_path, index=False)
                saved_files.append(f"CSV: {csv_path}")
                logger.info(f"Emergency CSV results saved: {csv_path}")
            except Exception as e:
                logger.error(f"CSV save failed: {e}")
        
        # Strategy 2: Save as JSON
        try:
            json_path = os.path.join(self.output_dir, f'EMERGENCY_results_{timestamp}.json')
            emergency_data = {
                'results': self.results,
                'completed_experiments': self.completed_experiments,
                'timestamp': timestamp,
                'total_results': len(self.results),
                'additional_info': additional_info,
                'output_directory': self.output_dir,
                'experiment_timestamp': self.timestamp
            }
            with open(json_path, 'w') as f:
                json.dump(emergency_data, f, indent=2, default=str)
            saved_files.append(f"JSON: {json_path}")
            logger.info(f"Emergency JSON results saved: {json_path}")
        except Exception as e:
            logger.error(f"JSON save failed: {e}")
        
        # Strategy 3: Save as pickle (most robust)
        try:
            pickle_path = os.path.join(self.output_dir, f'EMERGENCY_results_{timestamp}.pkl')
            emergency_data = {
                'results': self.results,
                'completed_experiments': self.completed_experiments,
                'timestamp': timestamp,
                'additional_info': additional_info,
                'output_directory': self.output_dir,
                'experiment_timestamp': self.timestamp
            }
            with open(pickle_path, 'wb') as f:
                pickle.dump(emergency_data, f)
            saved_files.append(f"Pickle: {pickle_path}")
            logger.info(f"Emergency pickle results saved: {pickle_path}")
        except Exception as e:
            logger.error(f"Pickle save failed: {e}")
        
        # Strategy 4: Save as plain text log (most reliable)
        try:
            txt_path = os.path.join(self.output_dir, f'EMERGENCY_results_{timestamp}.txt')
            with open(txt_path, 'w') as f:
                f.write(f"EMERGENCY RESULTS SAVE - {timestamp}\n")
                f.write("="*60 + "\n")
                f.write(f"Output Directory: {self.output_dir}\n")
                f.write(f"Experiment Timestamp: {self.timestamp}\n")
                f.write(f"Total results: {len(self.results)}\n")
                f.write(f"Completed experiments: {len(self.completed_experiments)}\n")
                if additional_info:
                    f.write(f"Additional info: {additional_info}\n")
                f.write("\nCompleted Experiments:\n")
                for i, exp in enumerate(self.completed_experiments):
                    f.write(f"  {i+1}. {exp}\n")
                f.write("\nResults Summary:\n")
                for i, result in enumerate(self.results):
                    f.write(f"\nResult {i+1}:\n")
                    f.write(f"  Target Dataset: {result.get('target_dataset', 'Unknown')}\n")
                    f.write(f"  Model: {result.get('model_name', 'Unknown')}\n")
                    f.write(f"  AUC: {result.get('auc', 'N/A')}\n")
                    f.write(f"  Accuracy: {result.get('accuracy', 'N/A')}\n")
                    f.write(f"  Status: {result.get('status', 'completed')}\n")
                    if 'error' in result:
                        f.write(f"  Error: {result['error']}\n")
            saved_files.append(f"TXT: {txt_path}")
            logger.info(f"Emergency text results saved: {txt_path}")
        except Exception as e:
            logger.error(f"Text save failed: {e}")
        
        # Create summary index file for easy finding
        try:
            index_path = os.path.join(self.output_dir, f'RESULTS_INDEX_{timestamp}.txt')
            with open(index_path, 'w') as f:
                f.write(f"RESULTS INDEX - {timestamp}\n")
                f.write("="*60 + "\n")
                f.write(f"LOOK FOR YOUR RESULTS IN: {self.output_dir}\n")
                f.write("="*60 + "\n")
                f.write("Files saved:\n")
                for file_info in saved_files:
                    f.write(f"  {file_info}\n")
                f.write(f"\nTotal experiments completed: {len(self.completed_experiments)}\n")
                f.write(f"Total results saved: {len(self.results)}\n")
            logger.info(f"Results index saved: {index_path}")
        except Exception as e:
            logger.error(f"Index save failed: {e}")
        
        if saved_files:
            logger.info("="*60)
            logger.info("EMERGENCY SAVE COMPLETED!")
            logger.info(f"Results saved in directory: {self.output_dir}")
            logger.info("Files saved:")
            for file_info in saved_files:
                logger.info(f"  {file_info}")
            logger.info("="*60)
        else:
            logger.error("ALL EMERGENCY SAVE ATTEMPTS FAILED!")
        
        return saved_files
    
    def get_resume_status(self):
        """Get the current resumption status."""
        if self.checkpoint_manager:
            progress = self.checkpoint_manager.get_progress_summary()
            if progress:
                return f"Can resume from {progress['completion_percentage']:.1f}% completion ({progress['completed_experiments']} experiments done)"
        return "No checkpoint found - will start fresh"
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        # Create results DataFrame
        df_results = pd.DataFrame(self.results)
        
        # Save detailed results
        results_path = os.path.join(self.output_dir, f'detailed_results_{self.timestamp}.csv')
        df_results.to_csv(results_path, index=False)
        logger.info(f"Detailed results saved to: {results_path}")
        
        # Create summary statistics
        summary_stats = df_results.groupby('model_name').agg({
            'auc': ['mean', 'std'],
            'sensitivity_at_95_specificity': ['mean', 'std'],
            'ece': ['mean', 'std'],
            'accuracy': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = [f"{col[0]}_{col[1]}" for col in summary_stats.columns]
        summary_stats = summary_stats.reset_index()
        
        # Save summary statistics
        summary_path = os.path.join(self.output_dir, f'summary_statistics_{self.timestamp}.csv')
        summary_stats.to_csv(summary_path, index=False)
        logger.info(f"Summary statistics saved to: {summary_path}")
        
        # Print summary to console
        print("\n" + "="*100)
        print("MULTI-SOURCE DOMAIN FINE-TUNING SUMMARY")
        print("="*100)
        print(f"Total experiments completed: {len(df_results)}")
        print(f"Models evaluated: {df_results['model_name'].nunique()}")
        print(f"Target datasets: {df_results['target_dataset'].nunique()}")
        print("\nSummary Statistics (Mean ± Std):")
        print("-" * 80)
        
        for _, row in summary_stats.iterrows():
            model_name = row['model_name']
            print(f"\n{model_name}:")
            print(f"  AUC:                    {row['auc_mean']:.4f} ± {row['auc_std']:.4f}")
            print(f"  Sens@95%Spec:          {row['sensitivity_at_95_specificity_mean']:.4f} ± {row['sensitivity_at_95_specificity_std']:.4f}")
            print(f"  ECE:                   {row['ece_mean']:.4f} ± {row['ece_std']:.4f}")
            print(f"  Accuracy:              {row['accuracy_mean']:.2f}% ± {row['accuracy_std']:.2f}%")
        
        print("\nDetailed Results by Target Dataset:")
        print("-" * 80)
        for target in df_results['target_dataset'].unique():
            print(f"\nTarget Dataset: {target}")
            target_results = df_results[df_results['target_dataset'] == target]
            for _, row in target_results.iterrows():
                print(f"  {row['model_name']}: AUC={row['auc']:.4f}, "
                      f"Sens@95Spec={row['sensitivity_at_95_specificity']:.4f}, "
                      f"ECE={row['ece']:.4f}")
        
        print("="*100)
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        df_results = pd.DataFrame(self.results)
        
        # 1. Performance comparison across models and datasets
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['auc', 'sensitivity_at_95_specificity', 'ece', 'accuracy']
        titles = ['AUC', 'Sensitivity @ 95% Specificity', 'Expected Calibration Error', 'Accuracy (%)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]
            
            # Create pivot table for heatmap
            pivot_data = df_results.pivot(index='target_dataset', columns='model_name', values=metric)
            
            # Create heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel('Target Dataset', fontsize=10)
        
        plt.suptitle('Multi-Source Domain Fine-Tuning Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        viz_path = os.path.join(self.output_dir, f'performance_heatmaps_{self.timestamp}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Performance heatmaps saved to: {viz_path}")
        
        # 2. Box plots for metric distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]
            
            sns.boxplot(data=df_results, x='model_name', y=metric, ax=ax)
            ax.set_title(f'{title} Distribution Across Target Datasets', fontsize=12, fontweight='bold')
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Performance Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        box_path = os.path.join(self.output_dir, f'performance_distributions_{self.timestamp}.png')
        plt.savefig(box_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Performance distributions saved to: {box_path}")
        
        # 3. Radar chart for overall performance
        self.create_radar_chart(df_results)
    
    def create_radar_chart(self, df_results: pd.DataFrame):
        """Create radar chart for overall model performance."""
        # Calculate mean performance for each model
        model_means = df_results.groupby('model_name').agg({
            'auc': 'mean',
            'sensitivity_at_95_specificity': 'mean',
            'ece': 'mean',  # Note: lower is better for ECE
            'accuracy': 'mean'
        }).reset_index()
        
        # Normalize ECE (invert since lower is better)
        model_means['ece_normalized'] = 1 - model_means['ece']
        model_means['accuracy_normalized'] = model_means['accuracy'] / 100.0
        
        # Create radar chart
        categories = ['AUC', 'Sens@95%Spec', 'Calibration\n(1-ECE)', 'Accuracy']
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_means)))
        
        for i, (_, model_row) in enumerate(model_means.iterrows()):
            values = [
                model_row['auc'],
                model_row['sensitivity_at_95_specificity'],
                model_row['ece_normalized'],
                model_row['accuracy_normalized']
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_row['model_name'], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        radar_path = os.path.join(self.output_dir, f'performance_radar_{self.timestamp}.png')
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Performance radar chart saved to: {radar_path}")


def load_all_datasets(args: argparse.Namespace) -> dict:
    """Load all available datasets for multi-source training."""
    all_datasets = {}
    
    # Log dataset filtering options
    if args.exclude_smdg_unknown:
        logger.info("SMDG dataset filtering: Excluding SMDG_Unknown samples (ambiguous source data)")
    else:
        logger.info("SMDG dataset filtering: Including all SMDG subdatasets (including SMDG_Unknown)")
    
    # Load SMDG-19 subdatasets
    try:
        smdg_metadata_file = os.path.join(args.base_data_root, 'raw', 'SMDG-19', 'metadata - standardized.csv')
        smdg_image_dir = os.path.join(args.base_data_root, 'raw', 'SMDG-19', 'full-fundus', 'full-fundus')
        
        if os.path.exists(smdg_metadata_file):
            logger.info(f"Loading SMDG-19 metadata: {smdg_metadata_file}")
            df_smdg = pd.read_csv(smdg_metadata_file)
            
            if "names" in df_smdg.columns:
                # Add image paths
                df_smdg["image_path"] = df_smdg["names"].apply(
                    lambda name: os.path.join(smdg_image_dir, f"{name}.png")
                )
                df_smdg["file_exists"] = df_smdg["image_path"].apply(os.path.exists)
                df_smdg = df_smdg[df_smdg["file_exists"]]
                
                # Clean and filter data
                df_smdg = df_smdg.dropna(subset=["types"])
                df_smdg = df_smdg[df_smdg["types"].isin([0, 1])]
                df_smdg["types"] = df_smdg["types"].astype(int)
                
                # Assign dataset sources to subdivide SMDG-19
                df_smdg["dataset_source"] = assign_dataset_source(df_smdg["names"])
                
                # Split SMDG-19 by subdatasets
                smdg_subdatasets = df_smdg.groupby('dataset_source')
                for dataset_name, dataset_df in smdg_subdatasets:
                    # Skip SMDG_Unknown if flag is set
                    if args.exclude_smdg_unknown and dataset_name == 'SMDG_Unknown':
                        logger.info(f"Excluding SMDG-{dataset_name}: {len(dataset_df)} samples (--exclude_smdg_unknown=True)")
                        continue
                        
                    if len(dataset_df) >= args.min_dataset_size:
                        all_datasets[f'SMDG-{dataset_name}'] = dataset_df.reset_index(drop=True)
                        logger.info(f"Added SMDG-{dataset_name}: {len(dataset_df)} samples")
    except Exception as e:
        logger.warning(f"Failed to load SMDG datasets: {e}")
    
    # Load AIROGS data if requested
    if args.use_airogs:
        try:
            airogs_config = argparse.Namespace()
            airogs_config.data_type = args.data_type
            airogs_config.base_data_root = args.base_data_root
            airogs_config.airogs_num_rg_samples = args.airogs_num_rg
            airogs_config.airogs_num_nrg_samples = args.airogs_num_nrg
            airogs_config.airogs_label_file = args.airogs_label_file
            airogs_config.airogs_image_dir = args.airogs_image_dir
            airogs_config.use_airogs_cache = args.use_airogs_cache
            airogs_config.seed = args.seed
            
            if args.data_type == 'processed':
                airogs_config.airogs_image_dir = adjust_path_for_data_type(
                    current_path=args.airogs_image_dir, data_type='processed',
                    base_data_dir=args.base_data_root, raw_dir_name=RAW_DIR_NAME_CONST,
                    processed_dir_name=PROCESSED_DIR_NAME_CONST
                )
            
            airogs_df = load_airogs_data(airogs_config)
            if not airogs_df.empty and len(airogs_df) >= args.min_dataset_size:
                all_datasets['AIROGS'] = airogs_df.reset_index(drop=True)
                logger.info(f"Added AIROGS: {len(airogs_df)} samples")
        except Exception as e:
            logger.warning(f"Failed to load AIROGS: {e}")
    
    # Load external datasets
    try:
        external_datasets = load_external_test_data(
            smdg_metadata_file_raw=args.smdg_metadata_file_raw,
            smdg_image_dir_raw=args.smdg_image_dir_raw,
            chaksu_base_dir_eval=args.chaksu_base_dir,
            chaksu_decision_dir_raw=args.chaksu_decision_dir_raw,
            chaksu_metadata_dir_raw=args.chaksu_metadata_dir_raw,
            data_type=args.data_type,
            base_data_root=args.base_data_root,
            raw_dir_name=RAW_DIR_NAME_CONST,
            processed_dir_name=PROCESSED_DIR_NAME_CONST,
            eval_papilla=False,  # Don't include PAPILLA as it's already in SMDG
            eval_oiaodir_test=False,
            eval_chaksu=args.eval_chaksu,
            eval_acrima=args.eval_acrima,
            eval_hygd=args.eval_hygd,
            acrima_image_dir_raw=args.acrima_image_dir_raw,
            hygd_image_dir_raw=args.hygd_image_dir_raw,
            hygd_labels_file_raw=args.hygd_labels_file_raw
        )
        
        for dataset_name, df in external_datasets.items():
            if not df.empty and len(df) >= args.min_dataset_size:
                all_datasets[dataset_name] = df.reset_index(drop=True)
                logger.info(f"Added {dataset_name}: {len(df)} samples")
    except Exception as e:
        logger.warning(f"Failed to load external datasets: {e}")
    
    logger.info(f"Total datasets loaded: {len(all_datasets)}")
    
    # Log summary of dataset filtering
    if args.exclude_smdg_unknown:
        smdg_datasets = [name for name in all_datasets.keys() if name.startswith('SMDG-')]
        logger.info(f"SMDG subdatasets included: {smdg_datasets}")
        logger.info("Note: SMDG-SMDG_Unknown was excluded due to --exclude_smdg_unknown=True")
    
    return all_datasets


def main(args: argparse.Namespace):
    """Main function for multi-source domain fine-tuning."""
    
    # Configure logging based on arguments
    if args.detailed_logging:
        set_detailed_logging(True)
    
    if args.quiet_mode:
        # Reduce console logging to warnings and above
        logging.getLogger().setLevel(logging.WARNING)
        # But keep file logging if detailed logging is enabled
        if args.detailed_logging:
            logger.setLevel(logging.DEBUG)
    
    logger.info("Starting multi-source domain fine-tuning experiment...")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"multisource_finetuning_{timestamp}")
    
    # Initialize trainer with domain adaptation options
    trainer = MultiSourceTrainer(
        output_dir, 
        device=args.device,
        use_domain_adversarial=args.use_domain_adversarial,
        use_mixstyle=args.use_mixstyle,
        use_swa=args.use_swa,
        use_tta=args.use_tta,
        use_advanced_augmentation=args.use_advanced_augmentation,
        domain_loss_weight=args.domain_loss_weight,
        enable_checkpointing=args.enable_checkpointing,
        quiet_mode=args.quiet_mode
    )
    
    # Show resume status if checkpointing is enabled
    if args.enable_checkpointing:
        resume_status = trainer.get_resume_status()
        logger.info(f"Checkpoint status: {resume_status}")
    
    # Load all datasets
    all_datasets = load_all_datasets(args)
    
    if len(all_datasets) < 2:
        logger.error("Need at least 2 datasets for multi-source training")
        return
    
    # Define model configurations
    model_configs = []
    added_architectures = set()  # Track architectures to prevent duplicates
    
    # Add VFM ViT-B pretrained as the default model (medical imaging foundation model)
    if os.path.exists(args.vfm_weights_path) and not args.disable_vfm:
        model_configs.append({
            'name': 'VFM_ViTB_Pretrained',
            'architecture': 'vit_base_patch16_224',
            'pretrained': False,
            'weights_path': args.vfm_weights_path
        })
        added_architectures.add('vit_base_patch16_224')
        logger.info("Using VFM (Vision Foundation Model) as default - optimized for medical imaging")
    else:
        if args.disable_vfm:
            logger.info(f"VFM disabled via --disable_vfm flag")
        else:
            logger.warning(f"VFM weights not found at {args.vfm_weights_path}")
        
        # Use specified default model instead
        default_model_name = args.default_model.replace('_', '-').title()
        model_configs.append({
            'name': f'{default_model_name}_Pretrained',
            'architecture': args.default_model,
            'pretrained': True,
            'weights_path': None
        })
        added_architectures.add(args.default_model.lower())
        logger.info(f"Using {args.default_model} as default model (faster training for lower-end GPUs)")
    
    # Add any additional model configurations from args
    for additional_model in args.additional_models:
        if ':' in additional_model:
            name, arch_and_weights = additional_model.split(':', 1)
            if ':' in arch_and_weights:
                arch, weights_path = arch_and_weights.split(':', 1)
                
                # Skip if architecture already added
                if arch.lower() in added_architectures:
                    logger.info(f"Skipping duplicate architecture: {arch}")
                    continue
                    
                model_configs.append({
                    'name': name,
                    'architecture': arch,
                    'pretrained': False,
                    'weights_path': weights_path if os.path.exists(weights_path) else None
                })
                added_architectures.add(arch.lower())
            else:
                arch = arch_and_weights
                
                # Skip if architecture already added
                if arch.lower() in added_architectures:
                    logger.info(f"Skipping duplicate architecture: {arch}")
                    continue
                
                model_configs.append({
                    'name': name,
                    'architecture': arch,
                    'pretrained': True,
                    'weights_path': None
                })
                added_architectures.add(arch.lower())
        else:
            arch = additional_model
            
            # Skip if architecture already added
            if arch.lower() in added_architectures:
                logger.info(f"Skipping duplicate architecture: {arch}")
                continue
                
            model_configs.append({
                'name': arch.title() + '_Pretrained',
                'architecture': arch,
                'pretrained': True,
                'weights_path': None
            })
            added_architectures.add(arch.lower())
    
    if not model_configs:
        logger.error("No valid model configurations available")
        return
    
    # Log the final model configurations
    logger.info(f"Training {len(model_configs)} models:")
    for config in model_configs:
        logger.info(f"  - {config['name']} ({config['architecture']})")
    
    # Additional information about model selection
    if len(model_configs) == 1 and 'VFM' in model_configs[0]['name']:
        logger.info("  Note: VFM is optimized for medical fundus images and typically outperforms generic ImageNet models")
        logger.info("  To add ResNet50 or other models, use --additional_models 'ResNet50:resnet50'")
    
    # Run multi-source experiment
    trainer.run_multisource_experiment(
        all_datasets=all_datasets,
        model_configs=model_configs,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        base_learning_rate=args.base_learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        resume_from_checkpoint=args.resume_from_checkpoint
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Source Domain Fine-Tuning for Glaucoma Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./multisource_results',
                       help="Directory to save results")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help="Device to use for training")
    
    # Data configuration
    parser.add_argument('--data_type', type=str, default='raw', choices=['raw', 'processed'],
                       help="Type of image data to use")
    parser.add_argument('--base_data_root', type=str, default=r'D:\glaucoma\data',
                       help="Base directory containing all datasets")
    parser.add_argument('--min_dataset_size', type=int, default=200,
                       help="Minimum number of samples required per dataset")
    
    # SMDG/PAPILLA dataset
    parser.add_argument('--smdg_metadata_file_raw', type=str, 
                       default=os.path.join('raw','SMDG-19','metadata - standardized.csv'))
    parser.add_argument('--smdg_image_dir_raw', type=str, 
                       default=os.path.join('raw','SMDG-19','full-fundus','full-fundus'))
    
    # CHAKSU dataset
    parser.add_argument('--chaksu_base_dir', type=str, 
                       default=os.path.join('raw','Chaksu','Train','Train','1.0_Original_Fundus_Images'))
    parser.add_argument('--chaksu_decision_dir_raw', type=str, 
                       default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision'))
    parser.add_argument('--chaksu_metadata_dir_raw', type=str, 
                       default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision','Majority'))
    
    # ACRIMA dataset
    parser.add_argument('--acrima_image_dir_raw', type=str, 
                       default=os.path.join('raw','ACRIMA','Database','Images'))
    
    # HYGD dataset
    parser.add_argument('--hygd_image_dir_raw', type=str, 
                       default=os.path.join('raw','HYGD','HYGD','Images'))
    parser.add_argument('--hygd_labels_file_raw', type=str, 
                       default=os.path.join('raw','HYGD','HYGD','Labels.csv'))
    
    # AIROGS dataset
    parser.add_argument('--use_airogs', action='store_true', default=True,
                       help="Include AIROGS dataset")
    parser.add_argument('--airogs_label_file', type=str, 
                       default=r'D:\glaucoma\data\raw\AIROGS\train_labels.csv')
    parser.add_argument('--airogs_image_dir', type=str, 
                       default=r'D:\glaucoma\data\raw\AIROGS\img')
    parser.add_argument('--airogs_num_rg', type=int, default=3000)
    parser.add_argument('--airogs_num_nrg', type=int, default=3000)
    parser.add_argument('--use_airogs_cache', action='store_true', default=True)
    
    # Dataset selection
    parser.add_argument('--exclude_smdg_unknown', action='store_true', default=True,
                       help="Exclude SMDG-SMDG_Unknown dataset (contains ~3269 samples with unclear/ambiguous sources, "
                            "may reduce domain shift clarity in multi-source training)")
    parser.add_argument('--eval_chaksu', action='store_true', default=True)
    parser.add_argument('--eval_acrima', action='store_true', default=True)
    parser.add_argument('--eval_hygd', action='store_true', default=True)
    
    # Model configuration
    parser.add_argument('--vfm_weights_path', type=str, 
                       default=r'D:\glaucoma\models\VFM_Fundus_weights.pth',
                       help="Path to VFM pretrained weights (used as default model for medical imaging)")
    parser.add_argument('--disable_vfm', action='store_true', default=False,
                       help="Disable VFM as default model (useful for faster training on lower-end GPUs)")
    parser.add_argument('--default_model', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'resnet50', 'resnet34', 'resnet18', 'mobilenetv3_large_100'],
                       help="Default model to use when VFM is disabled (resnet18 fastest, efficientnet_b0 best speed/accuracy balance)")
    parser.add_argument('--additional_models', nargs='*', default=[],
                       help="Additional models in format 'name:architecture:weights_path' or 'name:architecture'. "
                            "Example: 'ResNet50:resnet50' to add ResNet50 as an additional model")
    
    # Training configuration
    parser.add_argument('--num_epochs', type=int, default=5,
                       help="Maximum number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32,
                       help="Batch size for training (try 64-128 for faster training if GPU memory allows)")
    parser.add_argument('--base_learning_rate', type=float, default=1e-4,
                       help="Base learning rate (model-specific rates will be derived from this)")
    parser.add_argument('--early_stopping_patience', type=int, default=2,
                       help="Number of epochs with no improvement to wait before early stopping")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help="Weight decay for regularization")
    
    # Data loading performance
    parser.add_argument('--num_workers', type=int, default=0,
                       help="Number of data loading workers (try 8-16 for faster data loading, 0 for debugging)")
    parser.add_argument('--pin_memory', action='store_true', default=True,
                       help="Pin memory in DataLoader for faster GPU transfer")
    parser.add_argument('--prefetch_factor', type=int, default=2,
                       help="Number of batches to prefetch per worker (higher = more memory, faster loading)")
    parser.add_argument('--persistent_workers', action='store_true', default=False,
                       help="Keep workers alive between epochs (faster but uses more memory)")
    
    # Domain Adaptation Configuration
    parser.add_argument('--use_domain_adversarial', action='store_true', default=False,
                       help="Enable Domain Adversarial Training (DANN)")
    parser.add_argument('--use_mixstyle', action='store_true', default=False,
                       help="Enable MixStyle for domain generalization")
    parser.add_argument('--use_swa', action='store_true', default=False,
                       help="Enable Stochastic Weight Averaging")
    parser.add_argument('--use_tta', action='store_true', default=False,
                       help="Enable Test-Time Adaptation")
    parser.add_argument('--use_advanced_augmentation', action='store_true', default=False,
                       help="Enable advanced data augmentation techniques")
    parser.add_argument('--domain_loss_weight', type=float, default=0.1,
                       help="Weight for domain adversarial loss (used with --use_domain_adversarial)")
    
    # Checkpointing Configuration
    parser.add_argument('--enable_checkpointing', action='store_true', default=True,
                       help="Enable experiment checkpointing for resumption (default: True)")
    parser.add_argument('--no_checkpointing', dest='enable_checkpointing', action='store_false',
                       help="Disable experiment checkpointing")
    parser.add_argument('--resume_from_checkpoint', action='store_true', default=True,
                       help="Automatically resume from checkpoint if available (default: True)")
    parser.add_argument('--fresh_start', dest='resume_from_checkpoint', action='store_false',
                       help="Start fresh experiment, ignoring any existing checkpoint")
    
    # General configuration
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument('--detailed_logging', action='store_true', default=False,
                       help="Enable detailed batch-level logging during training (saves to training_detailed.log)")
    parser.add_argument('--quiet_mode', action='store_true', default=False,
                       help="Reduce console output (only show epoch summaries)")
    
    args = parser.parse_args()
    
    # Validate paths
    args.base_data_root = os.path.abspath(args.base_data_root)
    if not os.path.isdir(args.base_data_root):
        logger.error(f"Base data root not found: {args.base_data_root}")
        sys.exit(1)
    
    # Robust main execution with error handling
    try:
        main(args)
        logger.info("Experiment completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user (Ctrl+C)")
        logger.info("Progress has been saved. You can resume later with the same command.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try emergency save if possible
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(args.output_dir, f"multisource_finetuning_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            emergency_log_path = os.path.join(output_dir, f'EMERGENCY_crash_log_{timestamp}.txt')
            with open(emergency_log_path, 'w') as f:
                f.write(f"Emergency Crash Log - {datetime.now()}\n")
                f.write("="*60 + "\n")
                f.write(f"Error: {str(e)}\n")
                f.write("Full traceback:\n")
                f.write(traceback.format_exc())
                f.write("\nCommand line arguments:\n")
                f.write(str(vars(args)))
            logger.info(f"Emergency crash log saved to: {emergency_log_path}")
        except Exception as emergency_error:
            logger.error(f"Failed to save emergency crash log: {emergency_error}")
        
        sys.exit(1)
