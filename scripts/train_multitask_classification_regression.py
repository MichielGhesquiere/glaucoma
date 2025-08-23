#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Task Neural Network Training Script for Glaucoma Classification and vCDR Regression.

This script trains neural networks for glaucoma detection with two modes:
1. Single-task: Binary glaucoma classification only
2. Multi-task: Binary classification + vCDR regression

Key features:
- Single-task vs Multi-task comparison mode (--compare_tasks flag)
- Handles missing labels gracefully (binary-only, vCDR-only, or both)
- Multi-source domain adaptation with leave-one-dataset-out evaluation
- Support for ResNet18/50 and Vision Foundation Model (VFM) backbones
- Adaptive loss weighting for missing labels
- Comprehensive evaluation and visualization
- High-performance training optimizations

Research Question: Does adding vCDR regression improve binary classification performance?

Usage Examples:

1. Compare single-task vs multi-task with ResNet18:
python train_multitask_classification_regression.py --vcdr_csv path/to/vcdr_labels.csv --compare_tasks

2. Compare with VFM backbone:
python train_multitask_classification_regression.py --vcdr_csv path/to/vcdr_labels.csv --compare_tasks --backbone vfm

3. Train only multi-task model (original functionality):
python train_multitask_classification_regression.py --vcdr_csv path/to/vcdr_labels.csv

4. Force retrain comparison models:
python train_multitask_classification_regression.py --vcdr_csv path/to/vcdr_labels.csv --compare_tasks --force_retrain

5. Re-evaluate existing models with updated metrics (e.g., ECE):
python train_multitask_classification_regression.py --vcdr_csv path/to/vcdr_labels.csv --compare_tasks --reevaluate_only

Comparison Output:
- Individual results for single-task and multi-task models
- Side-by-side performance metrics (sensitivity, specificity, AUC, ECE for medical screening)
- Improvement statistics (absolute and percentage)
- Summary of how many datasets showed improvement

The comparison mode will show whether multi-task learning (adding vCDR regression) 
helps the primary binary classification task, which is a common research question 
in multi-task learning.

Re-evaluation Mode:
The --reevaluate_only flag allows you to re-evaluate existing trained models with 
updated metrics (such as ECE) without retraining. This is useful when you've added 
new evaluation metrics and want to update existing results without spending time 
on retraining models that are already converged.
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torchvision.transforms as transforms
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast

# Enable Flash Attention optimization
try:
    torch.backends.cuda.enable_flash_sdp(True)
    flash_enabled = torch.backends.cuda.flash_sdp_enabled() if hasattr(torch.backends.cuda, 'flash_sdp_enabled') else False
    print(f"Flash Attention enabled: {flash_enabled}")
except Exception as e:
    print(f"Flash Attention configuration failed: {e}")
    print("Continuing without Flash Attention optimization")

# Try to import timm for Vision Transformer models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. VFM backbone will not work. Install with: pip install timm")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins to use for calibration
        
    Returns:
        ece: Expected Calibration Error
    """
    import numpy as np
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    total_samples = len(y_true)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence for this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Add weighted contribution to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calculate_brier_score(y_true, y_prob):
    """
    Calculate Brier Score for binary classification.
    
    The Brier score measures the mean squared difference between predicted probabilities 
    and actual outcomes. Lower values indicate better calibration.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        
    Returns:
        brier_score: Brier Score (lower is better)
    """
    import numpy as np
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Calculate Brier score
    brier_score = np.mean((y_prob - y_true) ** 2)
    
    return brier_score

# Ensure custom modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.utils.helpers import set_seed, NpEncoder
except ImportError as e:
    print(f"Warning: Could not import custom helpers: {e}")
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            if hasattr(obj, 'item'):  # Handle numpy scalars
                try:
                    return obj.item()
                except:
                    pass
            # Skip torch tensors and other non-serializable objects
            if hasattr(obj, '__module__') and 'torch' in str(obj.__module__):
                return f"<torch object: {type(obj).__name__}>"
            return super(NpEncoder, self).default(obj)

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)


class MultiTaskDataset(Dataset):
    """Dataset class for multi-task learning with missing labels."""
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 transform: transforms.Compose = None,
                 vcdr_scaler: StandardScaler = None):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with columns ['image_path', 'binary_label', 'vcdr']
            transform: Image transformations
            vcdr_scaler: Fitted scaler for vCDR normalization
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.vcdr_scaler = vcdr_scaler
        
        # Handle missing labels
        self.has_binary = ~pd.isna(self.df['binary_label'])
        self.has_vcdr = ~pd.isna(self.df['vcdr'])
        
        logger.info(f"Dataset initialized: {len(self.df)} samples")
        logger.info(f"Binary labels available: {self.has_binary.sum()}")
        logger.info(f"vCDR labels available: {self.has_vcdr.sum()}")
        logger.info(f"Both labels available: {(self.has_binary & self.has_vcdr).sum()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = row['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Return a black image if loading fails
            image = torch.zeros(3, 224, 224)
        
        # Get labels and masks
        binary_label = row['binary_label'] if pd.notna(row['binary_label']) else -1
        vcdr_label = row['vcdr'] if pd.notna(row['vcdr']) else -1
        
        # Additional validation for binary label
        if binary_label != -1:
            if not isinstance(binary_label, (int, float)) or binary_label not in [0, 1]:
                logger.warning(f"Invalid binary label {binary_label} in {row['image_path']}, setting to -1")
                binary_label = -1
        
        # Normalize vCDR if scaler is provided and label is valid
        if self.vcdr_scaler is not None and vcdr_label != -1:
            vcdr_label = self.vcdr_scaler.transform([[vcdr_label]])[0, 0]
        
        # Create masks for valid labels
        binary_mask = 1.0 if binary_label != -1 else 0.0
        vcdr_mask = 1.0 if vcdr_label != -1 else 0.0
        
        return {
            'image': image,
            'binary_label': torch.tensor(binary_label, dtype=torch.float32),
            'vcdr_label': torch.tensor(vcdr_label, dtype=torch.float32),
            'binary_mask': torch.tensor(binary_mask, dtype=torch.float32),
            'vcdr_mask': torch.tensor(vcdr_mask, dtype=torch.float32),
            'dataset': row['dataset'],
            'image_path': img_path
        }


class MultiTaskModel(nn.Module):
    """Multi-task model for glaucoma classification and vCDR regression."""
    
    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True, dropout: float = 0.5, vfm_weights_path: str = None):
        super(MultiTaskModel, self).__init__()
        
        self.backbone_type = backbone
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'vfm':
            if not TIMM_AVAILABLE:
                raise ImportError("timm is required for VFM backbone. Install with: pip install timm")
            
            # Load VFM (Vision Foundation Model) - ViT-Base with 768 dimensions
            # Based on weight inspection: 768 dim, patch size 16, image size 224
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
            feature_dim = 768  # VFM uses 768-dimensional features
            
            # Load VFM weights if provided
            if vfm_weights_path:
                success = load_vfm_weights(self.backbone, vfm_weights_path)
                if success:
                    logger.info("VFM weights loaded successfully")
                else:
                    logger.warning("Failed to load VFM weights, using random initialization")
            else:
                logger.info("No VFM weights provided, using randomly initialized model")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Shared feature extractor
        self.feature_dim = feature_dim
        
        # Task-specific heads
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # Binary classification
        )
        
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # vCDR regression
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for task-specific heads."""
        for m in [self.classification_head, self.regression_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Task-specific predictions
        classification_logits = self.classification_head(features)
        regression_output = self.regression_head(features)
        
        return {
            'classification_logits': classification_logits.squeeze(),
            'regression_output': regression_output.squeeze(),
            'features': features
        }


class SingleTaskModel(nn.Module):
    """Single-task model for glaucoma classification only."""
    
    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True, dropout: float = 0.5, vfm_weights_path: str = None):
        super(SingleTaskModel, self).__init__()
        
        self.backbone_type = backbone
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'vfm':
            if not TIMM_AVAILABLE:
                raise ImportError("timm is required for VFM backbone. Install with: pip install timm")
            
            # Load VFM (Vision Foundation Model) - ViT-Base with 768 dimensions
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
            feature_dim = 768  # VFM uses 768-dimensional features
            
            # Load VFM weights if provided
            if vfm_weights_path:
                success = load_vfm_weights(self.backbone, vfm_weights_path)
                if success:
                    logger.info("VFM weights loaded successfully")
                else:
                    logger.warning("Failed to load VFM weights, using random initialization")
            else:
                logger.info("No VFM weights provided, using randomly initialized model")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Shared feature extractor
        self.feature_dim = feature_dim
        
        # Classification head only
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # Binary classification
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for classification head."""
        for m in self.classification_head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classification prediction only
        classification_logits = self.classification_head(features)
        
        return {
            'classification_logits': classification_logits.squeeze(),
            'features': features
        }


class MultiTaskLoss(nn.Module):
    """Multi-task loss with adaptive weighting for missing labels."""
    
    def __init__(self, 
                 classification_weight: float = 1.0,
                 regression_weight: float = 1.0,
                 adaptive_weighting: bool = True):
        super(MultiTaskLoss, self).__init__()
        
        self.classification_weight = nn.Parameter(torch.tensor(classification_weight))
        self.regression_weight = nn.Parameter(torch.tensor(regression_weight))
        self.adaptive_weighting = adaptive_weighting
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        Calculate multi-task loss.
        
        Args:
            predictions: Dict with 'classification_logits' and 'regression_output'
            targets: Dict with labels and masks
        """
        classification_logits = predictions['classification_logits']
        regression_output = predictions['regression_output']
        
        binary_labels = targets['binary_label']
        vcdr_labels = targets['vcdr_label']
        binary_masks = targets['binary_mask']
        vcdr_masks = targets['vcdr_mask']
        
        # Classification loss (only for samples with binary labels)
        classification_losses = self.bce_loss(classification_logits, binary_labels)
        masked_classification_loss = (classification_losses * binary_masks).sum()
        n_classification = binary_masks.sum()
        
        # Regression loss (only for samples with vCDR labels)
        regression_losses = self.mse_loss(regression_output, vcdr_labels)
        masked_regression_loss = (regression_losses * vcdr_masks).sum()
        n_regression = vcdr_masks.sum()
        
        # Normalize by number of valid samples
        if n_classification > 0:
            classification_loss = masked_classification_loss / n_classification
        else:
            classification_loss = torch.tensor(0.0, device=classification_logits.device)
        
        if n_regression > 0:
            regression_loss = masked_regression_loss / n_regression
        else:
            regression_loss = torch.tensor(0.0, device=regression_output.device)
        
        # Adaptive weighting
        if self.adaptive_weighting:
            w1 = torch.exp(-self.classification_weight)
            w2 = torch.exp(-self.regression_weight)
            total_loss = (w1 * classification_loss + self.classification_weight + 
                         w2 * regression_loss + self.regression_weight)
        else:
            total_loss = (self.classification_weight * classification_loss + 
                         self.regression_weight * regression_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'regression_loss': regression_loss,
            'n_classification': n_classification,
            'n_regression': n_regression
        }


class SingleTaskLoss(nn.Module):
    """Single-task loss for classification only."""
    
    def __init__(self):
        super(SingleTaskLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        Calculate classification loss only.
        
        Args:
            predictions: Dict with 'classification_logits'
            targets: Dict with labels and masks
        """
        classification_logits = predictions['classification_logits']
        binary_labels = targets['binary_label']
        binary_masks = targets['binary_mask']
        
        # Classification loss (only for samples with binary labels)
        classification_losses = self.bce_loss(classification_logits, binary_labels)
        masked_classification_loss = (classification_losses * binary_masks).sum()
        n_classification = binary_masks.sum()
        
        # Normalize by number of valid samples
        if n_classification > 0:
            classification_loss = masked_classification_loss / n_classification
        else:
            classification_loss = torch.tensor(0.0, device=classification_logits.device)
        
        return {
            'total_loss': classification_loss,
            'classification_loss': classification_loss,
            'n_classification': n_classification
        }


class TaskTrainer:
    """Trainer class for both single-task and multi-task learning."""
    
    def __init__(self, 
                 model: Union[SingleTaskModel, MultiTaskModel],
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 output_dir: str,
                 vcdr_scaler: StandardScaler = None,
                 backbone: str = 'resnet18',
                 use_mixed_precision: bool = True,
                 channels_last: bool = True,
                 vfm_backbone_lr: float = 1e-6,
                 vfm_head_lr: float = 5e-4,
                 task_mode: str = 'multitask'):
        
        self.model = model  # Model is already on device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.vcdr_scaler = vcdr_scaler
        self.backbone = backbone
        self.use_mixed_precision = use_mixed_precision and device.type == 'cuda'
        self.channels_last = channels_last and device.type == 'cuda'
        self.task_mode = task_mode
        
        # Training components - choose loss based on task mode
        if task_mode == 'singletask':
            self.criterion = SingleTaskLoss()
        else:
            self.criterion = MultiTaskLoss(adaptive_weighting=True)
        
        # Set up differential learning rates
        if hasattr(model, 'backbone_type') and model.backbone_type == 'vfm':
            # VFM: Use very low learning rate for backbone, higher for task heads
            backbone_lr = vfm_backbone_lr  # Use parameter instead of hardcoded value
            head_lr = vfm_head_lr          # Use parameter instead of hardcoded value
            
            # Create parameter groups with different learning rates
            backbone_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            
            param_groups = [
                {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': 1e-4},
                {'params': head_params, 'lr': head_lr, 'weight_decay': 1e-4}
            ]
            
            logger.info(f"Using differential learning rates for VFM:")
            logger.info(f"  Backbone (encoder): {backbone_lr}")
            logger.info(f"  Task heads: {head_lr}")
            logger.info(f"  Backbone parameters: {len(backbone_params)}")
            logger.info(f"  Head parameters: {len(head_params)}")
            
        else:
            # ResNet models: use single learning rate
            base_lr = 1e-4
            param_groups = [{'params': self.model.parameters(), 'lr': base_lr, 'weight_decay': 1e-4}]
            logger.info(f"Using standard learning rate for ResNet: {base_lr}")
        
        self.optimizer = optim.AdamW(param_groups)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=3, verbose=True, min_lr=1e-7
        )
        
        # Add warmup scheduler for VFM
        if hasattr(model, 'backbone_type') and model.backbone_type == 'vfm':
            self.warmup_epochs = 3
            self.current_epoch = 0
            logger.info(f"Using {self.warmup_epochs} warmup epochs for VFM training")
        else:
            self.warmup_epochs = 0
            self.current_epoch = 0
        
        # Mixed precision training
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled with GradScaler")
        else:
            self.scaler = None
            logger.info("Mixed precision training disabled")
        
        # Training history - adapt based on task mode
        if task_mode == 'singletask':
            self.history = {
                'train_loss': [], 'train_classification_loss': [],
                'val_loss': [], 'val_classification_loss': [],
                'val_accuracy': [], 'val_auc': []
            }
        else:
            self.history = {
                'train_loss': [], 'train_classification_loss': [], 'train_regression_loss': [],
                'val_loss': [], 'val_classification_loss': [], 'val_regression_loss': [],
                'val_accuracy': [], 'val_auc': [], 'val_mse': [], 'val_mae': [], 'val_r2': []
            }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def train_epoch(self) -> Dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_classification_loss = 0
        total_regression_loss = 0
        n_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device and apply channels_last format for better performance
            images = batch['image'].to(self.device)
            if self.channels_last:
                images = images.to(memory_format=torch.channels_last)
            
            targets = {
                'binary_label': batch['binary_label'].to(self.device),
                'vcdr_label': batch['vcdr_label'].to(self.device),
                'binary_mask': batch['binary_mask'].to(self.device),
                'vcdr_mask': batch['vcdr_mask'].to(self.device)
            }
            
            # Forward pass with optional mixed precision
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision and self.scaler is not None:
                with autocast():
                    predictions = self.model(images)
                    loss_dict = self.criterion(predictions, targets)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss_dict['total_loss']).backward()
                
                # Gradient clipping with scaled gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                predictions = self.model(images)
                loss_dict = self.criterion(predictions, targets)
                
                # Backward pass
                loss_dict['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Track losses
            total_loss += loss_dict['total_loss'].item()
            total_classification_loss += loss_dict['classification_loss'].item()
            
            # Only track regression loss for multitask mode
            if self.task_mode == 'multitask' and 'regression_loss' in loss_dict:
                total_regression_loss += loss_dict['regression_loss'].item()
            
            n_batches += 1
            
            # Update progress bar - adapt based on task mode
            if self.task_mode == 'singletask':
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total_loss'].item():.4f}",
                    'cls_loss': f"{loss_dict['classification_loss'].item():.4f}"
                })
            else:
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total_loss'].item():.4f}",
                    'cls_loss': f"{loss_dict['classification_loss'].item():.4f}",
                    'reg_loss': f"{loss_dict['regression_loss'].item():.4f}"
                })
        
        # Return metrics based on task mode
        if self.task_mode == 'singletask':
            return {
                'loss': total_loss / n_batches,
                'classification_loss': total_classification_loss / n_batches
            }
        else:
            return {
                'loss': total_loss / n_batches,
                'classification_loss': total_classification_loss / n_batches,
                'regression_loss': total_regression_loss / n_batches
            }
    
    def validate(self, loader: DataLoader = None) -> Dict:
        """Validate the model."""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        
        total_loss = 0
        total_classification_loss = 0
        total_regression_loss = 0
        n_batches = 0
        
        # For metrics calculation
        binary_predictions = []
        binary_targets = []
        regression_predictions = []
        regression_targets = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                # Move to device and apply channels_last format
                images = batch['image'].to(self.device)
                if self.channels_last:
                    images = images.to(memory_format=torch.channels_last)
                    
                targets = {
                    'binary_label': batch['binary_label'].to(self.device),
                    'vcdr_label': batch['vcdr_label'].to(self.device),
                    'binary_mask': batch['binary_mask'].to(self.device),
                    'vcdr_mask': batch['vcdr_mask'].to(self.device)
                }
                
                # Forward pass
                predictions = self.model(images)
                loss_dict = self.criterion(predictions, targets)
                
                # Track losses
                total_loss += loss_dict['total_loss'].item()
                total_classification_loss += loss_dict['classification_loss'].item()
                
                # Only track regression loss for multitask mode
                if self.task_mode == 'multitask' and 'regression_loss' in loss_dict:
                    total_regression_loss += loss_dict['regression_loss'].item()
                
                n_batches += 1
                
                # Collect predictions for metrics
                binary_mask = batch['binary_mask'].bool()
                
                if binary_mask.any():
                    binary_preds = torch.sigmoid(predictions['classification_logits'][binary_mask])
                    binary_targs = batch['binary_label'][binary_mask]
                    binary_predictions.extend(binary_preds.cpu().numpy())
                    binary_targets.extend(binary_targs.cpu().numpy())
                
                # Only collect regression data for multitask mode
                if self.task_mode == 'multitask' and 'regression_output' in predictions:
                    vcdr_mask = batch['vcdr_mask'].bool()
                    if vcdr_mask.any():
                        reg_preds = predictions['regression_output'][vcdr_mask]
                        reg_targs = batch['vcdr_label'][vcdr_mask]
                        regression_predictions.extend(reg_preds.cpu().numpy())
                        regression_targets.extend(reg_targs.cpu().numpy())
        
        # Calculate metrics - adapt based on task mode
        if self.task_mode == 'singletask':
            metrics = {
                'loss': float(total_loss / n_batches),
                'classification_loss': float(total_classification_loss / n_batches)
            }
        else:
            metrics = {
                'loss': float(total_loss / n_batches),
                'classification_loss': float(total_classification_loss / n_batches),
                'regression_loss': float(total_regression_loss / n_batches)
            }
        
        # Binary classification metrics
        if binary_predictions:
            binary_predictions = np.array(binary_predictions)
            binary_targets = np.array(binary_targets)
            binary_pred_labels = (binary_predictions > 0.5).astype(int)
            
            # Filter out any invalid targets (non-finite or outside [0,1])
            valid_mask = np.isfinite(binary_targets) & np.isin(binary_targets, [0, 1])
            
            if valid_mask.sum() > 0:
                binary_targets_filtered = binary_targets[valid_mask]
                binary_predictions_filtered = binary_predictions[valid_mask]
                binary_pred_labels_filtered = binary_pred_labels[valid_mask]
                
                # Check if we have binary classification (only 0s and 1s)
                unique_targets = np.unique(binary_targets_filtered)
                unique_preds = np.unique(binary_pred_labels_filtered)
                
                logger.info(f"Filtered binary classification: {len(binary_targets_filtered)} valid samples")
                logger.info(f"Unique targets: {unique_targets}, Unique predictions: {unique_preds}")
                
                try:
                    # Always use weighted average for robustness
                    average_param = 'weighted' if len(unique_targets) > 1 else 'binary'
                    
                    # Calculate basic accuracy first
                    accuracy = accuracy_score(binary_targets_filtered, binary_pred_labels_filtered)
                    
                    # Try to calculate other metrics
                    precision = recall = f1 = auc = sensitivity = specificity = 0.0
                    
                    if len(unique_targets) > 1:  # Only calculate if we have both classes
                        try:
                            precision = precision_score(binary_targets_filtered, binary_pred_labels_filtered, 
                                                       average=average_param, zero_division=0)
                            recall = recall_score(binary_targets_filtered, binary_pred_labels_filtered, 
                                                 average=average_param, zero_division=0)
                            f1 = f1_score(binary_targets_filtered, binary_pred_labels_filtered, 
                                         average=average_param, zero_division=0)
                            
                            # Calculate sensitivity and specificity from confusion matrix
                            tn, fp, fn, tp = confusion_matrix(binary_targets_filtered, binary_pred_labels_filtered).ravel()
                            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall for positive class
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True negative rate
                            
                        except Exception as e:
                            logger.warning(f"Failed to calculate precision/recall/f1/sensitivity/specificity: {e}")
                        
                        try:
                            auc = roc_auc_score(binary_targets_filtered, binary_predictions_filtered)
                        except Exception as e:
                            logger.warning(f"Failed to calculate AUC: {e}")
                        
                        # Calculate Expected Calibration Error (ECE)
                        try:
                            ece = calculate_ece(binary_targets_filtered, binary_predictions_filtered)
                        except Exception as e:
                            logger.warning(f"Failed to calculate ECE: {e}")
                            ece = 0.0
                        
                        # Calculate Brier Score
                        try:
                            brier_score = calculate_brier_score(binary_targets_filtered, binary_predictions_filtered)
                        except Exception as e:
                            logger.warning(f"Failed to calculate Brier Score: {e}")
                            brier_score = 1.0  # Worst possible Brier score as fallback
                    else:
                        ece = 0.0
                        brier_score = 1.0
                    
                    metrics.update({
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'sensitivity': float(sensitivity),  # Same as recall for binary classification
                        'specificity': float(specificity),
                        'f1': float(f1),
                        'auc': float(auc),
                        'ece': float(ece),
                        'brier_score': float(brier_score)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate classification metrics: {e}")
                    # Fallback to basic metrics only
                    try:
                        accuracy = accuracy_score(binary_targets_filtered, binary_pred_labels_filtered)
                    except:
                        accuracy = 0.0
                    
                    metrics.update({
                        'accuracy': float(accuracy),
                        'precision': 0.0,
                        'recall': 0.0,
                        'sensitivity': 0.0,
                        'specificity': 0.0,
                        'f1': 0.0,
                        'auc': 0.0,
                        'ece': 0.0
                    })
            else:
                logger.warning("No valid binary classification samples found.")
                metrics.update({
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'auc': 0.0
                })
        
        # Regression metrics (only for multitask mode)
        if self.task_mode == 'multitask' and regression_predictions:
            regression_predictions = np.array(regression_predictions)
            regression_targets = np.array(regression_targets)
            
            # Denormalize if scaler is available
            if self.vcdr_scaler is not None:
                regression_predictions = self.vcdr_scaler.inverse_transform(regression_predictions.reshape(-1, 1)).flatten()
                regression_targets = self.vcdr_scaler.inverse_transform(regression_targets.reshape(-1, 1)).flatten()
            
            metrics.update({
                'mse': float(mean_squared_error(regression_targets, regression_predictions)),
                'mae': float(mean_absolute_error(regression_targets, regression_predictions)),
                'r2': float(r2_score(regression_targets, regression_predictions))
            })
        
        return metrics
    
    def train(self, num_epochs: int = 50, early_stopping_patience: int = 10):
        """Train the model."""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            self.current_epoch = epoch
            
            # Apply warmup for VFM
            if hasattr(self, 'warmup_epochs') and epoch < self.warmup_epochs:
                warmup_factor = (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    original_lr = param_group.get('original_lr', param_group['lr'])
                    if 'original_lr' not in param_group:
                        param_group['original_lr'] = param_group['lr']
                    param_group['lr'] = original_lr * warmup_factor
                logger.info(f"Warmup epoch {epoch+1}/{self.warmup_epochs}, warmup factor: {warmup_factor:.3f}")
            elif hasattr(self, 'warmup_epochs') and epoch == self.warmup_epochs:
                # Restore original learning rates after warmup
                for param_group in self.optimizer.param_groups:
                    if 'original_lr' in param_group:
                        param_group['lr'] = param_group['original_lr']
                logger.info("Warmup completed, restored original learning rates")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate (only after warmup for VFM)
            if not hasattr(self, 'warmup_epochs') or epoch >= self.warmup_epochs:
                self.scheduler.step(val_metrics['loss'])
            
            # Update history - adapt based on task mode
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_classification_loss'].append(train_metrics['classification_loss'])
            
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_classification_loss'].append(val_metrics['classification_loss'])
            
            if self.task_mode == 'multitask':
                self.history['train_regression_loss'].append(train_metrics['regression_loss'])
                self.history['val_regression_loss'].append(val_metrics['regression_loss'])
            
            if 'accuracy' in val_metrics:
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                self.history['val_auc'].append(val_metrics['auc'])
            
            if self.task_mode == 'multitask' and 'mse' in val_metrics:
                self.history['val_mse'].append(val_metrics['mse'])
                self.history['val_mae'].append(val_metrics['mae'])
                self.history['val_r2'].append(val_metrics['r2'])
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            if 'accuracy' in val_metrics:
                logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}, ECE: {val_metrics['ece']:.4f}")
                logger.info(f"Val Sensitivity: {val_metrics['sensitivity']:.4f}, Specificity: {val_metrics['specificity']:.4f}")
            if self.task_mode == 'multitask' and 'mse' in val_metrics:
                logger.info(f"Val MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                best_epoch = epoch
                patience_counter = 0
                
                self.best_model_path = os.path.join(self.output_dir, f'best_model_{self.backbone}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'metrics': val_metrics,
                    'vcdr_scaler': self.vcdr_scaler
                }, self.best_model_path)
                
                logger.info(f"New best model saved with validation loss: {val_metrics['loss']:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        logger.info(f"Training completed. Best epoch: {best_epoch+1}")
        
        # Load best model for final evaluation
        if self.best_model_path:
            checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded best model for final evaluation")
    
    def evaluate(self, loader: DataLoader = None, save_predictions: bool = True) -> Dict:
        """Evaluate the model and save predictions."""
        if loader is None:
            loader = self.test_loader
        
        logger.info("Evaluating model...")
        metrics = self.validate(loader)
        
        if save_predictions:
            self.save_predictions(loader)
        
        return metrics
    
    def save_predictions(self, loader: DataLoader):
        """Save detailed predictions."""
        self.model.eval()
        
        predictions_data = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Saving predictions"):
                images = batch['image'].to(self.device)
                predictions = self.model(images)
                
                binary_probs = torch.sigmoid(predictions['classification_logits']).cpu().numpy()
                
                # Only get regression predictions for multitask mode
                if self.task_mode == 'multitask' and 'regression_output' in predictions:
                    regression_preds = predictions['regression_output'].cpu().numpy()
                    
                    # Denormalize regression predictions
                    if self.vcdr_scaler is not None:
                        regression_preds = self.vcdr_scaler.inverse_transform(regression_preds.reshape(-1, 1)).flatten()
                else:
                    regression_preds = [None] * len(batch['image_path'])
                
                for i in range(len(batch['image_path'])):
                    pred_data = {
                        'image_path': batch['image_path'][i],
                        'dataset': batch['dataset'][i],
                        'true_binary_label': batch['binary_label'][i].item() if batch['binary_mask'][i] else None,
                        'pred_binary_prob': binary_probs[i],
                        'pred_binary_label': int(binary_probs[i] > 0.5),
                        'has_binary_label': bool(batch['binary_mask'][i])
                    }
                    
                    # Add regression data only for multitask mode
                    if self.task_mode == 'multitask':
                        pred_data.update({
                            'true_vcdr': batch['vcdr_label'][i].item() if batch['vcdr_mask'][i] else None,
                            'pred_vcdr': regression_preds[i] if regression_preds[i] is not None else None,
                            'has_vcdr_label': bool(batch['vcdr_mask'][i])
                        })
                    
                    predictions_data.append(pred_data)
        
        # Save predictions
        predictions_df = pd.DataFrame(predictions_data)
        task_suffix = self.task_mode
        predictions_path = os.path.join(self.output_dir, f'predictions_{self.backbone}_{task_suffix}.csv')
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to: {predictions_path}")
        
        return predictions_df
    
    def plot_training_history(self):
        """Plot training history - adapt to task mode."""
        if self.task_mode == 'singletask':
            # Simplified plots for single-task mode
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Loss plots
            axes[0, 0].plot(self.history['train_loss'], label='Train')
            axes[0, 0].plot(self.history['val_loss'], label='Validation')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(self.history['train_classification_loss'], label='Train')
            axes[0, 1].plot(self.history['val_classification_loss'], label='Validation')
            axes[0, 1].set_title('Classification Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Metric plots
            if self.history['val_accuracy']:
                axes[1, 0].plot(self.history['val_accuracy'])
                axes[1, 0].set_title('Validation Accuracy')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].grid(True)
                
                axes[1, 1].plot(self.history['val_auc'])
                axes[1, 1].set_title('Validation AUC')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('AUC')
                axes[1, 1].grid(True)
                
        else:
            # Full plots for multi-task mode
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Loss plots
            axes[0, 0].plot(self.history['train_loss'], label='Train')
            axes[0, 0].plot(self.history['val_loss'], label='Validation')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(self.history['train_classification_loss'], label='Train')
            axes[0, 1].plot(self.history['val_classification_loss'], label='Validation')
            axes[0, 1].set_title('Classification Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            axes[0, 2].plot(self.history['train_regression_loss'], label='Train')
            axes[0, 2].plot(self.history['val_regression_loss'], label='Validation')
            axes[0, 2].set_title('Regression Loss')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
            
            # Metric plots
            if self.history['val_accuracy']:
                axes[1, 0].plot(self.history['val_accuracy'])
                axes[1, 0].set_title('Validation Accuracy')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].grid(True)
                
                axes[1, 1].plot(self.history['val_auc'])
                axes[1, 1].set_title('Validation AUC')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('AUC')
                axes[1, 1].grid(True)
            
            if self.history['val_mse']:
                axes[1, 2].plot(self.history['val_mse'])
                axes[1, 2].set_title('Validation MSE')
                axes[1, 2].set_xlabel('Epoch')
                axes[1, 2].set_ylabel('MSE')
                axes[1, 2].grid(True)
        
        plt.tight_layout()
        task_suffix = self.task_mode
        plot_path = os.path.join(self.output_dir, f'training_history_{self.backbone}_{task_suffix}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved to: {plot_path}")


def load_vcdr_data(vcdr_csv_path: str) -> pd.DataFrame:
    """Load vCDR data from CSV file."""
    logger.info(f"Loading vCDR data from: {vcdr_csv_path}")
    
    df = pd.read_csv(vcdr_csv_path)
    logger.info(f"Loaded {len(df)} samples from {df['dataset'].nunique()} datasets")
    
    # Rename columns for consistency
    if 'original_label' in df.columns:
        df['binary_label'] = df['original_label']
    
    # Ensure required columns exist
    required_cols = ['dataset', 'image_path', 'vcdr']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")
    
    # Clean binary labels - ensure they are 0/1 only
    if 'binary_label' in df.columns:
        initial_binary_samples = df['binary_label'].notna().sum()
        
        # Check for non-binary values
        binary_df = df.dropna(subset=['binary_label'])
        if len(binary_df) > 0:
            unique_labels = binary_df['binary_label'].unique()
            logger.info(f"Unique binary labels found: {sorted(unique_labels)}")
            
            # Convert any numeric values to binary if possible
            def convert_to_binary(val):
                if pd.isna(val):
                    return np.nan
                try:
                    val = float(val)
                    if val == 0 or val == 0.0:
                        return 0
                    elif val == 1 or val == 1.0:
                        return 1
                    elif val == 2:  # Sometimes glaucoma is labeled as 2
                        return 1
                    else:
                        return np.nan  # Invalid value
                except:
                    return np.nan  # Non-numeric value
            
            # Apply conversion
            df['binary_label'] = df['binary_label'].apply(convert_to_binary)
            
            # Count valid samples after conversion
            final_binary_samples = df['binary_label'].notna().sum()
            invalid_count = initial_binary_samples - final_binary_samples
            
            if invalid_count > 0:
                logger.warning(f"Converted/removed {invalid_count} samples with non-binary labels.")
            
            logger.info(f"Binary labels: {initial_binary_samples} → {final_binary_samples} valid samples")
            
            # Final check for valid binary values
            valid_binary_df = df.dropna(subset=['binary_label'])
            if len(valid_binary_df) > 0:
                final_unique = valid_binary_df['binary_label'].unique()
                logger.info(f"Final unique binary labels: {sorted(final_unique)}")
                if not set(final_unique).issubset({0.0, 1.0}):
                    logger.error(f"Still have non-binary labels after conversion: {final_unique}")
                    # Force to binary
                    df.loc[~df['binary_label'].isin([0.0, 1.0]), 'binary_label'] = np.nan
    
    # Filter valid samples
    initial_len = len(df)
    df = df[df['image_path'].apply(os.path.exists)]
    logger.info(f"Filtered to {len(df)} samples with existing images (removed {initial_len - len(df)})")
    
    # Log dataset-wise label availability
    logger.info("Dataset-wise label availability:")
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        binary_count = dataset_df['binary_label'].notna().sum()
        vcdr_count = dataset_df['vcdr'].notna().sum()
        both_count = (dataset_df['binary_label'].notna() & dataset_df['vcdr'].notna()).sum()
        logger.info(f"  {dataset}: {len(dataset_df)} total, {binary_count} binary, {vcdr_count} vCDR, {both_count} both")
    
    return df


def create_domain_splits(df: pd.DataFrame, test_dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits using leave-one-dataset-out strategy."""
    logger.info(f"Creating domain splits with test dataset: {test_dataset}")
    
    # Test set: specified dataset
    test_df = df[df['dataset'] == test_dataset].copy()
    
    # Train/val set: all other datasets
    train_val_df = df[df['dataset'] != test_dataset].copy()
    
    if len(test_df) == 0:
        raise ValueError(f"Test dataset '{test_dataset}' not found in data")
    
    if len(train_val_df) == 0:
        raise ValueError(f"No training data available when excluding '{test_dataset}'")
    
    # Split train_val into train and validation (80/20 split)
    # Try to stratify by binary label if available
    if 'binary_label' in train_val_df.columns and train_val_df['binary_label'].notna().sum() > 10:
        stratify_df = train_val_df.dropna(subset=['binary_label'])
        if len(stratify_df) >= 10 and len(stratify_df['binary_label'].unique()) > 1:
            train_stratified, val_stratified = train_test_split(
                stratify_df, test_size=0.2, stratify=stratify_df['binary_label'], random_state=42
            )
            # Add remaining samples without labels to training set
            remaining_df = train_val_df[train_val_df['binary_label'].isna()]
            train_df = pd.concat([train_stratified, remaining_df], ignore_index=True)
            val_df = val_stratified
        else:
            # Fallback to random split
            train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
    else:
        # Random split if no binary labels available
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
    
    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Train datasets: {train_df['dataset'].value_counts().to_dict()}")
    logger.info(f"Val datasets: {val_df['dataset'].value_counts().to_dict()}")
    
    return train_df, val_df, test_df


def get_data_transforms(augment: bool = True, backbone: str = 'resnet18'):
    """Get data transformations."""
    
    # Different normalization for different backbones
    if backbone == 'vfm':
        # ViT models often use ImageNet normalization but may have been trained with different stats
        # Using standard ImageNet normalization as a safe default
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        # ResNet models use ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def train_single_fold(df: pd.DataFrame, 
                     test_dataset: str, 
                     output_dir: str, 
                     args: argparse.Namespace) -> Dict:
    """Train and evaluate a single fold."""
    # Include backbone in folder name for better organization
    fold_dir = os.path.join(output_dir, f'fold_{test_dataset}_{args.backbone}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # Check if model and results already exist
    best_model_path = os.path.join(fold_dir, f'best_model_{args.backbone}.pth')
    results_path = os.path.join(fold_dir, f'results_{args.backbone}.json')
    model_and_results_exist = (os.path.exists(best_model_path) and 
                              os.path.exists(results_path) and 
                              not args.force_retrain)
    
    if model_and_results_exist:
        logger.info(f"Found existing {args.backbone} model and results for {test_dataset}")
        logger.info(f"Model: {best_model_path}")
        logger.info(f"Results: {results_path}")
        logger.info("Loading existing results instead of retraining...")
        
        # Load existing results
        try:
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
            logger.info(f"Loaded existing results. Test metrics:")
            if 'metrics' in existing_results:
                for metric, value in existing_results['metrics'].items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {metric}: {value:.4f}")
            return existing_results
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load existing results file: {e}")
            logger.info("Results file appears corrupted. Will re-evaluate the existing model...")
    elif args.force_retrain and os.path.exists(best_model_path):
        logger.info(f"Force retrain flag set. Will retrain existing {args.backbone} model for {test_dataset}")
    else:
        logger.info(f"No existing {args.backbone} model or results found for {test_dataset}. Starting training...")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING FOLD: TEST DATASET = {test_dataset}")
    logger.info(f"Using backbone: {args.backbone}")
    if args.backbone == 'vfm':
        logger.info(f"VFM weights path: {args.vfm_weights_path}")
    logger.info(f"{'='*80}")
    
    # Create splits
    train_df, val_df, test_df = create_domain_splits(df, test_dataset)
    
    # Normalize vCDR values
    vcdr_scaler = StandardScaler()
    train_vcdr_values = train_df['vcdr'].dropna().values.reshape(-1, 1)
    if len(train_vcdr_values) > 0:
        vcdr_scaler.fit(train_vcdr_values)
    else:
        logger.warning("No vCDR values available for scaling")
        vcdr_scaler = None
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(augment=True, backbone=args.backbone)
    
    # Create datasets
    train_data = MultiTaskDataset(train_df, train_transform, vcdr_scaler)
    val_data = MultiTaskDataset(val_df, val_transform, vcdr_scaler)
    test_data = MultiTaskDataset(test_df, val_transform, vcdr_scaler)
    
    # Create data loaders with performance optimizations
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, 
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else 2
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else 2
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else 2
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskModel(
        backbone=args.backbone, 
        pretrained=True, 
        dropout=args.dropout,
        vfm_weights_path=args.vfm_weights_path if args.backbone == 'vfm' else None
    )
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: {args.backbone}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Feature dimension: {model.feature_dim}")
    
    # Apply performance optimizations
    model = model.to(device)
    
    # Use channels_last memory format for better performance on modern GPUs
    if device.type == 'cuda' and args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        logger.info("Using channels_last memory format for better GPU performance")
    
    # Compile model for faster inference (PyTorch 2.0+)
    if args.compile_model:
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode='reduce-overhead')
                logger.info("Model compiled with torch.compile for faster training")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}. Continuing without compilation.")
    
    # Create trainer (model is already moved to device)
    trainer = TaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=fold_dir,
        vcdr_scaler=vcdr_scaler,
        backbone=args.backbone,
        use_mixed_precision=args.use_mixed_precision,
        channels_last=args.channels_last,
        vfm_backbone_lr=args.vfm_backbone_lr,
        vfm_head_lr=args.vfm_head_lr,
        task_mode='multitask'  # Original function remains multitask
    )
    
    # Train or load existing model
    if model_and_results_exist and os.path.exists(best_model_path):
        logger.info("Loading existing trained model...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model = model.to(device)
        
        # Load the vCDR scaler if available
        if 'vcdr_scaler' in checkpoint and checkpoint['vcdr_scaler'] is not None:
            vcdr_scaler = checkpoint['vcdr_scaler']
            trainer.vcdr_scaler = vcdr_scaler
            logger.info("Loaded vCDR scaler from checkpoint")
        else:
            logger.warning("No vCDR scaler found in checkpoint, using newly fitted scaler")
        
        logger.info("Successfully loaded existing model")
    else:
        # Adjust early stopping patience for VFM
        if args.backbone == 'vfm':
            effective_patience = min(args.early_stopping_patience, 3)  # VFM should converge faster
            logger.info(f"Using reduced early stopping patience for VFM: {effective_patience}")
        else:
            effective_patience = args.early_stopping_patience
        
        # Train the model
        trainer.train(num_epochs=args.num_epochs, early_stopping_patience=effective_patience)
        # Plot training history
        trainer.plot_training_history()
    
    # Evaluate
    test_metrics = trainer.evaluate(save_predictions=True)
    
    # Save fold results - ensure all values are JSON serializable
    serializable_metrics = {}
    for key, value in test_metrics.items():
        if isinstance(value, (int, float, str, bool, list, dict)) or value is None:
            serializable_metrics[key] = value
        elif isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif hasattr(value, 'item'):  # numpy scalars
            serializable_metrics[key] = value.item()
        else:
            # Skip non-serializable objects
            logger.warning(f"Skipping non-serializable metric: {key} (type: {type(value)})")
    
    fold_results = {
        'test_dataset': test_dataset,
        'train_datasets': list(train_df['dataset'].unique()),
        'backbone': args.backbone,
        'metrics': serializable_metrics,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df)
    }
    
    with open(os.path.join(fold_dir, f'results_{args.backbone}.json'), 'w') as f:
        json.dump(fold_results, f, indent=4, cls=NpEncoder)
    
    logger.info(f"Fold {test_dataset} completed. Test metrics:")
    for metric, value in test_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    return fold_results


def train_single_fold_comparison(df: pd.DataFrame, 
                                test_dataset: str, 
                                output_dir: str, 
                                args: argparse.Namespace) -> Dict:
    """Train and evaluate both single-task and multi-task models for comparison."""
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPARISON TRAINING: TEST DATASET = {test_dataset}")
    logger.info(f"Using backbone: {args.backbone}")
    logger.info(f"Training both single-task (classification only) and multi-task (classification + regression)")
    logger.info(f"{'='*80}")
    
    # Include backbone and comparison in folder name
    fold_dir = os.path.join(output_dir, f'fold_{test_dataset}_{args.backbone}_comparison')
    os.makedirs(fold_dir, exist_ok=True)
    
    # Check if comparison is already complete
    comparison_results_path = os.path.join(fold_dir, f'comparison_{args.backbone}.json')
    if os.path.exists(comparison_results_path) and not args.force_retrain:
        logger.info(f"Found existing comparison results for {test_dataset} at: {comparison_results_path}")
        logger.info("Loading existing comparison results instead of retraining...")
        
        try:
            with open(comparison_results_path, 'r') as f:
                existing_results = json.load(f)
            logger.info(f"Loaded existing comparison results")
            
            # Print summary of existing results
            if 'comparison' in existing_results:
                logger.info("Existing comparison summary:")
                comparison = existing_results['comparison']
                for metric in ['sensitivity', 'specificity', 'auc', 'ece']:
                    if metric in comparison:
                        st_val = comparison[metric]['singletask']
                        mt_val = comparison[metric]['multitask']
                        improvement = comparison[metric]['improvement']
                        improvement_pct = comparison[metric]['improvement_pct']
                        logger.info(f"  {metric}: ST={st_val:.4f}, MT={mt_val:.4f}, Δ={improvement:+.4f} ({improvement_pct:+.1f}%)")
            
            return existing_results
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load existing comparison results: {e}")
            logger.info("Comparison results file appears corrupted. Will re-run comparison...")
    elif args.force_retrain and os.path.exists(comparison_results_path):
        logger.info(f"Force retrain flag set. Will re-run comparison for {test_dataset}")
    else:
        logger.info(f"No existing comparison results found for {test_dataset}. Starting comparison training...")
    
    # Create splits (same for both models)
    train_df, val_df, test_df = create_domain_splits(df, test_dataset)
    
    # Normalize vCDR values
    vcdr_scaler = StandardScaler()
    train_vcdr_values = train_df['vcdr'].dropna().values.reshape(-1, 1)
    if len(train_vcdr_values) > 0:
        vcdr_scaler.fit(train_vcdr_values)
    else:
        logger.warning("No vCDR values available for scaling")
        vcdr_scaler = None
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(augment=True, backbone=args.backbone)
    
    # Create datasets
    train_data = MultiTaskDataset(train_df, train_transform, vcdr_scaler)
    val_data = MultiTaskDataset(val_df, val_transform, vcdr_scaler)
    test_data = MultiTaskDataset(test_df, val_transform, vcdr_scaler)
    
    # Create data loaders
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, 
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else 2
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else 2
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else 2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Train both models
    for task_mode in ['singletask', 'multitask']:
        logger.info(f"\n{'-'*60}")
        logger.info(f"Training {task_mode.upper()} model")
        logger.info(f"{'-'*60}")
        
        # Check if model already exists
        best_model_path = os.path.join(fold_dir, f'best_model_{args.backbone}_{task_mode}.pth')
        results_path = os.path.join(fold_dir, f'results_{args.backbone}_{task_mode}.json')
        model_exists = os.path.exists(best_model_path) and not args.force_retrain
        
        if model_exists:
            logger.info(f"Found existing {task_mode} {args.backbone} model for {test_dataset}")
            
            # Load existing results if available
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        existing_results = json.load(f)
                    logger.info(f"Loaded existing {task_mode} results")
                    results[task_mode] = existing_results
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load existing {task_mode} results: {e}")
        
        # Create model based on task mode
        if task_mode == 'singletask':
            model = SingleTaskModel(
                backbone=args.backbone, 
                pretrained=True, 
                dropout=args.dropout,
                vfm_weights_path=args.vfm_weights_path if args.backbone == 'vfm' else None
            )
        else:
            model = MultiTaskModel(
                backbone=args.backbone, 
                pretrained=True, 
                dropout=args.dropout,
                vfm_weights_path=args.vfm_weights_path if args.backbone == 'vfm' else None
            )
        
        # Apply performance optimizations
        model = model.to(device)
        
        if device.type == 'cuda' and args.channels_last:
            model = model.to(memory_format=torch.channels_last)
        
        if args.compile_model:
            try:
                if hasattr(torch, 'compile'):
                    model = torch.compile(model, mode='reduce-overhead')
                    logger.info(f"Compiled {task_mode} model with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile {task_mode} model: {e}")
        
        # Create task-specific output directory
        task_output_dir = os.path.join(fold_dir, task_mode)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # Create trainer
        trainer = TaskTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            output_dir=task_output_dir,
            vcdr_scaler=vcdr_scaler,
            backbone=args.backbone,
            use_mixed_precision=args.use_mixed_precision,
            channels_last=args.channels_last,
            vfm_backbone_lr=args.vfm_backbone_lr,
            vfm_head_lr=args.vfm_head_lr,
            task_mode=task_mode
        )
        
        # Train or load existing model
        if model_exists:
            logger.info(f"Loading existing {task_mode} model...")
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.model = model.to(device)
            
            if 'vcdr_scaler' in checkpoint and checkpoint['vcdr_scaler'] is not None:
                vcdr_scaler = checkpoint['vcdr_scaler']
                trainer.vcdr_scaler = vcdr_scaler
        else:
            # Adjust early stopping patience for VFM
            if args.backbone == 'vfm':
                effective_patience = min(args.early_stopping_patience, 3)
            else:
                effective_patience = args.early_stopping_patience
            
            # Train the model
            trainer.train(num_epochs=args.num_epochs, early_stopping_patience=effective_patience)
            trainer.plot_training_history()
        
        # Evaluate
        test_metrics = trainer.evaluate(save_predictions=True)
        
        # Save task-specific results
        task_results = {
            'test_dataset': test_dataset,
            'train_datasets': list(train_df['dataset'].unique()),
            'backbone': args.backbone,
            'task_mode': task_mode,
            'metrics': {k: v.item() if hasattr(v, 'item') else v for k, v in test_metrics.items()},
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df)
        }
        
        with open(results_path, 'w') as f:
            json.dump(task_results, f, indent=4, cls=NpEncoder)
        
        results[task_mode] = task_results
        
        logger.info(f"{task_mode.upper()} - Test metrics:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
    
    # Compare results
    if 'singletask' in results and 'multitask' in results:
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPARISON RESULTS FOR {test_dataset}")
        logger.info(f"{'='*60}")
        
        st_metrics = results['singletask']['metrics']
        mt_metrics = results['multitask']['metrics']
        
        # Focus on medical screening metrics for comparison
        classification_metrics = ['sensitivity', 'specificity', 'auc', 'ece', 'brier_score']
        
        comparison = {}
        logger.info("Medical Screening Performance Comparison (Primary Metrics):")
        logger.info(f"{'Metric':<12} {'Single-Task':<12} {'Multi-Task':<12} {'Improvement':<12}")
        logger.info("-" * 50)
        
        for metric in classification_metrics:
            if metric in st_metrics and metric in mt_metrics:
                st_val = st_metrics[metric]
                mt_val = mt_metrics[metric]
                improvement = mt_val - st_val
                improvement_pct = (improvement / st_val) * 100 if st_val > 0 else 0
                
                comparison[metric] = {
                    'singletask': st_val,
                    'multitask': mt_val,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                }
                
                logger.info(f"{metric:<12} {st_val:<12.4f} {mt_val:<12.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        # Also calculate secondary metrics for completeness
        secondary_metrics = ['accuracy', 'precision', 'recall', 'f1']
        logger.info("\nSecondary Metrics (for reference):")
        logger.info(f"{'Metric':<12} {'Single-Task':<12} {'Multi-Task':<12} {'Improvement':<12}")
        logger.info("-" * 50)
        
        for metric in secondary_metrics:
            if metric in st_metrics and metric in mt_metrics:
                st_val = st_metrics[metric]
                mt_val = mt_metrics[metric]
                improvement = mt_val - st_val
                improvement_pct = (improvement / st_val) * 100 if st_val > 0 else 0
                
                comparison[metric] = {
                    'singletask': st_val,
                    'multitask': mt_val,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                }
                
                logger.info(f"{metric:<12} {st_val:<12.4f} {mt_val:<12.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        # Save comparison results
        comparison_results = {
            'test_dataset': test_dataset,
            'backbone': args.backbone,
            'singletask_results': results['singletask'],
            'multitask_results': results['multitask'],
            'comparison': comparison
        }
        
        comparison_path = os.path.join(fold_dir, f'comparison_{args.backbone}.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=4, cls=NpEncoder)
        
        logger.info(f"\nComparison results saved to: {comparison_path}")
        
        return comparison_results
    
    return results
    

def reevaluate_existing_models(df: pd.DataFrame, 
                              test_dataset: str, 
                              output_dir: str, 
                              args: argparse.Namespace) -> Dict:
    """Re-evaluate existing trained models with updated metrics (e.g., ECE)."""
    logger.info(f"\n{'='*80}")
    logger.info(f"RE-EVALUATING EXISTING MODELS: TEST DATASET = {test_dataset}")
    logger.info(f"Using backbone: {args.backbone}")
    logger.info(f"This will update metrics without retraining")
    logger.info(f"{'='*80}")
    
    # Determine the comparison folder structure
    if args.compare_tasks:
        fold_dir = os.path.join(output_dir, f'fold_{test_dataset}_{args.backbone}_comparison')
        comparison_results_path = os.path.join(fold_dir, f'comparison_{args.backbone}.json')
    else:
        fold_dir = os.path.join(output_dir, f'fold_{test_dataset}_{args.backbone}')
        comparison_results_path = None
    
    if not os.path.exists(fold_dir):
        logger.error(f"No existing models found for {test_dataset} at: {fold_dir}")
        return None
    
    # Create splits for evaluation
    train_df, val_df, test_df = create_domain_splits(df, test_dataset)
    
    # Normalize vCDR values (same as training)
    vcdr_scaler = StandardScaler()
    train_vcdr_values = train_df['vcdr'].dropna().values.reshape(-1, 1)
    if len(train_vcdr_values) > 0:
        vcdr_scaler.fit(train_vcdr_values)
    else:
        logger.warning("No vCDR values available for scaling")
        vcdr_scaler = None
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(augment=True, backbone=args.backbone)
    
    # Create test dataset
    test_data = MultiTaskDataset(test_df, val_transform, vcdr_scaler)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else 2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Determine which models to re-evaluate
    if args.compare_tasks:
        task_modes = ['singletask', 'multitask']
    else:
        task_modes = ['multitask']  # Default mode
    
    for task_mode in task_modes:
        logger.info(f"\n{'-'*60}")
        logger.info(f"Re-evaluating {task_mode.upper()} model")
        logger.info(f"{'-'*60}")
        
        # Determine model path
        if args.compare_tasks:
            # Models are stored in task-specific subdirectories
            best_model_path = os.path.join(fold_dir, task_mode, f'best_model_{args.backbone}.pth')
        else:
            best_model_path = os.path.join(fold_dir, f'best_model_{args.backbone}.pth')
        
        if not os.path.exists(best_model_path):
            logger.warning(f"Model not found: {best_model_path}")
            continue
        
        logger.info(f"Loading model from: {best_model_path}")
        
        # Create model based on task mode
        if task_mode == 'singletask':
            model = SingleTaskModel(
                backbone=args.backbone, 
                pretrained=True, 
                dropout=args.dropout,
                vfm_weights_path=args.vfm_weights_path if args.backbone == 'vfm' else None
            )
        else:
            model = MultiTaskModel(
                backbone=args.backbone, 
                pretrained=True, 
                dropout=args.dropout,
                vfm_weights_path=args.vfm_weights_path if args.backbone == 'vfm' else None
            )
        
        # Load trained weights
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load vCDR scaler from checkpoint if available
            if 'vcdr_scaler' in checkpoint and checkpoint['vcdr_scaler'] is not None:
                vcdr_scaler = checkpoint['vcdr_scaler']
                logger.info("Loaded vCDR scaler from checkpoint")
            
            logger.info(f"Successfully loaded {task_mode} model")
        except Exception as e:
            logger.error(f"Failed to load {task_mode} model: {e}")
            continue
        
        # Apply performance optimizations
        model = model.to(device)
        if device.type == 'cuda' and args.channels_last:
            model = model.to(memory_format=torch.channels_last)
        
        # Create trainer for evaluation
        task_output_dir = os.path.join(fold_dir, task_mode) if args.compare_tasks else fold_dir
        os.makedirs(task_output_dir, exist_ok=True)
        
        trainer = TaskTrainer(
            model=model,
            train_loader=None,  # Not needed for evaluation
            val_loader=None,    # Not needed for evaluation
            test_loader=test_loader,
            device=device,
            output_dir=task_output_dir,
            vcdr_scaler=vcdr_scaler,
            backbone=args.backbone,
            use_mixed_precision=args.use_mixed_precision,
            channels_last=args.channels_last,
            vfm_backbone_lr=args.vfm_backbone_lr,
            vfm_head_lr=args.vfm_head_lr,
            task_mode=task_mode
        )
        
        # Re-evaluate with updated metrics
        logger.info(f"Re-evaluating {task_mode} model with updated metrics...")
        test_metrics = trainer.evaluate(save_predictions=True)
        
        # Save updated results
        task_results = {
            'test_dataset': test_dataset,
            'train_datasets': list(train_df['dataset'].unique()),
            'backbone': args.backbone,
            'task_mode': task_mode,
            'metrics': {k: v.item() if hasattr(v, 'item') else v for k, v in test_metrics.items()},
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'reevaluated': True,  # Mark as re-evaluated
            'reevaluation_timestamp': datetime.now().isoformat()
        }
        
        # Save individual task results
        if args.compare_tasks:
            results_path = os.path.join(fold_dir, f'results_{args.backbone}_{task_mode}.json')
        else:
            results_path = os.path.join(fold_dir, f'results_{args.backbone}.json')
        
        with open(results_path, 'w') as f:
            json.dump(task_results, f, indent=4, cls=NpEncoder)
        
        results[task_mode] = task_results
        
        logger.info(f"{task_mode.upper()} - Updated metrics:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
    
    # Update comparison results if doing comparison
    if args.compare_tasks and 'singletask' in results and 'multitask' in results:
        logger.info(f"\n{'='*60}")
        logger.info(f"UPDATING COMPARISON RESULTS FOR {test_dataset}")
        logger.info(f"{'='*60}")
        
        st_metrics = results['singletask']['metrics']
        mt_metrics = results['multitask']['metrics']
        
        # Focus on medical screening metrics for comparison
        classification_metrics = ['sensitivity', 'specificity', 'auc', 'ece', 'brier_score']
        secondary_metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        comparison = {}
        logger.info("Medical Screening Performance Comparison (Updated with ECE & Brier Score):")
        logger.info(f"{'Metric':<12} {'Single-Task':<12} {'Multi-Task':<12} {'Improvement':<12}")
        logger.info("-" * 50)
        
        for metric in classification_metrics:
            if metric in st_metrics and metric in mt_metrics:
                st_val = st_metrics[metric]
                mt_val = mt_metrics[metric]
                improvement = mt_val - st_val
                improvement_pct = (improvement / st_val) * 100 if st_val > 0 else 0
                
                comparison[metric] = {
                    'singletask': st_val,
                    'multitask': mt_val,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                }
                
                logger.info(f"{metric:<12} {st_val:<12.4f} {mt_val:<12.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        # Also calculate secondary metrics for completeness
        logger.info("\nSecondary Metrics (for reference):")
        logger.info(f"{'Metric':<12} {'Single-Task':<12} {'Multi-Task':<12} {'Improvement':<12}")
        logger.info("-" * 50)
        
        for metric in secondary_metrics:
            if metric in st_metrics and metric in mt_metrics:
                st_val = st_metrics[metric]
                mt_val = mt_metrics[metric]
                improvement = mt_val - st_val
                improvement_pct = (improvement / st_val) * 100 if st_val > 0 else 0
                
                comparison[metric] = {
                    'singletask': st_val,
                    'multitask': mt_val,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                }
                
                logger.info(f"{metric:<12} {st_val:<12.4f} {mt_val:<12.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        # Save updated comparison results
        comparison_results = {
            'test_dataset': test_dataset,
            'backbone': args.backbone,
            'singletask_results': results['singletask'],
            'multitask_results': results['multitask'],
            'comparison': comparison,
            'reevaluated': True,
            'reevaluation_timestamp': datetime.now().isoformat()
        }
        
        with open(comparison_results_path, 'w') as f:
            json.dump(comparison_results, f, indent=4, cls=NpEncoder)
        
        logger.info(f"\nUpdated comparison results saved to: {comparison_results_path}")
        
        return comparison_results
    
    # If no models were found for re-evaluation, return None
    if not results:
        logger.warning(f"No models found for re-evaluation in {test_dataset}")
        return None
    
    return results


def main(args: argparse.Namespace):
    """Main training function."""
    if args.reevaluate_only:
        if args.compare_tasks:
            logger.info("Re-evaluating existing single-task vs multi-task comparison models with updated metrics")
        else:
            logger.info("Re-evaluating existing multi-task models with updated metrics")
    else:
        if args.compare_tasks:
            logger.info("Starting single-task vs multi-task comparison with leave-one-dataset-out evaluation")
        else:
            logger.info("Starting multi-task training with leave-one-dataset-out evaluation")
    
    # Set random seed
    set_seed(args.seed)
    
    # Use simple output directory structure
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_vcdr_data(args.vcdr_csv)
    
    # Filter datasets with sufficient samples
    dataset_counts = df['dataset'].value_counts()
    valid_datasets = dataset_counts[dataset_counts >= args.min_samples_per_dataset].index.tolist()
    
    # Exclude specified datasets
    if args.exclude_datasets:
        excluded_found = [d for d in args.exclude_datasets if d in valid_datasets]
        if excluded_found:
            logger.info(f"Excluding datasets: {excluded_found}")
            valid_datasets = [d for d in valid_datasets if d not in args.exclude_datasets]
        else:
            logger.info(f"Specified exclusion datasets {args.exclude_datasets} not found in valid datasets")
    
    logger.info(f"Valid datasets (>= {args.min_samples_per_dataset} samples, after exclusions): {valid_datasets}")
    df = df[df['dataset'].isin(valid_datasets)]
    
    if len(valid_datasets) < 2:
        logger.error("Need at least 2 datasets for leave-one-out evaluation")
        return
    
    # Train one fold per dataset
    all_results = []
    
    for test_dataset in valid_datasets:
        try:
            logger.info(f"\n{'='*60}")
            if args.reevaluate_only:
                logger.info(f"Re-evaluating fold: {test_dataset}")
            else:
                logger.info(f"Starting fold: {test_dataset}")
            logger.info(f"{'='*60}")
            
            # Choose between re-evaluation or training
            if args.reevaluate_only:
                # Re-evaluate existing models
                fold_results = reevaluate_existing_models(df, test_dataset, output_dir, args)
                if fold_results is None:
                    logger.warning(f"Skipping {test_dataset} - no existing models found for re-evaluation")
                    continue
            else:
                # Use comparison training if requested
                if args.compare_tasks:
                    fold_results = train_single_fold_comparison(df, test_dataset, output_dir, args)
                else:
                    fold_results = train_single_fold(df, test_dataset, output_dir, args)
            
            # Only add successful results (not None)
            if fold_results is not None:
                all_results.append(fold_results)
            
            if args.reevaluate_only:
                logger.info(f"✅ Successfully re-evaluated fold: {test_dataset}")
            else:
                logger.info(f"✅ Successfully completed fold: {test_dataset}")
            
        except Exception as e:
            action = "re-evaluate" if args.reevaluate_only else "train"
            logger.error(f"❌ Failed to {action} fold {test_dataset}: {e}")
            
            # Log more detailed error information
            import traceback
            logger.error(f"Detailed error traceback:\n{traceback.format_exc()}")
            
            # Check if it's a label-related issue
            if "multiclass" in str(e).lower() or "binary" in str(e).lower():
                logger.error(f"This appears to be a label format issue. Checking {test_dataset} labels...")
                test_subset = df[df['dataset'] == test_dataset]
                if 'binary_label' in test_subset.columns:
                    unique_labels = test_subset['binary_label'].dropna().unique()
                    logger.error(f"Unique labels in {test_dataset}: {sorted(unique_labels)}")
            
            continue
    
    # Aggregate results
    if all_results:
        if args.compare_tasks:
            aggregate_comparison_results(all_results, output_dir)
        else:
            aggregate_results(all_results, output_dir)
    else:
        logger.error("No folds completed successfully")


def aggregate_results(all_results: List[Dict], output_dir: str):
    """Aggregate results across all folds."""
    logger.info("\n" + "="*80)
    logger.info("AGGREGATING RESULTS ACROSS ALL FOLDS")
    logger.info("="*80)
    
    # Collect metrics - prioritize medical screening metrics
    primary_metrics = ['sensitivity', 'specificity', 'auc', 'ece', 'brier_score']
    secondary_metrics = ['accuracy', 'precision', 'recall', 'f1']
    classification_metrics = primary_metrics + secondary_metrics
    regression_metrics = ['mse', 'mae', 'r2']
    
    results_summary = []
    
    for result in all_results:
        test_dataset = result['test_dataset']
        metrics = result['metrics']
        
        summary_row = {'test_dataset': test_dataset}
        
        # Add classification metrics
        for metric in classification_metrics:
            if metric in metrics:
                summary_row[f'classification_{metric}'] = metrics[metric]
        
        # Add regression metrics
        for metric in regression_metrics:
            if metric in metrics:
                summary_row[f'regression_{metric}'] = metrics[metric]
        
        summary_row.update({
            'train_size': result['train_size'],
            'val_size': result['val_size'],
            'test_size': result['test_size']
        })
        
        results_summary.append(summary_row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results_summary)
    
    # Calculate mean and std across folds
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['train_size', 'val_size', 'test_size']]
    
    aggregated_stats = {}
    for col in numeric_cols:
        values = summary_df[col].dropna()
        if len(values) > 0:
            aggregated_stats[col] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'count': len(values)
            }
    
    # Save results
    summary_df.to_csv(os.path.join(output_dir, 'all_folds_summary.csv'), index=False)
    
    with open(os.path.join(output_dir, 'aggregated_results.json'), 'w') as f:
        json.dump({
            'individual_results': all_results,
            'aggregated_stats': aggregated_stats
        }, f, indent=4, cls=NpEncoder)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print(f"Total folds completed: {len(all_results)}")
    
    if aggregated_stats:
        print("\nPrimary Medical Screening Metrics (Mean ± Std):")
        for metric in primary_metrics:
            col = f'classification_{metric}'
            if col in aggregated_stats:
                stats = aggregated_stats[col]
                print(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print("\nSecondary Classification Metrics (Mean ± Std):")
        for metric in secondary_metrics:
            col = f'classification_{metric}'
            if col in aggregated_stats:
                stats = aggregated_stats[col]
                print(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print("\nRegression Metrics (Mean ± Std):")
        for metric in regression_metrics:
            col = f'regression_{metric}'
            if col in aggregated_stats:
                stats = aggregated_stats[col]
                print(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print(f"\nDetailed results saved to: {output_dir}")
    print("="*80)


def aggregate_comparison_results(all_results: List[Dict], output_dir: str):
    """Aggregate comparison results across all folds."""
    logger.info("\n" + "="*80)
    logger.info("AGGREGATING COMPARISON RESULTS ACROSS ALL FOLDS")
    logger.info("="*80)
    
    # Collect metrics for both task modes
    classification_metrics = ['sensitivity', 'specificity', 'auc', 'ece', 'brier_score']
    secondary_metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    singletask_results = []
    multitask_results = []
    comparison_summaries = []
    
    for result in all_results:
        test_dataset = result['test_dataset']
        comparison = result.get('comparison', {})
        
        # Extract individual task results if available
        if 'singletask_results' in result:
            st_metrics = result['singletask_results']['metrics']
            st_row = {'test_dataset': test_dataset}
            for metric in classification_metrics + secondary_metrics:
                if metric in st_metrics:
                    st_row[metric] = st_metrics[metric]
            singletask_results.append(st_row)
        
        if 'multitask_results' in result:
            mt_metrics = result['multitask_results']['metrics']
            mt_row = {'test_dataset': test_dataset}
            for metric in classification_metrics + secondary_metrics:
                if metric in mt_metrics:
                    mt_row[metric] = mt_metrics[metric]
            multitask_results.append(mt_row)
        
        # Extract comparison metrics
        comp_row = {'test_dataset': test_dataset}
        for metric in classification_metrics + secondary_metrics:
            if metric in comparison:
                comp_row[f'{metric}_singletask'] = comparison[metric]['singletask']
                comp_row[f'{metric}_multitask'] = comparison[metric]['multitask']
                comp_row[f'{metric}_improvement'] = comparison[metric]['improvement']
                comp_row[f'{metric}_improvement_pct'] = comparison[metric]['improvement_pct']
        
        comparison_summaries.append(comp_row)
    
    # Create DataFrames
    if singletask_results:
        singletask_df = pd.DataFrame(singletask_results)
        singletask_df.to_csv(os.path.join(output_dir, 'singletask_summary.csv'), index=False)
    
    if multitask_results:
        multitask_df = pd.DataFrame(multitask_results)
        multitask_df.to_csv(os.path.join(output_dir, 'multitask_summary.csv'), index=False)
    
    if comparison_summaries:
        comparison_df = pd.DataFrame(comparison_summaries)
        comparison_df.to_csv(os.path.join(output_dir, 'comparison_summary.csv'), index=False)
    
    # Calculate aggregated statistics
    aggregated_stats = {
        'singletask': {},
        'multitask': {},
        'improvements': {}
    }
    
    # Single-task stats
    if singletask_results:
        st_df = pd.DataFrame(singletask_results)
        for metric in classification_metrics + secondary_metrics:
            if metric in st_df.columns:
                values = st_df[metric].dropna()
                if len(values) > 0:
                    aggregated_stats['singletask'][metric] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'count': len(values)
                    }
    
    # Multi-task stats
    if multitask_results:
        mt_df = pd.DataFrame(multitask_results)
        for metric in classification_metrics + secondary_metrics:
            if metric in mt_df.columns:
                values = mt_df[metric].dropna()
                if len(values) > 0:
                    aggregated_stats['multitask'][metric] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'count': len(values)
                    }
    
    # Improvement stats
    if comparison_summaries:
        comp_df = pd.DataFrame(comparison_summaries)
        for metric in classification_metrics + secondary_metrics:
            improvement_col = f'{metric}_improvement'
            improvement_pct_col = f'{metric}_improvement_pct'
            
            if improvement_col in comp_df.columns:
                values = comp_df[improvement_col].dropna()
                pct_values = comp_df[improvement_pct_col].dropna() if improvement_pct_col in comp_df.columns else []
                
                if len(values) > 0:
                    # For ECE and Brier score, lower values are better, so negative improvements are actually good
                    if metric in ['ece', 'brier_score']:
                        improved_count = (values < 0).sum()  # Negative change is improvement for ECE/Brier
                        degraded_count = (values > 0).sum()  # Positive change is degradation for ECE/Brier
                    else:
                        improved_count = (values > 0).sum()  # Positive change is improvement for other metrics
                        degraded_count = (values < 0).sum()  # Negative change is degradation for other metrics
                    
                    aggregated_stats['improvements'][metric] = {
                        'mean_improvement': values.mean(),
                        'std_improvement': values.std(),
                        'mean_improvement_pct': pct_values.mean() if len(pct_values) > 0 else 0,
                        'std_improvement_pct': pct_values.std() if len(pct_values) > 0 else 0,
                        'positive_improvements': improved_count,
                        'negative_improvements': degraded_count,
                        'count': len(values)
                    }
    
    # Save aggregated results
    with open(os.path.join(output_dir, 'aggregated_comparison_results.json'), 'w') as f:
        json.dump({
            'individual_results': all_results,
            'aggregated_stats': aggregated_stats
        }, f, indent=4, cls=NpEncoder)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS SUMMARY")
    print("="*80)
    
    print(f"Total folds completed: {len(all_results)}")
    
    # Print single-task vs multi-task comparison
    print("\nMEDICAL SCREENING PERFORMANCE COMPARISON (PRIMARY METRICS):")
    print("="*70)
    print(f"{'Metric':<12} {'Single-Task':<15} {'Multi-Task':<15} {'Mean Δ':<12} {'% Improved':<12}")
    print("-" * 70)
    
    # Primary metrics first
    for metric in classification_metrics + secondary_metrics:
        if (metric in aggregated_stats['singletask'] and 
            metric in aggregated_stats['multitask'] and 
            metric in aggregated_stats['improvements']):
            
            st_stats = aggregated_stats['singletask'][metric]
            mt_stats = aggregated_stats['multitask'][metric]
            imp_stats = aggregated_stats['improvements'][metric]
            
            st_mean = st_stats['mean']
            mt_mean = mt_stats['mean']
            mean_improvement = imp_stats['mean_improvement']
            mean_improvement_pct = imp_stats['mean_improvement_pct']
            positive_count = imp_stats['positive_improvements']
            total_count = imp_stats['count']
            pct_improved = (positive_count / total_count * 100) if total_count > 0 else 0
            
            # Add asterisk for primary metrics
            metric_label = f"{metric.upper()}*" if metric in ['sensitivity', 'specificity', 'auc', 'ece'] else metric.upper()
            print(f"{metric_label:<12} {st_mean:<15.4f} {mt_mean:<15.4f} {mean_improvement:+.4f}      {pct_improved:<12.1f}%")
    
    print("\n* Primary metrics for medical screening applications")
    
    print("\nDETAILED IMPROVEMENT STATISTICS:")
    print("="*50)
    
    # Show primary metrics first
    print("PRIMARY MEDICAL SCREENING METRICS:")
    for metric in ['sensitivity', 'specificity', 'auc', 'ece']:
        if metric in aggregated_stats['improvements']:
            imp_stats = aggregated_stats['improvements'][metric]
            print(f"\n{metric.upper()}* (Medical Priority):")
            print(f"  Mean improvement: {imp_stats['mean_improvement']:+.4f} ({imp_stats['mean_improvement_pct']:+.1f}%)")
            print(f"  Std improvement:  ±{imp_stats['std_improvement']:.4f} (±{imp_stats['std_improvement_pct']:.1f}%)")
            print(f"  Datasets improved: {imp_stats['positive_improvements']}/{imp_stats['count']} ({imp_stats['positive_improvements']/imp_stats['count']*100:.1f}%)")
            print(f"  Datasets degraded: {imp_stats['negative_improvements']}/{imp_stats['count']} ({imp_stats['negative_improvements']/imp_stats['count']*100:.1f}%)")
    
    # Show secondary metrics
    print("\nSECONDARY METRICS (for reference):")
    for metric in secondary_metrics:
        if metric in aggregated_stats['improvements']:
            imp_stats = aggregated_stats['improvements'][metric]
            print(f"\n{metric.upper()}:")
            print(f"  Mean improvement: {imp_stats['mean_improvement']:+.4f} ({imp_stats['mean_improvement_pct']:+.1f}%)")
            print(f"  Std improvement:  ±{imp_stats['std_improvement']:.4f} (±{imp_stats['std_improvement_pct']:.1f}%)")
            print(f"  Datasets improved: {imp_stats['positive_improvements']}/{imp_stats['count']} ({imp_stats['positive_improvements']/imp_stats['count']*100:.1f}%)")
            print(f"  Datasets degraded: {imp_stats['negative_improvements']}/{imp_stats['count']} ({imp_stats['negative_improvements']/imp_stats['count']*100:.1f}%)")
    
    print(f"\nDetailed results saved to: {output_dir}")
    print("Files created:")
    print("  - singletask_summary.csv")
    print("  - multitask_summary.csv") 
    print("  - comparison_summary.csv")
    print("  - aggregated_comparison_results.json")
    print("="*80)


def load_vfm_weights(model, weights_path: str) -> bool:
    """
    Load VFM weights with robust error handling.
    
    Args:
        model: The ViT model to load weights into
        weights_path: Path to the weights file
        
    Returns:
        bool: True if weights loaded successfully, False otherwise
    """
    if not os.path.exists(weights_path):
        logger.warning(f"VFM weights file not found: {weights_path}")
        return False
    
    try:
        logger.info(f"Loading VFM weights from: {weights_path}")
        
        # Load the state dict
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Handle VFM checkpoint format (has 'student' key)
        if isinstance(checkpoint, dict):
            if 'student' in checkpoint:
                # VFM checkpoint format
                state_dict = checkpoint['student']
                logger.info("Loading from VFM 'student' model")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume the entire checkpoint is the state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.backbone.' prefix from VFM weights to match our model
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.backbone.' prefix if present
            if k.startswith('module.backbone.'):
                new_key = k[16:]  # Remove 'module.backbone.'
                new_state_dict[new_key] = v
            elif k.startswith('module.'):
                new_key = k[7:]  # Remove 'module.'
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        # Remove any keys that don't match our model architecture
        model_keys = set(model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_keys}
        
        # Load weights, allowing for missing keys (e.g., classifier head)
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        # Count actual parameter values loaded
        loaded_param_count = sum(v.numel() for v in filtered_state_dict.values())
        total_param_count = sum(p.numel() for p in model.parameters())
        loaded_percentage = (loaded_param_count / total_param_count) * 100
        
        logger.info(f"Loaded {len(filtered_state_dict)} parameter tensors from VFM weights")
        logger.info(f"Loaded {loaded_param_count:,} individual parameters ({loaded_percentage:.1f}% of model)")
        logger.info(f"Model has {total_param_count:,} total parameters")
        
        if missing_keys:
            missing_param_count = sum(p.numel() for name, p in model.named_parameters() if name in missing_keys)
            logger.info(f"Missing {len(missing_keys)} parameter tensors ({missing_param_count:,} individual parameters)")
            logger.debug(f"Missing keys: {missing_keys}")
        
        if unexpected_keys:
            logger.info(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
            logger.debug(f"Unexpected keys: {unexpected_keys}")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to load VFM weights: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-task training for glaucoma classification and vCDR regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument('--vcdr_csv', type=str, 
                       default=r'D:\glaucoma\vcdr_extraction_results\vcdr_extraction_20250711_170055\vcdr_labels_20250711_170055.csv',
                       help="Path to CSV file with vCDR labels")
    parser.add_argument('--min_samples_per_dataset', type=int, default=100,
                       help="Minimum samples per dataset to include in training")
    parser.add_argument('--exclude_datasets', type=str, nargs='*', default=['G1020_all'],
                       help="List of datasets to exclude from training (default: ['G1020_all'])")
    
    # Model configuration
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'vfm'],
                       help="Backbone architecture")
    parser.add_argument('--vfm_weights_path', type=str, 
                       default=r'D:\glaucoma\models\VFM_Fundus_weights.pth',
                       help="Path to VFM pre-trained weights")
    parser.add_argument('--dropout', type=float, default=0.5,
                       help="Dropout rate")
    parser.add_argument('--vfm_backbone_lr', type=float, default=1e-6,
                       help="Learning rate for VFM backbone/encoder")
    parser.add_argument('--vfm_head_lr', type=float, default=5e-4,
                       help="Learning rate for VFM task heads")
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=128,
                       help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument('--early_stopping_patience', type=int, default=4,
                       help="Early stopping patience (3 for VFM, 5 for ResNet)")
    parser.add_argument('--num_workers', type=int, default=4,
                       help="Number of data loader workers")
    
    # Performance optimization arguments
    parser.add_argument('--use_mixed_precision', action='store_true', default=True,
                       help="Use mixed precision training for faster VFM training")
    parser.add_argument('--compile_model', action='store_true', default=False,
                       help="Use torch.compile for faster inference (PyTorch 2.0+)")
    parser.add_argument('--channels_last', action='store_true', default=True,
                       help="Use channels_last memory format for better GPU performance")
    
    # Comparison configuration
    parser.add_argument('--compare_tasks', action='store_true',
                       help="Compare single-task vs multi-task performance")
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default=r'D:\glaucoma\multitask_results',
                       help="Output directory for results")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed")
    parser.add_argument('--force_retrain', action='store_true',
                       help="Force retraining even if existing model found")
    parser.add_argument('--reevaluate_only', action='store_true',
                       help="Only re-evaluate existing models without retraining (useful for adding new metrics like ECE)")
    
    args = parser.parse_args()
    
    # Validate flag combinations
    if args.reevaluate_only and args.force_retrain:
        logger.error("Cannot use both --reevaluate_only and --force_retrain flags together")
        sys.exit(1)
    
    # Validate inputs
    if not os.path.exists(args.vcdr_csv):
        logger.error(f"vCDR CSV file not found: {args.vcdr_csv}")
        sys.exit(1)
    
    # Validate VFM weights if using VFM backbone
    if args.backbone == 'vfm':
        if not TIMM_AVAILABLE:
            logger.error("timm is required for VFM backbone. Install with: pip install timm")
            sys.exit(1)
        
        if not os.path.exists(args.vfm_weights_path):
            logger.warning(f"VFM weights file not found: {args.vfm_weights_path}")
            logger.info("Will use randomly initialized VFM model instead")
    
    main(args)
