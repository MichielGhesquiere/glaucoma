#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VCDR Regression Model Evaluation Script for Binary Glaucoma Detection.

This script loads a pre-trained VCDR regression model and evaluates it on 
the same test datasets used by other evaluation scripts, converting VCDR 
scores to binary predictions using threshold-based classification.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_curve
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import seaborn as sns
import timm


# Ensure custom modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.utils.helpers import NpEncoder, set_seed
    from src.data.utils import get_eval_transforms
    from src.data.external_loader import load_external_test_data
    from src.data.utils import adjust_path_for_data_type 
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    sys.exit(1)

RAW_DIR_NAME_CONST = "raw"
PROCESSED_DIR_NAME_CONST = "processed"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

class VCDRRegressor(nn.Module):
    """VCDR Regression model using ResNet-18 (corrected from ResNet-50)"""
    def __init__(self, pretrained=True):
        super(VCDRRegressor, self).__init__()
        # Load ResNet-18 (corrected architecture based on error analysis)
        self.resnet = models.resnet18(pretrained=False)
        
        # Replace the final classification layer with regression layer
        num_features = self.resnet.fc.in_features  # Should be 512 for ResNet-18
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )
    
    def forward(self, x):
        return self.resnet(x)

class VCDRViTRegressor(nn.Module):
    """VCDR Regression model using Vision Transformer (ViT) Base - matches training structure"""
    def __init__(self, pretrained=False):
        super(VCDRViTRegressor, self).__init__()
        
        # Create ViT model exactly like in training - this will be the entire model
        self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        
        # Get the feature dimension (768 for ViT-Base)
        num_features = self.model.num_features  # Should be 768 for ViT-Base
        
        # Replace the head with regression head (matching your training setup exactly)
        head_dp1, head_dp2, head_dp3 = 0.5, 0.3, 0.2  # Match your training dropout values
        self.model.head = nn.Sequential(
            nn.Dropout(head_dp1), 
            nn.Linear(num_features, 512), 
            nn.ReLU(),
            nn.Dropout(head_dp2), 
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(head_dp3), 
            nn.Linear(256, 1),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )
    
    def forward(self, x):
        # The model already has the ViT backbone + custom head
        return self.model(x)

class GlaucomaSubgroupDataset(Dataset):
    """
    Dataset for loading glaucoma images and associated sensitive attributes.
    Ensures labels are binary (0 or 1).
    """
    def __init__(self,
                 df: pd.DataFrame,
                 transform: Optional[Callable] = None,
                 sensitive_attributes: Optional[List[str]] = None,
                 require_attributes: bool = False,
                 label_col: str = 'types',
                 path_col: str = 'image_path',
                 expected_labels: Optional[List[int]] = None):

        super().__init__()
        self.transform = transform
        self.sensitive_attributes = sensitive_attributes if sensitive_attributes else []
        self.label_col = label_col
        self.path_col = path_col
        self.expected_labels = [0, 1] if expected_labels is None else expected_labels

        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.warning("Input DataFrame is empty or not a DataFrame. Dataset will be empty.")
            self.df = pd.DataFrame(columns=[self.path_col, self.label_col] + self.sensitive_attributes)
            return

        # --- Input Validation ---
        required_cols = [self.path_col, self.label_col]
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            raise ValueError(f"Input DataFrame is missing required columns: {missing_required}")

        missing_attributes = [attr for attr in self.sensitive_attributes if attr not in df.columns]
        if missing_attributes:
            logger.warning(f"Requested sensitive attributes not found in DataFrame columns: {missing_attributes}. They will be ignored.")
            self.sensitive_attributes = [attr for attr in self.sensitive_attributes if attr in df.columns]

        # --- Data Filtering & Label Processing ---
        df_processed = df.copy()

        # 1. Handle missing labels in the label column
        initial_len_before_nan_drop = len(df_processed)
        df_processed = df_processed.dropna(subset=[self.label_col])
        dropped_for_nan_label = initial_len_before_nan_drop - len(df_processed)
        if dropped_for_nan_label > 0:
            logger.info(f"Dropped {dropped_for_nan_label} rows due to missing values in label column '{self.label_col}'.")

        if df_processed.empty:
            logger.warning(f"DataFrame became empty after dropping rows with missing labels. Dataset will be empty.")
            self.df = pd.DataFrame(columns=[self.path_col, self.label_col] + self.sensitive_attributes)
            return
            
        # 2. Convert label column to numeric, coercing errors to NaN
        original_labels_sample = df_processed[self.label_col].unique()[:5]
        df_processed[self.label_col] = pd.to_numeric(df_processed[self.label_col], errors='coerce')
        
        # 3. Drop rows where label could not be converted to numeric (became NaN)
        initial_len_before_coerce_drop = len(df_processed)
        df_processed = df_processed.dropna(subset=[self.label_col])
        dropped_for_coerce = initial_len_before_coerce_drop - len(df_processed)
        if dropped_for_coerce > 0:
            logger.info(f"Dropped {dropped_for_coerce} rows because their labels (e.g., {original_labels_sample}) could not be converted to numeric.")

        if df_processed.empty:
            logger.warning(f"DataFrame became empty after dropping rows with non-numeric-convertible labels. Dataset will be empty.")
            self.df = pd.DataFrame(columns=[self.path_col, self.label_col] + self.sensitive_attributes)
            return

        # 4. Convert to integer type
        try:
            df_processed[self.label_col] = df_processed[self.label_col].astype(int)
        except Exception as e:
            logger.error(f"Error converting label column '{self.label_col}' to integer after numeric conversion: {e}")

        # 5. Filter for expected labels (e.g., [0, 1])
        initial_len_before_expected_label_filter = len(df_processed)
        df_processed = df_processed[df_processed[self.label_col].isin(self.expected_labels)]
        dropped_for_unexpected_label = initial_len_before_expected_label_filter - len(df_processed)
        
        if dropped_for_unexpected_label > 0:
            unique_labels_before_filter = df[self.label_col].unique()
            logger.warning(f"Dropped {dropped_for_unexpected_label} rows because their integer labels were not in "
                           f"the expected set: {self.expected_labels}. Original unique labels included: {unique_labels_before_filter}")

        # Log unique labels after all filtering
        final_unique_labels_with_counts = df_processed[self.label_col].value_counts().to_dict()
        logger.info(f"Final unique labels in dataset after all processing (expected: {self.expected_labels}): {final_unique_labels_with_counts}")

        if df_processed.empty:
            logger.warning(f"DataFrame is empty after all label processing steps. Dataset will be empty.")
            self.df = pd.DataFrame(columns=[self.path_col, self.label_col] + self.sensitive_attributes)
            return
            
        # --- Attribute Filtering (after label filtering to save processing) ---
        initial_len_before_attr_drop = len(df_processed)
        if require_attributes and self.sensitive_attributes:
            df_processed = df_processed.dropna(subset=self.sensitive_attributes)
            dropped_for_attr = initial_len_before_attr_drop - len(df_processed)
            if dropped_for_attr > 0:
                logger.info(f"Dropped {dropped_for_attr} additional rows due to missing required sensitive attributes: {self.sensitive_attributes}")
        elif require_attributes and not self.sensitive_attributes:
             logger.warning("'require_attributes' is True, but no 'sensitive_attributes' were provided or found.")

        # --- File Existence Check (optional) ---
        initial_len_before_file_check = len(df_processed)
        if 'file_exists' in df_processed.columns:
             df_processed = df_processed[df_processed['file_exists'] == True]
             dropped_for_file = initial_len_before_file_check - len(df_processed)
             if dropped_for_file > 0:
                  logger.info(f"Dropped {dropped_for_file} additional rows due to missing image files (based on 'file_exists' column).")

        self.df = df_processed.reset_index(drop=True)

        if len(self.df) == 0:
            logger.warning(f"Dataset is empty after all filtering stages.")
        else:
            logger.info(f"Dataset initialized with {len(self.df)} samples.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        if idx >= len(self.df):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.df)}")

        row = self.df.iloc[idx]
        img_path = row[self.path_col]
        label = row[self.label_col]

        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            logger.error(f"Image file not found at path: {img_path}. Returning dummy data.")
            return torch.zeros((3, 224, 224)), torch.tensor(-1, dtype=torch.long), {} 
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}", exc_info=True)
            return torch.zeros((3, 224, 224)), torch.tensor(-1, dtype=torch.long), {}

        if self.transform:
            img = self.transform(img)

        attributes = {attr_name: row[attr_name] for attr_name in self.sensitive_attributes}
        
        # Convert label to tensor
        label_tensor = torch.tensor(int(label), dtype=torch.long)
        
        return img, label_tensor, attributes

def safe_collate(batch):
    """Safe collate function similar to your evaluate_id.py"""
    if not batch:
        return None
    
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    try:
        images, labels, metadata_list = zip(*batch)
        
        # Stack images
        images = torch.stack(images)
        
        # Convert labels to tensors if they aren't already
        label_tensors = []
        for label in labels:
            if isinstance(label, torch.Tensor):
                label_tensors.append(label)
            else:
                # Convert numpy/int to tensor
                label_tensors.append(torch.tensor(int(label), dtype=torch.long))
        
        labels = torch.stack(label_tensors)
        
        # Combine metadata
        combined_metadata = {}
        if metadata_list and len(metadata_list) > 0:
            for key in metadata_list[0].keys():
                combined_metadata[key] = [item[key] for item in metadata_list]
        
        return images, labels, combined_metadata
    except Exception as e:
        logger.warning(f"Error in collate function: {e}")
        return None

#def load_vcdr_model(model_path, device):
#    """Load the trained VCDR regression model"""
#    logger.info(f"Loading VCDR regression model from: {model_path}")
#    
#    # Initialize model (ResNet-18 based on error analysis)
#    model = VCDRRegressor(pretrained=False)
#    
#    try:
#        # Load checkpoint
#        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
#        
#        # Extract state dict (handle different checkpoint formats)
#        state_dict_keys_to_try = ['model_state_dict', 'state_dict', 'model']
#        state_dict = None
#        
#        for key in state_dict_keys_to_try:
#            if key in checkpoint:
#                state_dict = checkpoint[key]
#                logger.info(f"Found state_dict under key: {key}")
#                break
#        
#        if state_dict is None:
#            if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
#                state_dict = checkpoint
#                logger.info("Using checkpoint directly as state_dict")
#            else:
#                raise ValueError("Could not find valid state_dict in checkpoint")
#        
#        # Load weights
#        load_result = model.load_state_dict(state_dict, strict=False)
#        logger.info(f"Model weights loaded. Missing keys: {load_result.missing_keys}, "
#                   f"Unexpected keys: {load_result.unexpected_keys}")
#        
#        model.to(device)
#        model.eval()
#        
#        logger.info("VCDR model loaded successfully!")
#        return model
#        
#    except Exception as e:
#        logger.error(f"Error loading model: {e}")
#        raise

def load_vcdr_model(model_path, device):
    """Load the trained VCDR regression model"""
    logger.info(f"Loading VCDR regression model from: {model_path}")
    
    # Initialize ViT model to match training structure
    model = VCDRViTRegressor(pretrained=False)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract state dict (handle different checkpoint formats)
        state_dict_keys_to_try = ['model_state_dict', 'state_dict', 'model']
        state_dict = None
        
        for key in state_dict_keys_to_try:
            if key in checkpoint:
                state_dict = checkpoint[key]
                logger.info(f"Found state_dict under key: {key}")
                break
        
        if state_dict is None:
            if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
                logger.info("Using checkpoint directly as state_dict")
            else:
                raise ValueError("Could not find valid state_dict in checkpoint")
        
        # The checkpoint should have a flat structure that matches our model.model structure
        # Map keys to the correct structure
        model_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('head.'):
                # Head layers - map to model.head
                model_state_dict[f'model.{key}'] = value
            elif key in ['cls_token', 'pos_embed'] or key.startswith(('patch_embed.', 'blocks.')):
                # ViT backbone layers - map to model.{key}
                model_state_dict[f'model.{key}'] = value
            elif key.startswith('fc_norm.'):
                # Handle fc_norm -> norm mapping
                new_key = key.replace('fc_norm.', 'norm.')
                model_state_dict[f'model.{new_key}'] = value
                logger.info(f"Mapped {key} -> model.{new_key}")
            elif key.startswith('norm.'):
                # Direct norm mapping
                model_state_dict[f'model.{key}'] = value
            else:
                # Other layers - keep as is but add model prefix
                model_state_dict[f'model.{key}'] = value
                logger.warning(f"Unmapped key: {key} -> model.{key}")
        
        # Load weights with the corrected keys
        load_result = model.load_state_dict(model_state_dict, strict=True)
        logger.info(f"Model weights loaded successfully!")
        
        model.to(device)
        model.eval()
        
        logger.info("VCDR ViT model loaded successfully!")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def find_optimal_threshold(y_true, y_scores, metric='youden'):
    """
    Find optimal threshold for binary classification.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        metric: Optimization metric ('youden', 'f1', 'accuracy')
    
    Returns:
        dict: Contains optimal threshold and associated metrics
    """
    thresholds = np.linspace(0.1, 0.9, 100)
    best_threshold = 0.5
    best_score = 0
    threshold_metrics = []
    
    for thresh in thresholds:
        y_pred = (y_scores > thresh).astype(int)
        
        if len(np.unique(y_true)) < 2:
            continue
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        if (tp + fn) == 0 or (tn + fp) == 0:
            continue
            
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        else:
            precision = 0
            f1 = 0
        
        # Calculate optimization metric
        if metric == 'youden':
            score = sensitivity + specificity - 1  # Youden's J statistic
        elif metric == 'f1':
            score = f1
        elif metric == 'accuracy':
            score = accuracy
        else:
            score = accuracy
        
        threshold_metrics.append({
            'threshold': thresh,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'precision': precision,
            'f1': f1,
            'youden': sensitivity + specificity - 1,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return {
        'optimal_threshold': best_threshold,
        'best_score': best_score,
        'metric_used': metric,
        'all_thresholds': threshold_metrics
    }

def create_dataset_specific_plots(df_results, results_dir_path):
    """Create comprehensive dataset-specific analysis plots"""
    logger.info("Creating dataset-specific analysis plots...")
    
    # Get unique datasets
    datasets = sorted([d for d in df_results['dataset_source'].unique() 
                      if str(d).lower() not in ['nan', 'unknown_source', 'none']])
    
    if len(datasets) == 0:
        logger.warning("No valid datasets found for plotting")
        return {}
    
    # Color palette for datasets
    colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
    dataset_colors = dict(zip(datasets, colors))
    
    optimal_thresholds = {}
    
    # 1. Combined VCDR distribution plot
    plt.figure(figsize=(20, 12))
    
    # Plot 1: Overall distribution
    plt.subplot(3, 4, 1)
    normal_all = df_results[df_results['label'] == 0]['vcdr_score']
    glaucoma_all = df_results[df_results['label'] == 1]['vcdr_score']
    
    plt.hist(normal_all, bins=40, alpha=0.6, label=f'Normal (n={len(normal_all)})', 
             color='green', density=True)
    plt.hist(glaucoma_all, bins=40, alpha=0.6, label=f'Glaucoma (n={len(glaucoma_all)})', 
             color='red', density=True)
    plt.xlabel('VCDR Score')
    plt.ylabel('Density')
    plt.title('Overall VCDR Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Box plots by dataset
    plt.subplot(3, 4, 2)
    data_for_box = []
    labels_for_box = []
    positions = []
    
    for i, dataset in enumerate(datasets):
        dataset_data = df_results[df_results['dataset_source'] == dataset]
        normal_data = dataset_data[dataset_data['label'] == 0]['vcdr_score']
        glaucoma_data = dataset_data[dataset_data['label'] == 1]['vcdr_score']
        
        data_for_box.extend([normal_data, glaucoma_data])
        labels_for_box.extend([f'{dataset}\nNormal', f'{dataset}\nGlaucoma'])
        positions.extend([i*2, i*2+1])
    
    box_plot = plt.boxplot(data_for_box, positions=positions, patch_artist=True)
    
    # Color the boxes
    for i, patch in enumerate(box_plot['boxes']):
        if i % 2 == 0:  # Normal
            patch.set_facecolor('lightgreen')
        else:  # Glaucoma
            patch.set_facecolor('lightcoral')
    
    plt.xticks(positions, labels_for_box, rotation=45, ha='right')
    plt.ylabel('VCDR Score')
    plt.title('VCDR Distribution by Dataset')
    plt.grid(True, alpha=0.3)
    
    # Plots 3-6: Individual dataset distributions
    for i, dataset in enumerate(datasets[:4]):  # Show first 4 datasets
        plt.subplot(3, 4, i+3)
        dataset_data = df_results[df_results['dataset_source'] == dataset]
        
        normal_data = dataset_data[dataset_data['label'] == 0]['vcdr_score']
        glaucoma_data = dataset_data[dataset_data['label'] == 1]['vcdr_score']
        
        if len(normal_data) > 0:
            plt.hist(normal_data, bins=20, alpha=0.6, label=f'Normal (n={len(normal_data)})', 
                    color='green', density=True)
        if len(glaucoma_data) > 0:
            plt.hist(glaucoma_data, bins=20, alpha=0.6, label=f'Glaucoma (n={len(glaucoma_data)})', 
                    color='red', density=True)
        
        plt.xlabel('VCDR Score')
        plt.ylabel('Density')
        plt.title(f'{dataset}')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    # Plot 7: ROC curves by dataset
    plt.subplot(3, 4, 7)
    for i, dataset in enumerate(datasets):
        dataset_data = df_results[df_results['dataset_source'] == dataset]
        if len(dataset_data) < 10 or len(dataset_data['label'].unique()) < 2:
            continue
            
        fpr, tpr, _ = roc_curve(dataset_data['label'], dataset_data['vcdr_score'])
        auc_score = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=dataset_colors[dataset], linewidth=2,
                label=f'{dataset} (AUC={auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Dataset')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Dataset statistics
    plt.subplot(3, 4, 8)
    dataset_stats = []
    for dataset in datasets:
        dataset_data = df_results[df_results['dataset_source'] == dataset]
        normal_count = len(dataset_data[dataset_data['label'] == 0])
        glaucoma_count = len(dataset_data[dataset_data['label'] == 1])
        
        dataset_stats.append({
            'Dataset': dataset,
            'Normal': normal_count,
            'Glaucoma': glaucoma_count,
            'Total': normal_count + glaucoma_count,
            'Glaucoma %': (glaucoma_count / (normal_count + glaucoma_count)) * 100 if (normal_count + glaucoma_count) > 0 else 0
        })
    
    stats_df = pd.DataFrame(dataset_stats)
    
    # Create stacked bar chart
    bottom_vals = np.zeros(len(datasets))
    plt.bar(range(len(datasets)), stats_df['Normal'], label='Normal', color='green', alpha=0.7)
    plt.bar(range(len(datasets)), stats_df['Glaucoma'], bottom=stats_df['Normal'], 
            label='Glaucoma', color='red', alpha=0.7)
    
    plt.xticks(range(len(datasets)), datasets, rotation=45, ha='right')
    plt.ylabel('Sample Count')
    plt.title('Sample Distribution by Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plots 9-12: Optimal threshold analysis per dataset
    for i, dataset in enumerate(datasets[:4]):
        plt.subplot(3, 4, i+9)
        dataset_data = df_results[df_results['dataset_source'] == dataset]
        
        if len(dataset_data) < 10 or len(dataset_data['label'].unique()) < 2:
            plt.text(0.5, 0.5, f'Insufficient data\nfor {dataset}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{dataset} - Threshold Analysis')
            continue
        
        # Find optimal threshold
        threshold_analysis = find_optimal_threshold(
            dataset_data['label'], dataset_data['vcdr_score'], metric='youden'
        )
        optimal_thresholds[dataset] = threshold_analysis
        
        # Plot threshold curves
        thresholds = [t['threshold'] for t in threshold_analysis['all_thresholds']]
        accuracies = [t['accuracy'] for t in threshold_analysis['all_thresholds']]
        sensitivities = [t['sensitivity'] for t in threshold_analysis['all_thresholds']]
        specificities = [t['specificity'] for t in threshold_analysis['all_thresholds']]
        
        plt.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
        plt.plot(thresholds, sensitivities, label='Sensitivity', linewidth=2)
        plt.plot(thresholds, specificities, label='Specificity', linewidth=2)
        
        # Mark optimal threshold
        opt_thresh = threshold_analysis['optimal_threshold']
        plt.axvline(x=opt_thresh, color='red', linestyle='--', alpha=0.7,
                   label=f'Optimal ({opt_thresh:.3f})')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'{dataset} - Threshold Analysis')
        plt.legend(fontsize=7)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_path, 'dataset_specific_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional plots for remaining datasets if more than 4
    if len(datasets) > 4:
        remaining_datasets = datasets[4:]
        n_remaining = len(remaining_datasets)
        n_cols = min(4, n_remaining)
        n_rows = (n_remaining + n_cols - 1) // n_cols
        
        plt.figure(figsize=(5*n_cols, 4*n_rows))
        
        for i, dataset in enumerate(remaining_datasets):
            plt.subplot(n_rows, n_cols, i+1)
            dataset_data = df_results[df_results['dataset_source'] == dataset]
            
            normal_data = dataset_data[dataset_data['label'] == 0]['vcdr_score']
            glaucoma_data = dataset_data[dataset_data['label'] == 1]['vcdr_score']
            
            if len(normal_data) > 0:
                plt.hist(normal_data, bins=20, alpha=0.6, label=f'Normal (n={len(normal_data)})', 
                        color='green', density=True)
            if len(glaucoma_data) > 0:
                plt.hist(glaucoma_data, bins=20, alpha=0.6, label=f'Glaucoma (n={len(glaucoma_data)})', 
                        color='red', density=True)
            
            plt.xlabel('VCDR Score')
            plt.ylabel('Density')
            plt.title(f'{dataset}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Find optimal threshold for remaining datasets
            if len(dataset_data) >= 10 and len(dataset_data['label'].unique()) >= 2:
                threshold_analysis = find_optimal_threshold(
                    dataset_data['label'], dataset_data['vcdr_score'], metric='youden'
                )
                optimal_thresholds[dataset] = threshold_analysis
                
                opt_thresh = threshold_analysis['optimal_threshold']
                plt.axvline(x=opt_thresh, color='black', linestyle='--', alpha=0.7,
                           label=f'Opt: {opt_thresh:.3f}')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir_path, 'additional_datasets_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Dataset-specific analysis plots saved. Found optimal thresholds for {len(optimal_thresholds)} datasets.")
    return optimal_thresholds

def create_optimal_threshold_summary(df_results, optimal_thresholds, results_dir_path):
    """Create summary table of optimal thresholds and performance"""
    logger.info("Creating optimal threshold summary...")
    
    summary_data = []
    
    # Overall performance first
    if len(df_results['label'].unique()) >= 2:
        overall_analysis = find_optimal_threshold(
            df_results['label'], df_results['vcdr_score'], metric='youden'
        )
        
        overall_optimal = overall_analysis['optimal_threshold']
        overall_preds = (df_results['vcdr_score'] > overall_optimal).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(df_results['label'], overall_preds).ravel()
        overall_metrics = {
            'Dataset': 'OVERALL',
            'N_Total': len(df_results),
            'N_Normal': len(df_results[df_results['label'] == 0]),
            'N_Glaucoma': len(df_results[df_results['label'] == 1]),
            'Mean_VCDR_Normal': df_results[df_results['label'] == 0]['vcdr_score'].mean(),
            'Mean_VCDR_Glaucoma': df_results[df_results['label'] == 1]['vcdr_score'].mean(),
            'Std_VCDR_Normal': df_results[df_results['label'] == 0]['vcdr_score'].std(),
            'Std_VCDR_Glaucoma': df_results[df_results['label'] == 1]['vcdr_score'].std(),
            'Optimal_Threshold': overall_optimal,
            'Accuracy': (tp + tn) / (tp + tn + fp + fn),
            'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'F1_Score': 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0,
            'AUC': auc(*roc_curve(df_results['label'], df_results['vcdr_score'])[:2])
        }
        summary_data.append(overall_metrics)
    
    # Dataset-specific performance
    datasets = sorted([d for d in df_results['dataset_source'].unique() 
                      if str(d).lower() not in ['nan', 'unknown_source', 'none']])
    
    for dataset in datasets:
        dataset_data = df_results[df_results['dataset_source'] == dataset]
        
        if len(dataset_data) < 10 or len(dataset_data['label'].unique()) < 2:
            continue
        
        # Use optimal threshold if found, otherwise use Youden's J
        if dataset in optimal_thresholds:
            optimal_thresh = optimal_thresholds[dataset]['optimal_threshold']
        else:
            threshold_analysis = find_optimal_threshold(
                dataset_data['label'], dataset_data['vcdr_score'], metric='youden'
            )
            optimal_thresh = threshold_analysis['optimal_threshold']
        
        # Calculate metrics with optimal threshold
        optimal_preds = (dataset_data['vcdr_score'] > optimal_thresh).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(dataset_data['label'], optimal_preds).ravel()
        
        dataset_metrics = {
            'Dataset': dataset,
            'N_Total': len(dataset_data),
            'N_Normal': len(dataset_data[dataset_data['label'] == 0]),
            'N_Glaucoma': len(dataset_data[dataset_data['label'] == 1]),
            'Mean_VCDR_Normal': dataset_data[dataset_data['label'] == 0]['vcdr_score'].mean(),
            'Mean_VCDR_Glaucoma': dataset_data[dataset_data['label'] == 1]['vcdr_score'].mean(),
            'Std_VCDR_Normal': dataset_data[dataset_data['label'] == 0]['vcdr_score'].std(),
            'Std_VCDR_Glaucoma': dataset_data[dataset_data['label'] == 1]['vcdr_score'].std(),
            'Optimal_Threshold': optimal_thresh,
            'Accuracy': (tp + tn) / (tp + tn + fp + fn),
            'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'F1_Score': 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0,
            'AUC': auc(*roc_curve(dataset_data['label'], dataset_data['vcdr_score'])[:2])
        }
        summary_data.append(dataset_metrics)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Round numerical columns
    numerical_cols = ['Mean_VCDR_Normal', 'Mean_VCDR_Glaucoma', 'Std_VCDR_Normal', 'Std_VCDR_Glaucoma',
                     'Optimal_Threshold', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1_Score', 'AUC']
    for col in numerical_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(4)
    
    # Save to CSV
    summary_df.to_csv(os.path.join(results_dir_path, 'optimal_thresholds_summary.csv'), index=False)
    
    # Create a formatted table plot
    plt.figure(figsize=(20, max(8, len(summary_df) * 0.5)))
    
    # Create table
    ax = plt.subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Select key columns for display
    display_cols = ['Dataset', 'N_Total', 'N_Glaucoma', 'Mean_VCDR_Normal', 'Mean_VCDR_Glaucoma', 
                   'Optimal_Threshold', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC']
    display_df = summary_df[display_cols].copy()
    
    # Rename columns for better display
    display_df.columns = ['Dataset', 'Total N', 'Glaucoma N', 'Mean VCDR (Normal)', 'Mean VCDR (Glaucoma)',
                         'Optimal Threshold', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC']
    
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color the header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color the overall row if present
    if len(summary_df) > 0 and summary_df.iloc[0]['Dataset'] == 'OVERALL':
        for i in range(len(display_df.columns)):
            table[(1, i)].set_facecolor('#d4edda')
    
    plt.title('VCDR Model Performance Summary - Optimal Thresholds by Dataset', 
             fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(results_dir_path, 'optimal_thresholds_table.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary to console
    logger.info("\n" + "="*80)
    logger.info("OPTIMAL THRESHOLD ANALYSIS SUMMARY")
    logger.info("="*80)
    
    for _, row in summary_df.iterrows():
        if row['Dataset'] == 'OVERALL':
            logger.info(f"\n{row['Dataset']}:")
            logger.info(f"  Samples: {row['N_Total']} (Normal: {row['N_Normal']}, Glaucoma: {row['N_Glaucoma']})")
            logger.info(f"  Mean VCDR - Normal: {row['Mean_VCDR_Normal']:.3f}, Glaucoma: {row['Mean_VCDR_Glaucoma']:.3f}")
            logger.info(f"  Optimal Threshold: {row['Optimal_Threshold']:.3f}")
            logger.info(f"  Performance - AUC: {row['AUC']:.3f}, Accuracy: {row['Accuracy']:.3f}")
            logger.info(f"  Sensitivity: {row['Sensitivity']:.3f}, Specificity: {row['Specificity']:.3f}")
    
    logger.info(f"\nDATASET-SPECIFIC RESULTS:")
    for _, row in summary_df.iterrows():
        if row['Dataset'] != 'OVERALL':
            logger.info(f"\n{row['Dataset']}:")
            logger.info(f"  Samples: {row['N_Total']} (Glaucoma: {row['N_Glaucoma']})")
            logger.info(f"  Optimal Threshold: {row['Optimal_Threshold']:.3f} (vs global)")
            logger.info(f"  AUC: {row['AUC']:.3f}, Accuracy: {row['Accuracy']:.3f}")
    
    logger.info("="*80)
    
    return summary_df

def evaluate_vcdr_on_test_set(
    model_to_eval: nn.Module,
    loader: DataLoader,
    dataset_obj: GlaucomaSubgroupDataset,
    dataset_name_suffix: str,
    results_dir_path: str,
    epoch_num_str: str,
    device_to_use: torch.device,
    vcdr_threshold: float = 0.6,
    plot_per_source_roc: bool = True,
    roc_min_samples_per_source: int = 50
) -> tuple[dict | None, pd.DataFrame, dict | None]:
    """
    Evaluates VCDR regression model on test set, converting VCDR scores to binary predictions.
    Adapted from your evaluate_on_test_set function.
    """
    if not loader or not dataset_obj or len(dataset_obj) == 0:
        logger.info(f"Skipping evaluation for {dataset_name_suffix} (loader or dataset is None or empty).")
        return None, pd.DataFrame(), None

    logger.info(f"\n--- Evaluating VCDR Model on {dataset_name_suffix} ({len(dataset_obj)} samples) ---")
    logger.info(f"Using VCDR threshold: {vcdr_threshold} for binary classification")
    
    model_to_eval.eval()
    all_labels, all_vcdr_scores, all_binary_predictions, all_sources_eval, all_image_paths_eval = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{dataset_name_suffix} Evaluation Progress"):
            if batch is None: 
                continue

            inputs, labels, metadata_dict = batch[0], batch[1], batch[2] if len(batch) > 2 else {}

            # Handle image paths (ensure it's a list of strings)
            img_paths_batch = metadata_dict.get('image_path', ['Unknown_Path'] * len(labels))
            if not isinstance(img_paths_batch, list):
                img_paths_batch = [str(img_paths_batch)] * len(labels)
            all_image_paths_eval.extend(img_paths_batch)

            inputs = inputs.to(device_to_use)
            
            # Get VCDR predictions from regression model
            vcdr_outputs = model_to_eval(inputs)
            vcdr_scores = vcdr_outputs.cpu().numpy().flatten()
            
            # Convert VCDR scores to binary predictions using threshold
            binary_predictions = (vcdr_scores > vcdr_threshold).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_vcdr_scores.extend(vcdr_scores)
            all_binary_predictions.extend(binary_predictions)

            # Handle dataset sources (ensure it's a list of strings)
            sources_batch = metadata_dict.get('dataset_source', ['Unknown_Source'] * len(labels))
            if not isinstance(sources_batch, list):
                sources_batch = [str(sources_batch)] * len(labels)
            else:
                sources_batch = [str(s) for s in sources_batch]
            all_sources_eval.extend(sources_batch)

    if not all_labels:
        logger.warning(f"No labels collected during evaluation for {dataset_name_suffix}. Skipping results processing.")
        return None, pd.DataFrame(), None

    # Create DataFrame with results (using VCDR scores as probabilities for compatibility)
    df_res = pd.DataFrame({
        "image_path": all_image_paths_eval,
        "label": all_labels,
        "vcdr_score": all_vcdr_scores,
        "binary_prediction": all_binary_predictions,
        "probability_class1": all_vcdr_scores,  # Use VCDR scores as probability for ROC analysis
        "dataset_source": all_sources_eval
    })
    
    # Save raw results to CSV
    results_csv_filename = f"vcdr_test_{dataset_name_suffix.lower().replace(' ', '_')}_results_epoch{epoch_num_str}.csv"
    df_res.to_csv(os.path.join(results_dir_path, results_csv_filename), index=False)
    logger.info(f"VCDR evaluation results for {dataset_name_suffix} saved to {results_csv_filename}")

    # CREATE DATASET-SPECIFIC ANALYSIS
    try:
        logger.info("Performing dataset-specific threshold analysis...")
        optimal_thresholds = create_dataset_specific_plots(df_res, results_dir_path)
        summary_df = create_optimal_threshold_summary(df_res, optimal_thresholds, results_dir_path)
    except Exception as e:
        logger.error(f"Error in dataset-specific analysis: {e}", exc_info=True)
        optimal_thresholds = {}

    metrics_summary = {}
    overall_roc_data_for_combined_plot = None

    try:
        # Calculate overall metrics
        if len(np.unique(df_res["label"])) < 2:
            logger.warning(f"Overall {dataset_name_suffix} dataset has only one class. AUC cannot be computed.")
            accuracy = accuracy_score(df_res["label"], df_res["binary_prediction"]) if len(df_res["label"]) > 0 else np.nan
            auc_value = np.nan
            fpr_overall, tpr_overall = None, None
        else:
            # Use binary predictions for accuracy, VCDR scores for ROC
            accuracy = accuracy_score(df_res["label"], df_res["binary_prediction"])
            fpr_overall, tpr_overall, _ = roc_curve(df_res["label"], df_res["probability_class1"])
            auc_value = auc(fpr_overall, tpr_overall)
            overall_roc_data_for_combined_plot = {'fpr': fpr_overall, 'tpr': tpr_overall, 'auc': auc_value}

        # Additional detailed metrics
        if len(np.unique(df_res["label"])) >= 2:
            precision, recall, _ = precision_recall_curve(df_res["label"], df_res["probability_class1"])
            avg_precision = average_precision_score(df_res["label"], df_res["probability_class1"])
            
            tn, fp, fn, tp = confusion_matrix(df_res["label"], df_res["binary_prediction"]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        else:
            avg_precision = sensitivity = specificity = ppv = npv = np.nan
            tn = fp = fn = tp = 0

        auc_log_str = f"{auc_value:.4f}" if not np.isnan(auc_value) else "N/A"
        avg_precision_str = f"{avg_precision:.4f}" if not np.isnan(avg_precision) else "N/A"
        logger.info(f"{dataset_name_suffix} VCDR Model - Accuracy: {accuracy:.4f}, AUC: {auc_log_str}, "
                   f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
        logger.info(f"VCDR Threshold used: {vcdr_threshold}, Avg Precision: {avg_precision_str}")
        
        metrics_summary["overall"] = {
            "accuracy": accuracy, 
            "auc": auc_value, 
            "average_precision": avg_precision,
            "sensitivity": sensitivity,
            "specificity": specificity, 
            "ppv": ppv,
            "npv": npv,
            "vcdr_threshold": vcdr_threshold,
            "num_samples": len(df_res),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "optimal_thresholds": optimal_thresholds
        }

        # Create enhanced plots - wrap in try/except to not fail the whole evaluation
        try:
            plt.figure(figsize=(15, 10))
            
            # Main ROC plot
            plt.subplot(2, 2, 1)
            if fpr_overall is not None and tpr_overall is not None:
                plot_label = f'VCDR Model {dataset_name_suffix} (AUC={auc_value:.3f})' if not np.isnan(auc_value) else f'VCDR Model {dataset_name_suffix} (AUC=N/A)'
                plt.plot(fpr_overall, tpr_overall, color='darkorange', lw=2.5, label=plot_label)

            # Plot per-source ROC curves if enabled
            if plot_per_source_roc and 'dataset_source' in df_res.columns:
                unique_sources = sorted([s for s in df_res["dataset_source"].astype(str).unique() if s.lower() not in ['nan', 'unknown_source', 'none']])
                if unique_sources:
                    cmap_name = 'tab10' if len(unique_sources) <= 10 else 'tab20'
                    colors_for_sources = plt.cm.get_cmap(cmap_name, len(unique_sources))
                    for i, src_name in enumerate(unique_sources):
                        src_df = df_res[df_res["dataset_source"].astype(str) == src_name]
                        if len(src_df) < roc_min_samples_per_source or len(src_df["label"].unique()) < 2:
                            logger.info(f"Skipping ROC for source '{src_name}' (N={len(src_df)}), less than {roc_min_samples_per_source} samples or <2 unique classes.")
                            continue

                        fpr_s, tpr_s, _ = roc_curve(src_df["label"], src_df["probability_class1"])
                        auc_s = auc(fpr_s, tpr_s)
                        src_accuracy = accuracy_score(src_df["label"], src_df["binary_prediction"])
                        metrics_summary[src_name] = {
                            "accuracy": src_accuracy, 
                            "auc": auc_s, 
                            "num_samples": len(src_df),
                            "vcdr_threshold": vcdr_threshold
                        }
                        plt.plot(fpr_s, tpr_s, color=colors_for_sources(i), lw=1.5, linestyle='--',
                                 label=f'{src_name} (AUC={auc_s:.3f}, N={len(src_df)})')

            plt.plot([0, 1], [0, 1], 'k:', lw=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate (FPR)")
            plt.ylabel("True Positive Rate (TPR)")
            plt.title(f'VCDR Model ROC Curves (Threshold={vcdr_threshold})')
            plt.legend(loc="lower right", fontsize='small')
            plt.grid(True, alpha=0.7)

            # VCDR Score Distribution
            plt.subplot(2, 2, 2)
            normal_scores = df_res[df_res["label"] == 0]["vcdr_score"]
            glaucoma_scores = df_res[df_res["label"] == 1]["vcdr_score"]
            
            if len(normal_scores) > 0:
                plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='green', density=True)
            if len(glaucoma_scores) > 0:
                plt.hist(glaucoma_scores, bins=30, alpha=0.7, label='Glaucoma', color='red', density=True)
            plt.axvline(x=vcdr_threshold, color='black', linestyle='--', label=f'Threshold ({vcdr_threshold})')
            plt.xlabel('VCDR Score')
            plt.ylabel('Density')
            plt.title('VCDR Score Distribution')
            plt.legend()
            plt.grid(True, alpha=0.7)

            # Confusion Matrix
            plt.subplot(2, 2, 3)
            cm = confusion_matrix(df_res["label"], df_res["binary_prediction"])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix (Threshold={vcdr_threshold})')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            # Performance vs Threshold
            plt.subplot(2, 2, 4)
            thresholds = np.linspace(0.3, 0.9, 50)
            accuracies, sensitivities, specificities = [], [], []
            
            for thresh in thresholds:
                preds = (df_res["probability_class1"] > thresh).astype(int)
                
                if len(np.unique(df_res["label"])) >= 2:
                    tn_t, fp_t, fn_t, tp_t = confusion_matrix(df_res["label"], preds).ravel()
                    
                    acc_t = (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t)
                    sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
                    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
                else:
                    acc_t = sens_t = spec_t = np.nan
                
                accuracies.append(acc_t)
                sensitivities.append(sens_t)
                specificities.append(spec_t)
            
            plt.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
            plt.plot(thresholds, sensitivities, label='Sensitivity', linewidth=2)
            plt.plot(thresholds, specificities, label='Specificity', linewidth=2)
            plt.axvline(x=vcdr_threshold, color='black', linestyle='--', alpha=0.7, 
                       label=f'Current Threshold ({vcdr_threshold})')
            plt.xlabel('VCDR Threshold')
            plt.ylabel('Score')
            plt.title('Performance vs Threshold')
            plt.legend()
            plt.grid(True, alpha=0.7)

            plt.tight_layout()
            roc_plot_filename = f'vcdr_test_{dataset_name_suffix.lower().replace(" ", "_")}_analysis_epoch{epoch_num_str}.png'
            plt.savefig(os.path.join(results_dir_path, roc_plot_filename), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"VCDR analysis plots for {dataset_name_suffix} saved to {roc_plot_filename}")
        except Exception as plot_error:
            logger.error(f"Error creating plots for {dataset_name_suffix}: {plot_error}", exc_info=True)
            logger.info("Continuing without plots...")

        # Save metrics summary to JSON - also wrap in try/except
        try:
            metrics_json_filename = f'vcdr_test_{dataset_name_suffix.lower().replace(" ", "_")}_metrics_epoch{epoch_num_str}.json'
            with open(os.path.join(results_dir_path, metrics_json_filename), 'w') as f_json:
                json.dump(metrics_summary, f_json, indent=4, cls=NpEncoder)
            logger.info(f"VCDR metrics summary for {dataset_name_suffix} saved to {metrics_json_filename}")
        except Exception as json_error:
            logger.error(f"Error saving metrics JSON for {dataset_name_suffix}: {json_error}")

        return metrics_summary, df_res, overall_roc_data_for_combined_plot
        
    except Exception as e:
        logger.error(f"Error during VCDR metrics calculation or plotting for {dataset_name_suffix}: {e}", exc_info=True)
        
        # Still return basic metrics even if plotting fails
        basic_metrics = {
            "overall": {
                "accuracy": accuracy_score(df_res["label"], df_res["binary_prediction"]) if len(df_res) > 0 else np.nan,
                "vcdr_threshold": vcdr_threshold,
                "num_samples": len(df_res),
                "error": str(e)
            }
        }
        return basic_metrics, df_res, None

def main(args: argparse.Namespace):
    """
    Main function to orchestrate VCDR model evaluation.
    Now supports both manifest-based and external dataset evaluation.
    """
    start_time = datetime.now()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load the VCDR model
    model = load_vcdr_model(args.model_path, device)

    # Setup output directory
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    eval_output_base_dir = os.path.join(args.output_dir, f"vcdr_evaluation_{timestamp}")
    os.makedirs(eval_output_base_dir, exist_ok=True)

    # Setup file logging
    log_file_path = os.path.join(eval_output_base_dir, f"vcdr_evaluation_{timestamp}.log")
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s"
    ))
    logger.addHandler(file_handler)

    try:
        # Create eval transforms
        eval_transforms = get_eval_transforms(args.image_size, 'vit_base_patch16_224')
        
        # Prepare datasets to evaluate
        datasets_to_evaluate = {}
        
        # Load manifest-based test set if provided
        if args.test_manifest_path and os.path.exists(args.test_manifest_path):
            try:
                df_test_manifest = pd.read_csv(args.test_manifest_path)
                if {'image_path', 'types'}.issubset(df_test_manifest.columns) and not df_test_manifest.empty:
                    datasets_to_evaluate["ID-Test"] = df_test_manifest
                    logger.info(f"Loaded test manifest with {len(df_test_manifest)} samples")
                else:
                    logger.warning("Test set manifest is empty or missing required columns ('image_path', 'types').")
            except Exception as e:
                logger.error(f"Error loading test set manifest: {e}")
        
        # Load external test datasets if enabled
        if args.use_external_datasets:
            # Correct CHAKSU path handling (matching your working script)
            chaksu_base_dir_for_loader = args.chaksu_base_dir_eval
            if args.data_type == PROCESSED_DIR_NAME_CONST:
                # Use the adjust_path_for_data_type function with correct parameter names
                chaksu_base_dir_for_loader = adjust_path_for_data_type(
                    current_path=args.chaksu_base_dir_eval,  # Changed from relative_path_part
                    data_type='raw',                   # Changed from target_data_type
                    base_data_dir=args.base_data_root,       # Changed from base_data_dir_abs
                    raw_dir_name=RAW_DIR_NAME_CONST,         # Changed from raw_subdir_name
                    processed_dir_name=PROCESSED_DIR_NAME_CONST  # Changed from processed_subdir_name
                )
                logger.info(f"Adjusted CHAKSU Base Dir for 'raw' evaluation: {chaksu_base_dir_for_loader}")
            
            external_test_sets_map = load_external_test_data(
                smdg_metadata_file_raw=args.smdg_metadata_file_raw,
                smdg_image_dir_raw=args.smdg_image_dir_raw,
                chaksu_base_dir_eval=chaksu_base_dir_for_loader,
                chaksu_decision_dir_raw=args.chaksu_decision_dir_raw,
                chaksu_metadata_dir_raw=args.chaksu_metadata_dir_raw,
                data_type=args.data_type,
                base_data_root=args.base_data_root,
                raw_dir_name=RAW_DIR_NAME_CONST,
                processed_dir_name=PROCESSED_DIR_NAME_CONST,
                eval_papilla=args.eval_papilla,
                eval_oiaodir_test=args.eval_oiaodir_test,
                eval_chaksu=args.eval_chaksu
            )
            
            # Add non-empty external datasets
            for dataset_name, df_external in external_test_sets_map.items():
                if not df_external.empty:
                    datasets_to_evaluate[dataset_name] = df_external
                    logger.info(f"Loaded external dataset '{dataset_name}' with {len(df_external)} samples")
                else:
                    logger.warning(f"External dataset '{dataset_name}' is empty, skipping")
        
        # Check if we have any datasets to evaluate
        if not datasets_to_evaluate:
            logger.error("No datasets loaded for evaluation. Exiting.")
            return
        
        logger.info(f"Will evaluate on {len(datasets_to_evaluate)} dataset(s): {list(datasets_to_evaluate.keys())}")
        
        # Evaluate each dataset
        all_results = {}
        combined_results = []
        
        for dataset_name, df_dataset in datasets_to_evaluate.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"EVALUATING DATASET: {dataset_name}")
            logger.info(f"{'='*60}")
            
            # Check if dataset has enough samples for meaningful evaluation
            min_samples_for_evaluation = 50  # You can make this configurable
            if len(df_dataset) < min_samples_for_evaluation:
                logger.warning(f"Dataset '{dataset_name}' has only {len(df_dataset)} samples "
                              f"(< {min_samples_for_evaluation}). Skipping individual evaluation.")
                logger.info(f"Adding '{dataset_name}' to combined analysis only.")
                
                # Still add to combined analysis but skip individual evaluation
                df_dataset_copy = df_dataset.copy()
                df_dataset_copy['evaluation_dataset'] = dataset_name
                combined_results.append(df_dataset_copy)
                continue
            
            # Prepare dataset
            attributes_to_load = ['dataset_source', 'image_path']
            sensitive_attributes_in_manifest = [attr for attr in attributes_to_load if attr in df_dataset.columns]
            if 'image_path' in df_dataset.columns and 'image_path' not in sensitive_attributes_in_manifest:
                sensitive_attributes_in_manifest.append('image_path')

            test_dataset = GlaucomaSubgroupDataset(
                df_dataset,
                transform=eval_transforms,
                sensitive_attributes=sensitive_attributes_in_manifest,
                require_attributes=False
            )
            
            if len(test_dataset) == 0:
                logger.warning(f"Dataset '{dataset_name}' is empty after initialization. Skipping.")
                continue

            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                collate_fn=safe_collate, persistent_workers=(args.num_workers > 0)
            )

            logger.info(f"Created dataset '{dataset_name}' with {len(test_dataset)} samples")

            # Perform VCDR evaluation
            metrics_summary, df_results, overall_roc_data = evaluate_vcdr_on_test_set(
                model, test_loader, test_dataset,
                f"{dataset_name}_VCDR_Model", eval_output_base_dir,
                f"{dataset_name.lower()}", device, args.vcdr_threshold, 
                args.plot_roc_per_source, args.roc_min_samples_per_source
            )

            # Store results
            if metrics_summary is not None and not df_results.empty:
                all_results[dataset_name] = {
                    'metrics': metrics_summary,
                    'results_df': df_results,
                    'roc_data': overall_roc_data
                }
                
                # Add dataset identifier and collect for combined analysis
                df_results_copy = df_results.copy()
                df_results_copy['evaluation_dataset'] = dataset_name
                combined_results.append(df_results_copy)
                
                # Print classification report for this dataset
                if len(np.unique(df_results["label"])) >= 2:
                    logger.info(f"\n{dataset_name} CLASSIFICATION REPORT:")
                    logger.info("="*50)
                    report = classification_report(df_results["label"], df_results["binary_prediction"], 
                                                 target_names=['Normal', 'Glaucoma'])
                    logger.info(f"\n{report}")
                    
                    # Save individual classification report
                    with open(os.path.join(eval_output_base_dir, f"{dataset_name}_vcdr_classification_report.txt"), 'w') as f:
                        f.write(f"VCDR Regression Model - {dataset_name} Results\n")
                        f.write("="*60 + "\n\n")
                        f.write(f"Model: {args.model_path}\n")
                        f.write(f"Dataset: {dataset_name}\n")
                        f.write(f"VCDR Threshold: {args.vcdr_threshold}\n")
                        f.write(f"Evaluation Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        f.write("Classification Report:\n")
                        f.write(report)
                        f.write("\n\nDetailed Metrics:\n")
                        if 'overall' in metrics_summary:
                            for metric, value in metrics_summary['overall'].items():
                                if metric != 'optimal_thresholds':
                                    f.write(f"{metric}: {value}\n")
            else:
                logger.error(f"VCDR evaluation failed for dataset '{dataset_name}'")
        
        # Combined analysis if we have multiple datasets
        if len(combined_results) > 1:
            logger.info(f"\n{'='*60}")
            logger.info("COMBINED ANALYSIS ACROSS ALL DATASETS")
            logger.info(f"{'='*60}")
            
            df_combined = pd.concat(combined_results, ignore_index=True)
            
            # Create combined analysis plots
            try:
                optimal_thresholds_combined = create_dataset_specific_plots(df_combined, eval_output_base_dir)
                summary_df_combined = create_optimal_threshold_summary(df_combined, optimal_thresholds_combined, eval_output_base_dir)
                
                # Save combined results
                df_combined.to_csv(os.path.join(eval_output_base_dir, "combined_all_datasets_results.csv"), index=False)
                logger.info("Combined analysis completed and saved")
                
            except Exception as e:
                logger.error(f"Error in combined analysis: {e}", exc_info=True)
        
        # Final summary across all datasets
        logger.info(f"\n{'='*60}")
        logger.info("FINAL SUMMARY - ALL DATASETS")
        logger.info(f"{'='*60}")
        
        for dataset_name, result_data in all_results.items():
            if 'overall' in result_data['metrics']:
                metrics = result_data['metrics']['overall']
                logger.info(f"\n{dataset_name}:")
                logger.info(f"  Samples: {metrics.get('num_samples', 0)}")
                logger.info(f"  VCDR Threshold: {args.vcdr_threshold}")
                logger.info(f"  AUC-ROC: {metrics.get('auc', 0):.3f}")
                logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
                logger.info(f"  Sensitivity: {metrics.get('sensitivity', 0):.3f}")
                logger.info(f"  Specificity: {metrics.get('specificity', 0):.3f}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up file handler
        if 'file_handler' in locals():
            logger.removeHandler(file_handler)
            file_handler.close()

    # Calculate total execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    logger.info(f"\nTotal execution time: {execution_time}")
    logger.info(f"Evaluation completed successfully!")
    logger.info(f"Results saved to: {eval_output_base_dir}")

def parse_arguments():
    """Parse command line arguments for VCDR model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate VCDR regression model for binary glaucoma detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the trained VCDR regression model checkpoint"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save evaluation results"
    )
    
    # Test data options - make manifest optional when using external datasets
    parser.add_argument(
        "--test_manifest_path", 
        type=str, 
        default=None,
        help="Path to the test set manifest CSV file (optional if using external datasets)"
    )
    
    # External dataset arguments
    parser.add_argument(
        "--use_external_datasets",
        action="store_true",
        help="Load external test datasets (PAPILLA, OIA-ODIR, CHAKSU)"
    )
    parser.add_argument(
        "--base_data_root",
        type=str,
        default="D:/glaucoma/data",
        help="Base data root directory for external datasets"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["raw", "processed"],
        default="processed",
        help="Data type to use for external datasets"
    )
    
    # SMDG/PAPILLA/OIA-ODIR arguments (corrected paths to match your working script)
    parser.add_argument(
        "--smdg_metadata_file_raw",
        type=str,
        default=os.path.join('raw','SMDG-19','metadata - standardized.csv'),
        help="Path to SMDG metadata file (relative to base_data_root)"
    )
    parser.add_argument(
        "--smdg_image_dir_raw",
        type=str,
        default=os.path.join('raw','SMDG-19','full-fundus','full-fundus'),
        help="Path to SMDG image directory (relative to base_data_root)"
    )
    parser.add_argument(
        "--eval_papilla",
        action="store_true",
        help="Evaluate on PAPILLA dataset"
    )
    parser.add_argument(
        "--eval_oiaodir_test",
        action="store_true",
        help="Evaluate on OIA-ODIR test dataset"
    )
    
    # CHAKSU arguments (corrected paths to match your working script)
    parser.add_argument(
        "--chaksu_base_dir_eval",
        type=str,
        default=os.path.join('raw','Chaksu','Train','Train','1.0_Original_Fundus_Images'),
        help="CHAKSU base directory for evaluation (relative to base_data_root)"
    )
    parser.add_argument(
        "--chaksu_decision_dir_raw",
        type=str,
        default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision'),
        help="CHAKSU decision directory (relative to base_data_root)"
    )
    parser.add_argument(
        "--chaksu_metadata_dir_raw",
        type=str,
        default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision','Majority'),
        help="CHAKSU metadata directory (relative to base_data_root)"
    )
    parser.add_argument(
        "--eval_chaksu",
        action="store_true",
        help="Evaluate on CHAKSU dataset"
    )
    
    # Model and evaluation parameters
    parser.add_argument(
        "--vcdr_threshold", 
        type=float, 
        default=0.6,
        help="VCDR threshold for binary classification (glaucoma if VCDR > threshold)"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=224,
        help="Input image size for the model"
    )
    
    # Data loading parameters
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of worker processes for data loading"
    )
    
    # Plotting and analysis options
    parser.add_argument(
        "--plot_roc_per_source", 
        action="store_true",
        help="Plot ROC curves for each dataset source"
    )
    parser.add_argument(
        "--roc_min_samples_per_source", 
        type=int, 
        default=50,
        help="Minimum samples required per source to plot ROC curve"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()



def validate_arguments(args):
    """Validate parsed arguments."""
    # Check if model path exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    # Check test data sources
    if not args.use_external_datasets and not args.test_manifest_path:
        logger.error("Either --test_manifest_path or --use_external_datasets must be specified")
        sys.exit(1)
    
    if args.test_manifest_path and not os.path.exists(args.test_manifest_path):
        logger.error(f"Test manifest path does not exist: {args.test_manifest_path}")
        sys.exit(1)
    
    # Validate external dataset arguments
    if args.use_external_datasets:
        if not args.base_data_root:
            logger.error("--base_data_root is required when using external datasets")
            sys.exit(1)
        
        if not any([args.eval_papilla, args.eval_oiaodir_test, args.eval_chaksu]):
            logger.error("At least one external dataset (--eval_papilla, --eval_oiaodir_test, --eval_chaksu) must be enabled")
            sys.exit(1)
        
        # Check SMDG requirements for PAPILLA/OIA-ODIR
        if (args.eval_papilla or args.eval_oiaodir_test):
            if not args.smdg_metadata_file_raw or not args.smdg_image_dir_raw:
                logger.error("--smdg_metadata_file_raw and --smdg_image_dir_raw are required for PAPILLA/OIA-ODIR evaluation")
                sys.exit(1)
        
        # Check CHAKSU requirements
        if args.eval_chaksu:
            if not all([args.chaksu_base_dir_eval, args.chaksu_decision_dir_raw, args.chaksu_metadata_dir_raw]):
                logger.error("CHAKSU arguments (--chaksu_base_dir_eval, --chaksu_decision_dir_raw, --chaksu_metadata_dir_raw) are required for CHAKSU evaluation")
                sys.exit(1)
    
    # Validate other parameters
    if not (0.0 <= args.vcdr_threshold <= 1.0):
        logger.error(f"VCDR threshold must be between 0.0 and 1.0, got: {args.vcdr_threshold}")
        sys.exit(1)
    
    if args.image_size <= 0:
        logger.error(f"Image size must be positive, got: {args.image_size}")
        sys.exit(1)
    
    if args.batch_size <= 0:
        logger.error(f"Batch size must be positive, got: {args.batch_size}")
        sys.exit(1)
    
    if args.num_workers < 0:
        logger.error(f"Number of workers must be non-negative, got: {args.num_workers}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create output directory {args.output_dir}: {e}")
        sys.exit(1)
    
    logger.info("All arguments validated successfully")


if __name__ == "__main__":
    """Main entry point for the VCDR evaluation script."""
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)
        
        # Log script start
        logger.info("="*60)
        logger.info("VCDR REGRESSION MODEL EVALUATION SCRIPT")
        logger.info("="*60)
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Test Manifest: {args.test_manifest_path}")
        logger.info(f"Output Directory: {args.output_dir}")
        logger.info(f"VCDR Threshold: {args.vcdr_threshold}")
        logger.info(f"Image Size: {args.image_size}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Random Seed: {args.seed}")
        logger.info("="*60)
        
        # Run main evaluation
        main(args)
        
    except KeyboardInterrupt:
        logger.info("\nScript interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in main script: {e}", exc_info=True)
        sys.exit(1)
