#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Analysis Script for Glaucoma Classification Project.

This script loads and analyzes all datasets used in the glaucoma classification project,
including training, validation, test, and external OOD datasets. It generates comprehensive
statistics and summary tables for:
1. Sample counts per dataset and split
2. Glaucoma prevalence (positive class distribution)
3. Additional metadata analysis (age, eye, camera, etc.)
4. Data quality metrics
5. Comparative analysis across datasets

The script produces:
- Comprehensive summary tables (CSV and visual)
- Distribution plots and visualizations
- Statistical comparisons between datasets
- Data quality reports
"""

import argparse
import json
import logging
import os
import sys
import importlib.util
from datetime import datetime
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tqdm import tqdm
import cv2
import random

# Ensure custom modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.utils.helpers import NpEncoder, set_seed
    from src.data.utils import adjust_path_for_data_type
    from src.data.external_loader import load_external_test_data
    
    # Import U-Net model and metrics
    from src.models.segmentation.unet import MultiTaskUNet, OpticDiscCupPredictor
    from src.features.metrics import GlaucomaMetrics
    
    # Import from train_classification.py
    import sys
    import importlib.util
    train_classification_path = os.path.join(os.path.dirname(__file__), "train_classification.py")
    spec = importlib.util.spec_from_file_location("train_classification", train_classification_path)
    train_classification = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_classification)
    load_and_split_data = train_classification.load_and_split_data
    load_chaksu_data = train_classification.load_chaksu_data
    load_airogs_data = train_classification.load_airogs_data
    assign_dataset_source = train_classification.assign_dataset_source
    PAPILLA_PREFIX = train_classification.PAPILLA_PREFIX
    
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Current sys.path:", sys.path)
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

# Global constants
RAW_DIR_NAME_CONST = "raw"
PROCESSED_DIR_NAME_CONST = "processed"

class DatasetAnalyzer:
    """Class to handle comprehensive dataset analysis."""
    
    def __init__(self, output_dir: str, unet_model_path: str = None):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary_data = []
        self.vcdr_results = []  # Store vCDR analysis results
        
        # U-Net segmentation setup
        self.unet_model_path = unet_model_path
        self.predictor = None
        self.metrics_calculator = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Initialize U-Net if model path provided
        if self.unet_model_path and os.path.exists(self.unet_model_path):
            try:
                logger.info(f"Initializing U-Net predictor with model: {self.unet_model_path}")
                self.predictor = OpticDiscCupPredictor(model_path=self.unet_model_path)
                self.metrics_calculator = GlaucomaMetrics()
                logger.info("U-Net predictor and metrics calculator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize U-Net predictor: {e}")
                self.predictor = None
                self.metrics_calculator = None
        else:
            logger.info("No U-Net model path provided or file doesn't exist. Skipping vCDR analysis.")
        
    def analyze_dataframe(self, df: pd.DataFrame, dataset_name: str, split: str = "Unknown") -> dict:
        """Analyze a single DataFrame and return statistics."""
        if df is None or df.empty:
            return {
                'dataset_name': dataset_name,
                'split': split,
                'total_samples': 0,
                'glaucoma_positive': 0,
                'glaucoma_negative': 0,
                'glaucoma_prevalence': 0.0,
                'missing_labels': 0,
                'has_age': False,
                'has_eye': False,
                'has_camera': False,
                'age_mean': None,
                'age_std': None,
                'unique_eyes': None,
                'unique_cameras': None,
                'error': 'Empty or None DataFrame'
            }
        
        # Basic sample counts
        total_samples = len(df)
        
        # Find label column (could be 'label', 'types', 'glaucoma', etc.)
        label_col = None
        for col in ['types', 'label', 'glaucoma', 'diagnosis']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            logger.warning(f"No label column found in {dataset_name} - {split}")
            return {
                'dataset_name': dataset_name,
                'split': split,
                'total_samples': total_samples,
                'glaucoma_positive': 0,
                'glaucoma_negative': 0,
                'glaucoma_prevalence': 0.0,
                'missing_labels': total_samples,
                'has_age': 'age' in df.columns,
                'has_eye': 'eye' in df.columns,
                'has_camera': 'camera' in df.columns,
                'age_mean': None,
                'age_std': None,
                'unique_eyes': None,
                'unique_cameras': None,
                'error': 'No label column found'
            }
        
        # Analyze labels
        labels = df[label_col].dropna()
        missing_labels = total_samples - len(labels)
        
        # Count positive/negative cases (assuming 1=glaucoma, 0=normal)
        glaucoma_positive = np.sum(labels == 1)
        glaucoma_negative = np.sum(labels == 0)
        glaucoma_prevalence = glaucoma_positive / len(labels) if len(labels) > 0 else 0.0
        
        # Analyze metadata
        has_age = 'age' in df.columns
        has_eye = 'eye' in df.columns
        has_camera = 'camera' in df.columns
        
        age_mean = df['age'].mean() if has_age else None
        age_std = df['age'].std() if has_age else None
        
        unique_eyes = list(df['eye'].unique()) if has_eye else None
        unique_cameras = list(df['camera'].unique()) if has_camera else None
        
        return {
            'dataset_name': dataset_name,
            'split': split,
            'total_samples': total_samples,
            'glaucoma_positive': glaucoma_positive,
            'glaucoma_negative': glaucoma_negative,
            'glaucoma_prevalence': glaucoma_prevalence,
            'missing_labels': missing_labels,
            'has_age': has_age,
            'has_eye': has_eye,
            'has_camera': has_camera,
            'age_mean': age_mean,
            'age_std': age_std,
            'unique_eyes': unique_eyes,
            'unique_cameras': unique_cameras,
            'error': None
        }
    
    def add_dataset_analysis(self, df: pd.DataFrame, dataset_name: str, split: str = "Unknown"):
        """Add analysis of a dataset to the summary."""
        analysis = self.analyze_dataframe(df, dataset_name, split)
        
        # Filter out datasets with less than 100 samples
        if analysis['total_samples'] < 100:
            logger.info(f"Skipping {dataset_name} ({split}): only {analysis['total_samples']} samples (< 100)")
            return
        
        # Filter out SMDG-SMDG_Unknown dataset
        if 'SMDG_Unknown' in dataset_name or 'SMDG-SMDG_Unknown' in dataset_name:
            logger.info(f"Skipping {dataset_name} ({split}): SMDG_Unknown dataset excluded")
            return
            
        self.summary_data.append(analysis)
        
        # Log summary
        if analysis['error']:
            logger.warning(f"{dataset_name} ({split}): {analysis['error']}")
        else:
            logger.info(f"{dataset_name} ({split}): {analysis['total_samples']} samples, "
                       f"{analysis['glaucoma_prevalence']:.1%} glaucoma prevalence")
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create a comprehensive summary table."""
        df_summary = pd.DataFrame(self.summary_data)
        
        # Reorder columns for better presentation
        column_order = [
            'dataset_name', 'split', 'total_samples', 
            'glaucoma_positive', 'glaucoma_negative', 'glaucoma_prevalence',
            'missing_labels', 'has_age', 'has_eye', 'has_camera',
            'age_mean', 'age_std', 'unique_eyes', 'unique_cameras', 'error'
        ]
        
        df_summary = df_summary[column_order]
        
        # Format numerical columns
        df_summary['glaucoma_prevalence'] = df_summary['glaucoma_prevalence'].round(4)
        df_summary['age_mean'] = df_summary['age_mean'].round(1)
        df_summary['age_std'] = df_summary['age_std'].round(1)
        
        return df_summary
    
    def save_summary_table(self, df_summary: pd.DataFrame):
        """Save summary table to CSV and create visual table."""
        # Save CSV
        csv_path = os.path.join(self.output_dir, f'dataset_summary_{self.timestamp}.csv')
        df_summary.to_csv(csv_path, index=False)
        logger.info(f"Summary table saved to: {csv_path}")
        
        # Create visual table
        self.create_visual_summary_table(df_summary)
    
    def create_visual_summary_table(self, df_summary: pd.DataFrame):
        """Create a visual summary table as an image."""
        # Prepare data for visual table
        display_df = df_summary.copy()
        
        # Format columns for display
        display_df['Prevalence %'] = (display_df['glaucoma_prevalence'] * 100).round(1).astype(str) + '%'
        display_df['Age (Mean±SD)'] = display_df.apply(
            lambda row: f"{row['age_mean']:.1f}±{row['age_std']:.1f}" 
            if pd.notna(row['age_mean']) else 'N/A', axis=1
        )
        
        # Select and rename columns for display
        display_cols = {
            'dataset_name': 'Dataset',
            'split': 'Split',
            'total_samples': 'Total\nSamples',
            'glaucoma_positive': 'Glaucoma\nPositive',
            'glaucoma_negative': 'Normal',
            'Prevalence %': 'Glaucoma\nPrevalence',
            'has_age': 'Has\nAge',
            'has_eye': 'Has\nEye',
            'has_camera': 'Has\nCamera',
            'Age (Mean±SD)': 'Age\n(Mean±SD)'
        }
        
        plot_df = display_df[list(display_cols.keys())].rename(columns=display_cols)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, max(8, len(plot_df) * 0.4 + 2)))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=plot_df.values,
            colLabels=plot_df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Header styling
        for i in range(len(plot_df.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.1)
        
        # Row styling
        for i in range(1, len(plot_df) + 1):
            row_color = '#F8F9FA' if i % 2 == 0 else '#FFFFFF'
            for j in range(len(plot_df.columns)):
                table[(i, j)].set_facecolor(row_color)
                table[(i, j)].set_height(0.08)
        
        # Title
        plt.suptitle('Glaucoma Dataset Analysis Summary', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.title(f'Generated: {self.timestamp}', fontsize=10, color='gray', y=0.90)
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'dataset_summary_table_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Visual summary table saved to: {plot_path}")
    
    def create_prevalence_plot(self):
        """Create prevalence comparison plot."""
        df_summary = pd.DataFrame(self.summary_data)
        
        # Filter out datasets with errors or no data
        df_plot = df_summary[
            (df_summary['error'].isna()) & 
            (df_summary['total_samples'] > 0)
        ].copy()
        
        if df_plot.empty:
            logger.warning("No valid data for prevalence plot")
            return
        
        # Create combined dataset-split labels
        df_plot['dataset_split'] = df_plot['dataset_name'] + ' (' + df_plot['split'] + ')'
        
        # Clean dataset names by removing SMDG- prefix
        df_plot['dataset_split_clean'] = df_plot['dataset_split'].str.replace('SMDG-', '')
        
        # Create plot
        plt.figure(figsize=(14, 8))
        
        # Bar plot
        bars = plt.bar(range(len(df_plot)), df_plot['glaucoma_prevalence'], 
                      color='lightcoral', alpha=0.7, edgecolor='black')
        
        # Add sample size annotations
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            plt.text(i, row['glaucoma_prevalence'] + 0.01, 
                    f"n={row['total_samples']}", 
                    ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Dataset (Split)', fontsize=12)
        plt.ylabel('Glaucoma Prevalence', fontsize=12)
        plt.title('Glaucoma Prevalence Across Datasets', fontsize=14, fontweight='bold')
        plt.xticks(range(len(df_plot)), df_plot['dataset_split_clean'], rotation=45, ha='right')
        plt.ylim(0, max(df_plot['glaucoma_prevalence']) * 1.1)
        
        # Add horizontal line at 50%
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% prevalence')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'glaucoma_prevalence_comparison_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Prevalence comparison plot saved to: {plot_path}")
    
    def create_sample_size_plot(self):
        """Create sample size comparison plot."""
        df_summary = pd.DataFrame(self.summary_data)
        
        # Filter out datasets with errors
        df_plot = df_summary[
            (df_summary['error'].isna()) & 
            (df_summary['total_samples'] > 0)
        ].copy()
        
        if df_plot.empty:
            logger.warning("No valid data for sample size plot")
            return
        
        # Create combined dataset-split labels
        df_plot['dataset_split'] = df_plot['dataset_name'] + ' (' + df_plot['split'] + ')'
        
        # Clean dataset names by removing SMDG- prefix and split labels (all), (test), etc.
        df_plot['dataset_name_clean'] = df_plot['dataset_name'].str.replace('SMDG-', '')
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        # Stacked bar plot for positive/negative samples
        bottom_bars = plt.bar(range(len(df_plot)), df_plot['glaucoma_negative'], 
                             color='lightblue', label='Normal', alpha=0.8)
        top_bars = plt.bar(range(len(df_plot)), df_plot['glaucoma_positive'], 
                          bottom=df_plot['glaucoma_negative'], 
                          color='lightcoral', label='Glaucoma', alpha=0.8)
        
        # Add text annotations on top of bars with G/Total ratio and prevalence
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            total_samples = row['total_samples']
            glaucoma_samples = row['glaucoma_positive']
            prevalence = row['glaucoma_prevalence']
            
            # Position text above the bar
            bar_height = total_samples
            plt.text(i, bar_height + max(df_plot['total_samples']) * 0.02, 
                    f'{glaucoma_samples}/{total_samples}\n({prevalence:.1%})', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'))
        
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title('Sample Distribution Across Datasets', fontsize=14, fontweight='bold')
        plt.xticks(range(len(df_plot)), df_plot['dataset_name_clean'], rotation=45, ha='right')
        plt.legend()
        
        # Adjust y-limit to accommodate text annotations
        plt.ylim(0, max(df_plot['total_samples']) * 1.15)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'sample_distribution_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sample distribution plot saved to: {plot_path}")
    
    def create_metadata_availability_plot(self):
        """Create metadata availability heatmap."""
        df_summary = pd.DataFrame(self.summary_data)
        
        # Filter out datasets with errors
        df_plot = df_summary[
            (df_summary['error'].isna()) & 
            (df_summary['total_samples'] > 0)
        ].copy()
        
        if df_plot.empty:
            logger.warning("No valid data for metadata availability plot")
            return
        
        # Create metadata availability matrix
        metadata_cols = ['has_age', 'has_eye', 'has_camera']
        metadata_matrix = df_plot[metadata_cols].astype(int)
        
        # Clean dataset names by removing SMDG- prefix
        clean_index = (df_plot['dataset_name'] + ' (' + df_plot['split'] + ')').str.replace('SMDG-', '')
        metadata_matrix.index = clean_index
        metadata_matrix.columns = ['Age', 'Eye', 'Camera']
        
        # Create heatmap
        plt.figure(figsize=(8, max(6, len(df_plot) * 0.4)))
        sns.heatmap(metadata_matrix, annot=True, cmap='RdYlGn', 
                   cbar_kws={'label': 'Available (1) / Not Available (0)'})
        plt.title('Metadata Availability Across Datasets', fontsize=14, fontweight='bold')
        plt.xlabel('Metadata Type', fontsize=12)
        plt.ylabel('Dataset (Split)', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'metadata_availability_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metadata availability plot saved to: {plot_path}")
    
    def create_pixel_distribution_plots(self, datasets_dict: dict):
        """Create pixel distribution comparison plots for datasets with 100+ samples."""
        logger.info("Creating pixel distribution analysis...")
        
        # Filter datasets with sufficient samples (changed from 200+ to 100+)
        valid_datasets = []
        for dataset_type, splits in datasets_dict.items():
            for split_name, df in splits.items():
                if df is not None and not df.empty and len(df) >= 100:
                    # Skip SMDG_Unknown datasets
                    dataset_full_name = f"{dataset_type}_{split_name}"
                    if 'SMDG_Unknown' not in dataset_full_name and 'SMDG-SMDG_Unknown' not in dataset_full_name:
                        valid_datasets.append((dataset_full_name, df))
        
        # Also check external datasets
        if hasattr(self, '_external_datasets'):
            for dataset_name, df in self._external_datasets.items():
                if df is not None and not df.empty and len(df) >= 100:
                    # Skip SMDG_Unknown datasets
                    if 'SMDG_Unknown' not in dataset_name and 'SMDG-SMDG_Unknown' not in dataset_name:
                        valid_datasets.append((dataset_name, df))
        
        if not valid_datasets:
            logger.warning("No datasets with 100+ samples found for pixel distribution analysis")
            return
        
        logger.info(f"Analyzing pixel distributions for {len(valid_datasets)} datasets")
        
        # Sample images from each dataset for pixel analysis
        n_samples_per_dataset = min(100, 200)  # Sample up to 100 images per dataset
        pixel_data = {}
        
        # Define image transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        for dataset_name, df in valid_datasets:
            logger.info(f"Sampling images from {dataset_name}...")
            
            # Sample random images
            sample_df = df.sample(n=min(n_samples_per_dataset, len(df)), random_state=42)
            pixel_values = []
            
            for _, row in sample_df.iterrows():
                try:
                    img_path = row.get('image_path', '')
                    if os.path.exists(img_path):
                        # Load and preprocess image
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = transform(img)
                        
                        # Get pixel values (mean across channels)
                        pixel_vals = img_tensor.mean(dim=0).flatten().numpy()
                        pixel_values.extend(pixel_vals)
                        
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")
                    continue
            
            if pixel_values:
                pixel_data[dataset_name] = np.array(pixel_values)
                logger.info(f"Collected {len(pixel_values)} pixel values from {dataset_name}")
        
        if not pixel_data:
            logger.warning("No pixel data collected")
            return
        
        # Create pixel distribution comparison plot
        plt.figure(figsize=(15, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(pixel_data)))
        
        for i, (dataset_name, pixels) in enumerate(pixel_data.items()):
            # Clean dataset name by removing SMDG- prefix
            clean_dataset_name = dataset_name.replace('SMDG-', '')
            # Create histogram
            plt.hist(pixels, bins=50, alpha=0.6, label=clean_dataset_name, 
                    color=colors[i], density=True)
        
        plt.xlabel('Normalized Pixel Intensity', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Pixel Intensity Distribution Comparison Across Datasets', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'pixel_distribution_comparison_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Pixel distribution plot saved to: {plot_path}")
        
        # Create summary statistics plot
        plt.figure(figsize=(12, 8))
        
        stats_data = []
        for dataset_name, pixels in pixel_data.items():
            # Clean dataset name by removing SMDG- prefix
            clean_dataset_name = dataset_name.replace('SMDG-', '')
            stats_data.append({
                'Dataset': clean_dataset_name,
                'Mean': np.mean(pixels),
                'Std': np.std(pixels),
                'Median': np.median(pixels),
                'Min': np.min(pixels),
                'Max': np.max(pixels)
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Plot statistics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mean intensity
        axes[0,0].bar(range(len(stats_df)), stats_df['Mean'], color=colors[:len(stats_df)])
        axes[0,0].set_title('Mean Pixel Intensity', fontweight='bold')
        axes[0,0].set_xticks(range(len(stats_df)))
        axes[0,0].set_xticklabels(stats_df['Dataset'], rotation=45, ha='right')
        
        # Standard deviation
        axes[0,1].bar(range(len(stats_df)), stats_df['Std'], color=colors[:len(stats_df)])
        axes[0,1].set_title('Pixel Intensity Standard Deviation', fontweight='bold')
        axes[0,1].set_xticks(range(len(stats_df)))
        axes[0,1].set_xticklabels(stats_df['Dataset'], rotation=45, ha='right')
        
        # Range (Max - Min)
        range_vals = stats_df['Max'] - stats_df['Min']
        axes[1,0].bar(range(len(stats_df)), range_vals, color=colors[:len(stats_df)])
        axes[1,0].set_title('Pixel Intensity Range', fontweight='bold')
        axes[1,0].set_xticks(range(len(stats_df)))
        axes[1,0].set_xticklabels(stats_df['Dataset'], rotation=45, ha='right')
        
        # Median
        axes[1,1].bar(range(len(stats_df)), stats_df['Median'], color=colors[:len(stats_df)])
        axes[1,1].set_title('Median Pixel Intensity', fontweight='bold')
        axes[1,1].set_xticks(range(len(stats_df)))
        axes[1,1].set_xticklabels(stats_df['Dataset'], rotation=45, ha='right')
        
        plt.suptitle('Pixel Intensity Statistics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save statistics plot
        stats_plot_path = os.path.join(self.output_dir, f'pixel_statistics_comparison_{self.timestamp}.png')
        plt.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Pixel statistics plot saved to: {stats_plot_path}")
    
    def create_umap_embeddings_visualization(self, datasets_dict: dict, model_path: str = None):
        """Create UMAP visualization using pre-trained encoder embeddings."""
        logger.info("Creating UMAP embeddings visualization...")
        
        if model_path is None:
            model_path = r'D:\glaucoma\models\VFM_Fundus_weights.pth'
        
        if not os.path.exists(model_path):
            logger.warning(f"Model weights not found at {model_path}. Skipping UMAP analysis.")
            return
        
        # Filter datasets with sufficient samples (changed from 200+ to 100+)
        valid_datasets = []
        for dataset_type, splits in datasets_dict.items():
            for split_name, df in splits.items():
                if df is not None and not df.empty and len(df) >= 100:
                    # Skip SMDG_Unknown datasets
                    dataset_full_name = f"{dataset_type}_{split_name}"
                    if 'SMDG_Unknown' not in dataset_full_name and 'SMDG-SMDG_Unknown' not in dataset_full_name:
                        valid_datasets.append((dataset_full_name, df))
        
        # Also check external datasets
        if hasattr(self, '_external_datasets'):
            for dataset_name, df in self._external_datasets.items():
                if df is not None and not df.empty and len(df) >= 100:
                    # Skip SMDG_Unknown datasets
                    if 'SMDG_Unknown' not in dataset_name and 'SMDG-SMDG_Unknown' not in dataset_name:
                        valid_datasets.append((dataset_name, df))
        
        if not valid_datasets:
            logger.warning("No datasets with 100+ samples found for UMAP analysis")
            return
        
        logger.info(f"Creating UMAP visualization for {len(valid_datasets)} datasets")
        
        try:
            # Load pre-trained ViT-B model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            # Create ViT-B model (assuming it's a Vision Transformer)
            try:
                # Try to load using the project's model building function
                from src.models.classification.build_model import build_classifier_model
                model = build_classifier_model(
                    model_name='vit_base_patch16_224',
                    num_classes=2,
                    pretrained=False,
                    custom_weights_path=model_path
                )
                # Remove classification head to get features
                if hasattr(model, 'classifier'):
                    model.classifier = nn.Identity()
                elif hasattr(model, 'head'):
                    model.head = nn.Identity()
                elif hasattr(model, 'fc'):
                    model.fc = nn.Identity()
                    
            except Exception as e:
                logger.warning(f"Failed to load with project model builder: {e}")
                # Fallback: create a simple ViT model
                import timm
                model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                
                # Try different checkpoint keys
                state_dict = None
                for key in ['model_state_dict', 'state_dict', 'model']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
                if state_dict is None:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                
                model.load_state_dict(state_dict, strict=False)
            
            model = model.to(device)
            model.eval()
            
            # Define transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Extract features for each dataset
            all_features = []
            all_labels = []
            all_datasets = []
            dataset_colors = {}
            
            max_samples_per_dataset = 300  # Limit for computational efficiency
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(valid_datasets)))
            
            for i, (dataset_name, df) in enumerate(valid_datasets):
                logger.info(f"Extracting features from {dataset_name}...")
                
                # Sample images
                sample_size = min(max_samples_per_dataset, len(df))
                sample_df = df.sample(n=sample_size, random_state=42)
                
                dataset_features = []
                dataset_labels = []
                
                with torch.no_grad():
                    for idx, (_, row) in enumerate(sample_df.iterrows()):
                        if idx % 50 == 0:
                            logger.info(f"Processing {idx}/{len(sample_df)} images from {dataset_name}")
                        
                        try:
                            img_path = row.get('image_path', '')
                            if os.path.exists(img_path):
                                # Load and preprocess image
                                img = Image.open(img_path).convert('RGB')
                                img_tensor = transform(img).unsqueeze(0).to(device)
                                
                                # Extract features
                                features = model(img_tensor)
                                dataset_features.append(features.cpu().numpy().flatten())
                                
                                # Get label
                                label = row.get('types', row.get('label', 0))
                                dataset_labels.append(int(label))
                                
                        except Exception as e:
                            logger.warning(f"Failed to process image {img_path}: {e}")
                            continue
                
                if dataset_features:
                    all_features.extend(dataset_features)
                    all_labels.extend(dataset_labels)
                    all_datasets.extend([dataset_name] * len(dataset_features))
                    dataset_colors[dataset_name] = colors[i]
                    
                    logger.info(f"Extracted {len(dataset_features)} features from {dataset_name}")
            
            if not all_features:
                logger.warning("No features extracted. Skipping UMAP visualization.")
                return
            
            # Convert to numpy arrays
            features_array = np.array(all_features)
            labels_array = np.array(all_labels)
            datasets_array = np.array(all_datasets)
            
            logger.info(f"Total features extracted: {features_array.shape}")
            
            # Apply UMAP
            logger.info("Computing UMAP embeddings...")
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
            embedding = reducer.fit_transform(features_array)
            
            # Create UMAP visualization
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # Plot by dataset
            for dataset_name in np.unique(datasets_array):
                mask = datasets_array == dataset_name
                # Clean dataset name by removing SMDG- prefix for legend
                clean_dataset_name = dataset_name.replace('SMDG-', '')
                axes[0].scatter(embedding[mask, 0], embedding[mask, 1], 
                              c=[dataset_colors[dataset_name]], label=clean_dataset_name, 
                              alpha=0.6, s=20)
            
            axes[0].set_title('UMAP Embeddings by Dataset', fontsize=14, fontweight='bold')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].set_xlabel('UMAP 1')
            axes[0].set_ylabel('UMAP 2')
            
            # Plot by label (glaucoma vs normal)
            normal_mask = labels_array == 0
            glaucoma_mask = labels_array == 1;
            
            axes[1].scatter(embedding[normal_mask, 0], embedding[normal_mask, 1], 
                          c='lightblue', label='Normal', alpha=0.6, s=20)
            axes[1].scatter(embedding[glaucoma_mask, 0], embedding[glaucoma_mask, 1], 
                          c='lightcoral', label='Glaucoma', alpha=0.6, s=20)
            
            axes[1].set_title('UMAP Embeddings by Label', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].set_xlabel('UMAP 1')
            axes[1].set_ylabel('UMAP 2')
            
            plt.suptitle('UMAP Visualization of Dataset Embeddings (VFM Pre-trained Features)', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            umap_path = os.path.join(self.output_dir, f'umap_embeddings_{self.timestamp}.png')
            plt.savefig(umap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"UMAP visualization saved to: {umap_path}")
            
            # Create per-dataset UMAP plots
            n_datasets = len(np.unique(datasets_array))
            if n_datasets > 1:
                fig, axes = plt.subplots(2, (n_datasets + 1) // 2, figsize=(5 * ((n_datasets + 1) // 2), 10))
                if n_datasets == 1:
                    axes = [axes]
                elif (n_datasets + 1) // 2 == 1:
                    axes = axes.reshape(-1)
                else:
                    axes = axes.flatten()
                
                for i, dataset_name in enumerate(np.unique(datasets_array)):
                    if i >= len(axes):
                        break
                        
                    mask = datasets_array == dataset_name
                    dataset_embedding = embedding[mask]
                    dataset_labels = labels_array[mask]
                    
                    # Plot this dataset with labels
                    normal_mask = dataset_labels == 0
                    glaucoma_mask = dataset_labels == 1
                    
                    if np.any(normal_mask):
                        axes[i].scatter(dataset_embedding[normal_mask, 0], dataset_embedding[normal_mask, 1], 
                                      c='lightblue', label='Normal', alpha=0.7, s=25)
                    if np.any(glaucoma_mask):
                        axes[i].scatter(dataset_embedding[glaucoma_mask, 0], dataset_embedding[glaucoma_mask, 1], 
                                      c='lightcoral', label='Glaucoma', alpha=0.7, s=25)
                    
                    # Clean dataset name by removing SMDG- prefix for title
                    clean_dataset_name = dataset_name.replace('SMDG-', '')
                    axes[i].set_title(f'{clean_dataset_name}', fontweight='bold')
                    axes[i].legend()
                    axes[i].set_xlabel('UMAP 1')
                    axes[i].set_ylabel('UMAP 2')
                
                # Hide unused subplots
                for i in range(len(np.unique(datasets_array)), len(axes)):
                    axes[i].set_visible(False)
                
                plt.suptitle('UMAP Embeddings per Dataset', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                # Save per-dataset plot
                umap_per_dataset_path = os.path.join(self.output_dir, f'umap_per_dataset_{self.timestamp}.png')
                plt.savefig(umap_per_dataset_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Per-dataset UMAP visualization saved to: {umap_per_dataset_path}")
            
        except Exception as e:
            logger.error(f"Failed to create UMAP visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_vcdr_metrics(self, df: pd.DataFrame, dataset_name: str, max_samples: int = None):
        """Calculate vCDR metrics for a dataset using U-Net segmentation."""
        if self.predictor is None or self.metrics_calculator is None:
            logger.warning(f"U-Net predictor not available for {dataset_name}. Skipping vCDR analysis.")
            return None
        
        # Use all samples for comprehensive analysis
        sample_size = len(df) if max_samples is None else min(max_samples, len(df))
        logger.info(f"Calculating vCDR metrics for {dataset_name} (processing {sample_size} samples)...")
        
        if sample_size < 50:
            logger.warning(f"Dataset {dataset_name} has only {sample_size} samples. Skipping vCDR analysis.")
            return None
        
        # For comprehensive analysis, use all samples (no sampling unless max_samples is specified)
        if max_samples is not None and sample_size < len(df):
            # Only sample if explicitly limited by max_samples parameter
            try:
                label_col = None
                for col in ['types', 'label', 'glaucoma', 'diagnosis']:
                    if col in df.columns:
                        label_col = col
                        break
                
                if label_col and df[label_col].nunique() >= 2:
                    _, sample_df = train_test_split(
                        df, test_size=sample_size, stratify=df[label_col], random_state=42
                    )
                else:
                    sample_df = df.sample(n=sample_size, random_state=42)
            except:
                sample_df = df.sample(n=sample_size, random_state=42)
        else:
            # Use all samples for comprehensive analysis
            sample_df = df.copy()
            sample_size = len(sample_df)
        
        results = []
        processed_count = 0
        
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), 
                          desc=f"Processing {dataset_name}"):
            try:
                img_path = row.get('image_path', '')
                if not os.path.exists(img_path):
                    continue
                
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Predict disc and cup masks
                disc_mask, cup_mask = self.predictor.predict(img, refine_smooth=False)
                
                # Calculate metrics
                metrics = self.metrics_calculator.extract_metrics(disc_mask, cup_mask)
                
                if metrics and 'vcdr' in metrics:
                    label = row.get('types', row.get('label', 0))
                    results.append({
                        'dataset_name': dataset_name,
                        'image_name': row.get('names', ''),
                        'image_path': img_path,
                        'label': int(label),
                        'vcdr': metrics['vcdr'],
                        'hcdr': metrics.get('hcdr', None),
                        'area_ratio': metrics.get('area_ratio', None)
                    })
                    processed_count += 1
                
            except Exception as e:
                logger.debug(f"Failed to process image {img_path}: {e}")
                continue
        
        if processed_count == 0:
            logger.warning(f"No valid vCDR measurements for {dataset_name}")
            return None
        
        logger.info(f"Successfully processed {processed_count} images for {dataset_name}")
        return pd.DataFrame(results)
    
    def find_optimal_threshold(self, vcdr_values: np.array, labels: np.array):
        """Find optimal vCDR threshold using ROC curve analysis."""
        try:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(labels, vcdr_values)
            auc_score = roc_auc_score(labels, vcdr_values)
            
            # Find optimal threshold using Youden's J statistic
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            return {
                'optimal_threshold': optimal_threshold,
                'auc': auc_score,
                'sensitivity': tpr[optimal_idx],
                'specificity': 1 - fpr[optimal_idx],
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            }
        except Exception as e:
            logger.error(f"Failed to calculate optimal threshold: {e}")
            return None
    
    def create_vcdr_analysis(self, datasets_dict: dict, max_samples: int = None):
        """Perform comprehensive vCDR analysis across all datasets."""
        if self.predictor is None:
            logger.warning("U-Net predictor not available. Skipping vCDR analysis.")
            return
        
        logger.info("Starting comprehensive vCDR analysis...")
        if max_samples is None:
            logger.info("Processing ALL samples for comprehensive analysis")
        else:
            logger.info(f"Processing up to {max_samples} samples per dataset")
        
        # Collect all datasets for vCDR analysis
        all_datasets = []
        
        # Training datasets
        for dataset_type, splits in datasets_dict.items():
            for split_name, df in splits.items():
                if df is not None and not df.empty and len(df) >= 100:
                    dataset_full_name = f"{dataset_type}_{split_name}"
                    if 'SMDG_Unknown' not in dataset_full_name:
                        all_datasets.append((dataset_full_name, df))
        
        # External datasets
        if hasattr(self, '_external_datasets'):
            for dataset_name, df in self._external_datasets.items():
                if df is not None and not df.empty and len(df) >= 100:
                    if 'SMDG_Unknown' not in dataset_name:
                        all_datasets.append((dataset_name, df))
        
        if not all_datasets:
            logger.warning("No suitable datasets found for vCDR analysis")
            return
        
        logger.info(f"Performing vCDR analysis on {len(all_datasets)} datasets")
        
        # Process each dataset
        vcdr_summary_data = []
        
        for dataset_name, df in all_datasets:
            vcdr_df = self.calculate_vcdr_metrics(df, dataset_name, max_samples)
            
            if vcdr_df is not None and len(vcdr_df) >= 20:  # Minimum samples for meaningful analysis
                # Calculate optimal threshold
                threshold_results = self.find_optimal_threshold(
                    vcdr_df['vcdr'].values, vcdr_df['label'].values
                )
                
                if threshold_results:
                    clean_dataset_name = dataset_name.replace('SMDG-', '')
                    
                    summary_entry = {
                        'dataset_name': clean_dataset_name,
                        'sample_size': len(vcdr_df),
                        'glaucoma_samples': sum(vcdr_df['label']),
                        'normal_samples': len(vcdr_df) - sum(vcdr_df['label']),
                        'vcdr_optimal_threshold': threshold_results['optimal_threshold'],
                        'auc': threshold_results['auc'],
                        'sensitivity': threshold_results['sensitivity'],
                        'specificity': threshold_results['specificity'],
                        'mean_vcdr_normal': vcdr_df[vcdr_df['label'] == 0]['vcdr'].mean(),
                        'mean_vcdr_glaucoma': vcdr_df[vcdr_df['label'] == 1]['vcdr'].mean(),
                        'std_vcdr_normal': vcdr_df[vcdr_df['label'] == 0]['vcdr'].std(),
                        'std_vcdr_glaucoma': vcdr_df[vcdr_df['label'] == 1]['vcdr'].std()
                    }
                    
                    vcdr_summary_data.append(summary_entry)
                    
                    # Store detailed results for plotting
                    self.vcdr_results.append({
                        'dataset_name': clean_dataset_name,
                        'vcdr_df': vcdr_df,
                        'threshold_results': threshold_results
                    })
        
        if not vcdr_summary_data:
            logger.warning("No vCDR analysis results generated")
            return
        
        # Create summary table
        vcdr_summary_df = pd.DataFrame(vcdr_summary_data)
        
        # Save summary table
        csv_path = os.path.join(self.output_dir, f'vcdr_analysis_summary_{self.timestamp}.csv')
        vcdr_summary_df.to_csv(csv_path, index=False)
        logger.info(f"vCDR analysis summary saved to: {csv_path}")
        
        # Create visualizations
        self._create_vcdr_distribution_plots()
        self._create_vcdr_summary_table(vcdr_summary_df)
        
        logger.info("vCDR analysis completed successfully")
    
    def _create_vcdr_distribution_plots(self):
        """Create vCDR distribution plots for each dataset with optimal thresholds."""
        if not self.vcdr_results:
            return
        
        # Create individual distribution plots
        n_datasets = len(self.vcdr_results)
        cols = min(3, n_datasets)
        rows = (n_datasets + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_datasets == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_datasets > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, result in enumerate(self.vcdr_results):
            if i >= len(axes):
                break
                
            dataset_name = result['dataset_name']
            vcdr_df = result['vcdr_df']
            threshold_results = result['threshold_results']
            
            ax = axes[i]
            
            # Plot distributions
            normal_vcdr = vcdr_df[vcdr_df['label'] == 0]['vcdr']
            glaucoma_vcdr = vcdr_df[vcdr_df['label'] == 1]['vcdr']
            
            ax.hist(normal_vcdr, bins=20, alpha=0.6, label='Normal', color='lightblue', density=True)
            ax.hist(glaucoma_vcdr, bins=20, alpha=0.6, label='Glaucoma', color='lightcoral', density=True)
            
            # Add optimal threshold line
            ax.axvline(threshold_results['optimal_threshold'], color='red', linestyle='--', 
                      linewidth=2, label=f'Optimal Threshold: {threshold_results["optimal_threshold"]:.3f}')
            
            ax.set_xlabel('vCDR')
            ax.set_ylabel('Density')
            ax.set_title(f'{dataset_name}\nAUC: {threshold_results["auc"]:.3f}, n={len(vcdr_df)}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.vcdr_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('vCDR Distribution Analysis with Optimal Thresholds', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'vcdr_distributions_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"vCDR distribution plots saved to: {plot_path}")
        
        # Create combined ROC curves
        self._create_roc_curves_plot()
    
    def _create_roc_curves_plot(self):
        """Create ROC curves for all datasets."""
        if not self.vcdr_results:
            return
        
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.vcdr_results)))
        
        for i, result in enumerate(self.vcdr_results):
            dataset_name = result['dataset_name']
            threshold_results = result['threshold_results']
            
            plt.plot(threshold_results['fpr'], threshold_results['tpr'], 
                    color=colors[i], linewidth=2,
                    label=f'{dataset_name} (AUC = {threshold_results["auc"]:.3f})')
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for vCDR-based Glaucoma Classification', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        roc_path = os.path.join(self.output_dir, f'vcdr_roc_curves_{self.timestamp}.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves plot saved to: {roc_path}")
    
    def _create_vcdr_summary_table(self, vcdr_summary_df: pd.DataFrame):
        """Create visual summary table for vCDR analysis."""
        # Prepare data for display
        display_df = vcdr_summary_df.copy()
        
        # Format columns for display
        display_df['AUC'] = display_df['auc'].round(3)
        display_df['Threshold'] = display_df['vcdr_optimal_threshold'].round(3)
        display_df['Sensitivity'] = (display_df['sensitivity'] * 100).round(1).astype(str) + '%'
        display_df['Specificity'] = (display_df['specificity'] * 100).round(1).astype(str) + '%'
        display_df['Normal vCDR'] = (display_df['mean_vcdr_normal'].round(3).astype(str) + '±' + 
                                   display_df['std_vcdr_normal'].round(3).astype(str))
        display_df['Glaucoma vCDR'] = (display_df['mean_vcdr_glaucoma'].round(3).astype(str) + '±' + 
                                     display_df['std_vcdr_glaucoma'].round(3).astype(str))
        
        # Select columns for display
        display_cols = {
            'dataset_name': 'Dataset',
            'sample_size': 'Sample\nSize',
            'Threshold': 'Optimal\nThreshold',
            'AUC': 'AUC',
            'Sensitivity': 'Sensitivity',
            'Specificity': 'Specificity',
            'Normal vCDR': 'Normal vCDR\n(Mean±SD)',
            'Glaucoma vCDR': 'Glaucoma vCDR\n(Mean±SD)'
        }
        
        plot_df = display_df[list(display_cols.keys())].rename(columns=display_cols)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, max(6, len(plot_df) * 0.4 + 2)))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=plot_df.values,
            colLabels=plot_df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Header styling
        for i in range(len(plot_df.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.12)
        
        # Row styling with AUC-based coloring
        for i in range(1, len(plot_df) + 1):
            auc_val = float(plot_df.iloc[i-1]['AUC'])
            if auc_val >= 0.8:
                row_color = '#E8F5E8'  # Light green for good AUC
            elif auc_val >= 0.7:
                row_color = '#FFF8DC'  # Light yellow for moderate AUC
            else:
                row_color = '#FFE4E1'  # Light red for poor AUC
                
            for j in range(len(plot_df.columns)):
                table[(i, j)].set_facecolor(row_color)
                table[(i, j)].set_height(0.1)
        
        # Title
        plt.suptitle('vCDR Analysis Summary - Optimal Thresholds and Performance', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.title(f'Generated: {self.timestamp}', fontsize=10, color='gray', y=0.88)
        
        # Save plot
        table_path = os.path.join(self.output_dir, f'vcdr_summary_table_{self.timestamp}.png')
        plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"vCDR summary table saved to: {table_path}")
    

def load_training_datasets(args: argparse.Namespace) -> dict:
    """Load training, validation, and test datasets."""
    logger.info("Loading training/validation/test datasets...")
    
    datasets = {}
    
    try:
        # Load SMDG-19 data directly and analyze by subdatasets
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
                
                logger.info(f"Loaded {len(df_smdg)} SMDG-19 samples")
                
                # Split SMDG-19 by subdatasets for analysis
                smdg_subdatasets = df_smdg.groupby('dataset_source')
                for dataset_name, dataset_df in smdg_subdatasets:
                    datasets[f'SMDG-{dataset_name}'] = {'all': dataset_df}
            
                
                logger.info(f"SMDG-19 subdatasets found: {list(smdg_subdatasets.groups.keys())}")
            else:
                logger.warning("'names' column missing in SMDG metadata")
        else:
            logger.warning(f"SMDG-19 metadata file not found: {smdg_metadata_file}")
        
        # Load AIROGS data separately if requested
        if args.use_airogs:
            try:
                logger.info(f"Loading AIROGS dataset (RG: {args.airogs_num_rg}, NRG: {args.airogs_num_nrg})...")
                
                # Create minimal config for AIROGS loading
                airogs_config = argparse.Namespace()
                airogs_config.data_type = args.data_type
                airogs_config.base_data_root = args.base_data_root
                
                # Set AIROGS-specific parameters with correct naming
                airogs_config.airogs_num_rg_samples = args.airogs_num_rg
                airogs_config.airogs_num_nrg_samples = args.airogs_num_nrg
                airogs_config.airogs_label_file = args.airogs_label_file
                airogs_config.airogs_image_dir = args.airogs_image_dir
                airogs_config.use_airogs_cache = args.use_airogs_cache
                airogs_config.seed = args.seed
                
                # Adjust AIROGS image directory path if needed
                if args.data_type == 'processed':
                    airogs_config.airogs_image_dir = adjust_path_for_data_type(
                        current_path=args.airogs_image_dir, data_type='processed',
                        base_data_dir=args.base_data_root, raw_dir_name=RAW_DIR_NAME_CONST,
                        processed_dir_name=PROCESSED_DIR_NAME_CONST
                    )
                
                # Load AIROGS data using the function from train_classification
                airogs_df = load_airogs_data(airogs_config)
                
                if not airogs_df.empty:
                    datasets['AIROGS'] = {'all': airogs_df}
                    logger.info(f"Successfully loaded {len(airogs_df)} AIROGS samples")
                else:
                    logger.warning("AIROGS dataset is empty")
                    
            except Exception as e:
                logger.error(f"Failed to load AIROGS dataset: {e}")
                import traceback
                traceback.print_exc()
        
        # Try to load the full training data using the original function
        try:
            # Create a minimal config object
            config = argparse.Namespace()
            config.data_type = args.data_type
            config.base_data_root = args.base_data_root
            config.smdg_metadata_file = smdg_metadata_file
            config.smdg_image_dir = smdg_image_dir
            
            # Set required parameters for other datasets
            config.use_smdg = True
            config.use_chaksu = False  # Don't include CHAKSU in training
            config.use_oiaodir = False  # Disable for now
            config.use_airogs = False   # Disable for now
            config.exclude_eyepacs = True
            config.save_data_samples = False
            config.train_val_test_split = [0.6, 0.2, 0.2]
            config.oiaodir_train_val_split = 0.8
            config.max_samples_per_dataset = None
            config.random_seed = 42
            config.min_samples_per_group_for_stratify = 2
            
            # Try to load with full function
            train_df, val_df, test_df = load_and_split_data(config)
            
            if not train_df.empty or not val_df.empty or not test_df.empty:
                datasets['Training-Full'] = {
                    'train': train_df,
                    'validation': val_df,
                    'test': test_df
                }
                logger.info("Successfully loaded full training/validation/test datasets")
            
        except Exception as e:
            logger.warning(f"Failed to load full training datasets: {e}")
        
    except Exception as e:
        logger.warning(f"Failed to load any training data: {e}")
        logger.info("Continuing with external datasets only...")
    
    return datasets


def load_external_datasets(args: argparse.Namespace) -> dict:
    """Load external OOD test datasets."""
    logger.info("Loading external test datasets...")
    
    # Adjust CHAKSU path if needed
    chaksu_base_dir_for_loader = args.chaksu_base_dir
    if args.data_type == 'processed':
        chaksu_base_dir_for_loader = adjust_path_for_data_type(
            current_path=args.chaksu_base_dir, data_type='processed',
            base_data_dir=args.base_data_root, raw_dir_name=RAW_DIR_NAME_CONST,
            processed_dir_name=PROCESSED_DIR_NAME_CONST
        )
    try:
        external_datasets = load_external_test_data(
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
            eval_oiaodir_test=False,  # We don't want OIA-ODIR in OOD evaluation
            eval_chaksu=args.eval_chaksu,
            eval_acrima=args.eval_acrima,
            eval_hygd=args.eval_hygd,
            acrima_image_dir_raw=args.acrima_image_dir_raw,
            hygd_image_dir_raw=args.hygd_image_dir_raw,
            hygd_labels_file_raw=args.hygd_labels_file_raw
        )
        
        logger.info(f"Successfully loaded external datasets: {list(external_datasets.keys())}")
        return external_datasets
        
    except Exception as e:
        logger.error(f"Failed to load external datasets: {e}")
        return {}


def main(args: argparse.Namespace):
    """Main function to run comprehensive dataset analysis."""
    logger.info("Starting comprehensive dataset analysis...")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"dataset_analysis_{timestamp}")
    
    # Initialize analyzer with U-Net model path
    unet_model_path = args.unet_model_path if hasattr(args, 'unet_model_path') else None
    analyzer = DatasetAnalyzer(output_dir, unet_model_path)
    
    # Load training datasets
    training_datasets = load_training_datasets(args)
    
    # Analyze training datasets
    for dataset_type, splits in training_datasets.items():
        for split_name, df in splits.items():
            analyzer.add_dataset_analysis(df, dataset_type, split_name)
    
    # Load and analyze external datasets
    external_datasets = load_external_datasets(args)
    
    # Store external datasets in analyzer for pixel/UMAP analysis
    analyzer._external_datasets = external_datasets
    
    for dataset_name, df in external_datasets.items():
        analyzer.add_dataset_analysis(df, dataset_name, "test")
    
    # Create summary table
    summary_df = analyzer.create_summary_table()
    analyzer.save_summary_table(summary_df)
    
    # Create basic visualizations
    analyzer.create_prevalence_plot()
    analyzer.create_sample_size_plot()
    analyzer.create_metadata_availability_plot()
    
    # Create advanced visualizations if requested
    if args.pixel_analysis:
        logger.info("\n--- Creating Pixel Distribution Analysis ---")
        analyzer.create_pixel_distribution_plots(training_datasets)
    
    if args.umap_analysis:
        logger.info("\n--- Creating UMAP Embeddings Visualization ---")
        model_path = args.model_path if args.model_path else r'D:\glaucoma\models\VFM_Fundus_weights.pth'
        analyzer.create_umap_embeddings_visualization(training_datasets, model_path)
    
    # Create vCDR analysis if U-Net model is available
    if args.vcdr_analysis and analyzer.predictor is not None:
        logger.info("\n--- Creating vCDR Analysis ---")
        analyzer.create_vcdr_analysis(training_datasets, args.vcdr_max_samples)
    elif args.vcdr_analysis:
        logger.warning("vCDR analysis requested but U-Net model not available")
    
    # Save detailed analysis as JSON
    json_path = os.path.join(output_dir, f'detailed_analysis_{analyzer.timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(analyzer.summary_data, f, indent=4, cls=NpEncoder)
    
    logger.info(f"Dataset analysis completed. Results saved to: {output_dir}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("DATASET ANALYSIS SUMMARY")
    print("="*80)
    
    total_samples = summary_df['total_samples'].sum()
    total_glaucoma = summary_df['glaucoma_positive'].sum()
    overall_prevalence = total_glaucoma / total_samples if total_samples > 0 else 0
    
    print(f"Total samples across all datasets: {total_samples:,}")
    print(f"Total glaucoma cases: {total_glaucoma:,}")
    print(f"Overall glaucoma prevalence: {overall_prevalence:.1%}")
    print(f"Number of datasets analyzed: {len(summary_df)}")
    
    # Dataset-specific summary
    print("\nDataset-specific summary:")
    for _, row in summary_df.iterrows():
        if row['error']:
            print(f"  {row['dataset_name']} ({row['split']}): ERROR - {row['error']}")
        else:
            print(f"  {row['dataset_name']} ({row['split']}): "
                  f"{row['total_samples']:,} samples, "
                  f"{row['glaucoma_prevalence']:.1%} prevalence")
    
    # Print info about advanced analyses
    analyses_completed = []
    if args.pixel_analysis:
        analyses_completed.append("✓ Pixel distribution analysis")
    if args.umap_analysis:
        analyses_completed.append("✓ UMAP embeddings visualization")
    if args.vcdr_analysis and analyzer.predictor is not None:
        analyses_completed.append("✓ vCDR analysis with optimal thresholds")
    
    if analyses_completed:
        print(f"\nAdvanced analyses completed:")
        for analysis in analyses_completed:
            print(f"  {analysis}")
    else:
        print(f"\nTo enable advanced analysis, use:")
        print("  --pixel_analysis    for pixel distribution comparison")
        print("  --umap_analysis     for UMAP embeddings visualization")
        print("  --vcdr_analysis     for vCDR analysis with optimal thresholds")
    
    # Print vCDR analysis summary if available
    if hasattr(analyzer, 'vcdr_results') and analyzer.vcdr_results:
        print(f"\nvCDR Analysis Summary:")
        print(f"  Datasets analyzed: {len(analyzer.vcdr_results)}")
        for result in analyzer.vcdr_results:
            dataset_name = result['dataset_name']
            auc = result['threshold_results']['auc']
            threshold = result['threshold_results']['optimal_threshold']
            sample_size = len(result['vcdr_df'])
            print(f"  {dataset_name}: AUC={auc:.3f}, Threshold={threshold:.3f}, n={sample_size}")
    
    print(f"\nDetailed results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive Dataset Analysis for Glaucoma Classification Project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./dataset_analysis_results',
                       help="Directory to save analysis results")
    
    # Training data configuration
    parser.add_argument('--data_config_path', type=str, 
                       default=os.path.join('configs', 'data_paths.yaml'),
                       help="Path to data configuration file")
    
    # External data configuration
    parser.add_argument('--data_type', type=str, default='raw', 
                       choices=['raw', 'processed'],
                       help="Target type of image data ('raw' or 'processed')")
    parser.add_argument('--base_data_root', type=str, default=r'D:\glaucoma\data',
                       help="Absolute base root directory for all datasets")
    
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
    
    # AIROGS dataset configuration
    parser.add_argument('--use_airogs', action='store_true', default=True,
                       help="Include AIROGS dataset in analysis")
    parser.add_argument('--airogs_label_file', type=str, 
                       default=r'D:\glaucoma\data\raw\AIROGS\train_labels.csv',
                       help="Path to AIROGS training labels CSV file")
    parser.add_argument('--airogs_image_dir', type=str, 
                       default=r'D:\glaucoma\data\raw\AIROGS\img',
                       help="Path to AIROGS image directory")
    parser.add_argument('--airogs_num_rg', type=int, default=3000,
                       help="Number of RG (glaucoma) samples to load from AIROGS")
    parser.add_argument('--airogs_num_nrg', type=int, default=3000,
                       help="Number of NRG (normal) samples to load from AIROGS")
    parser.add_argument('--use_airogs_cache', action='store_true', default=True,
                       help="Use cached AIROGS manifest if available")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducible sampling")
    
    # Advanced analysis options
    parser.add_argument('--pixel_analysis', action='store_true', default=False,
                       help="Enable pixel distribution analysis (requires datasets with 200+ samples)")
    parser.add_argument('--umap_analysis', action='store_true', default=False,
                       help="Enable UMAP embeddings visualization using pre-trained model")
    parser.add_argument('--vcdr_analysis', action='store_true', default=True,
                       help="Enable vCDR analysis using U-Net segmentation with optimal threshold calculation")
    parser.add_argument('--vcdr_max_samples', type=int, default=None,
                       help="Maximum samples per dataset for vCDR analysis (default: None = all samples)")
    parser.add_argument('--model_path', type=str, default=r'D:\glaucoma\models\VFM_Fundus_weights.pth',
                       help="Path to pre-trained model weights for UMAP analysis")
    parser.add_argument('--unet_model_path', type=str, 
                       default=r'D:\glaucoma\models\best_multitask_model_epoch_25.pth',
                       help="Path to U-Net model weights for vCDR analysis")
    
    # Dataset selection
    parser.add_argument('--eval_papilla', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--eval_chaksu', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--eval_acrima', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--eval_hygd', action=argparse.BooleanOptionalAction, default=True)

    
    args = parser.parse_args()
    
    # Validate paths
    args.base_data_root = os.path.abspath(args.base_data_root)
    if not os.path.isdir(args.base_data_root):
        logger.error(f"Base data root not found: {args.base_data_root}")
        sys.exit(1)
    
    main(args)
