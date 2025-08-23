#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vCDR Label Extraction Script for Glaucoma Classification Project.

This script extracts vertical Cup-to-Disc Ratio (vCDR) values from all available datasets
using the trained U-Net segmentation model. The extracted vCDR values serve as continuous
labels for disease severity and can be used for regression-based deep learning models.

The script processes all datasets and creates a comprehensive CSV file containing:
- Dataset name
- Image path
- Original binary label (if available)
- Extracted vCDR value
- Additional metrics (hCDR, area ratio)

Output CSV format:
dataset, image_path, original_label, vcdr, hcdr, area_ratio, image_name
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


class VCDRExtractor:
    """Class to extract vCDR values from all available datasets."""
    
    def __init__(self, unet_model_path: str, output_dir: str):
        self.unet_model_path = unet_model_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize U-Net predictor
        self.predictor = None
        self.metrics_calculator = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize U-Net model
        if os.path.exists(self.unet_model_path):
            try:
                logger.info(f"Initializing U-Net predictor with model: {self.unet_model_path}")
                self.predictor = OpticDiscCupPredictor(model_path=self.unet_model_path)
                self.metrics_calculator = GlaucomaMetrics()
                logger.info("U-Net predictor and metrics calculator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize U-Net predictor: {e}")
                raise e
        else:
            logger.error(f"U-Net model not found at: {self.unet_model_path}")
            raise FileNotFoundError(f"U-Net model not found at: {self.unet_model_path}")
    
    def extract_vcdr_from_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Extract vCDR values from a single dataset."""
        logger.info(f"Extracting vCDR values from {dataset_name} ({len(df)} samples)...")
        
        results = []
        vcdr_success_count = 0
        vcdr_failed_count = 0
        image_load_failed_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
            img_path = row.get('image_path', '')
            
            # Get original label if available
            original_label = None
            for label_col in ['types', 'label', 'glaucoma', 'diagnosis']:
                if label_col in row and pd.notna(row[label_col]):
                    original_label = int(row[label_col])
                    break
            
            # Get image name
            image_name = row.get('names', row.get('image_name', os.path.basename(img_path)))
            
            # Initialize result entry - always create an entry for each sample
            result_entry = {
                'dataset': dataset_name,
                'image_path': img_path,
                'image_name': image_name,
                'original_label': original_label,
                'vcdr': np.nan,  # Default to NaN if extraction fails
                'hcdr': np.nan,
                'area_ratio': np.nan
            }
            
            # Try to extract vCDR if image exists and can be loaded
            if img_path and os.path.exists(img_path):
                try:
                    # Load image
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Predict disc and cup masks
                        disc_mask, cup_mask = self.predictor.predict(img, refine_smooth=False)
                        
                        # Calculate metrics
                        metrics = self.metrics_calculator.extract_metrics(disc_mask, cup_mask)
                        
                        if metrics and 'vcdr' in metrics:
                            # Successfully extracted vCDR - update the entry
                            result_entry['vcdr'] = metrics['vcdr']
                            result_entry['hcdr'] = metrics.get('hcdr', np.nan)
                            result_entry['area_ratio'] = metrics.get('area_ratio', np.nan)
                            vcdr_success_count += 1
                        else:
                            logger.debug(f"Failed to extract vCDR metrics from: {img_path}")
                            vcdr_failed_count += 1
                    else:
                        logger.debug(f"Failed to load image: {img_path}")
                        image_load_failed_count += 1
                        
                except Exception as e:
                    logger.debug(f"Error processing {img_path}: {e}")
                    vcdr_failed_count += 1
            else:
                logger.debug(f"Image not found: {img_path}")
                image_load_failed_count += 1
            
            # Always add the result entry (with or without vCDR)
            results.append(result_entry)
        
        logger.info(f"Dataset {dataset_name}: {len(results)} total samples, {vcdr_success_count} successful vCDR extractions, {vcdr_failed_count} vCDR extraction failures, {image_load_failed_count} image load failures")
        
        return pd.DataFrame(results)
    
    def process_all_datasets(self, training_datasets: dict, external_datasets: dict) -> pd.DataFrame:
        """Process all datasets and extract vCDR values."""
        logger.info("Starting vCDR extraction from all datasets...")
        
        all_results = []
        
        # Process training datasets
        logger.info("Processing training datasets...")
        for dataset_type, splits in training_datasets.items():
            for split_name, df in splits.items():
                if df is not None and not df.empty:
                    dataset_full_name = f"{dataset_type}_{split_name}"
                    
                    # Skip datasets with insufficient samples or problematic names
                    if len(df) < 50:
                        logger.info(f"Skipping {dataset_full_name}: only {len(df)} samples (< 50)")
                        continue
                    
                    if 'SMDG_Unknown' in dataset_full_name:
                        logger.info(f"Skipping {dataset_full_name}: SMDG_Unknown dataset excluded")
                        continue
                    
                    # Clean dataset name
                    clean_dataset_name = dataset_full_name.replace('SMDG-', '')
                    
                    # Extract vCDR values
                    vcdr_df = self.extract_vcdr_from_dataset(df, clean_dataset_name)
                    if not vcdr_df.empty:
                        all_results.append(vcdr_df)
                    else:
                        logger.warning(f"No samples processed from {clean_dataset_name}")
        
        # Process external datasets
        logger.info("Processing external datasets...")
        for dataset_name, df in external_datasets.items():
            if df is not None and not df.empty:
                # Skip datasets with insufficient samples or problematic names
                if len(df) < 50:
                    logger.info(f"Skipping {dataset_name}: only {len(df)} samples (< 50)")
                    continue
                
                if 'SMDG_Unknown' in dataset_name:
                    logger.info(f"Skipping {dataset_name}: SMDG_Unknown dataset excluded")
                    continue
                
                # Clean dataset name
                clean_dataset_name = dataset_name.replace('SMDG-', '')
                
                # Extract vCDR values
                vcdr_df = self.extract_vcdr_from_dataset(df, clean_dataset_name)
                if not vcdr_df.empty:
                    all_results.append(vcdr_df)
                else:
                    logger.warning(f"No samples processed from {clean_dataset_name}")
        
        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Count successful vCDR extractions vs total samples
            total_samples = len(combined_df)
            successful_vcdr = combined_df['vcdr'].notna().sum()
            failed_vcdr = total_samples - successful_vcdr
            
            logger.info(f"Successfully processed {total_samples} images across {len(all_results)} datasets")
            logger.info(f"vCDR extraction: {successful_vcdr} successful, {failed_vcdr} failed (kept as NaN)")
            
            return combined_df
        else:
            logger.warning("No samples were processed from any dataset")
            return pd.DataFrame()
    
    def save_vcdr_labels(self, vcdr_df: pd.DataFrame):
        """Save vCDR labels to CSV and create summary statistics."""
        if vcdr_df.empty:
            logger.warning("No vCDR data to save")
            return
        
        # Save main CSV file
        csv_path = os.path.join(self.output_dir, f'vcdr_labels_{self.timestamp}.csv')
        vcdr_df.to_csv(csv_path, index=False)
        logger.info(f"vCDR labels saved to: {csv_path}")
        
        # Create summary statistics
        self.create_summary_statistics(vcdr_df)
        
        # Create visualizations
        self.create_vcdr_visualizations(vcdr_df)
        
        return csv_path
    
    def create_summary_statistics(self, vcdr_df: pd.DataFrame):
        """Create summary statistics for vCDR extraction."""
        summary_stats = []
        
        for dataset in vcdr_df['dataset'].unique():
            dataset_df = vcdr_df[vcdr_df['dataset'] == dataset]
            
            # Count samples with valid vCDR values
            valid_vcdr_df = dataset_df.dropna(subset=['vcdr'])
            
            # Basic statistics
            stats = {
                'dataset': dataset,
                'total_samples': len(dataset_df),
                'vcdr_extracted_samples': len(valid_vcdr_df),
                'vcdr_extraction_rate': len(valid_vcdr_df) / len(dataset_df) if len(dataset_df) > 0 else 0
            }
            
            # vCDR statistics (only for samples with valid vCDR)
            if len(valid_vcdr_df) > 0:
                stats.update({
                    'vcdr_mean': valid_vcdr_df['vcdr'].mean(),
                    'vcdr_std': valid_vcdr_df['vcdr'].std(),
                    'vcdr_min': valid_vcdr_df['vcdr'].min(),
                    'vcdr_max': valid_vcdr_df['vcdr'].max(),
                    'vcdr_median': valid_vcdr_df['vcdr'].median(),
                    'vcdr_q25': valid_vcdr_df['vcdr'].quantile(0.25),
                    'vcdr_q75': valid_vcdr_df['vcdr'].quantile(0.75)
                })
            else:
                stats.update({
                    'vcdr_mean': np.nan,
                    'vcdr_std': np.nan,
                    'vcdr_min': np.nan,
                    'vcdr_max': np.nan,
                    'vcdr_median': np.nan,
                    'vcdr_q25': np.nan,
                    'vcdr_q75': np.nan
                })
            
            # Label-specific statistics if available
            if 'original_label' in dataset_df.columns and dataset_df['original_label'].notna().any():
                labeled_df = dataset_df.dropna(subset=['original_label'])
                stats['labeled_samples'] = len(labeled_df)
                
                if len(labeled_df) > 0:
                    normal_df = labeled_df[labeled_df['original_label'] == 0]
                    glaucoma_df = labeled_df[labeled_df['original_label'] == 1]
                    
                    stats['normal_samples'] = len(normal_df)
                    stats['glaucoma_samples'] = len(glaucoma_df)
                    stats['glaucoma_prevalence'] = len(glaucoma_df) / len(labeled_df)
                    
                    # vCDR statistics by label (only for samples with both label and vCDR)
                    normal_with_vcdr = normal_df.dropna(subset=['vcdr'])
                    glaucoma_with_vcdr = glaucoma_df.dropna(subset=['vcdr'])
                    
                    if len(normal_with_vcdr) > 0:
                        stats['vcdr_mean_normal'] = normal_with_vcdr['vcdr'].mean()
                        stats['vcdr_std_normal'] = normal_with_vcdr['vcdr'].std()
                        stats['normal_samples_with_vcdr'] = len(normal_with_vcdr)
                    
                    if len(glaucoma_with_vcdr) > 0:
                        stats['vcdr_mean_glaucoma'] = glaucoma_with_vcdr['vcdr'].mean()
                        stats['vcdr_std_glaucoma'] = glaucoma_with_vcdr['vcdr'].std()
                        stats['glaucoma_samples_with_vcdr'] = len(glaucoma_with_vcdr)
            
            summary_stats.append(stats)
        
        # Save summary statistics
        summary_df = pd.DataFrame(summary_stats)
        summary_path = os.path.join(self.output_dir, f'vcdr_summary_statistics_{self.timestamp}.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary statistics saved to: {summary_path}")
        
        # Create visual summary table
        self.create_visual_summary_table(summary_df)
    
    def create_visual_summary_table(self, summary_df: pd.DataFrame):
        """Create a visual summary table."""
        # Prepare data for display
        display_df = summary_df.copy()
        
        # Format columns for display
        numeric_cols = ['vcdr_mean', 'vcdr_std', 'vcdr_mean_normal', 'vcdr_std_normal', 
                       'vcdr_mean_glaucoma', 'vcdr_std_glaucoma', 'vcdr_extraction_rate']
        for col in numeric_cols:
            if col in display_df.columns:
                if col == 'vcdr_extraction_rate':
                    display_df[col] = (display_df[col] * 100).round(1)  # Convert to percentage
                else:
                    display_df[col] = display_df[col].round(3)
        
        if 'glaucoma_prevalence' in display_df.columns:
            display_df['glaucoma_prevalence'] = (display_df['glaucoma_prevalence'] * 100).round(1)
        
        # Select key columns for display
        display_cols = ['dataset', 'total_samples', 'vcdr_extracted_samples', 'vcdr_extraction_rate', 'vcdr_mean', 'vcdr_std']
        if 'normal_samples' in display_df.columns:
            display_cols.extend(['normal_samples', 'glaucoma_samples'])
        if 'vcdr_mean_normal' in display_df.columns:
            display_cols.extend(['vcdr_mean_normal', 'vcdr_mean_glaucoma'])
        
        plot_df = display_df[display_cols].fillna('N/A')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, len(display_cols) * 1.5), max(6, len(plot_df) * 0.5 + 2)))
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
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Header styling
        for i in range(len(plot_df.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Row styling
        for i in range(1, len(plot_df) + 1):
            row_color = '#F8F9FA' if i % 2 == 0 else '#FFFFFF'
            for j in range(len(plot_df.columns)):
                table[(i, j)].set_facecolor(row_color)
        
        plt.suptitle('vCDR Extraction Summary Statistics', fontsize=16, fontweight='bold')
        plt.title(f'Generated: {self.timestamp}', fontsize=10, color='gray')
        
        # Save plot
        table_path = os.path.join(self.output_dir, f'vcdr_summary_table_{self.timestamp}.png')
        plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Visual summary table saved to: {table_path}")
    
    def create_vcdr_visualizations(self, vcdr_df: pd.DataFrame):
        """Create vCDR distribution visualizations."""
        # 1. vCDR distribution by dataset
        self.plot_vcdr_by_dataset(vcdr_df)
        
        # 2. vCDR distribution by label (if available)
        if 'original_label' in vcdr_df.columns and vcdr_df['original_label'].notna().any():
            self.plot_vcdr_by_label(vcdr_df)
        
        # 3. Dataset comparison boxplot
        self.plot_dataset_comparison(vcdr_df)
    
    def plot_vcdr_by_dataset(self, vcdr_df: pd.DataFrame):
        """Create vCDR distribution plot by dataset."""
        datasets = vcdr_df['dataset'].unique()
        n_datasets = len(datasets)
        
        # Create subplots
        cols = min(3, n_datasets)
        rows = (n_datasets + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        if n_datasets == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_datasets > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, dataset in enumerate(datasets):
            if i >= len(axes):
                break
            
            dataset_df = vcdr_df[vcdr_df['dataset'] == dataset]
            dataset_vcdr = dataset_df['vcdr'].dropna()  # Remove NaN values for plotting
            
            if len(dataset_vcdr) > 0:
                axes[i].hist(dataset_vcdr, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{dataset}\n(n={len(dataset_df)}, valid vCDR={len(dataset_vcdr)}, μ={dataset_vcdr.mean():.3f})')
            else:
                axes[i].text(0.5, 0.5, f'{dataset}\n(n={len(dataset_df)}, no valid vCDR)', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{dataset}\n(No valid vCDR measurements)')
            
            axes[i].set_xlabel('vCDR')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_datasets, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('vCDR Distribution by Dataset', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, f'vcdr_by_dataset_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"vCDR by dataset plot saved to: {plot_path}")
    
    def plot_vcdr_by_label(self, vcdr_df: pd.DataFrame):
        """Create vCDR distribution plot by original label."""
        labeled_df = vcdr_df.dropna(subset=['original_label'])
        
        if labeled_df.empty:
            logger.warning("No labeled data available for vCDR by label plot")
            return
        
        datasets = labeled_df['dataset'].unique()
        n_datasets = len(datasets)
        
        # Create subplots
        cols = min(3, n_datasets)
        rows = (n_datasets + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        if n_datasets == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_datasets > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, dataset in enumerate(datasets):
            if i >= len(axes):
                break
            
            dataset_df = labeled_df[labeled_df['dataset'] == dataset]
            
            # Get samples with both labels and valid vCDR
            normal_df = dataset_df[dataset_df['original_label'] == 0]
            glaucoma_df = dataset_df[dataset_df['original_label'] == 1]
            
            normal_vcdr = normal_df['vcdr'].dropna()
            glaucoma_vcdr = glaucoma_df['vcdr'].dropna()
            
            if len(normal_vcdr) > 0:
                axes[i].hist(normal_vcdr, bins=20, alpha=0.6, label='Normal', color='lightblue')
            if len(glaucoma_vcdr) > 0:
                axes[i].hist(glaucoma_vcdr, bins=20, alpha=0.6, label='Glaucoma', color='lightcoral')
            
            # Title with counts including total samples vs samples with vCDR
            axes[i].set_title(f'{dataset}\n(N: {len(normal_df)} total, {len(normal_vcdr)} w/vCDR; G: {len(glaucoma_df)} total, {len(glaucoma_vcdr)} w/vCDR)')
            axes[i].set_xlabel('vCDR')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_datasets, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('vCDR Distribution by Original Label', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, f'vcdr_by_label_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"vCDR by label plot saved to: {plot_path}")
    
    def plot_dataset_comparison(self, vcdr_df: pd.DataFrame):
        """Create boxplot comparing vCDR across datasets."""
        # Only plot datasets that have at least some valid vCDR values
        datasets_with_vcdr = []
        for dataset in vcdr_df['dataset'].unique():
            dataset_df = vcdr_df[vcdr_df['dataset'] == dataset]
            if dataset_df['vcdr'].notna().any():
                datasets_with_vcdr.append(dataset)
        
        if not datasets_with_vcdr:
            logger.warning("No datasets with valid vCDR values for comparison plot")
            return
        
        # Filter to only datasets with valid vCDR
        plot_df = vcdr_df[vcdr_df['dataset'].isin(datasets_with_vcdr) & vcdr_df['vcdr'].notna()]
        
        plt.figure(figsize=(max(10, len(datasets_with_vcdr) * 1.5), 8))
        
        # Create boxplot
        sns.boxplot(data=plot_df, x='dataset', y='vcdr')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Dataset')
        plt.ylabel('vCDR')
        plt.title('vCDR Comparison Across Datasets (Valid Measurements Only)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add sample size annotations (total samples vs samples with vCDR)
        for i, dataset in enumerate(datasets_with_vcdr):
            total_count = len(vcdr_df[vcdr_df['dataset'] == dataset])
            valid_count = len(plot_df[plot_df['dataset'] == dataset])
            plt.text(i, plt.ylim()[1] * 0.95, f'n={valid_count}/{total_count}', ha='center', va='top', fontweight='bold')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, f'vcdr_comparison_boxplot_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"vCDR comparison boxplot saved to: {plot_path}")


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
        
    except Exception as e:
        logger.warning(f"Failed to load training data: {e}")
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
    """Main function to extract vCDR labels from all datasets."""
    logger.info("Starting vCDR label extraction...")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"vcdr_extraction_{timestamp}")
    
    # Initialize vCDR extractor
    try:
        extractor = VCDRExtractor(args.unet_model_path, output_dir)
    except Exception as e:
        logger.error(f"Failed to initialize vCDR extractor: {e}")
        sys.exit(1)
    
    # Load training datasets
    training_datasets = load_training_datasets(args)
    
    # Load external datasets
    external_datasets = load_external_datasets(args)
    
    # Process all datasets and extract vCDR values
    vcdr_df = extractor.process_all_datasets(training_datasets, external_datasets)
    
    if vcdr_df.empty:
        logger.error("No samples were processed. Exiting.")
        sys.exit(1)
    
    # Save results
    csv_path = extractor.save_vcdr_labels(vcdr_df)
    
    # Print summary
    print("\n" + "="*80)
    print("vCDR LABEL EXTRACTION SUMMARY")
    print("="*80)
    
    total_samples = len(vcdr_df)
    datasets_processed = vcdr_df['dataset'].nunique()
    valid_vcdr_samples = vcdr_df['vcdr'].notna().sum()
    vcdr_extraction_rate = valid_vcdr_samples / total_samples if total_samples > 0 else 0
    
    print(f"Total samples processed: {total_samples:,}")
    print(f"Datasets processed: {datasets_processed}")
    print(f"Successful vCDR extractions: {valid_vcdr_samples:,} ({vcdr_extraction_rate:.1%})")
    
    if valid_vcdr_samples > 0:
        valid_vcdr_data = vcdr_df['vcdr'].dropna()
        print(f"vCDR range: {valid_vcdr_data.min():.3f} - {valid_vcdr_data.max():.3f}")
        print(f"Mean vCDR: {valid_vcdr_data.mean():.3f} ± {valid_vcdr_data.std():.3f}")
    else:
        print("No valid vCDR measurements extracted")
    
    # Dataset breakdown
    print(f"\nDataset breakdown:")
    for dataset in vcdr_df['dataset'].unique():
        dataset_df = vcdr_df[vcdr_df['dataset'] == dataset]
        dataset_valid_vcdr = dataset_df['vcdr'].notna().sum()
        dataset_extraction_rate = dataset_valid_vcdr / len(dataset_df) if len(dataset_df) > 0 else 0
        
        # vCDR mean only if we have valid measurements
        vcdr_info = ""
        if dataset_valid_vcdr > 0:
            dataset_vcdr_mean = dataset_df['vcdr'].dropna().mean()
            vcdr_info = f", mean vCDR: {dataset_vcdr_mean:.3f}"
        
        # Label information if available
        label_info = ""
        if 'original_label' in dataset_df.columns and dataset_df['original_label'].notna().any():
            labeled_df = dataset_df.dropna(subset=['original_label'])
            if len(labeled_df) > 0:
                glaucoma_count = sum(labeled_df['original_label'])
                normal_count = len(labeled_df) - glaucoma_count
                label_info = f" (Normal: {normal_count}, Glaucoma: {glaucoma_count})"
        
        print(f"  {dataset}: {len(dataset_df):,} total, {dataset_valid_vcdr:,} with vCDR ({dataset_extraction_rate:.1%}){vcdr_info}{label_info}")
    
    print(f"\nResults saved to: {csv_path}")
    print(f"Additional visualizations and statistics saved to: {output_dir}")
    print("="*80)
    
    # Save final JSON summary
    summary_info = {
        'extraction_timestamp': timestamp,
        'total_samples': int(total_samples),
        'datasets_processed': int(datasets_processed),
        'valid_vcdr_samples': int(valid_vcdr_samples),
        'vcdr_extraction_rate': float(vcdr_extraction_rate)
    }
    
    if valid_vcdr_samples > 0:
        valid_vcdr_data = vcdr_df['vcdr'].dropna()
        summary_info['vcdr_statistics'] = {
            'min': float(valid_vcdr_data.min()),
            'max': float(valid_vcdr_data.max()),
            'mean': float(valid_vcdr_data.mean()),
            'std': float(valid_vcdr_data.std()),
            'median': float(valid_vcdr_data.median())
        }
    else:
        summary_info['vcdr_statistics'] = None
    
    summary_info['dataset_breakdown'] = {}
    for dataset in vcdr_df['dataset'].unique():
        dataset_df = vcdr_df[vcdr_df['dataset'] == dataset]
        dataset_valid_vcdr = dataset_df['vcdr'].notna().sum()
        dataset_breakdown = {
            'total_samples': int(len(dataset_df)),
            'valid_vcdr_samples': int(dataset_valid_vcdr),
            'vcdr_extraction_rate': float(dataset_valid_vcdr / len(dataset_df)) if len(dataset_df) > 0 else 0.0
        }
        
        if dataset_valid_vcdr > 0:
            valid_data = dataset_df['vcdr'].dropna()
            dataset_breakdown.update({
                'vcdr_mean': float(valid_data.mean()),
                'vcdr_std': float(valid_data.std())
            })
        
        summary_info['dataset_breakdown'][dataset] = dataset_breakdown
    
    json_path = os.path.join(output_dir, f'extraction_summary_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(summary_info, f, indent=4, cls=NpEncoder)
    
    logger.info(f"Extraction summary saved to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract vCDR labels from all datasets for deep learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./vcdr_extraction_results',
                       help="Directory to save extraction results")
    
    # U-Net model configuration
    parser.add_argument('--unet_model_path', type=str, 
                       default=r'D:\glaucoma\models\best_multitask_model_epoch_25.pth',
                       help="Path to trained U-Net model for vCDR extraction")
    
    # Data configuration
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
                       help="Include AIROGS dataset in extraction")
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
    
    if not os.path.exists(args.unet_model_path):
        logger.error(f"U-Net model not found: {args.unet_model_path}")
        sys.exit(1)
    
    main(args)
