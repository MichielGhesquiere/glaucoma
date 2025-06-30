"""
Standalone evaluation script for already trained models.
This script can evaluate existing model checkpoints and generate missing results.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.datasets import GlaucomaSubgroupDataset, safe_collate
from src.data.transforms import get_transforms
from src.models.classification.build_model import build_classifier_model
from src.utils.helpers import set_seed


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    total_samples = len(y_true)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece


def sensitivity_at_specificity(y_true: np.ndarray, y_prob: np.ndarray, target_specificity: float = 0.95) -> float:
    """Calculate sensitivity at a given specificity threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    
    # Find the threshold that gives us the desired specificity
    idx = np.argmax(specificity >= target_specificity)
    
    if idx == 0 and specificity[0] < target_specificity:
        return 0.0
        
    return tpr[idx]


def load_test_dataset(dataset_name: str, base_data_root: str) -> pd.DataFrame:
    """Load the test dataset based on the dataset name."""
    
    if dataset_name == 'SMDG-BEH':
        # Load SMDG dataset and filter for BEH
        metadata_file = os.path.join(base_data_root, 'raw', 'SMDG-19', 'metadata - standardized.csv')
        smdg_base_dir = os.path.join(base_data_root, 'raw', 'SMDG-19')
        
        df = pd.read_csv(metadata_file)
        
        # Filter for BEH dataset first
        df = df[df['names'].str.contains('BEH', case=False, na=False)].copy()
        
        # Build proper image paths using the fundus column
        # Note: files are actually in full-fundus/full-fundus/ subdirectory
        df['image_path'] = df['fundus'].apply(
            lambda x: os.path.join(smdg_base_dir, x.lstrip('/').replace('full-fundus/', 'full-fundus/full-fundus/')) if pd.notna(x) else ''
        )
        df['file_exists'] = df['image_path'].apply(lambda x: os.path.exists(x) if x else False)
        df = df[df['file_exists']].copy()
        
        df['label'] = df['types'].astype(int)
        
    elif dataset_name == 'CHAKSU':
        # Load CHAKSU dataset - you'll need to adapt this based on your CHAKSU structure
        chaksu_base_dir = os.path.join(base_data_root, 'raw', 'Chaksu', 'Train', 'Train', '1.0_Original_Fundus_Images')
        chaksu_decision_dir = os.path.join(base_data_root, 'raw', 'Chaksu', 'Train', 'Train', '6.0_Glaucoma_Decision')
        
        # This is a simplified loader - you may need to adapt based on actual structure
        metadata_files = list(Path(chaksu_decision_dir).glob("*.csv"))
        dfs = []
        for file in metadata_files:
            df_part = pd.read_csv(file)
            dfs.append(df_part)
        
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df['image_path'] = df['image_name'].apply(lambda x: os.path.join(chaksu_base_dir, x))
            df['file_exists'] = df['image_path'].apply(os.path.exists)
            df = df[df['file_exists']].copy()
            df['label'] = df['glaucoma_risk'].astype(int)
        else:
            df = pd.DataFrame()
            
    elif dataset_name == 'AIROGS':
        # Load AIROGS dataset
        airogs_label_file = os.path.join(base_data_root, 'raw', 'AIROGS', 'train_labels.csv')
        airogs_image_dir = os.path.join(base_data_root, 'raw', 'AIROGS', 'img')
        
        df = pd.read_csv(airogs_label_file)
        df['image_path'] = df['challenge_id'].apply(lambda x: os.path.join(airogs_image_dir, f"{x}.jpg"))
        df['file_exists'] = df['image_path'].apply(os.path.exists)
        df = df[df['file_exists']].copy()
        df['label'] = df['class'].astype(int)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return df


def evaluate_model(model_path: str, dataset_name: str, base_data_root: str, device: str = 'cuda'):
    """Evaluate a trained model on a specific dataset."""
    
    logging.info(f"Evaluating model: {model_path}")
    logging.info(f"Dataset: {dataset_name}")
    
    # Load test dataset
    test_df = load_test_dataset(dataset_name, base_data_root)
    
    if test_df.empty:
        logging.error(f"No test data found for dataset: {dataset_name}")
        logging.error(f"Base data root: {base_data_root}")
        # Additional debugging for SMDG-BEH
        if dataset_name == 'SMDG-BEH':
            metadata_file = os.path.join(base_data_root, 'raw', 'SMDG-19', 'metadata - standardized.csv')
            logging.error(f"Metadata file exists: {os.path.exists(metadata_file)}")
            if os.path.exists(metadata_file):
                df_debug = pd.read_csv(metadata_file)
                beh_count = len(df_debug[df_debug['names'].str.contains('BEH', case=False, na=False)])
                logging.error(f"Total BEH samples in metadata: {beh_count}")
        raise ValueError(f"No test data found for dataset: {dataset_name}")
        
    logging.info(f"Loaded {len(test_df)} test samples for {dataset_name}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract model configuration from checkpoint if available
    model_name = 'vit_base_patch16_224'  # Default for VFM
    num_classes = 2
    
    # Build model architecture
    model = build_classifier_model(
        model_name=model_name,
        num_classes=num_classes,
        custom_weights_path=None  # We'll load from checkpoint
    )
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Get transforms
    _, eval_transforms = get_transforms(224, model_name, False)
    
    # Create dataset and loader
    test_dataset = GlaucomaSubgroupDataset(test_df, transform=eval_transforms)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4,
        collate_fn=safe_collate
    )
    
    # Evaluate
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    logging.info("Running evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # Handle the 3-tuple return from GlaucomaSubgroupDataset
            if len(batch_data) == 3:
                images, targets, _ = batch_data  # Ignore attributes for evaluation
            else:
                images, targets = batch_data
                
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            if batch_idx % 10 == 0:
                logging.info(f"Processed batch {batch_idx}/{len(test_loader)}")
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    auc = roc_auc_score(all_targets, all_probabilities)
    ece = compute_ece(np.array(all_targets), np.array(all_probabilities))
    sens_at_95_spec = sensitivity_at_specificity(
        np.array(all_targets), np.array(all_probabilities), target_specificity=0.95
    )
    
    results = {
        'model_path': model_path,
        'dataset': dataset_name,
        'num_samples': len(all_targets),
        'accuracy': accuracy,
        'auc': auc,
        'ece': ece,
        'sensitivity_at_95_specificity': sens_at_95_spec
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model (.pth file)')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Name of the dataset to evaluate on (e.g., SMDG-BEH, CHAKSU, AIROGS)')
    parser.add_argument('--base_data_root', type=str, default='D:/glaucoma/data',
                       help='Base directory for datasets')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for results (JSON)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run evaluation
        results = evaluate_model(
            args.model_path, 
            args.dataset_name, 
            args.base_data_root,
            args.device
        )
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Model: {args.model_path}")
        print(f"Dataset: {args.dataset_name}")
        print(f"Samples: {results['num_samples']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"AUC: {results['auc']:.4f}")
        print(f"ECE: {results['ece']:.4f}")
        print(f"Sensitivity@95%Spec: {results['sensitivity_at_95_specificity']:.4f}")
        print("="*60)
        
        # Save results if output file specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output_file}")
            
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
