"""
Temperature scaling calibration for trained models.

This script applies temperature scaling to existing trained models to improve calibration
without retraining. It saves calibrated models with "_calibrated" suffix.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import logging
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Add parent directory to path to import from train script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the training script
from train_multitask_classification_regression import (
    MultiTaskModel, SingleTaskModel, MultiTaskDataset,
    load_vcdr_data, create_domain_splits, get_data_transforms,
    calculate_ece, calculate_brier_score, set_seed, NpEncoder
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification model outputting the prediction logits
    """
    def __init__(self, model, task_mode='multitask'):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.task_mode = task_mode
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initialize to 1.5

    def forward(self, input):
        output = self.model(input)
        if self.task_mode == 'multitask':
            # For multitask models, apply temperature scaling only to classification logits
            if isinstance(output, tuple):
                class_logits, vcdr_pred = output
                return class_logits / self.temperature, vcdr_pred
            elif isinstance(output, dict):
                class_logits = output.get('classification_logits', output.get('logits'))
                vcdr_pred = output.get('regression_output', output.get('vcdr'))
                if class_logits is not None:
                    output_dict = output.copy()
                    output_dict['classification_logits'] = class_logits / self.temperature
                    return output_dict
                else:
                    return output
            else:
                return output / self.temperature, None
        else:
            # For single-task models, apply temperature scaling to the classification logits
            if isinstance(output, dict):
                class_logits = output.get('classification_logits', output.get('logits', output))
                if class_logits is not None:
                    output_dict = output.copy()
                    output_dict['classification_logits'] = class_logits / self.temperature
                    return output_dict
                else:
                    return output / self.temperature
            else:
                return output / self.temperature

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        if isinstance(logits, dict):
            scaled_dict = logits.copy()
            class_logits = logits.get('classification_logits', logits.get('logits'))
            if class_logits is not None:
                scaled_dict['classification_logits'] = class_logits / self.temperature
            return scaled_dict
        else:
            # Handle both multi-class logits and single sigmoid outputs
            return logits / self.temperature


def calibrate_model_on_validation(model, val_loader, device, task_mode='multitask', max_iter=50):
    """
    Tune the temperature of the model using the validation set.
    
    Args:
        model: The trained model to calibrate
        val_loader: Validation data loader
        device: Device to run on
        task_mode: 'multitask' or 'singletask'
        max_iter: Maximum iterations for temperature optimization
        
    Returns:
        calibrated_model: Model wrapped with temperature scaling
        temperature: Optimal temperature value found
        calibration_metrics: Before/after calibration metrics
    """
    logger.info("Starting temperature scaling calibration...")
    
    # Wrap model with temperature scaling
    calibrated_model = TemperatureScaling(model, task_mode)
    calibrated_model = calibrated_model.to(device)
    calibrated_model.eval()
    
    # Collect validation predictions and targets
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            targets = batch['binary_label'].to(device)
            
            # Filter out samples without binary labels
            valid_mask = targets != -1
            if not valid_mask.any():
                continue
                
            images = images[valid_mask]
            targets = targets[valid_mask].long()
            
            # Get logits from original model
            output = model(images)
            if task_mode == 'multitask':
                if isinstance(output, tuple):
                    logits, _ = output
                elif isinstance(output, dict):
                    logits = output.get('classification_logits', output.get('logits'))
                else:
                    logits = output
            else:
                if isinstance(output, dict):
                    logits = output.get('classification_logits', output.get('logits', output))
                else:
                    logits = output
            
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    
    if not all_logits:
        raise ValueError("No valid validation samples found for calibration")
    
    all_logits = torch.cat(all_logits, dim=0).to(device)
    all_targets = torch.cat(all_targets, dim=0).to(device)
    
    logger.info(f"Calibrating on {len(all_targets)} validation samples")
    
    # Calculate metrics before calibration
    with torch.no_grad():
        # Handle both 1D and 2D tensor outputs
        if all_logits.dim() == 1 or (all_logits.dim() == 2 and all_logits.size(1) == 1):
            # Single output (sigmoid)
            probs_before = torch.sigmoid(all_logits)
            if all_logits.dim() == 2:
                predicted_probs_before = probs_before.squeeze(1).cpu().numpy()
            else:
                predicted_probs_before = probs_before.cpu().numpy()
        else:
            # Multi-class logits
            probs_before = F.softmax(all_logits, dim=1)
            predicted_probs_before = probs_before[:, 1].cpu().numpy()  # Probability of positive class
        
        targets_numpy = all_targets.cpu().numpy()
        
        ece_before = calculate_ece(targets_numpy, predicted_probs_before)
        brier_before = calculate_brier_score(targets_numpy, predicted_probs_before)
    
    # Optimize temperature
    optimizer = torch.optim.LBFGS([calibrated_model.temperature], lr=0.01, max_iter=max_iter)
    
    def eval_loss():
        optimizer.zero_grad()
        scaled_output = calibrated_model.temperature_scale(all_logits)
        if isinstance(scaled_output, dict):
            scaled_logits = scaled_output.get('classification_logits', scaled_output.get('logits'))
        else:
            scaled_logits = scaled_output
        
        # Use appropriate loss function based on output dimensions
        if scaled_logits.dim() == 1 or (scaled_logits.dim() == 2 and scaled_logits.size(1) == 1):
            # Binary classification with single output
            criterion = nn.BCEWithLogitsLoss()
            if scaled_logits.dim() == 2:
                scaled_logits = scaled_logits.squeeze(1)
            loss = criterion(scaled_logits, all_targets.float())
        else:
            # Multi-class classification
            criterion = nn.CrossEntropyLoss()
            loss = criterion(scaled_logits, all_targets)
        
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    
    # Calculate metrics after calibration
    with torch.no_grad():
        calibrated_output = calibrated_model.temperature_scale(all_logits)
        if isinstance(calibrated_output, dict):
            calibrated_logits = calibrated_output.get('classification_logits', calibrated_output.get('logits'))
        else:
            calibrated_logits = calibrated_output
        
        # Handle both 1D and 2D tensor outputs
        if calibrated_logits.dim() == 1 or (calibrated_logits.dim() == 2 and calibrated_logits.size(1) == 1):
            # Single output (sigmoid)
            probs_after = torch.sigmoid(calibrated_logits)
            if calibrated_logits.dim() == 2:
                predicted_probs_after = probs_after.squeeze(1).cpu().numpy()
            else:
                predicted_probs_after = probs_after.cpu().numpy()
        else:
            # Multi-class logits
            probs_after = F.softmax(calibrated_logits, dim=1)
            predicted_probs_after = probs_after[:, 1].cpu().numpy()
        
        ece_after = calculate_ece(targets_numpy, predicted_probs_after)
        brier_after = calculate_brier_score(targets_numpy, predicted_probs_after)
    
    temperature_value = calibrated_model.temperature.item()
    
    calibration_metrics = {
        'temperature': temperature_value,
        'ece_before': ece_before,
        'ece_after': ece_after,
        'ece_improvement': ece_before - ece_after,
        'ece_improved': ece_after < ece_before,
        'brier_before': brier_before,
        'brier_after': brier_after,
        'brier_improvement': brier_before - brier_after,
        'brier_improved': brier_after < brier_before,
        'calibration_beneficial': (ece_after < ece_before) or (brier_after < brier_before),
        'validation_samples': len(all_targets)
    }
    
    logger.info(f"Temperature scaling completed:")
    logger.info(f"  Optimal temperature: {temperature_value:.4f}")
    
    # For calibration metrics, lower is better, so negative delta indicates improvement
    ece_change = ece_after - ece_before
    brier_change = brier_after - brier_before
    
    ece_status = "improved" if ece_change < 0 else "degraded"
    brier_status = "improved" if brier_change < 0 else "degraded"
    
    logger.info(f"  ECE {ece_status}: {ece_before:.4f} → {ece_after:.4f} (Δ={ece_change:+.4f})")
    logger.info(f"  Brier {brier_status}: {brier_before:.4f} → {brier_after:.4f} (Δ={brier_change:+.4f})")
    
    if ece_change > 0 and brier_change > 0:
        logger.info("  📊 Both metrics degraded - calibration may not be beneficial for this model")
    elif ece_change > 0:
        logger.info("  📊 ECE degraded, Brier improved - mixed calibration results")
    elif brier_change > 0:
        logger.info("  📊 Brier degraded, ECE improved - mixed calibration results")
    else:
        logger.info("  📊 Both metrics improved - calibration was beneficial")
    
    return calibrated_model, temperature_value, calibration_metrics


def evaluate_calibrated_model(calibrated_model, test_loader, device, task_mode='multitask'):
    """
    Evaluate the calibrated model on test set.
    
    Args:
        calibrated_model: Calibrated model with temperature scaling
        test_loader: Test data loader
        device: Device to run on
        task_mode: 'multitask' or 'singletask'
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    calibrated_model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_vcdr_predictions = []
    all_vcdr_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            binary_targets = batch['binary_label'].to(device)
            vcdr_targets = batch['vcdr'].to(device) if 'vcdr' in batch else None
            
            # Forward pass
            if task_mode == 'multitask':
                output = calibrated_model(images)
                if isinstance(output, tuple):
                    logits, vcdr_pred = output
                elif isinstance(output, dict):
                    logits = output.get('classification_logits', output.get('logits'))
                    vcdr_pred = output.get('regression_output', output.get('vcdr'))
                else:
                    logits = output
                    vcdr_pred = None
            else:
                output = calibrated_model(images)
                if isinstance(output, dict):
                    logits = output.get('classification_logits', output.get('logits', output))
                else:
                    logits = output
                vcdr_pred = None
            
            # Get probabilities and predictions
            if logits.dim() == 1 or logits.shape[-1] == 1:
                # Single output (sigmoid), convert to binary classification logits
                if logits.dim() == 1:
                    logits = logits.unsqueeze(-1)  # Add feature dimension
                # Convert sigmoid output to logits for binary classification
                pos_probs = torch.sigmoid(logits.squeeze(-1))
                neg_probs = 1 - pos_probs
                probabilities = torch.stack([neg_probs, pos_probs], dim=1)
                predictions = (pos_probs > 0.5).long()
            else:
                # Multi-class logits, apply softmax normally
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
            
            # Store results for samples with valid binary labels
            valid_binary_mask = binary_targets != -1
            if valid_binary_mask.any():
                all_predictions.extend(predictions[valid_binary_mask].cpu().numpy())
                all_probabilities.extend(probabilities[valid_binary_mask, 1].cpu().numpy())  # Positive class prob
                all_targets.extend(binary_targets[valid_binary_mask].cpu().numpy())
            
            # Store vCDR results if available
            if vcdr_pred is not None and vcdr_targets is not None:
                valid_vcdr_mask = vcdr_targets != -1
                if valid_vcdr_mask.any():
                    all_vcdr_predictions.extend(vcdr_pred[valid_vcdr_mask].cpu().numpy())
                    all_vcdr_targets.extend(vcdr_targets[valid_vcdr_mask].cpu().numpy())
    
    # Calculate classification metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {}
    
    if all_targets:
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(all_targets, all_predictions)
        metrics['precision'] = precision_score(all_targets, all_predictions, zero_division=0)
        metrics['recall'] = recall_score(all_targets, all_predictions, zero_division=0)
        metrics['sensitivity'] = metrics['recall']  # Sensitivity = Recall
        metrics['f1'] = f1_score(all_targets, all_predictions, zero_division=0)
        
        # Specificity
        tn = np.sum((all_targets == 0) & (all_predictions == 0))
        fp = np.sum((all_targets == 0) & (all_predictions == 1))
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # AUC
        if len(np.unique(all_targets)) > 1:
            metrics['auc'] = roc_auc_score(all_targets, all_probabilities)
        else:
            metrics['auc'] = 0.5
        
        # Calibration metrics
        metrics['ece'] = calculate_ece(all_targets, all_probabilities)
        metrics['brier_score'] = calculate_brier_score(all_targets, all_probabilities)
    
    # Calculate regression metrics if available
    if all_vcdr_targets and task_mode == 'multitask':
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        all_vcdr_targets = np.array(all_vcdr_targets)
        all_vcdr_predictions = np.array(all_vcdr_predictions)
        
        metrics['mse'] = mean_squared_error(all_vcdr_targets, all_vcdr_predictions)
        metrics['mae'] = mean_absolute_error(all_vcdr_targets, all_vcdr_predictions)
        metrics['r2'] = r2_score(all_vcdr_targets, all_vcdr_predictions)
    
    return metrics


def calibrate_single_fold(df: pd.DataFrame, 
                         test_dataset: str, 
                         output_dir: str, 
                         args: argparse.Namespace) -> Dict:
    """
    Calibrate an existing trained model for a single fold.
    
    Args:
        df: DataFrame with all data
        test_dataset: Name of test dataset
        output_dir: Output directory containing trained models
        args: Command line arguments
        
    Returns:
        Dictionary with calibration results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"CALIBRATING FOLD: TEST DATASET = {test_dataset}")
    logger.info(f"Using backbone: {args.backbone}")
    logger.info(f"{'='*80}")
    
    # Determine task modes to calibrate
    if args.compare_tasks:
        task_modes = ['singletask', 'multitask']
        fold_dir = os.path.join(output_dir, f'fold_{test_dataset}_{args.backbone}_comparison')
    else:
        task_modes = ['multitask']
        fold_dir = os.path.join(output_dir, f'fold_{test_dataset}_{args.backbone}')
    
    if not os.path.exists(fold_dir):
        logger.error(f"No existing models found for {test_dataset} at: {fold_dir}")
        return None
    
    # Create splits
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
    train_transform, val_transform = get_data_transforms(augment=False, backbone=args.backbone)
    
    # Create datasets
    val_data = MultiTaskDataset(val_df, val_transform, vcdr_scaler)
    test_data = MultiTaskDataset(test_df, val_transform, vcdr_scaler)
    
    # Create data loaders
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    for task_mode in task_modes:
        logger.info(f"\n{'-'*60}")
        logger.info(f"Calibrating {task_mode.upper()} model")
        logger.info(f"{'-'*60}")
        
        # Determine model path
        if args.compare_tasks:
            model_path = os.path.join(fold_dir, task_mode, f'best_model_{args.backbone}.pth')
        else:
            model_path = os.path.join(fold_dir, f'best_model_{args.backbone}.pth')
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            continue
        
        # Create and load model
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
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load vCDR scaler from checkpoint if available
            if 'vcdr_scaler' in checkpoint and checkpoint['vcdr_scaler'] is not None:
                vcdr_scaler = checkpoint['vcdr_scaler']
                logger.info("Loaded vCDR scaler from checkpoint")
            
            logger.info(f"Successfully loaded {task_mode} model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load {task_mode} model: {e}")
            continue
        
        model = model.to(device)
        model.eval()
        
        # Evaluate original model on test set for comparison
        logger.info("Evaluating original model on test set...")
        original_metrics = evaluate_calibrated_model(model, test_loader, device, task_mode)
        
        # Calibrate model using validation set
        try:
            calibrated_model, temperature, calibration_metrics = calibrate_model_on_validation(
                model, val_loader, device, task_mode, max_iter=args.max_iter
            )
        except Exception as e:
            logger.error(f"Calibration failed for {task_mode}: {e}")
            continue
        
        # Evaluate calibrated model on test set
        logger.info("Evaluating calibrated model on test set...")
        calibrated_metrics = evaluate_calibrated_model(calibrated_model, test_loader, device, task_mode)
        
        # Save calibrated model
        calibrated_model_path = model_path.replace('.pth', '_calibrated.pth')
        torch.save({
            'model_state_dict': calibrated_model.model.state_dict(),
            'temperature': temperature,
            'calibration_metrics': calibration_metrics,
            'vcdr_scaler': vcdr_scaler,
            'task_mode': task_mode,
            'backbone': args.backbone,
            'calibration_timestamp': datetime.now().isoformat()
        }, calibrated_model_path)
        
        logger.info(f"Calibrated model saved to: {calibrated_model_path}")
        
        # Store results
        task_results = {
            'test_dataset': test_dataset,
            'backbone': args.backbone,
            'task_mode': task_mode,
            'calibration_metrics': calibration_metrics,
            'original_test_metrics': original_metrics,
            'calibrated_test_metrics': calibrated_metrics,
            'model_path': model_path,
            'calibrated_model_path': calibrated_model_path,
            'val_size': len(val_df),
            'test_size': len(test_df)
        }
        
        # Save results
        results_path = calibrated_model_path.replace('.pth', '_results.json')
        with open(results_path, 'w') as f:
            json.dump(task_results, f, indent=4, cls=NpEncoder)
        
        results[task_mode] = task_results
        
        # Log comparison
        logger.info(f"\n{task_mode.upper()} Calibration Results:")
        logger.info(f"  Temperature: {temperature:.4f}")
        logger.info(f"  Original ECE: {original_metrics.get('ece', 'N/A'):.4f}")
        logger.info(f"  Calibrated ECE: {calibrated_metrics.get('ece', 'N/A'):.4f}")
        logger.info(f"  ECE improvement: {original_metrics.get('ece', 0) - calibrated_metrics.get('ece', 0):+.4f}")
        logger.info(f"  Original Brier: {original_metrics.get('brier_score', 'N/A'):.4f}")
        logger.info(f"  Calibrated Brier: {calibrated_metrics.get('brier_score', 'N/A'):.4f}")
        logger.info(f"  Brier improvement: {original_metrics.get('brier_score', 0) - calibrated_metrics.get('brier_score', 0):+.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Temperature scaling calibration for trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument('--vcdr_csv', type=str, 
                       default=r'D:\glaucoma\vcdr_extraction_results\vcdr_extraction_20250711_170055\vcdr_labels_20250711_170055.csv',
                       help="Path to CSV file with vCDR labels")
    parser.add_argument('--min_samples_per_dataset', type=int, default=100,
                       help="Minimum samples per dataset to include")
    parser.add_argument('--exclude_datasets', type=str, nargs='*', default=['G1020_all'],
                       help="List of datasets to exclude")
    
    # Model configuration
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'vfm'],
                       help="Backbone architecture")
    parser.add_argument('--vfm_weights_path', type=str, 
                       default=r'D:\glaucoma\models\VFM_Fundus_weights.pth',
                       help="Path to VFM pre-trained weights")
    parser.add_argument('--dropout', type=float, default=0.5,
                       help="Dropout rate")
    
    # Calibration configuration
    parser.add_argument('--max_iter', type=int, default=50,
                       help="Maximum iterations for temperature optimization")
    parser.add_argument('--compare_tasks', action='store_true', default=False,
                       help="Calibrate both single-task and multi-task models")
    
    # Data loading configuration
    parser.add_argument('--batch_size', type=int, default=128,
                       help="Batch size for evaluation")
    parser.add_argument('--num_workers', type=int, default=4,
                       help="Number of data loader workers")
    
    # I/O configuration
    parser.add_argument('--input_dir', type=str, 
                       default=r'D:\glaucoma\multitask_results',
                       help="Directory containing trained models")
    parser.add_argument('--datasets', type=str, nargs='*', default=None,
                       help="Specific datasets to calibrate (default: all)")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
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
    
    # Filter to specific datasets if requested
    if args.datasets:
        valid_datasets = [d for d in valid_datasets if d in args.datasets]
        logger.info(f"Calibrating specific datasets: {valid_datasets}")
    
    logger.info(f"Valid datasets for calibration: {valid_datasets}")
    df = df[df['dataset'].isin(valid_datasets)]
    
    if len(valid_datasets) < 1:
        logger.error("No valid datasets found for calibration")
        return
    
    # Calibrate models for each dataset
    all_results = []
    
    for test_dataset in valid_datasets:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting calibration for: {test_dataset}")
            logger.info(f"{'='*60}")
            
            fold_results = calibrate_single_fold(df, test_dataset, args.input_dir, args)
            
            if fold_results is not None:
                all_results.append(fold_results)
                logger.info(f"✅ Successfully calibrated fold: {test_dataset}")
            else:
                logger.warning(f"⚠️ Skipped {test_dataset} - no models found")
            
        except Exception as e:
            logger.error(f"❌ Failed to calibrate fold {test_dataset}: {e}")
            import traceback
            logger.error(f"Detailed error traceback:\n{traceback.format_exc()}")
            continue
    
    # Summary
    if all_results:
        logger.info(f"\n{'='*80}")
        logger.info("CALIBRATION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Successfully calibrated {len(all_results)} datasets")
        
        # Calculate average improvements and success rates
        ece_improvements = []
        brier_improvements = []
        temperatures = []
        calibration_beneficial_count = 0
        ece_improved_count = 0
        brier_improved_count = 0
        total_calibrations = 0
        
        for result in all_results:
            for task_mode in result:
                if isinstance(result[task_mode], dict):
                    calib_metrics = result[task_mode].get('calibration_metrics', {})
                    total_calibrations += 1
                    
                    if 'ece_improvement' in calib_metrics:
                        ece_improvements.append(calib_metrics['ece_improvement'])
                    if 'brier_improvement' in calib_metrics:
                        brier_improvements.append(calib_metrics['brier_improvement'])
                    if 'temperature' in calib_metrics:
                        temperatures.append(calib_metrics['temperature'])
                    
                    # Count successes
                    if calib_metrics.get('ece_improved', False):
                        ece_improved_count += 1
                    if calib_metrics.get('brier_improved', False):
                        brier_improved_count += 1
                    if calib_metrics.get('calibration_beneficial', False):
                        calibration_beneficial_count += 1
        
        if total_calibrations > 0:
            logger.info(f"Calibration Success Rates:")
            logger.info(f"  ECE improved: {ece_improved_count}/{total_calibrations} ({100*ece_improved_count/total_calibrations:.1f}%)")
            logger.info(f"  Brier improved: {brier_improved_count}/{total_calibrations} ({100*brier_improved_count/total_calibrations:.1f}%)")
            logger.info(f"  At least one metric improved: {calibration_beneficial_count}/{total_calibrations} ({100*calibration_beneficial_count/total_calibrations:.1f}%)")
            
        if ece_improvements:
            logger.info(f"Average ECE change: {np.mean(ece_improvements):+.4f} ± {np.std(ece_improvements):.4f}")
        if brier_improvements:
            logger.info(f"Average Brier change: {np.mean(brier_improvements):+.4f} ± {np.std(brier_improvements):.4f}")
        if temperatures:
            logger.info(f"Average temperature: {np.mean(temperatures):.4f} ± {np.std(temperatures):.4f}")
        
        logger.info("Calibrated models saved with '_calibrated' suffix")
        logger.info(f"{'='*80}")
    else:
        logger.error("No models were successfully calibrated")


if __name__ == "__main__":
    main()
