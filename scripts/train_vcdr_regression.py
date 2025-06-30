"""
Refactored VCDR Regression Model Training Script.

This script trains models for Vertical Cup-to-Disc Ratio (VCDR) regression
using the G-RISK dataset. It has been modularized for better maintainability.
"""

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modular components
from src.data.grisk_loader import load_and_preprocess_data
from src.data.vcdr_dataset import VCDRDataset
from src.data.vcdr_splits import split_data_by_patient
from src.data.transforms import get_vcdr_transforms
from src.models.regression.vcdr_model import build_regressor_model
from src.training.vcdr_trainer import train_epoch, validate_epoch, print_training_summary
from src.evaluation.vcdr_metrics import evaluate_model_performance
from src.utils.logging_utils import setup_logging


def setup_experiment_logging(base_output_dir: str, experiment_tag: str, model_name: str, seed: int) -> Path:
    """Setup experiment-specific logging and directory structure."""
    # Create experiment directory
    experiment_id = f"{experiment_tag}_{model_name}_seed{seed}"
    experiment_dir = Path(base_output_dir) / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging for this experiment
    log_file = experiment_dir / 'training_log.txt'
    setup_logging(log_level='INFO', log_file=str(log_file))
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {experiment_id}")
    logger.info(f"Experiment directory: {experiment_dir}")
    
    return experiment_dir


def prepare_data(args) -> tuple:
    """Prepare and split the dataset."""
    logger = logging.getLogger(__name__)
    
    # Determine processed data path
    processed_data_path = Path(args.base_data_root) / "processed" / "grisk_processed_metadata.csv"
    
    # Load or process data
    if args.use_processed_data and processed_data_path.exists():
        logger.info(f"Loading processed data from: {processed_data_path}")
        df = pd.read_csv(processed_data_path)
    else:
        logger.info("Processing raw data...")
        df = load_and_preprocess_data(args.grisk_data_dir, str(processed_data_path))
    
    # Split data by patient
    train_df, val_df, test_df = split_data_by_patient(
        df, 
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    
    return train_df, val_df, test_df


def create_data_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, args) -> tuple:
    """Create PyTorch DataLoaders."""
    # Get transforms
    train_transforms, val_transforms = get_vcdr_transforms(image_size=224)
    
    # Create datasets
    train_dataset = VCDRDataset(train_df, transform=train_transforms)
    val_dataset = VCDRDataset(val_df, transform=val_transforms)
    test_dataset = VCDRDataset(test_df, transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    return train_loader, val_loader, test_loader


def setup_model_and_training(args) -> tuple:
    """Setup model, optimizer, scheduler, and loss function."""
    logger = logging.getLogger(__name__)
    
    # Build model
    model = build_regressor_model(
        model_name=args.model_name,
        num_classes=1,
        custom_pretrained_path=args.custom_weights_path,
        use_timm_hub_pretrained=args.use_timm_pretrained_if_no_custom,
        dropout_rate=args.head_dropout_prob
    )
    
    # Setup device and move model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.2, 
        patience=args.lr_scheduler_patience, 
        verbose=True, 
        min_lr=1e-7
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model_name}, Total trainable parameters: {total_params:,}")
    logger.info(f"Device: {device}")
    
    return model, criterion, optimizer, scheduler, device


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args) -> dict:
    """Main training loop."""
    logger = logging.getLogger(__name__)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_mae': [], 'val_mae': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_completed = 0
    
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    training_start_time = time.time()
    
    for epoch in range(args.num_epochs):
        epochs_completed += 1
        epoch_start_time = time.time()
        
        logger.info(f"--- Epoch {epoch + 1}/{args.num_epochs} ---")
        
        # Train and validate
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            logger.info(f"🌟 New best validation loss: {best_val_loss:.6f}")
        
        # Log epoch results
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1} Summary ({epoch_duration:.1f}s):")
        logger.info(f"  Train -> Loss: {train_loss:.6f}, MAE: {train_mae:.6f}")
        logger.info(f"  Valid -> Loss: {val_loss:.6f}, MAE: {val_mae:.6f}")
        
        if optimizer.param_groups[0]['lr'] < current_lr:
            logger.info(f"  Learning rate reduced to {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping check
        if epoch >= args.early_stopping_patience:
            if len(history['val_loss']) > args.early_stopping_patience:
                best_val_loss_epoch_idx = history['val_loss'].index(min(history['val_loss']))
                if (epoch - best_val_loss_epoch_idx) >= args.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model state for evaluation")
    
    total_training_time = time.time() - training_start_time
    logger.info(f"Training completed in {total_training_time / 60:.1f} minutes")
    
    # Print training summary
    if epochs_completed > 0:
        print_training_summary(
            history['train_loss'], 
            history['val_loss'], 
            history['train_mae'], 
            history['val_mae']
        )
    
    return history, best_model_state, best_val_loss


def evaluate_test_set(model, test_loader, device) -> dict:
    """Evaluate model on test set."""
    logger = logging.getLogger(__name__)
    
    logger.info("Evaluating on test set...")
    
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images).squeeze()
            
            test_predictions.extend(outputs.cpu().numpy().tolist())
            test_targets.extend(targets.cpu().numpy().tolist())
    
    # Calculate metrics
    if len(test_targets) > 0:
        metrics = evaluate_model_performance(test_predictions, test_targets, "Test Set Performance")
        return metrics
    else:
        logger.warning("No test data available for evaluation")
        return {}


def save_experiment_results(experiment_dir: Path, model_name: str, best_model_state, optimizer, args, metrics: dict) -> str:
    """Save model and experiment results."""
    logger = logging.getLogger(__name__)
    
    if best_model_state:
        model_save_path = experiment_dir / f"{model_name}_best_regressor.pth"
        torch.save({
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'training_args': vars(args),
            'test_metrics': metrics
        }, model_save_path)
        logger.info(f"Best model saved to: {model_save_path}")
        return str(model_save_path)
    else:
        logger.warning("No best model state to save")
        return "N/A"


def run_experiment(args):
    """Run a complete VCDR regression experiment."""
    logger = logging.getLogger(__name__)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup experiment
    experiment_dir = setup_experiment_logging(
        args.base_output_dir, 
        args.experiment_tag, 
        args.model_name, 
        args.seed
    )
    
    try:
        # Prepare data
        train_df, val_df, test_df = prepare_data(args)
        train_loader, val_loader, test_loader = create_data_loaders(train_df, val_df, test_df, args)
        
        # Setup model and training
        model, criterion, optimizer, scheduler, device = setup_model_and_training(args)
        
        # Train model
        history, best_model_state, best_val_loss = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, device, args
        )
        
        # Evaluate on test set
        test_metrics = evaluate_test_set(model, test_loader, device)
        
        # Save results
        model_path = save_experiment_results(
            experiment_dir, args.model_name, best_model_state, optimizer, args, test_metrics
        )
        
        # Return experiment summary
        return {
            "model_name": args.model_name,
            "experiment_tag": args.experiment_tag,
            "seed": args.seed,
            "best_val_loss": best_val_loss,
            "test_mae": test_metrics.get('mae', float('inf')),
            "test_r2": test_metrics.get('r2', float('-inf')),
            "model_path": model_path,
            "experiment_dir": str(experiment_dir)
        }
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="VCDR Regression Model Training Script")
    
    # Data arguments
    data_group = parser.add_argument_group('Data Arguments')
    data_group.add_argument('--grisk_data_dir', type=str, default=r"data\raw\griskFundus",
                           help="Path to G-RISK raw fundus image directory")
    data_group.add_argument('--base_data_root', type=str, default=r"data",
                           help="Base root for data")
    data_group.add_argument('--use_processed_data', action='store_true',
                           help="Load from pre-processed CSV if available")
    data_group.add_argument('--train_ratio', type=float, default=0.7,
                           help="Proportion of patients for training")
    data_group.add_argument('--val_ratio', type=float, default=0.15,
                           help="Proportion of patients for validation")
    data_group.add_argument('--test_ratio', type=float, default=0.15,
                           help="Proportion of patients for testing")
    
    # Model arguments
    model_group = parser.add_argument_group('Model Arguments')
    model_group.add_argument('--model_name', type=str, required=True,
                            help="Model architecture name")
    model_group.add_argument('--custom_weights_path', type=str, default=None,
                            help="Path to custom pretrained weights")
    model_group.add_argument('--use_timm_pretrained_if_no_custom', action='store_true', default=True,
                            help="Use TIMM/Hub pretrained weights if no custom weights")
    model_group.add_argument('--head_dropout_prob', type=float, default=0.3,
                            help="Dropout probability for regression head")
    
    # Training arguments
    train_group = parser.add_argument_group('Training Arguments')
    train_group.add_argument('--num_epochs', type=int, default=50,
                            help="Number of training epochs")
    train_group.add_argument('--batch_size', type=int, default=16,
                            help="Training batch size")
    train_group.add_argument('--eval_batch_size', type=int, default=32,
                            help="Evaluation batch size")
    train_group.add_argument('--learning_rate', type=float, default=1e-5,
                            help="Initial learning rate")
    train_group.add_argument('--weight_decay', type=float, default=0.01,
                            help="Weight decay for optimizer")
    train_group.add_argument('--early_stopping_patience', type=int, default=5,
                            help="Early stopping patience")
    train_group.add_argument('--lr_scheduler_patience', type=int, default=5,
                            help="Learning rate scheduler patience")
    
    # System arguments
    sys_group = parser.add_argument_group('System Arguments')
    sys_group.add_argument('--seed', type=int, default=42,
                          help="Random seed")
    sys_group.add_argument('--num_workers', type=int, default=0,
                          help="Number of DataLoader workers")
    
    # Output arguments
    out_group = parser.add_argument_group('Output Arguments')
    out_group.add_argument('--base_output_dir', type=str, default="experiments_vcdr",
                          help="Base directory for experiment outputs")
    out_group.add_argument('--experiment_tag', type=str, required=True,
                          help="Descriptive tag for this experiment")
    
    args = parser.parse_args()
    
    try:
        result = run_experiment(args)
        print(f"Experiment completed successfully: {result}")
    except Exception as e:
        print(f"Experiment failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
