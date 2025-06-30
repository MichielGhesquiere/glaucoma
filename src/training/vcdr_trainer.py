"""
Training utilities for VCDR regression.
"""

import logging
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module, 
    train_loader: DataLoader, 
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer, 
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Tuple of (average_loss, average_mae)
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, targets in progress_bar:
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            mae = torch.mean(torch.abs(outputs - targets))
            
        total_loss += loss.item()
        total_mae += mae.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'MAE': f'{mae.item():.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    return avg_loss, avg_mae


def validate_epoch(
    model: nn.Module, 
    val_loader: DataLoader, 
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, List[float], List[float]]:
    """
    Validate the model for one epoch.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average_loss, average_mae, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images).squeeze()
            
            # Calculate loss
            loss = criterion(outputs, targets)
            mae = torch.mean(torch.abs(outputs - targets))
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
            
            # Store predictions and targets
            all_predictions.extend(outputs.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MAE': f'{mae.item():.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    return avg_loss, avg_mae, all_predictions, all_targets


def print_training_summary(
    train_losses: List[float], 
    val_losses: List[float], 
    train_maes: List[float], 
    val_maes: List[float],
    train_r2s: Optional[List[float]] = None,
    val_r2s: Optional[List[float]] = None
) -> None:
    """
    Print a summary of training metrics.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_maes: List of training MAEs per epoch
        val_maes: List of validation MAEs per epoch
        train_r2s: Optional list of training R² scores per epoch
        val_r2s: Optional list of validation R² scores per epoch
    """
    num_epochs = len(train_losses)
    
    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Epochs: {num_epochs}")
    
    if num_epochs > 0:
        # Best validation metrics
        best_val_loss_epoch = val_losses.index(min(val_losses)) + 1
        best_val_mae_epoch = val_maes.index(min(val_maes)) + 1
        
        logger.info(f"Best Validation Loss: {min(val_losses):.6f} (Epoch {best_val_loss_epoch})")
        logger.info(f"Best Validation MAE: {min(val_maes):.6f} (Epoch {best_val_mae_epoch})")
        
        if val_r2s:
            best_val_r2_epoch = val_r2s.index(max(val_r2s)) + 1
            logger.info(f"Best Validation R²: {max(val_r2s):.6f} (Epoch {best_val_r2_epoch})")
        
        # Final metrics
        logger.info(f"Final Training Loss: {train_losses[-1]:.6f}")
        logger.info(f"Final Training MAE: {train_maes[-1]:.6f}")
        logger.info(f"Final Validation Loss: {val_losses[-1]:.6f}")
        logger.info(f"Final Validation MAE: {val_maes[-1]:.6f}")
        
        if train_r2s and val_r2s:
            logger.info(f"Final Training R²: {train_r2s[-1]:.6f}")
            logger.info(f"Final Validation R²: {val_r2s[-1]:.6f}")
    
    logger.info("=" * 80)
