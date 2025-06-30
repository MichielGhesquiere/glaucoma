"""
Fine-tuning utilities for glaucoma classification models.

This module contains functions for different fine-tuning strategies including
linear probing, gradual unfreezing, and full fine-tuning with layer-wise learning rates.
"""

import argparse
import logging
import torch
import torch.optim as optim
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def freeze_backbone(model, model_name):
    """Freeze backbone parameters for initial fine-tuning epochs."""
    frozen_params = 0
    
    for name, param in model.named_parameters():
        # Don't freeze the classification head
        if not any(head_attr in name for head_attr in ['head.', 'fc.', 'classifier.']):
            param.requires_grad = False
            frozen_params += param.numel()
    
    logger.info(f"Frozen {frozen_params} backbone parameters")
    return frozen_params


def unfreeze_backbone(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Unfrozen all parameters")


def create_optimizer_for_strategy(model, config):
    """
    Create optimizer based on the fine-tuning strategy.
    
    Args:
        model: The model to create optimizer for
        config: Configuration object with strategy and learning parameters
    
    Returns:
        Optimizer instance
    """
    try:
        from src.utils.fine_tune_tools import param_groups_llrd, count_trainable_parameters
    except ImportError:
        logger.error("Could not import fine_tune_tools. Using standard optimizer.")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    
    if config.ft_strategy == 'full':
        # Use Layer-wise Learning Rate Decay (LLRD)
        param_groups = param_groups_llrd(
            model, config.learning_rate, config.weight_decay, decay=config.llrd_decay
        )
        optimizer = optim.AdamW(param_groups)
        logger.info(f"Created LLRD optimizer with {len(param_groups)} parameter groups")
    else:
        # For 'linear' and 'gradual' strategies, use standard optimizer
        # Only trainable parameters will be included
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
        logger.info(f"Created standard optimizer with {len(trainable_params)} parameter groups")
    
    return optimizer


def setup_fine_tuning_strategy(model, config, current_epoch=0):
    """
    Set up the model for the specified fine-tuning strategy.
    
    Args:
        model: The model to configure
        config: Configuration object
        current_epoch: Current epoch number (for gradual unfreezing)
    
    Returns:
        Tuple of (optimizer, phase_info)
    """
    logger.info(f"Setting up fine-tuning strategy: {config.ft_strategy}")
    
    try:
        from src.utils.fine_tune_tools import (
            freeze_up_to, get_gradual_unfreeze_patterns, 
            count_trainable_parameters, print_parameter_status
        )
    except ImportError:
        logger.error("Could not import fine_tune_tools. Using basic strategy.")
        optimizer = create_optimizer_for_strategy(model, config)
        return optimizer, {'strategy': config.ft_strategy, 'epoch': current_epoch}
    
    if config.ft_strategy == 'linear':
        # Linear probing: only train the classification head
        freeze_up_to(model, r'head|fc|classifier')
        logger.info("Linear probing: Only classification head is trainable")
        
    elif config.ft_strategy == 'gradual':
        # Gradual unfreezing: start with head, gradually unfreeze more layers
        patterns = get_gradual_unfreeze_patterns(config.model_name)
        
        # Determine current phase based on epoch
        if current_epoch == 0:
            phase = 0
        elif current_epoch < config.gradual_patience:
            phase = 0
        elif current_epoch < 2 * config.gradual_patience:
            phase = 1
        else:
            phase = 2
        
        phase_pattern = patterns[f'phase_{min(phase, 2)}']
        freeze_up_to(model, phase_pattern)
        
        phase_names = ['head-only', 'head+top-layers', 'full-model']
        logger.info(f"Gradual unfreezing phase {phase} ({phase_names[phase]}): pattern '{phase_pattern}'")
        
    elif config.ft_strategy == 'full':
        # Full fine-tuning: all parameters trainable (LLRD will handle LR differences)
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Full fine-tuning: All parameters trainable with LLRD")
    
    # Print parameter status
    print_parameter_status(model)
    
    # Create optimizer
    optimizer = create_optimizer_for_strategy(model, config)
    
    phase_info = {
        'strategy': config.ft_strategy,
        'epoch': current_epoch,
        'trainable_params': count_trainable_parameters(model)
    }
    
    return optimizer, phase_info


def should_update_gradual_strategy(config, current_epoch, last_improvement_epoch, current_phase):
    """
    Determine if we should move to the next phase in gradual unfreezing.
    
    Args:
        config: Configuration object
        current_epoch: Current epoch number
        last_improvement_epoch: Epoch when validation loss last improved
        current_phase: Current unfreezing phase (0, 1, or 2)
    
    Returns:
        True if should advance to next phase
    """
    if config.ft_strategy != 'gradual' or current_phase >= 2:
        return False
    
    epochs_since_improvement = current_epoch - last_improvement_epoch
    return epochs_since_improvement >= config.gradual_patience


def create_optimizer_with_differential_lr(model, base_lr, backbone_lr_multiplier, weight_decay):
    """Create optimizer with different learning rates for backbone and head."""
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if any(head_attr in name for head_attr in ['head.', 'fc.', 'classifier.']):
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': base_lr * backbone_lr_multiplier, 'name': 'backbone'},
        {'params': head_params, 'lr': base_lr, 'name': 'head'}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    logger.info(f"Created optimizer with backbone LR: {base_lr * backbone_lr_multiplier:.2e}, head LR: {base_lr:.2e}")
    return optimizer
