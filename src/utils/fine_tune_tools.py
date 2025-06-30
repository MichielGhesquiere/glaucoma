"""
Fine-tuning utilities for different training strategies.

This module provides tools for implementing various fine-tuning approaches:
1. Linear probing (head-only training)
2. Gradual unfreezing (ULMFit-style)
3. Layer-wise learning rate decay (LLRD)
"""

import math
import re
from collections import defaultdict
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def freeze_up_to(model: nn.Module, regex: str) -> None:
    """
    Freeze all parameters whose name *does NOT* match `regex`.
    Only parameters matching the regex pattern will remain trainable.
    
    Args:
        model: PyTorch model to modify
        regex: Regular expression pattern. Parameters matching this will stay trainable.
               Example: r'head|classifier|fc' (keeps head & classifier trainable)
               Example: r'head|layer4|blocks\.11' (keeps head & last transformer block)
    """
    prog = re.compile(regex)
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if prog.search(name):
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    
    logger.info(f"Applied freeze pattern '{regex}': {trainable_count} trainable, {frozen_count} frozen parameters")


def param_groups_llrd(model: nn.Module, base_lr: float, weight_decay: float, decay: float = 0.95):
    """
    Create parameter groups with exponentially decayed learning rate per layer.
    This implements Layer-wise Learning Rate Decay (LLRD).
    
    Args:
        model: PyTorch model
        base_lr: Base learning rate for the top layers
        weight_decay: Weight decay to apply to all parameter groups
        decay: Decay factor per layer (e.g., 0.95 means each layer down gets 95% of the LR)
    
    Returns:
        List of parameter groups suitable for optimizer initialization
    """
    groups = defaultdict(list)
    
    # Group parameters by their top-level module name
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Extract the top-level module name (e.g., 'patch_embed', 'blocks', 'norm', 'head')
        layer_id = name.split('.')[0]
        groups[layer_id].append(param)
    
    # Sort layer names to establish depth order
    # This is a heuristic that works for most Vision Transformers and ResNets
    layer_names = list(groups.keys())
    layer_names.sort()
    
    # Special handling for common layer patterns
    # Put embedding/stem layers first, head/classifier last
    def layer_priority(layer_name):
        layer_lower = layer_name.lower()
        if any(x in layer_lower for x in ['embed', 'patch', 'stem', 'conv1']):
            return 0  # Earliest layers
        elif any(x in layer_lower for x in ['head', 'fc', 'classifier']):
            return 1000  # Latest layers
        elif 'norm' in layer_lower:
            return 999  # Just before head
        elif 'blocks' in layer_lower or 'layers' in layer_lower:
            return 500  # Middle layers
        else:
            return 100  # Default middle
    
    layer_names.sort(key=layer_priority)
    
    param_groups = []
    total_layers = len(layer_names)
    
    for depth, layer_name in enumerate(layer_names):
        # Calculate LR: deeper layers (higher depth) get lower LR
        lr_multiplier = decay ** (total_layers - depth - 1)
        layer_lr = base_lr * lr_multiplier
        
        param_groups.append({
            'params': groups[layer_name],
            'lr': layer_lr,
            'weight_decay': weight_decay,
            'layer_name': layer_name  # For debugging
        })
        
        logger.info(f"Layer '{layer_name}' (depth {depth}): LR = {layer_lr:.2e} ({len(groups[layer_name])} param tensors)")
    
    logger.info(f"Created {len(param_groups)} parameter groups with LLRD (decay={decay})")
    return param_groups


def get_gradual_unfreeze_patterns(model_name: str):
    """
    Get regex patterns for gradual unfreezing based on model architecture.
    
    Args:
        model_name: Name of the model (e.g., 'vit_base_patch16_224', 'resnet50')
    
    Returns:
        Dict with patterns for different unfreezing phases
    """
    model_lower = model_name.lower()
    
    if 'vit' in model_lower or 'deit' in model_lower or 'dinov2' in model_lower:
        # Vision Transformer patterns
        return {
            'phase_0': r'head|fc|classifier',  # Head only
            'phase_1': r'head|fc|classifier|norm|blocks\.1[01]',  # Head + last 2 blocks + norm
            'phase_2': r'.*'  # Everything
        }
    elif 'resnet' in model_lower:
        # ResNet patterns
        return {
            'phase_0': r'head|fc|classifier',  # Head only
            'phase_1': r'head|fc|classifier|layer4',  # Head + last residual block
            'phase_2': r'.*'  # Everything
        }
    elif 'efficientnet' in model_lower:
        # EfficientNet patterns
        return {
            'phase_0': r'head|fc|classifier',
            'phase_1': r'head|fc|classifier|features\.[6-9]',  # Head + last few blocks
            'phase_2': r'.*'
        }
    else:
        # Generic fallback
        logger.warning(f"Unknown model architecture '{model_name}', using generic unfreeze patterns")
        return {
            'phase_0': r'head|fc|classifier',
            'phase_1': r'head|fc|classifier|layer[3-4]|blocks\.(1[0-1]|[8-9])',
            'phase_2': r'.*'
        }


def count_trainable_parameters(model: nn.Module):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_parameter_status(model: nn.Module):
    """Print detailed information about which parameters are trainable."""
    total_params = 0
    trainable_params = 0
    
    logger.info("Parameter trainability status:")
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            status = "✓ TRAINABLE"
        else:
            status = "✗ FROZEN"
        
        logger.debug(f"  {name:50s} {param.numel():>8d} params {status}")
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    logger.info(f"Frozen parameters: {total_params-trainable_params:,} ({100*(total_params-trainable_params)/total_params:.1f}%)")
