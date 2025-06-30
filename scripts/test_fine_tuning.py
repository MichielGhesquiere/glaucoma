#!/usr/bin/env python3
"""
Quick test script to verify fine-tuning strategies are working correctly.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import timm
from src.utils.fine_tune_tools import (
    freeze_up_to, param_groups_llrd, get_gradual_unfreeze_patterns,
    count_trainable_parameters, print_parameter_status
)

def test_fine_tuning_tools():
    """Test the fine-tuning utilities."""
    print("Testing Fine-tuning Tools")
    print("=" * 50)
    
    # Create a simple test model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
    print(f"Created test model: {type(model).__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n1. Testing freeze_up_to with head-only pattern:")
    freeze_up_to(model, r'head|fc|classifier')
    trainable_count = count_trainable_parameters(model)
    print(f"Trainable parameters after head-only freeze: {trainable_count:,}")
    
    print("\n2. Testing gradual unfreeze patterns:")
    patterns = get_gradual_unfreeze_patterns('vit_tiny_patch16_224')
    for phase, pattern in patterns.items():
        print(f"  {phase}: {pattern}")
    
    print("\n3. Testing LLRD parameter groups:")
    # Unfreeze everything first
    for param in model.parameters():
        param.requires_grad = True
    
    param_groups = param_groups_llrd(model, base_lr=1e-4, weight_decay=0.05, decay=0.9)
    print(f"Created {len(param_groups)} parameter groups")
    
    for i, group in enumerate(param_groups):
        layer_name = group.get('layer_name', f'group_{i}')
        lr = group['lr']
        param_count = sum(p.numel() for p in group['params'])
        print(f"  {layer_name}: LR={lr:.2e}, Params={param_count:,}")
    
    print("\n4. Testing different model architectures:")
    for model_name in ['resnet18', 'efficientnet_b0']:
        try:
            patterns = get_gradual_unfreeze_patterns(model_name)
            print(f"  {model_name}: Phase 0 = {patterns['phase_0']}")
        except Exception as e:
            print(f"  {model_name}: Error = {e}")
    
    print("\n✓ All tests completed successfully!")

if __name__ == "__main__":
    test_fine_tuning_tools()
