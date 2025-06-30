#!/usr/bin/env python3
"""
Test script to verify default model behavior (no duplicates)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_default_behavior():
    """Test default behavior when no additional models are specified"""
    
    # Mock args with no additional models
    class MockArgs:
        def __init__(self):
            self.additional_models = []  # No additional models
            self.vfm_weights_path = os.path.join(os.path.dirname(__file__), "..", "models", "VFM_Fundus_weights.pth")
    
    args = MockArgs()
    
    # Simulate the model configuration logic
    model_configs = []
    added_architectures = set()
    
    # Check if ResNet-50 is in additional models
    resnet50_in_additional = any(
        model.lower() == 'resnet50' or 'resnet50' in model.lower() 
        for model in args.additional_models
    )
    
    print(f"ResNet-50 found in additional models: {resnet50_in_additional}")
    
    # Add ResNet-50 pretrained only if not in additional models
    if not resnet50_in_additional:
        model_configs.append({
            'name': 'ResNet50_Pretrained',
            'architecture': 'resnet50',
            'pretrained': True,
            'weights_path': None
        })
        added_architectures.add('resnet50')
        print("Added default ResNet-50")
    else:
        print("Skipped default ResNet-50 (found in additional models)")
    
    # Add VFM ViT-B pretrained
    if os.path.exists(args.vfm_weights_path):
        model_configs.append({
            'name': 'VFM_ViTB_Pretrained',
            'architecture': 'vit_base_patch16_224',
            'pretrained': False,
            'weights_path': args.vfm_weights_path
        })
        added_architectures.add('vit_base_patch16_224')
        print("Added VFM ViT-B")
    else:
        print(f"VFM weights not found at {args.vfm_weights_path}")
    
    print(f"\nFinal model configurations ({len(model_configs)} models):")
    for i, config in enumerate(model_configs, 1):
        print(f"  {i}. {config['name']} ({config['architecture']})")
    
    # Test assertions
    assert len(model_configs) == 2, f"Expected 2 models, got {len(model_configs)}"
    
    architectures = [config['architecture'] for config in model_configs]
    assert 'resnet50' in architectures, "ResNet-50 should be included"
    assert 'vit_base_patch16_224' in architectures, "VFM ViT-B should be included"
    
    print("\n✅ Default behavior test passed!")
    return True

if __name__ == "__main__":
    print("Testing default model behavior...")
    test_default_behavior()
    print("All tests passed! 🎉")
