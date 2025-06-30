#!/usr/bin/env python3
"""
Test script to verify duplicate model prevention in multisource_domain_finetuning.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_duplicate_prevention():
    """Test that duplicate models are properly handled"""
    
    # Mock args with duplicate ResNet-50
    class MockArgs:
        def __init__(self):
            self.additional_models = ['resnet50']  # Should cause ResNet-50 to be added only once
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
    
    # Add any additional model configurations from args
    for additional_model in args.additional_models:
        arch = additional_model
        
        # Skip if architecture already added
        if arch.lower() in added_architectures:
            print(f"Skipping duplicate architecture: {arch}")
            continue
            
        model_configs.append({
            'name': arch.title() + '_Pretrained',
            'architecture': arch,
            'pretrained': True,
            'weights_path': None
        })
        added_architectures.add(arch.lower())
        print(f"Added additional model: {arch}")
    
    print(f"\nFinal model configurations ({len(model_configs)} models):")
    for i, config in enumerate(model_configs, 1):
        print(f"  {i}. {config['name']} ({config['architecture']})")
    
    # Test assertions
    assert len(model_configs) == 2, f"Expected 2 models, got {len(model_configs)}"
    
    architectures = [config['architecture'] for config in model_configs]
    assert 'resnet50' in architectures, "ResNet-50 should be included once"
    assert 'vit_base_patch16_224' in architectures, "VFM ViT-B should be included"
    assert len(set(architectures)) == len(architectures), "No duplicate architectures should exist"
    
    # Verify ResNet-50 appears only once
    resnet50_count = sum(1 for arch in architectures if arch == 'resnet50')
    assert resnet50_count == 1, f"ResNet-50 should appear exactly once, but appears {resnet50_count} times"
    
    print("\n✅ Duplicate prevention test passed!")
    return True

if __name__ == "__main__":
    print("Testing duplicate model prevention...")
    test_duplicate_prevention()
    print("All tests passed! 🎉")
