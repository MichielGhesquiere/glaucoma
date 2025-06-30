#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple test script to verify multi-source domain fine-tuning setup.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_imports():
    """Test that all required imports work."""
    try:
        print("Testing imports...")
        
        # Core imports
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import numpy as np
        import pandas as pd
        print("✓ NumPy and Pandas")
        
        # Custom imports
        from src.utils.helpers import set_seed
        from src.data.datasets import GlaucomaClassificationDataset
        print("✓ Custom modules")
        
        # Check if weights file exists
        vfm_weights = r'D:\glaucoma\models\VFM_Fundus_weights.pth'
        if os.path.exists(vfm_weights):
            print(f"✓ VFM weights found: {vfm_weights}")
        else:
            print(f"⚠ VFM weights not found: {vfm_weights}")
        
        # Check data directory
        data_dir = r'D:\glaucoma\data'
        if os.path.exists(data_dir):
            print(f"✓ Data directory found: {data_dir}")
        else:
            print(f"⚠ Data directory not found: {data_dir}")
        
        print("\n✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_device():
    """Test GPU availability."""
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA not available, will use CPU")

def main():
    """Run all tests."""
    print("="*50)
    print("Multi-Source Domain Fine-Tuning Setup Test")
    print("="*50)
    
    success = test_imports()
    test_device()
    
    if success:
        print("\n🎉 Setup test passed! Ready to run multi-source domain fine-tuning.")
        print("\nNext steps:")
        print("1. Run: .\\run_multisource_finetuning.ps1")
        print("2. Or: python multisource_domain_finetuning.py --help")
    else:
        print("\n❌ Setup test failed. Please check your environment.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
