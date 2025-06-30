#!/usr/bin/env python3
"""
Quick test script to verify the evaluation script works correctly.
Tests the transforms loading and basic functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.data.transforms import get_transforms
    print("✓ Successfully imported get_transforms")
    
    # Test the function signature
    try:
        train_transforms, eval_transforms = get_transforms(224, 'vit_base_patch16_224', False)
        print("✓ get_transforms works with positional arguments")
        print(f"✓ Train transforms: {type(train_transforms)}")
        print(f"✓ Eval transforms: {type(eval_transforms)}")
    except Exception as e:
        print(f"✗ Error with get_transforms: {e}")
        
except ImportError as e:
    print(f"✗ Could not import get_transforms: {e}")
    print("Make sure you're running from the correct directory and src.data.transforms exists")

# Test other critical imports
try:
    from src.models.classification.build_model import build_classifier_model
    print("✓ Successfully imported build_classifier_model")
except ImportError as e:
    print(f"✗ Could not import build_classifier_model: {e}")

try:
    from src.data.datasets import GlaucomaSubgroupDataset
    print("✓ Successfully imported GlaucomaSubgroupDataset")
except ImportError as e:
    print(f"✗ Could not import GlaucomaSubgroupDataset: {e}")

try:
    from src.evaluation.metrics import calculate_ece
    print("✓ Successfully imported calculate_ece")
except ImportError as e:
    print(f"✗ Could not import calculate_ece: {e}")

print("\nTest completed. If all imports are successful, the evaluation script should work.")
