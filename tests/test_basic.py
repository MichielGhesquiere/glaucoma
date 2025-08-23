"""
Basic smoke tests for core functionality.
These tests verify that the most critical components can be imported and instantiated.
"""
import os
import sys
import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that core modules can be imported without errors."""
    try:
        # Core data modules
        from src.data.multisource_loader import load_all_datasets
        from src.data.loaders import load_chaksu_data, load_airogs_data
        
        # Model building
        from src.models.classification.build_model import build_classifier_model
        
        # Training components
        from src.training.engine import train_model
        
        # Evaluation metrics
        from src.evaluation.metrics import calculate_ece, calculate_sensitivity_at_specificity
        
        # Feature extraction
        from src.features.metrics import GlaucomaMetrics
        
        print("✓ All core imports successful")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_basic_model_creation():
    """Test that models can be created without custom weights."""
    from src.models.classification.build_model import build_classifier_model
    
    # Test ResNet18 (lightweight for testing)
    model = build_classifier_model(
        model_name='resnet18',
        num_classes=2,
        dropout_prob=0.3,
        pretrained=False,  # Avoid downloading weights in tests
        custom_weights_path=None
    )
    
    assert model is not None
    assert hasattr(model, 'fc') or hasattr(model, 'head') or hasattr(model, 'classifier')
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"
    print("✓ Basic model creation and forward pass successful")

def test_glaucoma_metrics():
    """Test glaucoma metrics calculation with synthetic data."""
    from src.features.metrics import GlaucomaMetrics
    
    metrics_calculator = GlaucomaMetrics()
    
    # Create synthetic disc and cup masks
    disc_mask = np.zeros((224, 224), dtype=np.uint8)
    cup_mask = np.zeros((224, 224), dtype=np.uint8)
    
    # Create circular disc
    center = (112, 112)
    radius_disc = 50
    radius_cup = 25
    
    y, x = np.ogrid[:224, :224]
    disc_region = (x - center[0])**2 + (y - center[1])**2 <= radius_disc**2
    cup_region = (x - center[0])**2 + (y - center[1])**2 <= radius_cup**2
    
    disc_mask[disc_region] = 1
    cup_mask[cup_region] = 1
    
    # Test metrics extraction
    metrics = metrics_calculator.extract_metrics(disc_mask, cup_mask)
    
    assert isinstance(metrics, dict)
    assert 'vcdr' in metrics  # lowercase in actual implementation
    assert 'disc_area' in metrics  # check actual column names
    assert 'cup_area' in metrics
    
    # Basic sanity checks
    assert 0 <= metrics['vcdr'] <= 1, f"vcdr should be between 0 and 1, got {metrics['vcdr']}"
    assert metrics['disc_area'] > metrics['cup_area'], "Disc area should be larger than cup area"
    
    print("✓ Glaucoma metrics calculation successful")

def test_data_structures():
    """Test that data loading functions return correct structures."""
    from src.data.multisource_loader import get_dataset_statistics, validate_dataset_integrity
    
    # Create mock dataset
    mock_dataset = pd.DataFrame({
        'image_path': ['/fake/path1.jpg', '/fake/path2.jpg'],
        'label': [0, 1],
        'dataset_source': ['TEST', 'TEST'],
        'names': ['image1.jpg', 'image2.jpg']
    })
    
    mock_datasets = {'TEST': mock_dataset}
    
    # Test statistics calculation
    stats = get_dataset_statistics(mock_datasets)
    assert 'TEST' in stats
    assert stats['TEST']['total_samples'] == 2
    assert stats['TEST']['num_classes'] == 2
    
    print("✓ Data structure tests successful")

def test_pytorch_setup():
    """Test PyTorch setup and GPU availability."""
    import torch
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    # Test tensor operations
    x = torch.randn(2, 3)
    y = torch.randn(3, 2)
    result = torch.mm(x, y)
    assert result.shape == (2, 2)
    
    print("✓ PyTorch setup test successful")

def test_project_structure():
    """Test that expected project directories exist."""
    project_root = Path(__file__).parent.parent
    
    # Check main directories exist
    assert (project_root / "src").exists()
    assert (project_root / "scripts").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "pyproject.toml").exists()

if __name__ == "__main__":
    test_imports()
    test_basic_model_creation()
    test_glaucoma_metrics()
    test_data_structures()
    test_pytorch_setup()
    test_project_structure()
    print("All basic tests passed!")
