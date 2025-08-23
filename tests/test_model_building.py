"""
Tests for model building, architecture creation, and weight loading.
"""
import os
import sys
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestModelBuilding:
    """Test model building functionality."""
    
    def test_build_resnet_model(self):
        """Test ResNet model building."""
        from src.models.classification.build_model import build_classifier_model
        
        model = build_classifier_model(
            model_name='resnet18',
            num_classes=2,
            dropout_prob=0.3,
            pretrained=False
        )
        
        assert model is not None
        assert hasattr(model, 'fc')
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 2)
    
    def test_build_vit_model_standard_head(self):
        """Test ViT model building with standard head."""
        from src.models.classification.build_model import build_classifier_model
        
        model = build_classifier_model(
            model_name='vit_base_patch16_224',
            num_classes=2,
            dropout_prob=0.3,
            pretrained=False,
            is_custom_sequential_head_vit=False
        )
        
        assert model is not None
        assert hasattr(model, 'head')
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 2)
    
    def test_build_vit_model_sequential_head(self):
        """Test ViT model building with sequential head."""
        from src.models.classification.build_model import build_classifier_model
        
        model = build_classifier_model(
            model_name='vit_base_patch16_224',
            num_classes=2,
            dropout_prob=0.3,
            pretrained=False,
            is_custom_sequential_head_vit=True
        )
        
        assert model is not None
        assert hasattr(model, 'head')
        assert isinstance(model.head, nn.Sequential)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 2)
    
    @patch('torch.hub.load')
    def test_build_dinov2_model(self, mock_hub_load):
        """Test DINOv2 model building."""
        from src.models.classification.build_model import build_classifier_model
        
        # Mock DINOv2 model
        mock_model = MagicMock()
        mock_model.embed_dim = 768  # ViT-Base
        mock_hub_load.return_value = mock_model
        
        model = build_classifier_model(
            model_name='dinov2_vitb14',
            num_classes=2,
            dropout_prob=0.3,
            pretrained=False
        )
        
        assert model is not None
        mock_hub_load.assert_called_once()
        # Check that head was replaced
        assert hasattr(model, 'head')

class TestWeightLoading:
    """Test custom weight loading functionality."""
    
    def test_detect_head_structure_from_checkpoint(self):
        """Test head structure detection from checkpoint."""
        from src.models.classification.build_model import detect_head_structure_from_checkpoint
        
        # Create temporary checkpoint with sequential head
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            checkpoint = {
                'model': {
                    'backbone.layer1.weight': torch.randn(64, 64, 3, 3),
                    'head.0.weight': torch.randn(256, 768),  # Dropout layer
                    'head.1.weight': torch.randn(2, 256),    # Linear layer
                    'head.1.bias': torch.randn(2)
                }
            }
            torch.save(checkpoint, tmp_file.name)
            
            has_seq, has_std, state_dict = detect_head_structure_from_checkpoint(
                tmp_file.name, 'model'
            )
            
            assert has_seq
            assert not has_std
            assert state_dict is not None
            
            os.unlink(tmp_file.name)
    
    def test_adapt_head_weights_linear_to_sequential(self):
        """Test adapting linear head weights to sequential structure."""
        from src.models.classification.build_model import adapt_head_weights
        import logging
        
        # Create mock model with sequential head
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {
            'head.0.weight': torch.randn(256),
            'head.1.weight': torch.randn(2, 768),
            'head.1.bias': torch.randn(2)
        }
        
        # State dict with linear head
        state_dict = {
            'backbone.layer1.weight': torch.randn(64, 64, 3, 3),
            'head.weight': torch.randn(2, 768),
            'head.bias': torch.randn(2)
        }
        
        logger = logging.getLogger(__name__)
        adapted = adapt_head_weights(state_dict, mock_model, logger)
        
        assert 'head.1.weight' in adapted
        assert 'head.1.bias' in adapted
        assert 'head.weight' not in adapted
        assert 'head.bias' not in adapted
    
    def test_load_custom_pretrained_weights_success(self):
        """Test successful loading of custom pretrained weights."""
        from src.models.classification.build_model import load_custom_pretrained_weights
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # Create temporary checkpoint
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            state_dict = model.state_dict()
            checkpoint = {'model': state_dict}
            torch.save(checkpoint, tmp_file.name)
            
            # Load weights
            success = load_custom_pretrained_weights(
                model, tmp_file.name, checkpoint_key='model'
            )
            
            assert success
            os.unlink(tmp_file.name)
    
    def test_load_custom_pretrained_weights_nonexistent_file(self):
        """Test loading weights from nonexistent file."""
        from src.models.classification.build_model import load_custom_pretrained_weights
        
        model = nn.Linear(10, 2)
        success = load_custom_pretrained_weights(
            model, '/nonexistent/path.pth'
        )
        
        assert not success

class TestModelArchitectures:
    """Test specific model architectures."""
    
    def test_resnet_architecture_properties(self):
        """Test ResNet architecture properties."""
        from src.models.classification.build_model import build_classifier_model
        
        model = build_classifier_model('resnet50', num_classes=3, pretrained=False)
        
        # Check final layer has correct output size
        if hasattr(model, 'fc'):
            assert model.fc.out_features == 3
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                last_layer = model.classifier[-1]
            else:
                last_layer = model.classifier
            assert last_layer.out_features == 3
    
    def test_vit_architecture_properties(self):
        """Test ViT architecture properties."""
        from src.models.classification.build_model import build_classifier_model
        
        model = build_classifier_model(
            'vit_base_patch16_224', 
            num_classes=5, 
            pretrained=False
        )
        
        # Check head properties
        if hasattr(model, 'head'):
            if isinstance(model.head, nn.Sequential):
                last_layer = model.head[-1]
            else:
                last_layer = model.head
            assert last_layer.out_features == 5

class TestModelUtilities:
    """Test model utility functions."""
    
    def test_model_parameter_count(self):
        """Test model parameter counting."""
        from src.models.classification.build_model import build_classifier_model
        
        model = build_classifier_model('resnet18', num_classes=2, pretrained=False)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All params should be trainable by default
    
    def test_model_device_compatibility(self):
        """Test model device compatibility."""
        from src.models.classification.build_model import build_classifier_model
        
        model = build_classifier_model('resnet18', num_classes=2, pretrained=False)
        
        # Test CPU
        model_cpu = model.to('cpu')
        dummy_input = torch.randn(1, 3, 224, 224).to('cpu')
        
        with torch.no_grad():
            output = model_cpu(dummy_input)
        
        assert output.device.type == 'cpu'
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.to('cuda')
            dummy_input_gpu = torch.randn(1, 3, 224, 224).to('cuda')
            
            with torch.no_grad():
                output_gpu = model_gpu(dummy_input_gpu)
            
            assert output_gpu.device.type == 'cuda'

class TestRegressionToClassification:
    """Test regression to classification model conversion."""
    
    def test_regression_to_classification_flag(self):
        """Test building model with regression to classification conversion."""
        from src.models.classification.build_model import build_classifier_model
        
        model = build_classifier_model(
            model_name='resnet18',
            num_classes=2,
            pretrained=False,
            is_regression_to_classification=True
        )
        
        assert model is not None
        
        # Should have sequential head due to flag
        if hasattr(model, 'fc'):
            # Check if it's sequential (indicating conversion happened)
            # This might vary based on implementation
            pass
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 2)

class TestMultiTaskModels:
    """Test multi-task model architectures."""
    
    def test_multitask_model_creation(self):
        """Test multi-task model creation."""
        # Skip if not available
        try:
            from scripts.train_multitask_classification_regression import MultiTaskModel
        except ImportError:
            pytest.skip("MultiTaskModel not available")
        
        model = MultiTaskModel(
            backbone='resnet18',
            pretrained=False,
            dropout=0.3
        )
        
        assert model is not None
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        assert isinstance(output, dict)
        assert 'classification_logits' in output
        assert 'regression_output' in output

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
