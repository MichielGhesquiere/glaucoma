"""
Integration tests for end-to-end workflows including
training pipelines, evaluation workflows, and data processing.
"""
import os
import sys
import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_data_loading_to_model_pipeline(self):
        """Test complete pipeline from data loading to model training."""
        # Create temporary directory with test images
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create dummy images
            for i in range(5):
                img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
                img.save(os.path.join(tmp_dir, f'test_img_{i}.jpg'))
            
            # Create test dataset
            test_df = pd.DataFrame({
                'image_path': [os.path.join(tmp_dir, f'test_img_{i}.jpg') for i in range(5)],
                'label': [0, 1, 0, 1, 1],
                'dataset_source': ['TEST'] * 5,
                'names': [f'test_img_{i}.jpg' for i in range(5)]
            })
            
            # Test data validation
            from src.data.multisource_loader import validate_dataset_integrity
            datasets = {'TEST': test_df}
            is_valid = validate_dataset_integrity(datasets)
            assert is_valid
            
            # Test model creation
            from src.models.classification.build_model import build_classifier_model
            model = build_classifier_model(
                model_name='resnet18',
                num_classes=2,
                pretrained=False
            )
            assert model is not None
    
    def test_training_evaluation_pipeline(self):
        """Test training and evaluation pipeline."""
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 2)
        )
        
        # Create synthetic dataset
        X = torch.randn(50, 100)
        y = torch.randint(0, 2, (50,))
        
        # Split data
        train_X, val_X = X[:40], X[40:]
        train_y, val_y = y[:40], y[40:]
        
        # Training setup
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Mini training loop
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(train_X)
            loss = criterion(outputs, train_y)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs, val_y)
            predictions = torch.argmax(val_outputs, dim=1)
            accuracy = (predictions == val_y).float().mean()
        
        assert val_loss.item() >= 0
        assert 0 <= accuracy.item() <= 1
    
    def test_multisource_domain_adaptation_workflow(self):
        """Test multi-source domain adaptation workflow."""
        # Skip if modules not available
        try:
            from src.data.multisource_loader import prepare_leave_one_out_splits
        except ImportError:
            pytest.skip("Multi-source modules not available")
        
        # Create mock multi-source datasets
        datasets = {}
        for domain in ['DOMAIN_A', 'DOMAIN_B', 'DOMAIN_C']:
            datasets[domain] = pd.DataFrame({
                'image_path': [f'/fake/{domain.lower()}_img_{i}.jpg' for i in range(10)],
                'label': np.random.randint(0, 2, 10),
                'dataset_source': [domain] * 10,
                'names': [f'{domain.lower()}_img_{i}.jpg' for i in range(10)]
            })
        
        # Test leave-one-out splits
        splits = prepare_leave_one_out_splits(datasets)
        assert len(splits) == 3
        
        for test_domain, train_domains in splits:
            assert test_domain in ['DOMAIN_A', 'DOMAIN_B', 'DOMAIN_C']
            assert len(train_domains) == 2
            assert test_domain not in train_domains

class TestDataPipelineIntegration:
    """Test data pipeline integration."""
    
    def test_transforms_to_dataloader_pipeline(self):
        """Test complete data transforms to DataLoader pipeline."""
        from torchvision import transforms
        from torch.utils.data import Dataset, DataLoader
        
        # Custom dataset for testing
        class MockDataset(Dataset):
            def __init__(self, size=20):
                self.size = size
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Create random image
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                label = idx % 2  # Alternate labels
                
                if self.transform:
                    img = self.transform(img)
                
                return img, label
        
        dataset = MockDataset(size=20)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Test batch loading
        for batch_idx, (images, labels) in enumerate(dataloader):
            assert images.shape == (4, 3, 224, 224)
            assert labels.shape == (4,)
            if batch_idx == 2:  # Test a few batches
                break
    
    def test_dataset_statistics_pipeline(self):
        """Test dataset statistics calculation pipeline."""
        from src.data.multisource_loader import get_dataset_statistics
        
        # Create diverse datasets
        datasets = {}
        
        # Balanced dataset
        datasets['BALANCED'] = pd.DataFrame({
            'label': [0, 1] * 50,  # 50% each class
            'dataset_source': ['BALANCED'] * 100
        })
        
        # Imbalanced dataset
        datasets['IMBALANCED'] = pd.DataFrame({
            'label': [0] * 90 + [1] * 10,  # 90% class 0, 10% class 1
            'dataset_source': ['IMBALANCED'] * 100
        })
        
        stats = get_dataset_statistics(datasets)
        
        # Check balanced dataset
        assert stats['BALANCED']['total_samples'] == 100
        assert stats['BALANCED']['class_distribution'][0] == 50
        assert stats['BALANCED']['class_distribution'][1] == 50
        
        # Check imbalanced dataset
        assert stats['IMBALANCED']['total_samples'] == 100
        assert stats['IMBALANCED']['class_distribution'][0] == 90
        assert stats['IMBALANCED']['class_distribution'][1] == 10

class TestModelEvaluationIntegration:
    """Test model evaluation integration."""
    
    def test_model_inference_pipeline(self):
        """Test complete model inference pipeline."""
        from src.models.classification.build_model import build_classifier_model
        
        # Build model
        model = build_classifier_model(
            model_name='resnet18',
            num_classes=2,
            pretrained=False
        )
        model.eval()
        
        # Simulate batch inference
        batch_size = 8
        dummy_batch = torch.randn(batch_size, 3, 224, 224)
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for i in range(0, batch_size, 2):  # Process in mini-batches
                mini_batch = dummy_batch[i:i+2]
                outputs = model(mini_batch)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        assert len(all_predictions) == batch_size
        assert len(all_probabilities) == batch_size
        assert all(0 <= pred <= 1 for pred in all_predictions)
    
    def test_metrics_calculation_pipeline(self):
        """Test complete metrics calculation pipeline."""
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
        
        # Simulate model predictions
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_scores = np.random.rand(n_samples)  # Prediction probabilities
        y_pred = (y_scores > 0.5).astype(int)  # Binary predictions
        
        # Calculate comprehensive metrics
        metrics = {}
        
        try:
            metrics['auc'] = roc_auc_score(y_true, y_scores)
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            
            # Sensitivity at 95% specificity
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            
            # Find threshold for 95% specificity (5% FPR)
            target_fpr = 0.05
            idx = np.argmin(np.abs(fpr - target_fpr))
            sensitivity_at_95_spec = tpr[idx]
            metrics['sensitivity_at_95_spec'] = sensitivity_at_95_spec
            
        except ValueError:
            # Handle cases with only one class
            metrics = {k: 0.0 for k in ['auc', 'accuracy', 'precision', 'recall', 'sensitivity_at_95_spec']}
        
        # Validate metric ranges
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} out of range: {value}"

class TestExperimentIntegration:
    """Test experiment management integration."""
    
    def test_experiment_logging_pipeline(self):
        """Test experiment logging and tracking."""
        # Create temporary directory for experiment logs
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Simulate experiment configuration
            experiment_config = {
                'model_name': 'resnet18',
                'num_classes': 2,
                'learning_rate': 1e-4,
                'batch_size': 32,
                'epochs': 10
            }
            
            # Simulate training metrics
            training_metrics = {
                'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
                'val_loss': [1.1, 0.9, 0.7, 0.6, 0.5],
                'val_accuracy': [0.5, 0.6, 0.7, 0.75, 0.8]
            }
            
            # Save experiment results
            experiment_file = os.path.join(tmp_dir, 'experiment_results.json')
            
            import json
            with open(experiment_file, 'w') as f:
                json.dump({
                    'config': experiment_config,
                    'metrics': training_metrics
                }, f, indent=2)
            
            # Verify saved results
            assert os.path.exists(experiment_file)
            
            with open(experiment_file, 'r') as f:
                loaded_results = json.load(f)
            
            assert 'config' in loaded_results
            assert 'metrics' in loaded_results
            assert loaded_results['config']['model_name'] == 'resnet18'

class TestMultitaskIntegration:
    """Test multi-task learning integration."""
    
    def test_multitask_training_pipeline(self):
        """Test multi-task training pipeline."""
        # Skip if not available
        try:
            from scripts.train_multitask_classification_regression import MultiTaskModel
        except ImportError:
            pytest.skip("Multi-task components not available")
        
        # Create multi-task model
        model = MultiTaskModel(
            backbone='resnet18',
            pretrained=False,
            dropout=0.3
        )
        
        # Create synthetic multi-task data
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        binary_labels = torch.randint(0, 2, (batch_size,)).float()
        vcdr_labels = torch.rand(batch_size, 1)
        
        # Indicate which samples have which labels
        has_binary = torch.ones(batch_size, dtype=torch.bool)
        has_vcdr = torch.ones(batch_size, dtype=torch.bool)
        
        # Forward pass
        model.train()
        outputs = model(images)
        
        assert isinstance(outputs, dict)
        assert 'classification_logits' in outputs
        assert 'regression_output' in outputs
        
        # Test loss calculation
        classification_loss = torch.nn.BCEWithLogitsLoss()(
            outputs['classification_logits'][has_binary],
            binary_labels[has_binary]
        )
        
        regression_loss = torch.nn.MSELoss()(
            outputs['regression_output'][has_vcdr],
            vcdr_labels[has_vcdr]
        )
        
        total_loss = classification_loss + regression_loss
        
        assert total_loss.item() >= 0

class TestRobustnessIntegration:
    """Test model robustness and error handling."""
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted or missing data."""
        # Test with missing image files
        test_df = pd.DataFrame({
            'image_path': ['/nonexistent/img1.jpg', '/nonexistent/img2.jpg'],
            'label': [0, 1],
            'dataset_source': ['TEST', 'TEST'],
            'names': ['img1.jpg', 'img2.jpg']
        })
        
        # This should be detected by validation
        from src.data.multisource_loader import validate_dataset_integrity
        datasets = {'TEST': test_df}
        is_valid = validate_dataset_integrity(datasets)
        assert not is_valid
    
    def test_model_error_handling(self):
        """Test model error handling with invalid inputs."""
        from src.models.classification.build_model import build_classifier_model
        
        model = build_classifier_model('resnet18', num_classes=2, pretrained=False)
        model.eval()
        
        # Test with wrong input dimensions
        with pytest.raises((RuntimeError, ValueError)):
            wrong_input = torch.randn(1, 3, 100, 100)  # Wrong size
            with torch.no_grad():
                model(wrong_input)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with gradient accumulation."""
        model = torch.nn.Linear(1000, 2)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Simulate gradient accumulation
        accumulation_steps = 4
        effective_batch_size = 16
        mini_batch_size = effective_batch_size // accumulation_steps
        
        model.train()
        optimizer.zero_grad()
        
        total_loss = 0
        for step in range(accumulation_steps):
            # Create mini-batch
            inputs = torch.randn(mini_batch_size, 1000)
            targets = torch.randint(0, 2, (mini_batch_size,))
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps  # Scale loss
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()
        
        # Update weights after accumulation
        optimizer.step()
        optimizer.zero_grad()
        
        assert total_loss >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
