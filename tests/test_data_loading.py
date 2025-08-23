"""
Tests for data loading functionality including multisource datasets,
transforms, and dataset classes.
"""
import os
import sys
import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from torchvision import transforms
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestMultisourceLoader:
    """Test the multisource data loader functionality."""
    
    def test_get_dataset_statistics(self):
        """Test dataset statistics calculation."""
        from src.data.multisource_loader import get_dataset_statistics
        
        # Create mock datasets
        dataset1 = pd.DataFrame({
            'image_path': ['/path/img1.jpg', '/path/img2.jpg'],
            'label': [0, 1],
            'dataset_source': ['DS1', 'DS1'],
            'names': ['img1.jpg', 'img2.jpg']
        })
        
        dataset2 = pd.DataFrame({
            'image_path': ['/path/img3.jpg', '/path/img4.jpg', '/path/img5.jpg'],
            'label': [1, 1, 0],
            'dataset_source': ['DS2', 'DS2', 'DS2'],
            'names': ['img3.jpg', 'img4.jpg', 'img5.jpg']
        })
        
        datasets = {'Dataset1': dataset1, 'Dataset2': dataset2}
        stats = get_dataset_statistics(datasets)
        
        assert 'Dataset1' in stats
        assert 'Dataset2' in stats
        assert stats['Dataset1']['total_samples'] == 2
        assert stats['Dataset2']['total_samples'] == 3
        assert stats['Dataset1']['num_classes'] == 2
        assert stats['Dataset2']['num_classes'] == 2
        assert stats['Dataset1']['class_distribution'] == {0: 1, 1: 1}
        assert stats['Dataset2']['class_distribution'] == {0: 1, 1: 2}
    
    def test_combine_train_datasets(self):
        """Test combining multiple training datasets."""
        from src.data.multisource_loader import combine_train_datasets
        
        dataset1 = pd.DataFrame({
            'image_path': ['/path/img1.jpg'],
            'label': [0],
            'dataset_source': ['DS1'],
            'names': ['img1.jpg']
        })
        
        dataset2 = pd.DataFrame({
            'image_path': ['/path/img2.jpg'],
            'label': [1],
            'dataset_source': ['DS2'],
            'names': ['img2.jpg']
        })
        
        train_datasets = {'Dataset1': dataset1, 'Dataset2': dataset2}
        combined = combine_train_datasets(train_datasets)
        
        assert len(combined) == 2
        assert set(combined['dataset_source']) == {'DS1', 'DS2'}
        assert list(combined['label']) == [0, 1]
    
    def test_prepare_leave_one_out_splits(self):
        """Test leave-one-dataset-out split preparation."""
        from src.data.multisource_loader import prepare_leave_one_out_splits
        
        datasets = {
            'DS1': pd.DataFrame({'label': [0, 1]}),
            'DS2': pd.DataFrame({'label': [1, 0]}),
            'DS3': pd.DataFrame({'label': [0, 0]})
        }
        
        splits = prepare_leave_one_out_splits(datasets)
        
        assert len(splits) == 3
        for test_dataset_name, train_datasets in splits:
            assert test_dataset_name in ['DS1', 'DS2', 'DS3']
            assert len(train_datasets) == 2
            assert test_dataset_name not in train_datasets

class TestDataLoaders:
    """Test dataset loading functions."""
    
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_load_airogs_dataset(self, mock_read_csv, mock_exists):
        """Test AIROGS dataset loading."""
        from src.data.multisource_loader import load_airogs_dataset
        
        # Mock CSV data
        mock_df = pd.DataFrame({
            'challenge_id': ['img1', 'img2'],
            'class': [0, 1]
        })
        mock_read_csv.return_value = mock_df
        mock_exists.return_value = True
        
        result = load_airogs_dataset('/fake/labels.csv', '/fake/images/')
        
        assert len(result) == 2
        assert 'dataset_source' in result.columns
        assert all(result['dataset_source'] == 'AIROGS')
        assert 'label' in result.columns
        assert 'image_path' in result.columns
    
    @patch('glob.glob')
    @patch('os.path.exists')
    def test_load_acrima_dataset(self, mock_exists, mock_glob):
        """Test ACRIMA dataset loading."""
        from src.data.multisource_loader import load_acrima_dataset
        
        # Mock file paths
        normal_files = ['/fake/Normal/img1.jpg', '/fake/Normal/img2.jpg']
        glaucoma_files = ['/fake/Glaucoma/img3.jpg']
        
        def glob_side_effect(pattern, recursive=False):
            if 'Normal' in pattern:
                return normal_files
            elif 'Glaucoma' in pattern:
                return glaucoma_files
            return []
        
        mock_glob.side_effect = glob_side_effect
        mock_exists.return_value = True
        
        result = load_acrima_dataset('/fake/acrima/')
        
        assert len(result) == 3
        assert len(result[result['label'] == 0]) == 2  # Normal cases
        assert len(result[result['label'] == 1]) == 1  # Glaucoma cases
        assert all(result['dataset_source'] == 'ACRIMA')

class TestTransforms:
    """Test data transforms and augmentations."""
    
    def test_basic_transforms(self):
        """Test basic image transforms."""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        transformed = transform(dummy_image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 224, 224)
        assert transformed.dtype == torch.float32
        # Check normalization range (approximately)
        assert -3 < transformed.min() < 3
        assert -3 < transformed.max() < 3

class TestDatasetClasses:
    """Test custom dataset classes."""
    
    def test_multitask_dataset_creation(self):
        """Test MultiTaskDataset from training script."""
        # Skip if MultiTaskDataset not available
        try:
            from scripts.train_multitask_classification_regression import MultiTaskDataset
        except ImportError:
            pytest.skip("MultiTaskDataset not available")
        
        # Create mock dataframe with correct column names
        df = pd.DataFrame({
            'image_path': ['/fake/img1.jpg', '/fake/img2.jpg'],
            'binary_label': [0, 1],  # Use the expected column name
            'vcdr': [0.3, 0.7]  # Use lowercase as in actual implementation
        })
        
        # Create basic transform
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # This would normally fail due to missing files, but we test structure
        try:
            dataset = MultiTaskDataset(df, transform=transform)
            assert len(dataset) == 2
            assert hasattr(dataset, 'has_binary')
            assert hasattr(dataset, 'has_vcdr')
        except (FileNotFoundError, AttributeError):
            # Expected since files don't exist
            pass

class TestDataValidation:
    """Test data validation and integrity checks."""
    
    def test_validate_dataset_integrity_valid(self):
        """Test dataset integrity validation with valid data."""
        from src.data.multisource_loader import validate_dataset_integrity
        
        # Create temporary files for testing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test images
            test_files = []
            for i in range(3):
                file_path = os.path.join(temp_dir, f'test_img_{i}.jpg')
                # Create dummy file
                with open(file_path, 'w') as f:
                    f.write('dummy content')
                test_files.append(file_path)
            
            dataset = pd.DataFrame({
                'image_path': test_files,
                'label': [0, 1, 0],
                'dataset_source': ['TEST', 'TEST', 'TEST'],
                'names': ['img1.jpg', 'img2.jpg', 'img3.jpg']
            })
            
            datasets = {'TEST': dataset}
            is_valid = validate_dataset_integrity(datasets)
            assert is_valid
    
    def test_validate_dataset_integrity_missing_columns(self):
        """Test dataset integrity validation with missing columns."""
        from src.data.multisource_loader import validate_dataset_integrity
        
        dataset = pd.DataFrame({
            'image_path': ['/fake/img1.jpg'],
            'label': [0],
            # Missing 'dataset_source' and 'names'
        })
        
        datasets = {'TEST': dataset}
        is_valid = validate_dataset_integrity(datasets)
        assert not is_valid
    
    def test_validate_dataset_integrity_empty_dataset(self):
        """Test dataset integrity validation with empty dataset."""
        from src.data.multisource_loader import validate_dataset_integrity
        
        dataset = pd.DataFrame()
        datasets = {'TEST': dataset}
        is_valid = validate_dataset_integrity(datasets)
        assert not is_valid

class TestDataLoaderUtils:
    """Test data loader utility functions."""
    
    def test_safe_collate_function(self):
        """Test safe collate function for handling errors."""
        try:
            from src.data.utils import safe_collate
        except ImportError:
            pytest.skip("safe_collate not available")
        
        # Test with valid batch
        batch = [
            (torch.randn(3, 224, 224), torch.tensor(0)),
            (torch.randn(3, 224, 224), torch.tensor(1))
        ]
        
        result = safe_collate(batch)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (2, 3, 224, 224)
        assert result[1].shape == (2,)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
