"""
Tests for training components including loss functions, 
optimizers, schedulers, and training loops.
"""
import os
import sys
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestLossFunctions:
    """Test custom loss functions."""
    
    def test_binary_classification_loss(self):
        """Test binary classification loss."""
        criterion = nn.BCEWithLogitsLoss()
        
        # Create test data
        logits = torch.randn(4, 1)  # 4 samples, 1 output
        targets = torch.randint(0, 2, (4,)).float()
        
        loss = criterion(logits.squeeze(), targets)
        
        assert loss.item() >= 0
        # Don't test requires_grad on loss result as it depends on context
    
    def test_multitask_loss_calculation(self):
        """Test multi-task loss calculation."""
        # Skip if not available
        try:
            from scripts.train_multitask_classification_regression import calculate_multitask_loss
        except ImportError:
            pytest.skip("Multi-task loss function not available")
        
        # Mock outputs and targets
        outputs = {
            'classification_logits': torch.randn(4, 1),
            'regression_output': torch.randn(4, 1)
        }
        
        binary_targets = torch.randint(0, 2, (4,)).float()
        vcdr_targets = torch.rand(4, 1)
        has_binary = torch.ones(4, dtype=torch.bool)
        has_vcdr = torch.ones(4, dtype=torch.bool)
        
        loss_weights = {'classification': 1.0, 'regression': 1.0}
        
        total_loss, loss_components = calculate_multitask_loss(
            outputs, binary_targets, vcdr_targets, 
            has_binary, has_vcdr, loss_weights
        )
        
        assert total_loss.item() >= 0
        assert 'classification_loss' in loss_components
        assert 'regression_loss' in loss_components
    
    def test_focal_loss(self):
        """Test focal loss for class imbalance."""
        # Simple focal loss implementation for testing
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ce_loss = nn.CrossEntropyLoss(reduction='none')
            
            def forward(self, inputs, targets):
                ce_loss = self.ce_loss(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                return focal_loss.mean()
        
        focal_criterion = FocalLoss(alpha=1, gamma=2)
        
        # Test with class imbalanced data
        logits = torch.randn(10, 2)
        targets = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8:2 ratio
        
        focal_loss = focal_criterion(logits, targets)
        ce_loss = nn.CrossEntropyLoss()(logits, targets)
        
        assert focal_loss.item() >= 0
        # Don't test requires_grad on loss result as it depends on context

class TestOptimizers:
    """Test optimizer configurations."""
    
    def test_adamw_optimizer(self):
        """Test AdamW optimizer setup."""
        model = nn.Linear(10, 2)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-2,
            betas=(0.9, 0.999)
        )
        
        assert optimizer.defaults['lr'] == 1e-4
        assert optimizer.defaults['weight_decay'] == 1e-2
        
        # Test optimizer step
        dummy_input = torch.randn(5, 10)
        dummy_target = torch.randint(0, 2, (5,))
        
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = nn.CrossEntropyLoss()(outputs, dummy_target)
        loss.backward()
        optimizer.step()
        
        # Loss should be computed successfully
        assert loss.item() >= 0
    
    def test_sgd_with_momentum(self):
        """Test SGD with momentum optimizer."""
        model = nn.Linear(10, 2)
        
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        assert optimizer.defaults['lr'] == 0.01
        assert optimizer.defaults['momentum'] == 0.9
        
        # Test parameter update
        initial_params = [p.clone() for p in model.parameters()]
        
        dummy_input = torch.randn(5, 10)
        dummy_target = torch.randint(0, 2, (5,))
        
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = nn.CrossEntropyLoss()(outputs, dummy_target)
        loss.backward()
        optimizer.step()
        
        # Parameters should change
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.equal(initial, current)

class TestSchedulers:
    """Test learning rate schedulers."""
    
    def test_reduce_lr_on_plateau(self):
        """Test ReduceLROnPlateau scheduler."""
        model = nn.Linear(10, 2)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Simulate no improvement for several epochs
        for epoch in range(5):
            loss = 1.0  # Constant loss (no improvement)
            scheduler.step(loss)
        
        # LR should be reduced after patience epochs
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr
    
    def test_cosine_annealing_scheduler(self):
        """Test CosineAnnealingLR scheduler."""
        model = nn.Linear(10, 2)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-6
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step through half the schedule
        for _ in range(5):
            scheduler.step()
        
        mid_lr = optimizer.param_groups[0]['lr']
        
        # Step to the end
        for _ in range(5):
            scheduler.step()
        
        final_lr = optimizer.param_groups[0]['lr']
        
        # LR should decrease significantly over the schedule
        assert final_lr < initial_lr

class TestTrainingComponents:
    """Test training loop components."""
    
    def test_training_step(self):
        """Test a single training step."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Create batch
        batch_size = 4
        inputs = torch.randn(batch_size, 10)
        targets = torch.randint(0, 2, (batch_size,))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Check gradients were computed
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_validation_step(self):
        """Test a single validation step."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Create batch
        batch_size = 4
        inputs = torch.randn(batch_size, 10)
        targets = torch.randint(0, 2, (batch_size,))
        
        # Validation step
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Check no gradients in validation
        for param in model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)
        
        assert loss.item() >= 0
    
    def test_mixed_precision_training(self):
        """Test mixed precision training with GradScaler."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        model = nn.Linear(10, 2).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        scaler = torch.cuda.amp.GradScaler()
        
        batch_size = 4
        inputs = torch.randn(batch_size, 10).cuda()
        targets = torch.randint(0, 2, (batch_size,)).cuda()
        
        # Mixed precision training step
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        assert loss.item() >= 0

class TestDomainAdaptationTraining:
    """Test domain adaptation training components."""
    
    def test_gradient_reversal_layer(self):
        """Test gradient reversal layer."""
        try:
            from src.models.dann_components import GradientReversalFunction
        except ImportError:
            pytest.skip("GradientReversalFunction not available")
        
        # Test forward pass
        x = torch.randn(4, 768, requires_grad=True)
        alpha = 1.0
        
        grl_output = GradientReversalFunction.apply(x, alpha)
        
        # Forward pass should be identity
        assert torch.equal(grl_output, x)
        
        # Test backward pass
        loss = grl_output.sum()
        loss.backward()
        
        # Gradients should be reversed (negated)
        expected_grad = -torch.ones_like(x) * alpha
        assert torch.allclose(x.grad, expected_grad)
    
    def test_domain_classifier_training(self):
        """Test domain classifier training."""
        # Simple domain classifier
        domain_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)  # 3 domains
        )
        
        # Create features and domain labels
        features = torch.randn(8, 768)
        domain_labels = torch.randint(0, 3, (8,))
        
        # Training step
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(domain_classifier.parameters())
        
        optimizer.zero_grad()
        domain_outputs = domain_classifier(features)
        domain_loss = criterion(domain_outputs, domain_labels)
        domain_loss.backward()
        optimizer.step()
        
        assert domain_loss.item() >= 0

class TestDataParallel:
    """Test data parallel training."""
    
    def test_data_parallel_model(self):
        """Test DataParallel model creation."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Multiple GPUs not available")
        
        model = nn.Linear(10, 2)
        model = nn.DataParallel(model)
        model = model.cuda()
        
        # Test forward pass
        inputs = torch.randn(8, 10).cuda()
        outputs = model(inputs)
        
        assert outputs.shape == (8, 2)
        assert outputs.device.type == 'cuda'

class TestEarlyStopping:
    """Test early stopping functionality."""
    
    def test_early_stopping_logic(self):
        """Test early stopping decision logic."""
        # Simple early stopping implementation
        class EarlyStopping:
            def __init__(self, patience=3, min_delta=0.001):
                self.patience = patience
                self.min_delta = min_delta
                self.best_loss = float('inf')
                self.counter = 0
                self.early_stop = False
            
            def __call__(self, val_loss):
                if val_loss < self.best_loss - self.min_delta:
                    self.best_loss = val_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
        
        early_stopping = EarlyStopping(patience=3)
        
        # Simulate improving losses
        losses = [1.0, 0.8, 0.6, 0.4]
        for loss in losses:
            early_stopping(loss)
            assert not early_stopping.early_stop
        
        # Simulate no improvement
        no_improvement_losses = [0.5, 0.6, 0.7, 0.8]
        for loss in no_improvement_losses:
            early_stopping(loss)
        
        assert early_stopping.early_stop

class TestCheckpointing:
    """Test model checkpointing functionality."""
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading model checkpoints."""
        model = nn.Linear(10, 2)
        optimizer = optim.Adam(model.parameters())
        epoch = 5
        loss = 0.5
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }
            torch.save(checkpoint, tmp_file.name)
            
            # Load checkpoint
            loaded_checkpoint = torch.load(tmp_file.name, map_location='cpu')
            
            assert loaded_checkpoint['epoch'] == epoch
            assert loaded_checkpoint['loss'] == loss
            assert 'model_state_dict' in loaded_checkpoint
            assert 'optimizer_state_dict' in loaded_checkpoint
            
            # Load into new model
            new_model = nn.Linear(10, 2)
            new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
            
            # Parameters should match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.equal(p1, p2)
            
            os.unlink(tmp_file.name)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
