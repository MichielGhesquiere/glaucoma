"""
Domain adaptation utilities for multi-source domain fine-tuning.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

logger = logging.getLogger(__name__)


class GradientReversalFunction(Function):
    """Gradient reversal layer implementation for domain adversarial training."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def gradient_reversal_layer(x, alpha=1.0):
    """Apply gradient reversal with given alpha."""
    return GradientReversalFunction.apply(x, alpha)


class DomainClassifier(nn.Module):
    """Domain classifier for DANN (Domain Adversarial Neural Networks)."""
    
    def __init__(self, input_dim: int, num_domains: int, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_domains)
        )
        
    def forward(self, x, alpha=1.0):
        reversed_features = gradient_reversal_layer(x, alpha)
        return self.classifier(reversed_features)


class MixStyle(nn.Module):
    """
    MixStyle module for domain generalization.
    Mixes feature statistics across different domains.
    """
    
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p      # Probability of applying MixStyle
        self.alpha = alpha  # Mixing strength
        self.eps = eps  # Small constant for numerical stability
        
    def forward(self, x):
        if not self.training:
            return x
            
        if torch.rand(1) > self.p:
            return x
            
        B = x.size(0)
        if B < 2:
            return x
            
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        
        mu_mix, sig_mix = self._mix_stats(mu, sig)
        
        return sig_mix * (x - mu) / sig + mu_mix
        
    def _mix_stats(self, mu, sig):
        """Mix statistics between different samples."""
        perm = torch.randperm(mu.size(0)).to(mu.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().to(mu.device)
        
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        
        return mu_mix, sig_mix


class StochasticWeightAveraging:
    """
    Stochastic Weight Averaging for improved generalization.
    """
    
    def __init__(self, model: nn.Module, swa_start: int = 10, swa_freq: int = 1):
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_n = 0
        self.swa_state = {}
        
    def update(self, epoch: int):
        """Update SWA state if conditions are met."""
        if epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0:
            self._update_swa()
            
    def _update_swa(self):
        """Update the SWA running average."""
        if self.swa_n == 0:
            # Initialize SWA state
            for name, param in self.model.named_parameters():
                self.swa_state[name] = param.data.clone()
        else:
            # Update running average
            for name, param in self.model.named_parameters():
                self.swa_state[name] += (param.data - self.swa_state[name]) / (self.swa_n + 1)
                
        self.swa_n += 1
        
    def apply_swa_weights(self):
        """Apply SWA weights to the model."""
        if self.swa_n > 0:
            for name, param in self.model.named_parameters():
                param.data.copy_(self.swa_state[name])
            logger.info(f"Applied SWA weights (averaged over {self.swa_n} checkpoints)")
        else:
            logger.warning("No SWA weights to apply")


class TestTimeAdaptation:
    """
    Test-Time Adaptation using entropy minimization.
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-3, steps: int = 1):
        self.model = model
        self.learning_rate = learning_rate
        self.steps = steps
        self.original_state = None
        
    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        """Perform test-time adaptation on input batch."""
        # Save original state
        self.original_state = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Create optimizer for adaptation
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Perform adaptation steps
        self.model.train()
        for _ in range(self.steps):
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            
            # Entropy loss
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        # Return adapted predictions
        self.model.eval()
        with torch.no_grad():
            adapted_logits = self.model(x)
            
        return adapted_logits
        
    def restore_original_weights(self):
        """Restore original model weights."""
        if self.original_state is not None:
            for name, param in self.model.named_parameters():
                param.data.copy_(self.original_state[name])


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins for calibration
        
    Returns:
        Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    total_samples = len(y_true)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece


def sensitivity_at_specificity(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    target_specificity: float = 0.95
) -> float:
    """
    Calculate sensitivity at a given specificity threshold.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        target_specificity: Target specificity level
        
    Returns:
        Sensitivity at the target specificity
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    
    # Find the threshold that gives us the desired specificity
    idx = np.argmax(specificity >= target_specificity)
    
    if idx == 0 and specificity[0] < target_specificity:
        # If we can't achieve the target specificity, return 0
        return 0.0
        
    return tpr[idx]
