"""
Simple test script to verify calibration works.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_multitask_classification_regression import SingleTaskModel, set_seed

def test_calibration():
    """Test basic temperature scaling functionality."""
    set_seed(42)
    
    # Create a simple model
    model = SingleTaskModel(backbone='resnet18', pretrained=False)
    model.eval()
    
    # Create some dummy data
    batch_size = 10
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(dummy_images)
        print(f"Model output type: {type(outputs)}")
        
        if isinstance(outputs, dict):
            logits = outputs.get('classification_logits', outputs.get('logits'))
            print(f"Logits shape from dict: {logits.shape if logits is not None else 'None'}")
        else:
            logits = outputs
            print(f"Logits shape direct: {logits.shape}")
            
        # Test softmax
        if logits is not None:
            probs = F.softmax(logits, dim=1)
            print(f"Probabilities shape: {probs.shape}")
            print(f"Sample probabilities: {probs[0]}")
            
            # Test temperature scaling
            temperature = 2.0
            scaled_logits = logits / temperature
            scaled_probs = F.softmax(scaled_logits, dim=1)
            print(f"Scaled probabilities: {scaled_probs[0]}")
            print(f"Temperature scaling working: {not torch.equal(probs[0], scaled_probs[0])}")
        else:
            print("Could not extract logits from model output")

if __name__ == "__main__":
    test_calibration()
