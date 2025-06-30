"""
Training utilities for glaucoma classification models.

This module contains utility functions for training including mixup sample saving
and other training-related helpers.
"""

import logging
import os
import torch
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


def save_mixup_samples_before_training(train_loader, mixup_fn, save_dir, num_samples=5, device='cpu'):
    """
    Save a few sample images after applying Mixup/CutMix transformations.
    
    Args:
        train_loader: Training DataLoader
        mixup_fn: Mixup function from timm
        save_dir: Directory to save sample images
        num_samples: Number of samples to save
        device: Device to run on
    """
    try:
        # Get one batch from the training loader
        batch_iter = iter(train_loader)
        batch_data = next(batch_iter)
        
        if batch_data is None or len(batch_data) < 2:
            logger.warning("Could not get valid batch for Mixup sampling")
            return
        
        if len(batch_data) == 3:
            inputs, labels, _ = batch_data
        else:
            inputs, labels = batch_data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Apply Mixup/CutMix
        mixed_inputs, mixed_labels = mixup_fn(inputs, labels)
        
        # Save the first few samples
        samples_to_save = min(num_samples, mixed_inputs.size(0))
        
        for i in range(samples_to_save):
            # Convert tensor to PIL Image
            # Denormalize first (assuming ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            
            img_tensor = mixed_inputs[i] * std + mean
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            # Convert to PIL and save
            pil_img = TF.to_pil_image(img_tensor.cpu())
            
            # Create filename with label info
            if hasattr(mixed_labels, 'shape') and len(mixed_labels.shape) > 1:
                # Soft labels from mixup
                label_info = f"soft_{mixed_labels[i, 0]:.2f}_{mixed_labels[i, 1]:.2f}"
            else:
                # Hard labels
                label_info = f"label_{mixed_labels[i].item()}"
            
            filename = f"mixup_sample_{i+1}_{label_info}.png"
            save_path = os.path.join(save_dir, filename)
            pil_img.save(save_path)
        
        logger.info(f"Saved {samples_to_save} Mixup/CutMix sample images to {save_dir}")
        
    except Exception as e:
        logger.error(f"Error saving Mixup samples: {e}")


def save_data_samples_for_verification(df_all, config):
    """
    Save data sample verification information to output directory.
    
    Args:
        df_all: Combined DataFrame with all data
        config: Configuration object with output directory
    """
    try:
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Sample a few rows for verification
        sample_size = min(10, len(df_all))
        df_sample = df_all.sample(n=sample_size, random_state=42)
        
        # Save sample data
        sample_file = os.path.join(config.output_dir, "data_verification_sample.csv")
        df_sample.to_csv(sample_file, index=False)
        
        # Save basic statistics
        stats_file = os.path.join(config.output_dir, "data_statistics.txt")
        with open(stats_file, 'w') as f:
            f.write(f"Total samples: {len(df_all)}\n")
            f.write(f"Columns: {list(df_all.columns)}\n")
            if 'glaucoma_risk' in df_all.columns:
                f.write(f"Glaucoma distribution:\n{df_all['glaucoma_risk'].value_counts()}\n")
            if 'dataset_source' in df_all.columns:
                f.write(f"Dataset source distribution:\n{df_all['dataset_source'].value_counts()}\n")
        
        logger.info(f"Saved data verification files to {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Error saving data verification: {e}")
