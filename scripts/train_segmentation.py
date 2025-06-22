import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import logging
import json

# Make sure Python can find the src modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logging
from src.utils.plotting import plot_training_history_segmentation # Import plotting function
from src.data.datasets import FundusMultiTaskDataset
# Choose appropriate transform function
from src.data.transforms import get_segmentation_transforms
# Or use separate transforms:
# from src.data.transforms import get_segmentation_img_transforms, get_segmentation_mask_transforms
from src.models.segmentation.unet import MultiTaskUNet
from src.training.losses import multi_task_segmentation_loss # Import the specific loss
from src.training.trainers import train_segmentation_model

def main(args):
    """Main function to run segmentation model training."""
    # Load configuration
    config = load_config(args.config)
    paths_config = config.get('paths', {})
    train_config = config.get('training', {}).get('segmentation', {}) # Segmentation specific training params
    model_config = config.get('model', {}).get('segmentation', {}) # Segmentation specific model params

    # Setup logging
    log_dir = paths_config.get('log_dir', 'logs')
    model_name = model_config.get('name', 'unet_segmentation')
    log_file = os.path.join(log_dir, f'train_{model_name}.log')
    setup_logging(log_level=config.get('log_level', 'INFO'), log_dir=log_dir, log_file=log_file)
    logging.info("Starting segmentation model training script...")
    logging.info(f"Loaded configuration from: {args.config}")

    # --- Setup ---
    processed_data_dir = paths_config.get('processed_data_dir')
    checkpoint_dir = paths_config.get('checkpoint_dir', 'checkpoints')
    results_dir = paths_config.get('results_dir', 'results')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # --- Load Data ---
    data_file = train_config.get('data_file', 'smdg_cleaned.csv') # Assumes paths are pre-generated
    data_path = os.path.join(processed_data_dir, data_file)
    logging.info(f"Loading processed data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        # Define expected columns based on dataset class
        img_col = train_config.get('image_col', 'fundus_path')
        oc_mask_col = train_config.get('oc_mask_col', 'oc_mask_path')
        od_mask_col = train_config.get('od_mask_col', 'od_mask_path')
        required_cols = [img_col, oc_mask_col, od_mask_col]
        if not all(col in df.columns for col in required_cols):
             raise ValueError(f"Required columns not found in {data_path}: Need {required_cols}")
        # Dataset class handles dropping NaNs in these columns
        logging.info(f"Loaded {len(df)} potential samples for segmentation.")
    except FileNotFoundError:
        logging.error(f"Data file not found: {data_path}")
        return
    except Exception as e:
        logging.error(f"Error loading or processing data file {data_path}: {e}", exc_info=True)
        return

    # --- Create Datasets and DataLoaders ---
    img_size = model_config.get('img_size', 256)
    # Using a single transform for image and mask for simplicity here
    # Ensure it works correctly (e.g., ToTensor handles L and RGB inputs)
    transform = get_segmentation_transforms(img_size=img_size)
    # If using separate transforms:
    # img_transform = get_segmentation_img_transforms(img_size=img_size)
    # mask_transform = get_segmentation_mask_transforms(img_size=img_size)

    # Split data
    test_size = train_config.get('val_split', 0.2)
    random_state = train_config.get('random_seed', 42)
    # No stratification needed typically for segmentation unless masks are highly imbalanced per image type
    train_indices, val_indices = train_test_split(range(len(df)), test_size=test_size, random_state=random_state)
    logging.info(f"Splitting data: Train={len(train_indices)}, Validation={len(val_indices)}")

    # Create Datasets using the *same* underlying DataFrame but different indices
    train_dataset = FundusMultiTaskDataset(
        df.iloc[train_indices].reset_index(drop=True),
        image_col=img_col, oc_mask_col=oc_mask_col, od_mask_col=od_mask_col,
        transform=transform # Or img_transform=img_transform, mask_transform=mask_transform
    )
    val_dataset = FundusMultiTaskDataset(
        df.iloc[val_indices].reset_index(drop=True),
        image_col=img_col, oc_mask_col=oc_mask_col, od_mask_col=od_mask_col,
        transform=transform # Apply same base transform for validation
    )

    batch_size = train_config.get('batch_size', 8)
    num_workers = train_config.get('num_workers', os.cpu_count() // 2)
    # Consider a custom collate_fn if datasets might return None on error
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    logging.info(f"Created DataLoaders: batch_size={batch_size}, num_workers={num_workers}")

    # --- Initialize Model, Loss, Optimizer, Scheduler ---
    n_channels = model_config.get('n_channels', 3)
    bilinear = model_config.get('bilinear', True)
    model = MultiTaskUNet(n_channels=n_channels, bilinear=bilinear)
    logging.info(f"Initialized model: {model_name} (bilinear={bilinear})")

    # Loss function parameters from config
    loss_alpha = train_config.get('loss_alpha', 0.5) # Weight for OC loss
    loss_type = train_config.get('loss_type', 'dice') # 'dice' or 'bce'
    logging.info(f"Using loss: {loss_type} with alpha={loss_alpha}")

    optimizer_name = train_config.get('optimizer', 'Adam').lower() # Adam often preferred for segmentation
    lr = train_config.get('learning_rate', 1e-4)
    weight_decay = train_config.get('weight_decay', 1e-5) # Often lower for segmentation
    if optimizer_name == 'adam':
         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
         optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Add other optimizers like RMSprop if needed
    else:
        logging.warning(f"Unsupported optimizer '{optimizer_name}'. Defaulting to Adam.")
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    logging.info(f"Using optimizer: {type(optimizer).__name__} (lr={lr}, weight_decay={weight_decay})")

    scheduler_config = train_config.get('scheduler', {})
    scheduler = None
    if scheduler_config.get('use', False):
        scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau').lower()
        if scheduler_type == 'reducelronplateau':
             scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get('mode', 'min'), # Usually minimize validation loss
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 10), # Often higher patience for segmentation
                verbose=True
             )
             logging.info("Using ReduceLROnPlateau scheduler.")
        # Add other schedulers if needed (e.g., CosineAnnealingLR)
        else:
             logging.warning(f"Unsupported scheduler type: {scheduler_type}. No scheduler used.")


    # --- Train Model ---
    num_epochs = train_config.get('epochs', 100)
    trained_model, history = train_segmentation_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=multi_task_segmentation_loss, # Pass the imported loss function
        loss_alpha=loss_alpha,
        loss_type=loss_type,
        num_epochs=num_epochs,
        device=device,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        log_file=log_file # Pass log file path to trainer
    )

    # --- Save Results ---
    # Best model is saved by the trainer
    history_file = os.path.join(results_dir, f'{model_name}_training_history.json')
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
        logging.info(f"Training history saved to: {history_file}")
    except Exception as e:
        logging.error(f"Failed to save training history: {e}")

    # Plot history
    history_plot_file = os.path.join(results_dir, f'{model_name}_training_history.png')
    try:
        plot_training_history_segmentation(history, save_path=history_plot_file)
    except Exception as e:
        logging.error(f"Failed to plot training history: {e}")

    logging.info("Segmentation model training script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Glaucoma Multi-Task Segmentation Model")
    parser.add_argument('--config', type=str, required=True, help='Path to the main configuration YAML file.')
    parser.add_argument('--device', type=str, help='Device to use (cuda or cpu). Overrides config if set.')
    args = parser.parse_args()
    main(args)