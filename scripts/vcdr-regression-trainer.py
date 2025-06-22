# vcdr_regression_trainer_multiarch.py

import argparse
import glob
import logging
import os
import re
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# torchvision.models is used if model_name is e.g. 'resnet18' from torchvision
import torchvision.models as torchvision_models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Attempt to import timm, crucial for build_regressor_model
try:
    import timm
except ImportError:
    print("WARNING: timm library not found. Please install it: pip install timm. Some models may not be available.")
    timm = None  # Set to None if not available

warnings.filterwarnings('ignore')

# --- Logger Setup ---
# Configure basic logging (console only initially, file handler added per experiment)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# --- Data Loading and Preprocessing ---
def parse_filename(filepath: Path) -> dict | None:
    """
    Parses a filename to extract metadata for the G-RISK dataset.

    Expected format: {eye_pos}_contact{id}_dag{id}_patient{id}_{gender}_{age}j_{gt}.jpg
    Example: LL_contact123_dag1_patient456_M_55j_75.jpg

    Args:
        filepath (Path): Path object of the image file.

    Returns:
        dict | None: A dictionary containing parsed metadata or None if parsing fails.
    """
    filename = filepath.name
    pattern = r"(LL|LR|RL|RR)_contact(\d+)_dag(\d+)_patient(\d+)_([MFV])_(\d+)j_(\d+)\.jpg"
    match = re.match(pattern, filename)
    if match:
        eye_pos, contact_id, dag_id, patient_id, gender_raw, age, gt_raw = match.groups()
        gender_std = 'F' if gender_raw == 'V' else gender_raw  # Standardize 'V' (Vrouw) to 'F'
        return {
            'filepath': str(filepath),
            'filename': filename,
            'eye_position': eye_pos,
            'contact_id': int(contact_id),
            'dag_id': int(dag_id),
            'patient_id': int(patient_id),
            'gender_raw': gender_raw,
            'gender': gender_std,
            'age': int(age),
            'ground_truth_raw': int(gt_raw),
            'ground_truth_vcdr': int(gt_raw) / 100.0  # VCDR on a 0-1 scale
        }
    return None


def load_and_preprocess_data(data_dir: str, processed_data_save_path: str) -> pd.DataFrame:
    """
    Loads image metadata from filenames in the G-RISK dataset, preprocesses it,
    and saves a CSV of the processed metadata.

    Args:
        data_dir (str): Path to the directory containing raw G-RISK JPG images.
        processed_data_save_path (str): Path to save the processed metadata CSV.

    Returns:
        pd.DataFrame: DataFrame containing the processed image metadata.
    """
    logger.info(f"Loading data from G-RISK directory: {data_dir}")
    image_files = list(Path(data_dir).glob("*.jpg"))
    logger.info(f"Found {len(image_files)} images.")

    parsed_data = []
    unparsed_count = 0
    for img_path in image_files:
        parsed_info = parse_filename(img_path)
        if parsed_info:
            parsed_data.append(parsed_info)
        else:
            unparsed_count += 1
    df = pd.DataFrame(parsed_data)

    logger.info(f"Successfully parsed {len(df)} files.")
    if unparsed_count > 0:
        logger.warning(f"Failed to parse {unparsed_count} files.")

    if df.empty:
        logger.error("No data parsed. Check data directory and filename format.")
        raise ValueError("No data parsed from G-RISK directory.")

    logger.info("\n=== G-RISK DATASET OVERVIEW ===")
    logger.info(f"Total images parsed: {len(df)}")
    logger.info(f"Unique patients: {df['patient_id'].nunique()}")
    logger.info(f"Unique contacts: {df['contact_id'].nunique()}")
    logger.info(f"Age range: {df['age'].min()} - {df['age'].max()} years")
    logger.info(f"Gender distribution (standardized):\n{df['gender'].value_counts().to_string()}")
    logger.info(f"Eye position distribution:\n{df['eye_position'].value_counts().to_string()}")

    logger.info("\n=== GROUND TRUTH VCDR STATISTICS ===")
    logger.info(f"VCDR range: {df['ground_truth_vcdr'].min():.2f} - {df['ground_truth_vcdr'].max():.2f}")
    logger.info(f"Mean VCDR: {df['ground_truth_vcdr'].mean():.3f}")
    logger.info(f"Median VCDR: {df['ground_truth_vcdr'].median():.3f}")
    logger.info(f"Std Dev VCDR: {df['ground_truth_vcdr'].std():.3f}")

    logger.info("\n=== DATA QUALITY ===")
    logger.info(f"Missing values per column:\n{df.isnull().sum().to_string()}")
    logger.info(f"Duplicate rows: {df.duplicated().sum()}")

    # Save the processed dataset
    save_path = Path(processed_data_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Processed G-RISK metadata saved to: {save_path}")
    return df


# --- Custom Weight Loading Utility ---
def load_custom_pretrained_weights(model: nn.Module,
                                   weights_path: str,
                                   checkpoint_key: str = 'model',
                                   model_name_for_adapter: str | None = None,
                                   strict: bool = False):
    """
    Loads custom pretrained weights into a model.

    Args:
        model (nn.Module): The model to load weights into.
        weights_path (str): Path to the .pth or .pt checkpoint file.
        checkpoint_key (str): Key in the checkpoint dictionary for model state_dict.
                              Common keys: 'model', 'state_dict', 'teacher'.
        model_name_for_adapter (str | None): Name of the model, used for logging.
        strict (bool): Whether to strictly enforce that the keys in state_dict
                       match the keys returned by this module’s state_dict() function.
    """
    weights_p = Path(weights_path)
    if not weights_p.exists():
        logger.error(f"Custom weights path does not exist: {weights_path}")
        raise FileNotFoundError(f"Custom weights path does not exist: {weights_path}")

    model_id = model_name_for_adapter or "model"
    logger.info(f"Loading custom weights for '{model_id}' from: {weights_path} "
                f"with key '{checkpoint_key}', strict={strict}")
    try:
        checkpoint = torch.load(weights_p, map_location='cpu')
        state_dict = None

        if isinstance(checkpoint, dict):
            if checkpoint_key in checkpoint:
                state_dict = checkpoint[checkpoint_key]
            elif 'state_dict' in checkpoint: # Common alternative
                state_dict = checkpoint['state_dict']
                logger.info(f"Used 'state_dict' key from checkpoint for '{model_id}'.")
            elif 'teacher' in checkpoint and 'dinov2' in (model_id.lower() or ""): # DINOv2 specific
                 state_dict = checkpoint['teacher']
                 logger.info(f"Used 'teacher' key from checkpoint for DINOv2 model '{model_id}'.")
            else: # Try to load the whole checkpoint if specific key not found and it's a flat dict
                if all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
                    state_dict = checkpoint
                    logger.info(f"Loaded entire checkpoint as state_dict for '{model_id}'.")
                else:
                    logger.warning(f"Checkpoint key '{checkpoint_key}' not found. Available keys: {list(checkpoint.keys())}. Trying common keys.")
                    # Try other common keys
                    for common_key in ['model', 'state_dict', 'teacher', 'student']:
                        if common_key in checkpoint:
                            state_dict = checkpoint[common_key]
                            logger.info(f"Found weights under fallback key '{common_key}' for '{model_id}'.")
                            break
        elif isinstance(checkpoint, nn.Module): # If entire model was saved
            state_dict = checkpoint.state_dict()
            logger.info(f"Loaded state_dict from a saved nn.Module instance for '{model_id}'.")
        else: # Assume checkpoint is the state_dict itself
            state_dict = checkpoint
            logger.info(f"Loaded checkpoint directly as state_dict for '{model_id}'.")


        if state_dict is None:
            err_msg = f"Could not extract state_dict for '{model_id}' from checkpoint. "
            if isinstance(checkpoint, dict):
                err_msg += f"Available keys: {list(checkpoint.keys())}"
            raise ValueError(err_msg)

        # Remove 'module.' prefix if present (from DataParallel/DDP)
        new_state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

        if any(k.startswith('backbone.') for k in new_state_dict.keys()):
            backbone_state_dict = {}
            for k, v in new_state_dict.items():
                if k.startswith('backbone.'):
                    # Remove 'backbone.' prefix
                    new_key = k.replace('backbone.', '', 1)
                    backbone_state_dict[new_key] = v
                elif not k.startswith('head.'):  # Skip head weights since we're replacing them
                    backbone_state_dict[k] = v

            if backbone_state_dict:
                if 'norm.weight' in backbone_state_dict and 'fc_norm.weight' not in backbone_state_dict:
                    # Map norm to fc_norm for TIMM ViT compatibility
                    backbone_state_dict['fc_norm.weight'] = backbone_state_dict.pop('norm.weight')
                    backbone_state_dict['fc_norm.bias'] = backbone_state_dict.pop('norm.bias')
                    logger.info("Mapped 'norm' to 'fc_norm' for TIMM ViT compatibility.")
                logger.info(f"Extracted 'backbone.' prefixed weights for '{model_id}'.")
                new_state_dict = backbone_state_dict
    

        # Specific DINOv2 handling for 'teacher.' prefix if not caught by checkpoint_key
        if 'dinov2' in (model_id.lower() or "") and any(k.startswith('teacher.') for k in new_state_dict.keys()):
            teacher_state_dict = {k.replace('teacher.', '', 1): v for k, v in new_state_dict.items() if k.startswith('teacher.')}
            if teacher_state_dict:
                logger.info(f"Extracted 'teacher.' prefixed weights for DINOv2 model '{model_id}'.")
                new_state_dict = teacher_state_dict
        
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)
        
        if missing_keys:
            logger.warning(f"Missing keys when loading weights for '{model_id}': {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading weights for '{model_id}': {unexpected_keys}")
        
        if not missing_keys and not unexpected_keys:
            logger.info(f"Successfully loaded weights into '{model_id}' (all keys matched).")
        else:
            logger.info(f"Weights loaded into '{model_id}' with some mismatched keys (strict={strict}).")

    except Exception as e:
        logger.error(f"Error loading custom weights for '{model_id}' from {weights_path}: {e}", exc_info=True)
        raise


# --- Generalized Model Building for Regression ---
def build_regressor_model(model_name: str,
                          num_outputs: int = 1,
                          head_dropout_prob: float = 0.5,
                          use_timm_hub_pretrained: bool = True,
                          custom_weights_path: str | None = None,
                          checkpoint_key: str = 'model') -> nn.Module:
    """
    Builds a regression model using various architectures (TIMM, DINOv2, TorchVision).

    Args:
        model_name (str): Name of the model architecture (e.g., "resnet50",
                          "dinov2_vitb14", "convnext_base", "vit_base_patch16_224").
        num_outputs (int): Number of output units for the regression head (typically 1).
        head_dropout_prob (float): Dropout probability for the first dropout layer in the custom head.
        use_timm_hub_pretrained (bool): Whether to use default pretrained weights
                                     (ImageNet for TIMM, Hub for DINOv2) if
                                     `custom_weights_path` is not provided.
        custom_weights_path (str | None): Path to custom pretrained backbone weights.
                                       If provided, these are loaded.
        checkpoint_key (str): Key in the checkpoint dictionary for model state_dict.

    Returns:
        nn.Module: The constructed regression model.
    """
    logger.info(f"--- Building Regressor Model: {model_name} ---")
    logger.info(f"Num Outputs: {num_outputs}, Head Dropout1: {head_dropout_prob}")
    logger.info(f"Use TIMM/Hub Default Pretrained Backbone: {use_timm_hub_pretrained}")
    logger.info(f"Custom Backbone Weights Path: {custom_weights_path}")
    if custom_weights_path:
        logger.info(f"Checkpoint Key for Custom Weights: '{checkpoint_key}'")

    model_backbone: nn.Module | None = None
    num_in_features: int = 0

    # --- DINOv2 Specific Handling ---
    if 'dinov2' in model_name.lower():
        dinov2_variant_map = {
            'vits14': ('dinov2_vits14', 384), 'vitb14': ('dinov2_vitb14', 768),
            'vitl14': ('dinov2_vitl14', 1024), 'vitg14': ('dinov2_vitg14', 1536),
        }
        dinov2_key = model_name.lower().split('dinov2_')[-1] if 'dinov2_' in model_name.lower() else 'vitb14'
        if dinov2_key not in dinov2_variant_map:
            logger.warning(f"DINOv2 variant '{dinov2_key}' not recognized. Defaulting to 'vitb14'.")
            dinov2_key = 'vitb14'
        
        hub_model_name, num_in_features = dinov2_variant_map[dinov2_key]
        
        load_fb_pretrained = use_timm_hub_pretrained and (custom_weights_path is None)
        logger.info(f"Loading DINOv2 Hub model: {hub_model_name}, Official FB Pretrained: {load_fb_pretrained}")
        model_backbone = torch.hub.load('facebookresearch/dinov2', hub_model_name, pretrained=load_fb_pretrained)

        if custom_weights_path:
            load_custom_pretrained_weights(model_backbone, custom_weights_path, checkpoint_key,
                                           model_name_for_adapter=hub_model_name, strict=False)
    
    # --- Standard TIMM / TorchVision Model Handling ---
    else:
        timm_pretrained_flag = use_timm_hub_pretrained and (custom_weights_path is None)
        
        # Prioritize TIMM if available and model is not a simple torchvision name
        if timm and not hasattr(torchvision_models, model_name):
            logger.info(f"Attempting to load TIMM model: {model_name}, TIMM Default Pretrained: {timm_pretrained_flag}")
            try:
                feature_extractor = timm.create_model(model_name, pretrained=timm_pretrained_flag, num_classes=0, global_pool='avg')
                num_in_features = feature_extractor.num_features
                
                # Create the model instance we'll actually use for modification
                # Load TIMM default pretrained here if no custom weights, otherwise load custom weights later
                model_backbone = timm.create_model(
                    model_name,
                    pretrained=timm_pretrained_flag if not custom_weights_path else False,
                    num_classes=0, # We will replace the classifier
                    global_pool='avg'
                )
                if custom_weights_path:
                    load_custom_pretrained_weights(model_backbone, custom_weights_path, checkpoint_key,
                                                   model_name_for_adapter=model_name, strict=False)
            except Exception as e_timm:
                logger.warning(f"Failed to load '{model_name}' with TIMM: {e_timm}. Trying TorchVision if applicable.")
                model_backbone = None # Ensure fallback
        
        # Fallback to TorchVision if TIMM failed or model_name suggests TorchVision
        if model_backbone is None and hasattr(torchvision_models, model_name):
            logger.info(f"Using TorchVision model: {model_name}, TorchVision Default Pretrained: {timm_pretrained_flag}")
            # Map common model names to their TorchVision weights enum if available
            weights_enum = None
            if timm_pretrained_flag: # Only load default weights if requested and no custom ones
                if model_name == 'resnet18': weights_enum = torchvision_models.ResNet18_Weights.DEFAULT
                elif model_name == 'resnet50': weights_enum = torchvision_models.ResNet50_Weights.DEFAULT
                # Add more mappings here as needed
            
            model_backbone = getattr(torchvision_models, model_name)(weights=weights_enum)
            
            # Determine num_in_features for TorchVision models
            if hasattr(model_backbone, 'fc') and isinstance(model_backbone.fc, nn.Linear):
                num_in_features = model_backbone.fc.in_features
            elif hasattr(model_backbone, 'classifier'):
                if isinstance(model_backbone.classifier, nn.Linear):
                    num_in_features = model_backbone.classifier.in_features
                elif isinstance(model_backbone.classifier, nn.Sequential) and \
                     hasattr(model_backbone.classifier[-1], 'in_features'): # e.g., EfficientNet, VGG
                    num_in_features = model_backbone.classifier[-1].in_features
            if num_in_features == 0:
                 raise ValueError(f"Cannot determine num_in_features for TorchVision model {model_name}")
            
            if custom_weights_path: # Load custom weights if provided, even for torchvision
                load_custom_pretrained_weights(model_backbone, custom_weights_path, checkpoint_key,
                                               model_name_for_adapter=model_name, strict=False)
        elif model_backbone is None: # If neither TIMM nor TorchVision could load it
            raise RuntimeError(f"Model '{model_name}' could not be loaded via TIMM or TorchVision.")

    if model_backbone is None:
        raise RuntimeError(f"Model backbone for '{model_name}' could not be constructed.")
    if num_in_features <= 0:
        raise ValueError(f"Could not determine num_in_features for '{model_name}'. Detected: {num_in_features}")

    # Define the custom regression head (consistent multi-layer structure)
    # Dropout probabilities for the head layers (can be further parameterized if needed)
    head_dp1, head_dp2, head_dp3 = head_dropout_prob, 0.3, 0.2
    regressor_head = nn.Sequential(
        nn.Dropout(head_dp1), nn.Linear(num_in_features, 512), nn.ReLU(),
        nn.Dropout(head_dp2), nn.Linear(512, 256), nn.ReLU(),
        nn.Dropout(head_dp3), nn.Linear(256, num_outputs),
        nn.Sigmoid()  # Sigmoid to ensure VCDR output is between 0 and 1
    )

    # Attach the new regression head
    classifier_replaced = False
    if 'dinov2' in model_name.lower() and hasattr(model_backbone, 'head'):
        model_backbone.head = regressor_head
        classifier_replaced = True
        logger.info("Replaced DINOv2 'model.head' with custom regressor head.")
    elif timm and hasattr(model_backbone, 'default_cfg'): # TIMM model
        clf_attr = model_backbone.default_cfg.get('classifier', 'fc')
        # Handle nested classifiers like ConvNeXt's model.head.fc
        if model_name.startswith('convnext') and hasattr(model_backbone, 'head') and hasattr(model_backbone.head, 'fc'):
            model_backbone.head.fc = regressor_head
            classifier_replaced = True
            logger.info(f"Replaced ConvNeXt 'model.head.fc' with custom regressor head.")
        elif hasattr(model_backbone, clf_attr):
            setattr(model_backbone, clf_attr, regressor_head)
            classifier_replaced = True
            logger.info(f"Replaced TIMM model '{clf_attr}' with custom regressor head.")
    
    if not classifier_replaced: # Fallback for TorchVision or TIMM models without clear default_cfg
        common_clf_attrs = ['fc', 'classifier', 'head'] # Order of preference
        for attr in common_clf_attrs:
            if hasattr(model_backbone, attr):
                # If attr is a Sequential (e.g. VGG classifier), better to replace whole thing
                if isinstance(getattr(model_backbone, attr), nn.Linear) or \
                   isinstance(getattr(model_backbone, attr), nn.Sequential) or \
                   attr == 'head': # DINOv2 head also okay to replace directly
                    setattr(model_backbone, attr, regressor_head)
                    classifier_replaced = True
                    logger.info(f"Replaced (fallback) '{attr}' on model with custom regressor head.")
                    break
    
    if not classifier_replaced:
        raise AttributeError(f"Model {model_name} does not have a recognizable classifier attribute "
                             f"(fc, classifier, head) to replace with the regressor head.")

    logger.info(f"--- Model {model_name} built and configured for regression successfully. ---")
    return model_backbone


# --- PyTorch Dataset ---
class VCDRDataset(Dataset):
    """Custom PyTorch Dataset for VCDR regression."""
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx: int):
        try:
            row = self.dataframe.iloc[idx]
            image_path = row['filepath']
            image = Image.open(image_path).convert('RGB')
            vcdr_score = row['ground_truth_vcdr']

            if self.transform:
                image = self.transform(image)
            return image, torch.FloatTensor([vcdr_score])
        except Exception as e:
            # Log error and return a dummy tensor to prevent training crash
            logger.error(f"Error loading image {self.dataframe.iloc[idx]['filepath']} at index {idx}: {e}")
            return torch.zeros(3, 224, 224), torch.FloatTensor([-1.0]) # Use -1.0 to flag error


# --- Image Transforms ---
def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Returns training and validation/test image transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform


# --- Data Splitting ---
def split_data_by_patient(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15,
                          test_ratio: float = 0.15, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits DataFrame by patient ID to prevent data leakage."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    
    unique_patients = df['patient_id'].unique()
    n_patients = len(unique_patients)
    logger.info(f"Total unique patients for split: {n_patients}")
    
    train_patients, temp_patients = train_test_split(
        unique_patients, test_size=(val_ratio + test_ratio), random_state=random_state
    )
    # Ensure val_ratio + test_ratio is not zero for division
    val_test_sum = val_ratio + test_ratio
    if val_test_sum == 0: # Handle case where only train_ratio is 1.0
        val_patients, test_patients = np.array([]), np.array([])
    else:
        val_patients, test_patients = train_test_split(
            temp_patients, test_size=test_ratio / (val_test_sum + 1e-9), random_state=random_state
        )
    
    train_df = df[df['patient_id'].isin(train_patients)].copy()
    val_df = df[df['patient_id'].isin(val_patients)].copy()
    test_df = df[df['patient_id'].isin(test_patients)].copy()
    
    logger.info(f"Train patients: {len(train_patients)} ({len(train_patients)/n_patients*100:.1f}%) - Images: {len(train_df)}")
    logger.info(f"Val patients: {len(val_patients)} ({len(val_patients)/n_patients*100:.1f}%) - Images: {len(val_df)}")
    logger.info(f"Test patients: {len(test_patients)} ({len(test_patients)/n_patients*100:.1f}%) - Images: {len(test_df)}")
    
    return train_df, val_df, test_df


# --- Training and Validation Epoch Functions ---
def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device,
                epoch_num: int, num_total_epochs: int) -> tuple[float, float, float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss, total_mae = 0.0, 0.0
    num_valid_batches = 0
    all_predictions, all_targets = [], []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch_num}/{num_total_epochs} - Training", unit="batch")
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        
        # Skip batch if it contains error indicator target from VCDRDataset
        if (targets == -1.0).any():
            logger.warning(f"Skipping training batch in epoch {epoch_num} due to VCDRDataset data loading error.")
            continue
        num_valid_batches += 1

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0) # Loss per sample
        with torch.no_grad():
            mae_batch = mean_absolute_error(targets.cpu().numpy().flatten(), outputs.cpu().numpy().flatten())
            total_mae += mae_batch * images.size(0) # MAE per sample
        
        all_predictions.extend(outputs.cpu().detach().numpy().flatten())
        all_targets.extend(targets.cpu().detach().numpy().flatten())
        
        # Fixed progress bar postfix - calculate running averages based on processed samples
        total_samples_so_far = len(all_targets)
        pbar.set_postfix({
            'Loss': f'{running_loss / total_samples_so_far:.4f}' if total_samples_so_far > 0 else 'N/A',
            'MAE': f'{total_mae / total_samples_so_far:.4f}' if total_samples_so_far > 0 else 'N/A',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    if num_valid_batches == 0:
        logger.warning(f"Epoch {epoch_num}: No valid batches processed in training.")
        return 0.0, 0.0, 0.0, 0.0

    epoch_loss = running_loss / len(all_targets) # Avg loss over all valid samples
    epoch_mae = total_mae / len(all_targets) # Avg MAE over all valid samples
    
    # Calculate MSE and R2 over all collected valid predictions and targets
    targets_np = np.array(all_targets)
    predictions_np = np.array(all_predictions)

    epoch_mse = mean_squared_error(targets_np, predictions_np)
    epoch_r2 = r2_score(targets_np, predictions_np) if len(np.unique(targets_np)) > 1 else 0.0
    
    return epoch_loss, epoch_mse, epoch_mae, epoch_r2


def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
                   device: torch.device, epoch_num: int, num_total_epochs: int) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Runs a single validation epoch."""
    model.eval()
    running_loss, total_mae = 0.0, 0.0
    num_valid_batches = 0
    all_predictions, all_targets = [], []
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch_num}/{num_total_epochs} - Validation", unit="batch")
    with torch.no_grad():
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            if (targets == -1.0).any():
                logger.warning(f"Skipping validation batch in epoch {epoch_num} due to VCDRDataset data loading error.")
                continue
            num_valid_batches += 1
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            mae_batch = mean_absolute_error(targets.cpu().numpy().flatten(), outputs.cpu().numpy().flatten())
            total_mae += mae_batch * images.size(0)
            
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            
            # Fixed progress bar postfix
            total_samples_so_far = len(all_targets)
            pbar.set_postfix({
                'Loss': f'{running_loss / total_samples_so_far:.4f}' if total_samples_so_far > 0 else 'N/A',
                'MAE': f'{total_mae / total_samples_so_far:.4f}' if total_samples_so_far > 0 else 'N/A',
            })

    if num_valid_batches == 0:
        logger.warning(f"Epoch {epoch_num}: No valid batches processed in validation.")
        return 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([])

    epoch_loss = running_loss / len(all_targets)
    epoch_mae = total_mae / len(all_targets)
    
    targets_np = np.array(all_targets)
    predictions_np = np.array(all_predictions)

    epoch_mse = mean_squared_error(targets_np, predictions_np)
    epoch_r2 = r2_score(targets_np, predictions_np) if len(np.unique(targets_np)) > 1 else 0.0
    
    return epoch_loss, epoch_mse, epoch_mae, epoch_r2, predictions_np, targets_np


def print_training_summary(train_losses: list, val_losses: list, train_maes: list, val_maes: list,
                           train_r2s: list, val_r2s: list, current_epoch_idx: int):
    """Prints a summary of training progress."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING SUMMARY - AFTER EPOCH {current_epoch_idx + 1}")
    logger.info(f"{'='*80}")

    header = f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Train MAE':<12} {'Val MAE':<12} {'Train R²':<10} {'Val R²':<10}"
    
    if len(train_losses) >= 5:
        logger.info("\nLAST 5 EPOCHS PERFORMANCE:")
        logger.info("-" * len(header))
        logger.info(header)
        logger.info("-" * len(header))
        for i in range(max(0, len(train_losses) - 5), len(train_losses)):
            logger.info(f"{i + 1:<6} {train_losses[i]:<12.4f} {val_losses[i]:<12.4f} "
                        f"{train_maes[i]:<12.4f} {val_maes[i]:<12.4f} "
                        f"{train_r2s[i]:<10.4f} {val_r2s[i]:<10.4f}")
    
    logger.info(f"\nCURRENT EPOCH ({current_epoch_idx + 1}) METRICS:")
    logger.info("-" * 40)
    logger.info(f"Training   - Loss (Avg): {train_losses[-1]:.4f}, MAE: {train_maes[-1]:.4f}, R²: {train_r2s[-1]:.4f}")
    logger.info(f"Validation - Loss (Avg): {val_losses[-1]:.4f}, MAE: {val_maes[-1]:.4f}, R²: {val_r2s[-1]:.4f}")
    
    if val_losses:
        best_val_loss_so_far = min(val_losses)
        best_epoch_idx = val_losses.index(best_val_loss_so_far)
        logger.info(f"\nBEST PERFORMANCE SO FAR (Validation):")
        logger.info("-" * 40)
        logger.info(f"Best Epoch: {best_epoch_idx + 1}")
        logger.info(f"Best Val Loss (Avg): {val_losses[best_epoch_idx]:.4f}")
        logger.info(f"Corresponding Val MAE: {val_maes[best_epoch_idx]:.4f}")
        logger.info(f"Corresponding Val R²: {val_r2s[best_epoch_idx]:.4f}")
    logger.info(f"{'='*80}")


# --- Evaluation Functions ---
@torch.no_grad()
def evaluate_model_on_test(model: nn.Module, test_loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates the model on the test set."""
    model.eval()
    all_predictions, all_targets = [], []
    for images, targets in tqdm(test_loader, desc="Testing", unit="batch"):
        images, targets = images.to(device), targets.to(device)
        if (targets == -1.0).any():
            logger.warning("Skipping test batch due to VCDRDataset data loading error.")
            continue
        outputs = model(images)
        all_predictions.extend(outputs.cpu().numpy().flatten())
        all_targets.extend(targets.cpu().numpy().flatten())  # Fixed: added () after cpu
    return np.array(all_predictions), np.array(all_targets)


def analyze_performance_by_range(targets: np.ndarray, predictions: np.ndarray,
                                 ranges: list[tuple[float, float, str]]) -> pd.DataFrame:
    """Analyzes model performance across different VCDR ranges."""
    results = []
    for i, (min_val, max_val, label) in enumerate(ranges):
        # Ensure the last range includes the max_val (e.g., 1.0 for VCDR)
        if i == len(ranges) - 1:
            mask = (targets >= min_val) & (targets <= max_val)
        else:
            mask = (targets >= min_val) & (targets < max_val)
        
        if np.sum(mask) > 0:
            range_targets = targets[mask]
            range_predictions = predictions[mask]
            mse = mean_squared_error(range_targets, range_predictions)
            mae = mean_absolute_error(range_targets, range_predictions)
            # R2 score requires at least two unique target values for a meaningful score
            r2 = r2_score(range_targets, range_predictions) if len(np.unique(range_targets)) > 1 else 0.0
            results.append({
                'Range': label, 'Count': np.sum(mask), 'MSE': mse,
                'MAE': mae, 'R²': r2, 'Mean_Target': np.mean(range_targets),
                'Mean_Prediction': np.mean(range_predictions)
            })
    return pd.DataFrame(results)


# --- Plotting Functions ---
def plot_training_curves(epochs_ran: int, train_losses: list, val_losses: list,
                         train_maes: list, val_maes: list, train_r2s: list, val_r2s: list,
                         val_targets_final: np.ndarray | None, val_preds_final: np.ndarray | None,
                         save_dir: Path, model_name_tag: str):
    """Plots and saves training progress curves."""
    epochs_range = range(1, epochs_ran + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Training Progress: {model_name_tag}', fontsize=16, fontweight='bold')

    # Loss curves (Average Loss per Epoch)
    axes[0, 0].plot(epochs_range, train_losses, 'b-o', label='Train Avg Loss', linewidth=2, markersize=4)
    axes[0, 0].plot(epochs_range, val_losses, 'r-o', label='Val Avg Loss', linewidth=2, markersize=4)
    axes[0, 0].set_title('Average Loss per Epoch'); axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.4)

    # MAE curves
    axes[0, 1].plot(epochs_range, train_maes, 'b-o', label='Train MAE', linewidth=2, markersize=4)
    axes[0, 1].plot(epochs_range, val_maes, 'r-o', label='Val MAE', linewidth=2, markersize=4)
    axes[0, 1].set_title('Mean Absolute Error'); axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.4)

    # R² curves
    axes[1, 0].plot(epochs_range, train_r2s, 'b-o', label='Train R²', linewidth=2, markersize=4)
    axes[1, 0].plot(epochs_range, val_r2s, 'r-o', label='Val R²', linewidth=2, markersize=4)
    axes[1, 0].set_title('R² Score'); axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('R²')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.4)

    # Validation predictions vs actual (from last epoch)
    if val_targets_final is not None and val_preds_final is not None and len(val_targets_final) > 0:
        axes[1, 1].scatter(val_targets_final, val_preds_final, alpha=0.6, s=20, edgecolors='k', linewidths=0.5)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=2, label="Perfect Prediction") # Diagonal line
        axes[1, 1].set_title('Validation: Predicted vs Actual (Last Epoch)')
        axes[1, 1].set_xlabel('Actual VCDR'); axes[1, 1].set_ylabel('Predicted VCDR')
        axes[1, 1].set_xlim(0, 1); axes[1, 1].set_ylim(0, 1)
        val_r2_final = r2_score(val_targets_final, val_preds_final) if len(np.unique(val_targets_final)) > 1 else 0.0
        axes[1, 1].text(0.05, 0.95, f'R² = {val_r2_final:.3f}', transform=axes[1, 1].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.4)
    else:
        axes[1, 1].text(0.5, 0.5, "No validation data for scatter plot", ha='center', va='center', fontsize=10)
        axes[1, 1].set_title('Validation: Predicted vs Actual'); axes[1, 1].grid(True, alpha=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    save_path = save_dir / f"{model_name_tag}_training_curves.png"
    plt.savefig(save_path, dpi=150)
    logger.info(f"Training curves saved to {save_path}")
    plt.close(fig)


def plot_test_evaluation(test_targets: np.ndarray, test_predictions: np.ndarray,
                         test_r2: float, test_mae: float, save_dir: Path, model_name_tag: str):
    """Plots and saves test set evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Test Set Evaluation: {model_name_tag}', fontsize=16, fontweight='bold')

    # Scatter plot: Predicted vs Actual
    axes[0, 0].scatter(test_targets, test_predictions, alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual VCDR'); axes[0, 0].set_ylabel('Predicted VCDR')
    axes[0, 0].set_title('Predicted vs Actual VCDR'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].set_xlim(0, 1); axes[0, 0].set_ylim(0, 1)
    axes[0, 0].text(0.05, 0.95, f'R² = {test_r2:.3f}\nMAE = {test_mae:.3f}',
                    transform=axes[0, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # Residuals plot
    residuals = test_predictions - test_targets
    axes[0, 1].scatter(test_targets, residuals, alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Actual VCDR'); axes[0, 1].set_ylabel('Residuals (Predicted - Actual)')
    axes[0, 1].set_title('Residuals Plot'); axes[0, 1].grid(True, alpha=0.4)

    # Distribution of predictions vs actuals
    axes[1, 0].hist(test_targets, bins=30, alpha=0.7, label='Actual', color='royalblue', edgecolor='black')
    axes[1, 0].hist(test_predictions, bins=30, alpha=0.7, label='Predicted', color='orangered', edgecolor='black')
    axes[1, 0].set_xlabel('VCDR Score'); axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution Comparison'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.4)

    # Error distribution
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black')
    axes[1, 1].set_xlabel('Prediction Error (Residual)'); axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution'); axes[1, 1].axvline(x=0, color='k', linestyle='--', linewidth=2)
    axes[1, 1].grid(True, alpha=0.4)
    axes[1, 1].text(0.05, 0.95, f'Mean Error: {np.mean(residuals):.3f}\nStd Error: {np.std(residuals):.3f}',
                    transform=axes[1, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = save_dir / f"{model_name_tag}_test_evaluation.png"
    plt.savefig(save_path, dpi=150)
    logger.info(f"Test evaluation plots saved to {save_path}")
    plt.close(fig)


def plot_performance_by_range(performance_df: pd.DataFrame, save_dir: Path, model_name_tag: str):
    """Plots and saves performance metrics by VCDR range."""
    if performance_df.empty:
        logger.warning(f"Performance by range dataframe is empty for {model_name_tag}. Skipping plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # Wider for readability
    fig.suptitle(f'Model Performance by VCDR Range on Test Set: {model_name_tag}', fontsize=14, fontweight='bold')
    
    ranges_labels = performance_df['Range']
    x_pos = np.arange(len(ranges_labels))

    # MAE by range
    axes[0].bar(x_pos, performance_df['MAE'], alpha=0.8, color='coral', edgecolor='black')
    axes[0].set_title('Mean Absolute Error by Range'); axes[0].set_ylabel('MAE')
    axes[0].set_xticks(x_pos); axes[0].set_xticklabels(ranges_labels, rotation=45, ha="right")
    axes[0].grid(True, linestyle='--', alpha=0.6); axes[0].set_xlabel('VCDR Range')

    # R² by range
    axes[1].bar(x_pos, performance_df['R²'], alpha=0.8, color='skyblue', edgecolor='black')
    axes[1].set_title('R² Score by Range'); axes[1].set_ylabel('R² Score')
    axes[1].set_xticks(x_pos); axes[1].set_xticklabels(ranges_labels, rotation=45, ha="right")
    axes[1].grid(True, linestyle='--', alpha=0.6); axes[1].set_xlabel('VCDR Range')
    
    # Sample count by range
    ax2_count = axes[2].twinx() # Create a twin y-axis if needed for MAE vs Count, or just use one
    axes[2].bar(x_pos, performance_df['Count'], alpha=0.8, color='mediumpurple', edgecolor='black')
    axes[2].set_title('Sample Count by Range'); axes[2].set_ylabel('Number of Samples')
    axes[2].set_xticks(x_pos); axes[2].set_xticklabels(ranges_labels, rotation=45, ha="right")
    axes[2].grid(True, linestyle='--', alpha=0.6); axes[2].set_xlabel('VCDR Range')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_path = save_dir / f"{model_name_tag}_performance_by_range.png"
    plt.savefig(save_path, dpi=150)
    logger.info(f"Performance by range plots saved to {save_path}")
    plt.close(fig)


# --- Experiment Orchestration ---
def run_experiment(args: argparse.Namespace) -> dict:
    """
    Runs a complete training and evaluation experiment for a given configuration.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_clean = args.model_name.replace('/', '_') # For file naming
    experiment_id = f"{args.experiment_tag}_{model_name_clean}_seed{args.seed}"
    
    logger.info(f"===== Starting Experiment: {experiment_id} =====")
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True # May impact performance slightly
        torch.backends.cudnn.benchmark = False

    # Create output directory for this specific experiment run
    current_experiment_dir = Path(args.base_output_dir) / experiment_id
    current_experiment_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory for this experiment: {current_experiment_dir}")
    
    # Setup file handler for logging to a file within the experiment directory
    # Remove existing file handlers from root logger to avoid multiple file logs from previous runs
    for handler in logger.handlers[:]: 
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
            
    file_handler = logging.FileHandler(current_experiment_dir / f'training_log.txt', mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"Logging to: {current_experiment_dir / 'training_log.txt'}")
    logger.info(f"Run arguments: {vars(args)}")


    # 1. Load and preprocess data
    # Define path for processed data CSV based on the raw data dir name
    raw_data_dir_name = Path(args.grisk_data_dir).name
    processed_data_file = Path(args.base_data_root) / "processed" / f"{raw_data_dir_name}_dataset.csv"
    
    if args.use_processed_data and processed_data_file.exists():
        logger.info(f"Loading pre-processed data from {processed_data_file}")
        df = pd.read_csv(processed_data_file)
    else:
        if args.use_processed_data: # Tried to use it but not found
            logger.warning(f"Flag --use_processed_data was set, but pre-processed data not found at {processed_data_file}. Processing from raw.")
        df = load_and_preprocess_data(args.grisk_data_dir, str(processed_data_file))

    # 2. Split data and create DataLoaders
    train_df, val_df, test_df = split_data_by_patient(df, random_state=args.seed,
                                                      train_ratio=args.train_ratio,
                                                      val_ratio=args.val_ratio,
                                                      test_ratio=args.test_ratio)
    train_transform, val_test_transform = get_transforms()
    train_dataset = VCDRDataset(train_df, transform=train_transform)
    val_dataset = VCDRDataset(val_df, transform=val_test_transform)
    test_dataset = VCDRDataset(test_df, transform=val_test_transform)
    
    # DataLoader num_workers: 0 for Windows, can be >0 for Linux
    num_workers = args.num_workers if os.name != 'nt' else 0
    if os.name == 'nt' and args.num_workers > 0:
        logger.warning(f"On Windows, num_workers > 0 can cause issues. Setting to 0 from {args.num_workers}.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    logger.info(f"DataLoader sizes: Train={len(train_loader)} ({len(train_dataset)} imgs), Val={len(val_loader)} ({len(val_dataset)} imgs), Test={len(test_loader)} ({len(test_dataset)} imgs)")

    # 3. Initialize model, optimizer, criterion, scheduler
    model = build_regressor_model(
        model_name=args.model_name,
        num_outputs=1, # For VCDR regression
        head_dropout_prob=args.head_dropout_prob,
        use_timm_hub_pretrained=args.use_timm_pretrained_if_no_custom,
        custom_weights_path=args.custom_weights_path,
        checkpoint_key=args.checkpoint_key
    ).to(device)
    
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) # AdamW is often preferred
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=args.lr_scheduler_patience, verbose=True, min_lr=1e-7)
    logger.info(f"Model: {args.model_name}, Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"Optimizer: AdamW, LR: {args.learning_rate}, Weight Decay: {args.weight_decay}")
    logger.info(f"Scheduler: ReduceLROnPlateau, Factor: 0.2, Patience: {args.lr_scheduler_patience}")


    # 4. Training loop
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'train_r2': [], 'val_r2': []}
    best_val_loss = float('inf')
    best_model_state = None
    epochs_completed = 0
    val_preds_for_plot, val_targets_for_plot = None, None # For plotting last epoch's validation

    training_start_time = time.time()
    logger.info(f"\n🚀 Starting training for {args.model_name} (Experiment: {experiment_id})...")

    for epoch in range(args.num_epochs):
        epochs_completed += 1
        epoch_start_time = time.time()
        
        logger.info(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")
        
        # Train
        avg_train_loss, _, train_mae, train_r2 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1, args.num_epochs
        )
        # Validate
        avg_val_loss, _, val_mae, val_r2, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device, epoch + 1, args.num_epochs
        )
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        val_preds_for_plot, val_targets_for_plot = val_preds, val_targets
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss) # Scheduler steps on validation loss
        
        improvement_tag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy() # Save best model state
            improvement_tag = f"🌟 New best validation loss: {best_val_loss:.4f}!"
            logger.info(improvement_tag)
        
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1} Summary ({epoch_duration:.1f}s):")
        logger.info(f"  Train -> Avg Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        logger.info(f"  Valid -> Avg Loss: {avg_val_loss:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        if optimizer.param_groups[0]['lr'] < current_lr:
            logger.info(f"  Learning rate reduced to {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping check (based on validation loss not improving)
        if epoch >= args.early_stopping_patience: # Start checking after min epochs
            # Check if val_loss has not improved for `early_stopping_patience` epochs
            # from the point where best_val_loss was last updated
            if len(history['val_loss']) > args.early_stopping_patience:
                # Find epoch index of best_val_loss
                best_val_loss_epoch_idx = history['val_loss'].index(min(history['val_loss']))
                if (epoch - best_val_loss_epoch_idx) >= args.early_stopping_patience:
                    logger.info(f"\n⏹️ Early stopping triggered after {epoch + 1} epochs. "
                                f"Validation loss has not improved for {args.early_stopping_patience} epochs "
                                f"since best loss at epoch {best_val_loss_epoch_idx + 1}.")
                    break
    
    if best_model_state:
        model.load_state_dict(best_model_state) # Load the best model state for evaluation
        logger.info("Loaded best model state for final evaluation.")
    else:
        logger.warning("Best model state was not captured (e.g., training stopped early or no improvement). Using last epoch's model.")

    total_training_time = time.time() - training_start_time
    logger.info(f"\n{'='*40}\n🎉 TRAINING COMPLETED for {args.model_name}! 🎉\n{'='*40}")
    logger.info(f"Total training time: {total_training_time / 60:.1f} minutes.")
    if history['val_loss']:
        logger.info(f"🏆 Best validation loss: {min(history['val_loss']):.4f} at epoch {history['val_loss'].index(min(history['val_loss'])) + 1}")
    logger.info(f"📊 Total epochs run: {epochs_completed}")
    if epochs_completed > 0:
        print_training_summary(history['train_loss'], history['val_loss'], history['train_mae'], 
                               history['val_mae'], history['train_r2'], history['val_r2'], epochs_completed - 1)


    # 5. Plot training curves
    if epochs_completed > 0 and history['val_loss']:
        plot_training_curves(epochs_completed, history['train_loss'], history['val_loss'],
                             history['train_mae'], history['val_mae'], history['train_r2'], history['val_r2'],
                             val_targets_for_plot, val_preds_for_plot,
                             current_experiment_dir, model_name_clean)

    # 6. Evaluate on test set
    logger.info("\n🧪 Evaluating on test set...")
    test_predictions, test_targets = evaluate_model_on_test(model, test_loader, device)
    
    test_mse, test_rmse, test_mae, test_r2 = 0.0, 0.0, 0.0, 0.0 # Initialize
    if len(test_targets) > 0:
        test_mse = mean_squared_error(test_targets, test_predictions)
        test_mae = mean_absolute_error(test_targets, test_predictions)
        test_r2 = r2_score(test_targets, test_predictions) if len(np.unique(test_targets)) > 1 else 0.0
        test_rmse = np.sqrt(test_mse)

        logger.info("\n=== FINAL TEST SET RESULTS ===")
        logger.info(f"Test MSE: {test_mse:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.4f}")
        logger.info(f"Test MAE: {test_mae:.4f}")
        logger.info(f"Test R²: {test_r2:.4f}")
        plot_test_evaluation(test_targets, test_predictions, test_r2, test_mae,
                             current_experiment_dir, model_name_clean)

        # 7. Performance analysis by VCDR ranges
        vcdr_analysis_ranges = [
            (0.0, 0.2, "VL (0-0.2)"), (0.2, 0.4, "L (0.2-0.4)"),
            (0.4, 0.6, "M (0.4-0.6)"), (0.6, 0.8, "H (0.6-0.8)"),
            (0.8, 1.0, "VH (0.8-1)") # Ensure last range includes 1.0
        ]
        performance_df = analyze_performance_by_range(test_targets, test_predictions, vcdr_analysis_ranges)
        logger.info("\n=== PERFORMANCE BY VCDR RANGE (Test Set) ===")
        logger.info("\n" + performance_df.round(4).to_string())
        plot_performance_by_range(performance_df, current_experiment_dir, model_name_clean)
    else:
        logger.warning("No test data was processed (e.g., all test batches had errors). Skipping test evaluation plots and analysis.")

    # 8. Save the best model
    model_save_path = current_experiment_dir / f"{model_name_clean}_best_regressor.pth"
    if best_model_state:
        torch.save({
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(), # Save optimizer state for potential resuming
            'epoch_completed': epochs_completed,
            'best_val_loss': best_val_loss,
            'final_test_r2': test_r2, 'final_test_mae': test_mae, 'final_test_rmse': test_rmse,
            'model_architecture': args.model_name,
            'training_args': vars(args) # Save training arguments
        }, model_save_path)
        logger.info(f"Best model state saved to: {model_save_path}")
    else:
        logger.warning(f"No best model state to save for {model_name_clean}.")

    # Clean up file handler for this specific experiment log
    logger.removeHandler(file_handler)
    file_handler.close()
    
    logger.info(f"===== Finished Experiment: {experiment_id} =====")

    return {
        "model_name": args.model_name, "experiment_tag": args.experiment_tag, "seed": args.seed,
        "best_val_loss": best_val_loss if history['val_loss'] else float('inf'),
        "final_val_loss": history['val_loss'][-1] if history['val_loss'] else float('inf'),
        "final_val_mae": history['val_mae'][-1] if history['val_mae'] else float('inf'),
        "final_val_r2": history['val_r2'][-1] if history['val_r2'] else float('-inf'),
        "test_r2": test_r2, "test_mae": test_mae, "test_rmse": test_rmse,
        "epochs_ran": epochs_completed,
        "model_path": str(model_save_path) if best_model_state else "N/A",
        "log_path": str(current_experiment_dir / 'training_log.txt')
    }


def main():
    parser = argparse.ArgumentParser(description="VCDR Regression Model Training Script (Multi-Architecture)")
    
    # --- Data Arguments ---
    data_group = parser.add_argument_group('Data Arguments')
    data_group.add_argument('--grisk_data_dir', type=str, default=r"D:\glaucoma\data\raw\griskFundus",
                            help="Path to G-RISK raw fundus image directory.")
    data_group.add_argument('--base_data_root', type=str, default=r"D:\glaucoma\data",
                            help="Base root for data (used to construct processed data path).")
    data_group.add_argument('--use_processed_data', action='store_true',
                            help="Load data from pre-processed CSV if available, otherwise process from raw.")
    data_group.add_argument('--train_ratio', type=float, default=0.7, help="Proportion of patients for training.")
    data_group.add_argument('--val_ratio', type=float, default=0.15, help="Proportion of patients for validation.")
    data_group.add_argument('--test_ratio', type=float, default=0.15, help="Proportion of patients for testing.")

    # --- Model Arguments ---
    model_group = parser.add_argument_group('Model Arguments')
    model_group.add_argument('--model_name', type=str, required=True,
                             help="Name of the model architecture (e.g., resnet50, dinov2_vitb14, convnext_base).")
    model_group.add_argument('--custom_weights_path', type=str, default=None,
                             help="Path to custom pretrained backbone weights (.pth or .pt file).")
    model_group.add_argument('--checkpoint_key', type=str, default='teacher',
                             help="Key in the checkpoint dict for model state_dict (e.g., 'model', 'state_dict', 'teacher').")
    model_group.add_argument('--use_timm_pretrained_if_no_custom', action=argparse.BooleanOptionalAction, default=True,
                             help="Use TIMM/Hub default pretrained weights if no custom_weights_path is provided.")
    model_group.add_argument('--head_dropout_prob', type=float, default=0.3,
                             help="Dropout probability for the first layer of the custom regression head.")

    # --- Training Hyperparameters ---
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs.")
    train_group.add_argument('--batch_size', type=int, default=16, help="Training batch size.")
    train_group.add_argument('--eval_batch_size', type=int, default=32, help="Evaluation batch size.")
    train_group.add_argument('--learning_rate', type=float, default=1e-5, help="Initial learning rate for AdamW.")
    train_group.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for AdamW.") # Common for ViTs
    train_group.add_argument('--early_stopping_patience', type=int, default=5,
                             help="Patience (epochs) for early stopping based on validation loss.")
    train_group.add_argument('--lr_scheduler_patience', type=int, default=5,
                             help="Patience (epochs) for ReduceLROnPlateau scheduler.")
    
    # --- System Arguments ---
    sys_group = parser.add_argument_group('System Arguments')
    sys_group.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    sys_group.add_argument('--num_workers', type=int, default=0,
                           help="Number of workers for DataLoader. Set to 0 for Windows debugging if issues arise.")
    
    # --- Output Arguments ---
    out_group = parser.add_argument_group('Output Arguments')
    out_group.add_argument('--base_output_dir', type=str, default="experiments_vcdr_multiarch",
                           help="Base directory to save all experiment outputs.")
    out_group.add_argument('--experiment_tag', type=str, required=True,
                           help="A descriptive tag for this specific experiment configuration (e.g., ResNet50_timm, DINOv2_custom).")

    args = parser.parse_args()
    
    # Your PowerShell script will iterate through configurations and call this Python script.
    # This script itself runs ONE experiment based on the passed arguments.
    
    all_results = [] # In case this script is later modified to run multiple experiments internally
    try:
        result = run_experiment(args)
        all_results.append(result)
    except Exception as e:
        logger.error(f"Experiment {args.experiment_tag} with model {args.model_name} (seed {args.seed}) failed: {e}", exc_info=True)
        # Ensure any file handlers are closed if an error occurs mid-experiment
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()

    logger.info("\n\n=== OVERALL SCRIPT EXECUTION SUMMARY ===")
    if all_results:
        summary_df = pd.DataFrame(all_results)
        logger.info("\n" + summary_df.to_string())
        
        # Create a unique summary file for this specific run, or append to a global one
        # For a single run script, this will just be one line.
        summary_csv_path = Path(args.base_output_dir) / f"experiment_summary_{args.experiment_tag}_seed{args.seed}.csv"
        try:
            summary_df.to_csv(summary_csv_path, index=False)
            logger.info(f"Experiment run summary saved to {summary_csv_path}")
        except Exception as e_csv:
            logger.error(f"Could not save experiment run summary CSV: {e_csv}")
    else:
        logger.info("No experiments were completed successfully in this run.")


if __name__ == "__main__":
    main()