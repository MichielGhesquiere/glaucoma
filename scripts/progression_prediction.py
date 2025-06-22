# %% Import necessary libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
from pathlib import Path
import re
from typing import Dict, Any, Optional, List, Tuple

# Scikit-learn and Imblearn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression 
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve, auc
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import seaborn as sns
import traceback
import pickle 
from tqdm import tqdm 
from sklearn.base import clone 

# PyTorch & TIMM
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader, WeightedRandomSampler
from torchvision import models 
import torchvision.transforms as T 
import timm 
from timm.data import create_transform, resolve_data_config 
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Cache paths
FEATURES_CACHE_PATH = Path("D:/glaucoma/data/processed") 
VIT_EMBEDDING_CACHE_FILE = FEATURES_CACHE_PATH / "vit_base_embeddings.pkl" 
VIT_PREDICTION_CACHE_FILE = FEATURES_CACHE_PATH / "vit_base_predictions.pkl" 
FEATURES_CACHE_PATH.mkdir(parents=True, exist_ok=True)

# %% --- ViT Embedding Extractor (Corrected Loading) ---
class ViTEmbeddingExtractor(nn.Module): 
    def __init__(self, model_name_timm: str, pretrained_vit_path: Optional[str] = None):
        super().__init__()
        self.model_name_timm = model_name_timm
        # Create model structure first (num_classes=0 for embeddings)
        self.vit = timm.create_model(model_name_timm, pretrained=False, num_classes=0) 

        if pretrained_vit_path and Path(pretrained_vit_path).exists():
            try:
                # Load the entire checkpoint dictionary
                checkpoint = torch.load(pretrained_vit_path, map_location=DEVICE, weights_only=False) # Set weights_only=True if safe
                
                # *** CORRECTED: Extract state_dict from the checkpoint ***
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print("Extracted 'model_state_dict' from checkpoint file.")
                else:
                    # Assume the file *is* the state_dict if not a dict with the key
                    state_dict = checkpoint 
                    print("Checkpoint file does not contain 'model_state_dict' key, assuming it's the state_dict itself.")

                # Adjust for 'module.' prefix if saved with DataParallel/DDP
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
                
                # Load into the num_classes=0 model, ignoring head keys
                missing_keys, unexpected_keys = self.vit.load_state_dict(state_dict, strict=False)
                unexpected_keys = [k for k in unexpected_keys if not k.startswith('head.')] # Ignore actual head keys

                if missing_keys: print(f"ViT Embedding Extractor Warning: Missing keys: {missing_keys}")
                if unexpected_keys: print(f"ViT Embedding Extractor Warning: Unexpected (non-head) keys: {unexpected_keys}")
                print(f"Loaded custom pre-trained ViT ({model_name_timm}) weights for embedding extraction from: {pretrained_vit_path}")

            except Exception as e:
                print(f"Error loading custom ViT state_dict for embeddings: {e}. Using TIMM default.")
                self.vit = timm.create_model(model_name_timm, pretrained=True, num_classes=0) # Fallback
        else:
            print(f"No custom ViT path/file invalid. Loading ImageNet pre-trained {model_name_timm} (num_classes=0).")
            self.vit = timm.create_model(model_name_timm, pretrained=True, num_classes=0) 
        self.vit.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.vit(x) 
        return features

# %% --- ViT Prediction Extractor (Corrected Loading) ---
class ViTPredictionExtractor(nn.Module):
    def __init__(self, model_name_timm: str, pretrained_vit_path: str, num_original_classes: int = 1):
        super().__init__()
        # Create model structure with the original number of classes
        self.vit = timm.create_model(model_name_timm, pretrained=False, num_classes=num_original_classes)
        
        try:
            # Load the entire checkpoint dictionary
            checkpoint = torch.load(pretrained_vit_path, map_location=DEVICE, weights_only=False) # Set weights_only=True if safe
            
            # *** CORRECTED: Extract state_dict from the checkpoint ***
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Extracted 'model_state_dict' from checkpoint file for prediction extractor.")
            else:
                state_dict = checkpoint 
                print("Checkpoint file does not contain 'model_state_dict' key, assuming it's the state_dict itself for prediction extractor.")

            # Adjust for 'module.' prefix
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            
            # Load state dict strictly now, should match including the head
            self.vit.load_state_dict(state_dict, strict=True)
            print(f"Loaded custom pre-trained ViT ({model_name_timm}) with head for predictions from: {pretrained_vit_path}")
        
        except Exception as e:
            # Raise a more informative error if loading fails here, as predictions are crucial
            raise RuntimeError(f"CRITICAL Error loading ViT state_dict for predictions: {e}. Check path, num_original_classes, and checkpoint structure ('model_state_dict' key?).") from e
            
        self.vit.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.vit(x)
            if logits.shape[-1] == 1: 
                probs = torch.sigmoid(logits)
            elif logits.shape[-1] > 1 and logits.shape[-1] == self.vit.num_classes : 
                probs = torch.softmax(logits, dim=-1)
                if probs.shape[-1] == 2: # Binary task with 2 outputs, assume class 1 is positive
                    probs = probs[..., 1] 
                else: 
                    print(f"Warning: ViT prediction model has {probs.shape[-1]} output classes. Returning prob of class 1.")
                    probs = probs[..., 1] # Default to class 1 if > 2 classes - ADJUST AS NEEDED
            else: 
                print(f"Warning: ViT prediction logits shape {logits.shape} not handled as expected. Returning raw logits.")
                return logits.squeeze()
            return probs.squeeze()

def get_vit_image_transforms(
    model_name_timm: str, image_size: Optional[int] = None
) -> T.Compose:
    """Gets the appropriate torchvision transforms for a given ViT model."""
    dummy_model_for_config = timm.create_model(model_name_timm, pretrained=False)
    config = resolve_data_config({}, model=dummy_model_for_config)

    if image_size is not None:
        config['input_size'] = (3, image_size, image_size)

    transform = create_transform(
        input_size=config['input_size'],
        is_training=False,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
    )
    print(f"ViT transforms for {model_name_timm}: Input size {config['input_size']}")
    return transform


def extract_embeddings_for_image_vit(
    image_path: str,
    model: 'ViTEmbeddingExtractor', # Use quotes if class not defined yet
    transform: T.Compose,
    device: torch.device,
) -> Optional[np.ndarray]:
    """Extracts ViT embeddings for a single image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad(): # Important for inference
            features = model(img_tensor)
        return features.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path} for ViT embedding: {e}")
        return None


def extract_and_cache_all_embeddings_vit(
    df: pd.DataFrame,
    embedding_extractor: 'ViTEmbeddingExtractor',
    transform: T.Compose,
    device: torch.device,
    cache_file: Path, # Removed default to make it explicit, e.g. VIT_EMBEDDING_CACHE_FILE
    force_reextract: bool = False,
) -> pd.DataFrame:
    """Extracts and caches ViT embeddings for all images in the DataFrame."""
    embeddings_dict = {}
    if not force_reextract and cache_file.exists():
        print(f"Loading cached ViT embeddings from {cache_file}...")
        try:
            with open(cache_file, 'rb') as f:
                embeddings_dict = pickle.load(f)
            print(f"Successfully loaded {len(embeddings_dict)} cached ViT embeddings.")
        except Exception as e:
            print(f"Error loading ViT embedding cache: {e}. Re-extracting.")
            embeddings_dict = {}

    df['Image Path'] = df['Image Path'].astype(str)
    all_image_paths = df['Image Path'].unique()
    paths_to_extract = [p for p in all_image_paths if p not in embeddings_dict]

    if paths_to_extract:
        print(f"Extracting ViT embeddings for {len(paths_to_extract)} new/missing images...")
        for image_path_str in tqdm(paths_to_extract, desc="Extracting ViT Embeddings"):
            embedding = extract_embeddings_for_image_vit(
                image_path_str, embedding_extractor, transform, device
            )
            if embedding is not None:
                embeddings_dict[image_path_str] = embedding

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings_dict, f)
            print(f"Saved/Updated ViT embedding cache to {cache_file}")
        except Exception as e:
            print(f"Error saving ViT embedding cache: {e}")
    else:
        print("All ViT embeddings found in cache or no images to process.")

    df['vit_features'] = df['Image Path'].map(embeddings_dict)
    missing_embeddings_count = df['vit_features'].isnull().sum()
    if missing_embeddings_count > 0:
        print(f"Warning: {missing_embeddings_count} images lack ViT embeddings after processing.")
    return df

def extract_predictions_for_image_vit(
    image_path: str,
    model: 'ViTPredictionExtractor',
    transform: T.Compose,
    device: torch.device,
) -> Optional[float]:
    """Extracts ViT predictions (e.g., probability) for a single image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad(): # Important for inference
            prediction = model(img_tensor)

        if torch.is_tensor(prediction) and prediction.numel() == 1:
            return prediction.item()
        return prediction # Assuming it might be a raw float already
    except Exception as e:
        print(f"Error processing image {image_path} for ViT prediction: {e}")
        return None


def extract_and_cache_all_predictions_vit(
    df: pd.DataFrame,
    prediction_extractor: 'ViTPredictionExtractor',
    transform: T.Compose,
    device: torch.device,
    cache_file: Path, # Removed default, e.g. VIT_PREDICTION_CACHE_FILE
    force_reextract: bool = False,
) -> pd.DataFrame:
    """Extracts and caches ViT predictions for all images in the DataFrame."""
    predictions_dict = {}
    if not force_reextract and cache_file.exists():
        print(f"Loading cached ViT predictions from {cache_file}...")
        try:
            with open(cache_file, 'rb') as f:
                predictions_dict = pickle.load(f)
            print(f"Successfully loaded {len(predictions_dict)} cached ViT predictions.")
        except Exception as e:
            print(f"Error loading ViT prediction cache: {e}. Re-extracting.")
            predictions_dict = {}

    df['Image Path'] = df['Image Path'].astype(str)
    all_image_paths = df['Image Path'].unique()
    paths_to_extract = [p for p in all_image_paths if p not in predictions_dict]

    if paths_to_extract:
        print(f"Extracting ViT predictions for {len(paths_to_extract)} new/missing images...")
        for image_path_str in tqdm(paths_to_extract, desc="Extracting ViT Predictions"):
            prediction = extract_predictions_for_image_vit(
                image_path_str, prediction_extractor, transform, device
            )
            if prediction is not None:
                predictions_dict[image_path_str] = prediction
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(predictions_dict, f)
            print(f"Saved/Updated ViT prediction cache to {cache_file}")
        except Exception as e:
            print(f"Error saving ViT prediction cache: {e}")
    else:
        print("All ViT predictions found in cache or no images to process.")

    df['vit_prediction'] = df['Image Path'].map(predictions_dict)
    missing_predictions_count = df['vit_prediction'].isnull().sum()
    if missing_predictions_count > 0:
        print(f"Warning: {missing_predictions_count} images lack ViT predictions after processing.")
    return df


# %% --- GRAPE Data Loader ---

class GRAPEDataLoader:
    """Loads and preprocesses data from the GRAPE dataset structure."""
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.metadata_path = self.data_root / "VF and clinical information.xlsx"
        self.images_path = self.data_root / "CFPs" / "CFPs"

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")

    def load_metadata(self) -> pd.DataFrame:
        """Loads and cleans the metadata Excel file."""
        df = pd.read_excel(self.metadata_path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # Remove unnamed columns
        df = df.dropna(subset=['Subject Number'])
        df['Subject Number'] = df['Subject Number'].astype(int)

        if 'Progression Status' in df.columns:
            mapping = {'Non-Progressor': 0, 'Progressor': 1, 0: 0, 1: 1, '0': 0, '1': 1}
            df['Progression Status'] = df['Progression Status'].map(mapping)
            df.dropna(subset=['Progression Status'], inplace=True)
            df['Progression Status'] = df['Progression Status'].astype(int)
        else:
            print("Warning: 'Progression Status' column not found in metadata.")
        return df.reset_index(drop=True)

    def get_image_paths(self) -> pd.DataFrame:
        """Scans the image directory and extracts information from filenames."""
        image_data = []
        # Regex to capture Subject_Number, Laterality (OD/OS), Visit_Number
        pattern = re.compile(r'(\d+)_(OD|OS)_(\d+)\.jpg', re.IGNORECASE)

        for image_path_obj in self.images_path.glob('*.jpg'):
            match = pattern.match(image_path_obj.name)
            if match:
                image_data.append({
                    'Subject Number': int(match.group(1)),
                    'Laterality': match.group(2).upper(), # Standardize laterality
                    'Visit Number': int(match.group(3)),
                    'Image Path': str(image_path_obj)
                })
        if not image_data:
            print(f"Warning: No images found in {self.images_path} matching pattern.")
        return pd.DataFrame(image_data)

    def merge_data(self) -> pd.DataFrame:
        """Merges metadata with image paths."""
        metadata_df = self.load_metadata()
        image_df = self.get_image_paths()

        if image_df.empty:
            print("Error: No image data to merge. Returning empty DataFrame.")
            return pd.DataFrame()

        merged_df = pd.merge(image_df, metadata_df, on=['Subject Number', 'Laterality'], how='left')

        if 'Progression Status' in merged_df.columns:
            merged_df.dropna(subset=['Progression Status'], inplace=True)
            # Ensure it's int after potential NaNs introduced by left merge
            merged_df['Progression Status'] = merged_df['Progression Status'].astype(int)
        else:
            print("Warning: 'Progression Status' not in merged data after merge.")

        merged_df.sort_values(['Subject Number', 'Laterality', 'Visit Number'], inplace=True)
        merged_df.reset_index(drop=True, inplace=True)
        return merged_df

    def get_progression_stats(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculates progression status statistics."""
        if 'Progression Status' not in df.columns or df.empty or df['Progression Status'].isnull().all():
            return {
                'by_laterality': pd.Series(dtype='float64'),
                'overall': pd.Series(dtype='float64')
            }

        # Consider each subject-eye once for progression status
        unique_subject_eye_progression = df.drop_duplicates(subset=['Subject Number', 'Laterality'])
        
        label_map = {0: 'Non-Progressor', 1: 'Progressor'}

        by_laterality_stats = (
            unique_subject_eye_progression.groupby('Laterality')['Progression Status']
            .value_counts(normalize=True)
            .rename(index=label_map, level=1) # Rename the 'Progression Status' level
        )
        overall_stats = (
            unique_subject_eye_progression['Progression Status']
            .value_counts(normalize=True)
            .rename(index=label_map)
        )
        return {'by_laterality': by_laterality_stats, 'overall': overall_stats}

    def plot_progression_distribution(self, df: pd.DataFrame) -> None:
        """Plots the distribution of progression status by laterality."""
        if ('Progression Status' not in df.columns or
            df.empty or
            df['Progression Status'].isnull().all()):
            print("Warning: Cannot plot progression distribution due to missing or empty 'Progression Status'.")
            return

        plt.figure(figsize=(10, 5))
        # Ensure we count each subject-eye once
        plot_df = df.drop_duplicates(subset=['Subject Number', 'Laterality']).copy()
        plot_df['Progression Status Label'] = plot_df['Progression Status'].map({
            0: 'Non-Progressor', 1: 'Progressor'
        })

        progression_dist = (
            plot_df.groupby(['Laterality'])['Progression Status Label']
            .value_counts(normalize=True)
            .unstack(fill_value=0) # Fill with 0 if a category is missing
        )

        if progression_dist.empty:
            print("Warning: No data to plot for progression distribution after processing.")
            return

        ax = progression_dist.plot(kind='bar', width=0.8, figsize=(10, 5))
        plt.title('Distribution of Progression Status by Laterality (Unique Subject-Eyes)')
        plt.xlabel('Laterality')
        plt.ylabel('Proportion of Subject-Eyes')
        plt.legend(title='Progression Status', loc='upper right')
        plt.xticks(rotation=0)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
        
        plt.tight_layout()
        plt.show()

class GRAPEDataLoader:
    """Loads and preprocesses data from the GRAPE dataset structure."""
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.metadata_path = self.data_root / "VF and clinical information.xlsx"
        self.images_path = self.data_root / "CFPs" / "CFPs"

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")

    def load_metadata(self) -> pd.DataFrame:
        """Loads and cleans the metadata Excel file."""
        df = pd.read_excel(self.metadata_path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # Remove unnamed columns
        df = df.dropna(subset=['Subject Number'])
        df['Subject Number'] = df['Subject Number'].astype(int)

        if 'Progression Status' in df.columns:
            mapping = {'Non-Progressor': 0, 'Progressor': 1, 0: 0, 1: 1, '0': 0, '1': 1}
            df['Progression Status'] = df['Progression Status'].map(mapping)
            df.dropna(subset=['Progression Status'], inplace=True)
            df['Progression Status'] = df['Progression Status'].astype(int)
        else:
            print("Warning: 'Progression Status' column not found in metadata.")
        return df.reset_index(drop=True)

    def get_image_paths(self) -> pd.DataFrame:
        """Scans the image directory and extracts information from filenames."""
        image_data = []
        # Regex to capture Subject_Number, Laterality (OD/OS), Visit_Number
        pattern = re.compile(r'(\d+)_(OD|OS)_(\d+)\.jpg', re.IGNORECASE)

        for image_path_obj in self.images_path.glob('*.jpg'):
            match = pattern.match(image_path_obj.name)
            if match:
                image_data.append({
                    'Subject Number': int(match.group(1)),
                    'Laterality': match.group(2).upper(), # Standardize laterality
                    'Visit Number': int(match.group(3)),
                    'Image Path': str(image_path_obj)
                })
        if not image_data:
            print(f"Warning: No images found in {self.images_path} matching pattern.")
        return pd.DataFrame(image_data)

    def merge_data(self) -> pd.DataFrame:
        """Merges metadata with image paths."""
        metadata_df = self.load_metadata()
        image_df = self.get_image_paths()

        if image_df.empty:
            print("Error: No image data to merge. Returning empty DataFrame.")
            return pd.DataFrame()

        merged_df = pd.merge(image_df, metadata_df, on=['Subject Number', 'Laterality'], how='left')

        if 'Progression Status' in merged_df.columns:
            merged_df.dropna(subset=['Progression Status'], inplace=True)
            # Ensure it's int after potential NaNs introduced by left merge
            merged_df['Progression Status'] = merged_df['Progression Status'].astype(int)
        else:
            print("Warning: 'Progression Status' not in merged data after merge.")

        merged_df.sort_values(['Subject Number', 'Laterality', 'Visit Number'], inplace=True)
        merged_df.reset_index(drop=True, inplace=True)
        return merged_df

    def get_progression_stats(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculates progression status statistics."""
        if 'Progression Status' not in df.columns or df.empty or df['Progression Status'].isnull().all():
            return {
                'by_laterality': pd.Series(dtype='float64'),
                'overall': pd.Series(dtype='float64')
            }

        # Consider each subject-eye once for progression status
        unique_subject_eye_progression = df.drop_duplicates(subset=['Subject Number', 'Laterality'])
        
        label_map = {0: 'Non-Progressor', 1: 'Progressor'}

        by_laterality_stats = (
            unique_subject_eye_progression.groupby('Laterality')['Progression Status']
            .value_counts(normalize=True)
            .rename(index=label_map, level=1) # Rename the 'Progression Status' level
        )
        overall_stats = (
            unique_subject_eye_progression['Progression Status']
            .value_counts(normalize=True)
            .rename(index=label_map)
        )
        return {'by_laterality': by_laterality_stats, 'overall': overall_stats}

    def plot_progression_distribution(self, df: pd.DataFrame) -> None:
        """Plots the distribution of progression status by laterality."""
        if ('Progression Status' not in df.columns or
            df.empty or
            df['Progression Status'].isnull().all()):
            print("Warning: Cannot plot progression distribution due to missing or empty 'Progression Status'.")
            return

        plt.figure(figsize=(10, 5))
        # Ensure we count each subject-eye once
        plot_df = df.drop_duplicates(subset=['Subject Number', 'Laterality']).copy()
        plot_df['Progression Status Label'] = plot_df['Progression Status'].map({
            0: 'Non-Progressor', 1: 'Progressor'
        })

        progression_dist = (
            plot_df.groupby(['Laterality'])['Progression Status Label']
            .value_counts(normalize=True)
            .unstack(fill_value=0) # Fill with 0 if a category is missing
        )

        if progression_dist.empty:
            print("Warning: No data to plot for progression distribution after processing.")
            return

        ax = progression_dist.plot(kind='bar', width=0.8, figsize=(10, 5))
        plt.title('Distribution of Progression Status by Laterality (Unique Subject-Eyes)')
        plt.xlabel('Laterality')
        plt.ylabel('Proportion of Subject-Eyes')
        plt.legend(title='Progression Status', loc='upper right')
        plt.xticks(rotation=0)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
        
        plt.tight_layout()
        plt.show()

def train_baseline_vit_model(
    features_df: pd.DataFrame,
    model_name: str = "Baseline ViT-RF Model (Embeddings)",
    n_splits: int = 5,
    select_features: bool = True,
    selector_c: float = 0.1,
    selector_threshold: str = 'median',
    rf_n_estimators: int = 150,
    rf_max_depth: Optional[int] = 10,
    rf_min_samples_leaf: int = 5,
    scaler_type: str = 'standard',
) -> Optional[Dict[str, Any]]:
    """Trains a Random Forest model on ViT embeddings with GroupKFold cross-validation."""
    print(f"\n===== Training {model_name} =====")

    required_cols = ['Progression Status', 'Subject Number', 'Laterality', 'vit_features']
    if not all(col in features_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in features_df.columns]
        print(f"Error: DataFrame for {model_name} is missing required columns: {missing}.")
        return None

    features_df_clean = features_df.dropna(subset=['vit_features', 'Progression Status'])
    if features_df_clean.empty:
        print(f"Error: DataFrame for {model_name} is empty after dropping NaNs in essential columns.")
        return None

    try:
        X_features_orig = np.vstack(features_df_clean['vit_features'].values)
        X_orig_df = pd.DataFrame(
            X_features_orig, columns=[f'vit_emb_{i}' for i in range(X_features_orig.shape[1])]
        )
        y = features_df_clean['Progression Status'].astype(int)
        groups = features_df_clean['Subject Number']
    except Exception as e:
        print(f"Error preparing data for {model_name}: {e}")
        traceback.print_exc()
        return None

    print(f"Using {X_orig_df.shape[0]} samples, {X_orig_df.shape[1]} features.")
    print("Class distribution:\n", y.value_counts(normalize=True))

    if len(y.unique()) < 2:
        print(f"Error: Only one class present in the target variable for {model_name}. Cannot train.")
        return None

    unique_groups = len(groups.unique())
    if unique_groups < n_splits:
        old_n_splits = n_splits
        n_splits = max(2, unique_groups)
        print(f"Warning: Number of unique groups ({unique_groups}) is less than n_splits ({old_n_splits}). "
              f"Reducing n_splits to {n_splits}.")
    if n_splits < 2: # After potential reduction
        print(f"Error: Not enough groups ({unique_groups}) for GroupKFold cross-validation (min 2 splits required).")
        return None
        
    neg_count, pos_count = y.value_counts(sort=False).get(0, 0), y.value_counts(sort=False).get(1, 0)
    class_weight_for_1 = (neg_count / pos_count) if pos_count > 0 else 1.0
    rf_class_weight = {0: 1, 1: class_weight_for_1}
    print(f"RF class_weight: {rf_class_weight}")

    pipeline_steps = [('imputer', SimpleImputer(strategy='mean'))]
    if scaler_type == 'minmax':
        pipeline_steps.append(('scaler', MinMaxScaler()))
        print("Using MinMaxScaler.")
    else: # Default to StandardScaler
        pipeline_steps.append(('scaler', StandardScaler()))
        print("Using StandardScaler.")

    if select_features:
        print(f"Adding feature selection (LogisticRegression L1, C={selector_c}, threshold='{selector_threshold}').")
        selector_estimator = LogisticRegression(
            solver='liblinear', penalty='l1', C=selector_c,
            class_weight='balanced', random_state=42, max_iter=200
        )
        pipeline_steps.append(
            ('selector', SelectFromModel(selector_estimator, threshold=selector_threshold))
        )
    
    # Adjust SMOTE k_neighbors based on the smallest class count in the whole dataset initially
    # This will be further adjusted per fold.
    initial_smote_k = max(1, min(4, pos_count - 1, neg_count -1)) if min(pos_count, neg_count) > 1 else 1

    pipeline_steps.extend([
        ('smote', SMOTE(random_state=42, k_neighbors=initial_smote_k)),
        ('classifier', RandomForestClassifier(
            n_estimators=rf_n_estimators, class_weight=rf_class_weight,
            min_samples_leaf=rf_min_samples_leaf, max_depth=rf_max_depth,
            random_state=42, n_jobs=-1
        ))
    ])
    pipeline = ImbPipeline(pipeline_steps)
    group_kfold = GroupKFold(n_splits=n_splits)
    
    all_y_true, all_y_pred, all_y_prob = [], [], []
    all_feature_importances_list = []
    cv_metric_keys = [
        'accuracy', 'auc', 'auc_pr_positive', 
        'sensitivity_positive', 'specificity_negative', 
        'precision_positive', 'f1_positive'
    ]
    cv_scores = {key: [] for key in cv_metric_keys}
    fold_errors = 0

    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X_orig_df, y, groups)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_train, X_test = X_orig_df.iloc[train_idx], X_orig_df.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if X_train.empty or X_test.empty or len(y_train.unique()) < 2:
            print(f"Warning: Fold {fold + 1} skipped due to empty data or single class in y_train.")
            fold_errors += 1
            # Add NaNs for this fold's metrics
            for key in cv_scores: cv_scores[key].append(np.nan)
            continue

        current_pipeline = ImbPipeline(steps=[
            (name, clone(estimator)) for name, estimator in pipeline.steps
        ])
        
        # Dynamically adjust SMOTE k_neighbors for the current training fold
        minority_count_train = y_train.value_counts().min()
        if 'smote' in current_pipeline.named_steps:
            smote_estimator = current_pipeline.named_steps['smote']
            current_k = smote_estimator.k_neighbors
            if minority_count_train <= current_k:
                new_k_smote = max(1, minority_count_train - 1)
                if new_k_smote == 0: # Not enough samples for SMOTE
                    print(f"SMOTE k_neighbors adjustment: Minority count ({minority_count_train}) too low. Removing SMOTE for this fold.")
                    current_pipeline.steps = [
                        s_step for s_step in current_pipeline.steps if s_step[0] != 'smote'
                    ]
                else:
                    print(f"SMOTE k_neighbors adjustment: Minority count ({minority_count_train}) <= current_k ({current_k}). Setting k_neighbors to {new_k_smote}.")
                    current_pipeline.set_params(smote__k_neighbors=new_k_smote)
        
        try:
            current_pipeline.fit(X_train, y_train)
            y_pred_fold = current_pipeline.predict(X_test)
            y_prob_fold = current_pipeline.predict_proba(X_test)[:, 1]

            # Feature Importance handling
            feature_names_in_fold = X_train.columns
            if 'selector' in current_pipeline.named_steps:
                selector = current_pipeline.named_steps['selector']
                if hasattr(selector, 'get_support'):
                    selected_mask = selector.get_support()
                    feature_names_in_fold = X_train.columns[selected_mask]
            
            classifier_step = current_pipeline.named_steps['classifier']
            if hasattr(classifier_step, 'feature_importances_') and \
               len(feature_names_in_fold) == len(classifier_step.feature_importances_):
                importances = pd.Series(
                    classifier_step.feature_importances_, index=feature_names_in_fold
                )
                all_feature_importances_list.append(importances)

        except Exception as pipe_e:
            print(f"Error in Fold {fold + 1} pipeline: {pipe_e}")
            traceback.print_exc()
            fold_errors += 1
            for key in cv_scores: cv_scores[key].append(np.nan)
            continue

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_fold)
        all_y_prob.extend(y_prob_fold)

        try: # Metric calculation for the fold
            accuracy_val = accuracy_score(y_test, y_pred_fold)
            roc_auc_val = roc_auc_score(y_test, y_prob_fold) if len(np.unique(y_test)) > 1 else np.nan
            
            cm_fold = confusion_matrix(y_test, y_pred_fold, labels=[0, 1])
            tn, fp, fn, tp = cm_fold.ravel()

            sens_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1_pos_val = (2 * prec_pos * sens_pos) / (prec_pos + sens_pos) if (prec_pos + sens_pos) > 0 else 0.0
            
            auc_pr_val = np.nan
            if len(np.unique(y_test)) > 1 and 1 in y_test and 0 in y_test:
                precision_pr_f, recall_pr_f, _ = precision_recall_curve(y_test, y_prob_fold, pos_label=1)
                auc_pr_val = auc(recall_pr_f, precision_pr_f) # Using auc from sklearn.metrics

            metrics_this_fold = [
                accuracy_val, roc_auc_val, auc_pr_val, 
                sens_pos, spec_neg, prec_pos, f1_pos_val
            ]
            for key, value in zip(cv_scores.keys(), metrics_this_fold):
                cv_scores[key].append(value)
        except Exception as metric_e:
            print(f"Error calculating metrics for Fold {fold + 1}: {metric_e}")
            traceback.print_exc()
            for key_err in cv_scores: cv_scores[key_err].append(np.nan)

    if fold_errors == n_splits or not all_y_true:
        print(f"\nERROR: {model_name} training failed completely or produced no results.")
        return None

    feature_importance_df = pd.DataFrame({'Feature': [], 'Importance': []})
    if all_feature_importances_list:
        valid_importances = [s for s in all_feature_importances_list if isinstance(s, pd.Series) and not s.empty]
        if valid_importances:
            # Concatenate series, average across folds, then sort
            feature_importance_df = (
                pd.concat(valid_importances, axis=1).fillna(0).mean(axis=1)
                .sort_values(ascending=False).reset_index()
            )
            feature_importance_df.columns = ['Feature', 'Importance']
    
    # --- Plotting & Reporting Overall Results ---
    overall_roc_auc = np.nan
    if len(all_y_true) > 0 and len(np.unique(all_y_true)) > 1:
        overall_roc_auc = roc_auc_score(all_y_true, all_y_prob)
    
    overall_cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]) if len(all_y_true) > 0 else np.array([[0, 0], [0, 0]])
    target_names = ['Non-Progressor (0)', 'Progressor (1)']
    report_text = "No predictions generated."
    report_dict = {}
    if len(all_y_true) > 0:
        report_text = classification_report(
            all_y_true, all_y_pred, zero_division=0, labels=[0, 1], target_names=target_names
        )
        report_dict = classification_report(
            all_y_true, all_y_pred, output_dict=True, zero_division=0, labels=[0, 1]
        )

    auc_pr_score_overall = np.nan
    if len(all_y_true) > 0 and len(np.unique(all_y_true)) > 1 and 1 in all_y_true and 0 in all_y_true:
        precision_pr, recall_pr, _ = precision_recall_curve(all_y_true, all_y_prob, pos_label=1)
        auc_pr_score_overall = auc(recall_pr, precision_pr) # Using auc from sklearn.metrics
        plt.figure(figsize=(7, 6))
        plt.plot(recall_pr, precision_pr, lw=2.5, color='blueviolet', 
                 label=f'PR Curve (AUC-PR = {auc_pr_score_overall:.3f})')
        plt.xlabel('Recall (Progressor)')
        plt.ylabel('Precision (Progressor)')
        plt.title(f'Precision-Recall Curve (Progressor) - {model_name}')
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(7, 6))
    if not np.isnan(overall_roc_auc):
        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
        plt.plot(fpr, tpr, lw=2.5, color='darkorange', label=f'Overall ROC (AUC = {overall_roc_auc:.3f})')
    else:
        plt.text(0.5, 0.5, 'ROC AUC N/A', ha='center', va='center', fontsize=12, color='red')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Overall ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

    if not feature_importance_df.empty:
        plt.figure(figsize=(10, max(6, min(20, len(feature_importance_df)) // 2)))
        top_n = min(30, len(feature_importance_df))
        sns.barplot(data=feature_importance_df.head(top_n), x='Importance', y='Feature', palette='viridis')
        plt.title(f'Top {top_n} Features - {model_name}')
        plt.xlabel('Mean Importance (Gini)')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    print(f"\n--- {model_name}: Cross-Validation Summary ---")
    for metric_key, scores_list in cv_scores.items():
        valid_scores = [s for s in scores_list if not np.isnan(s)]
        mean_val = np.mean(valid_scores) if valid_scores else np.nan
        std_val = np.std(valid_scores) if valid_scores and len(valid_scores) > 1 else np.nan
        mean_std_str = f"{mean_val:.3f} ± {std_val:.3f}" if not np.isnan(mean_val) and not np.isnan(std_val) else "N/A"
        metric_name_fmt = metric_key.replace('_', ' ').capitalize()
        print(f"Mean {metric_name_fmt:<25}: {mean_std_str} ({len(valid_scores)}/{n_splits} valid folds)")

    print(f"\n--- {model_name}: Overall Performance (Aggregated over all test folds) ---")
    roc_auc_display = f"{overall_roc_auc:.3f}" if not np.isnan(overall_roc_auc) else "N/A"
    pr_auc_display = f"{auc_pr_score_overall:.3f}" if not np.isnan(auc_pr_score_overall) else "N/A"
    print(f"Overall ROC AUC              : {roc_auc_display}")
    print(f"Overall PR AUC (Progressor)  : {pr_auc_display}")
    if report_dict and 'accuracy' in report_dict:
        print(f"Overall Accuracy             : {report_dict['accuracy']:.3f}")
    print("Confusion Matrix (Overall):\n", overall_cm)
    print("Classification Report (Overall):\n", report_text)
    
    return {
        'model_name': model_name,
        'feature_importance': feature_importance_df,
        'overall_roc_auc': overall_roc_auc,
        'overall_auc_pr': auc_pr_score_overall,
        'overall_confusion_matrix': overall_cm,
        'overall_classification_report_dict': report_dict,
        'cv_scores': cv_scores
    }
def engineer_longitudinal_prediction_features(df_with_predictions: pd.DataFrame) -> pd.DataFrame:
    """Engineers longitudinal features from sequences of ViT predictions per subject-eye."""
    print("Engineering longitudinal features from ViT predictions...")
    
    # Ensure 'Visit Number' is numeric for sorting and regression
    df_with_predictions['Visit Number Int'] = pd.to_numeric(
        df_with_predictions['Visit Number'], errors='coerce'
    )
    
    engineered_features_list = []
    # Group by subject and eye to process each longitudinal sequence
    grouped = df_with_predictions.groupby(['Subject Number', 'Laterality'])
    
    for (subject, laterality), group in tqdm(grouped, desc="Engineering Prediction Features"):
        # Sort by visit number and drop if essential data is missing for this sequence
        group = group.sort_values('Visit Number Int').dropna(subset=['vit_prediction', 'Visit Number Int'])
        
        if len(group) < 1:
            continue
            
        predictions = group['vit_prediction'].values
        visits = group['Visit Number Int'].values # Visit numbers corresponding to predictions
        
        # Basic descriptive statistics
        features = {
            'Subject Number': subject,
            'Laterality': laterality,
            'Progression Status': group['Progression Status'].iloc[0], # Assuming status is constant per subject-eye
            'num_visits_pred_feat': len(predictions),
            'mean_pred': np.mean(predictions) if len(predictions) > 0 else np.nan,
            'median_pred': np.median(predictions) if len(predictions) > 0 else np.nan,
            'std_pred': np.std(predictions) if len(predictions) > 1 else 0.0,
            'min_pred': np.min(predictions) if len(predictions) > 0 else np.nan,
            'max_pred': np.max(predictions) if len(predictions) > 0 else np.nan,
            'first_visit_pred': predictions[0] if len(predictions) > 0 else np.nan,
            'last_visit_pred': predictions[-1] if len(predictions) > 0 else np.nan,
            'range_pred': (np.max(predictions) - np.min(predictions)) if len(predictions) > 0 else np.nan,
        }
        
        # Features requiring at least two data points
        if len(predictions) > 1:
            features['diff_last_first_pred'] = predictions[-1] - predictions[0]
            
            # Linear regression for slope of predictions over visits
            visits_reshaped = visits.reshape(-1, 1)
            lin_reg = LinearRegression()
            try:
                lin_reg.fit(visits_reshaped, predictions)
                features['slope_pred'] = lin_reg.coef_[0]
            except ValueError: # Handles cases like all visits being the same
                features['slope_pred'] = 0.0
        else: # Default values for single-visit cases
            features['diff_last_first_pred'] = 0.0
            features['slope_pred'] = 0.0
            
        engineered_features_list.append(features)
        
    return pd.DataFrame(engineered_features_list)

def train_rf_on_prediction_features(
    features_df: pd.DataFrame, # This df should be the output of engineer_longitudinal_prediction_features
    model_name: str = "RF on ViT Prediction Features",
    n_splits: int = 5
) -> Optional[Dict[str, Any]]:
    """Trains a Random Forest model on engineered longitudinal prediction features."""
    print(f"\n===== Training {model_name} =====")

    # Define essential columns for this model
    # Add more if other engineered features are critical
    required_cols = ['Progression Status', 'Subject Number', 'Laterality', 'mean_pred', 'slope_pred'] 
    if not all(col in features_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in features_df.columns]
        print(f"Error: DataFrame for {model_name} is missing required columns: {missing}. "
              f"Available columns: {features_df.columns.tolist()}")
        return None

    # Identify feature columns (excluding identifiers and target) to check for NaNs
    cols_to_check_for_nan = [
        col for col in features_df.columns 
        if col not in ['Subject Number', 'Laterality', 'Progression Status']
    ]
    features_df_clean = features_df.dropna(subset=['Progression Status'] + cols_to_check_for_nan)

    if features_df_clean.empty:
        print(f"Error: DataFrame for {model_name} is empty after dropping NaNs.")
        return None

    try:
        y = features_df_clean['Progression Status'].astype(int)
        X = features_df_clean.drop(columns=['Subject Number', 'Laterality', 'Progression Status'])
        groups = features_df_clean['Subject Number']
    except Exception as e:
        print(f"Error preparing data for {model_name}: {e}")
        traceback.print_exc()
        return None

    print(f"Using {X.shape[0]} samples, {X.shape[1]} engineered features.")
    print("Class distribution:\n", y.value_counts(normalize=True))

    if len(y.unique()) < 2:
        print(f"Error: Only one class present for {model_name}. Cannot train.")
        return None

    unique_groups = len(groups.unique())
    if unique_groups < n_splits:
        old_n_splits = n_splits
        n_splits = max(2, unique_groups)
        print(f"Warning: Number of unique groups ({unique_groups}) < n_splits ({old_n_splits}). "
              f"Reducing n_splits to {n_splits}.")
    if n_splits < 2:
        print(f"Error: Not enough groups ({unique_groups}) for CV (min 2 splits required).")
        return None

    neg_count, pos_count = y.value_counts(sort=False).get(0, 0), y.value_counts(sort=False).get(1, 0)
    class_weight_for_1 = (neg_count / pos_count) if pos_count > 0 else 1.0
    rf_class_weight = {0: 1, 1: class_weight_for_1}
    print(f"RF class_weight: {rf_class_weight}")

    # Adjust SMOTE k_neighbors based on the smallest class count in the whole dataset initially
    initial_smote_k_pred = max(1, min(4, pos_count - 1, neg_count -1)) if min(pos_count, neg_count) > 1 else 1

    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42, k_neighbors=initial_smote_k_pred)),
        ('classifier', RandomForestClassifier(
            n_estimators=200, class_weight=rf_class_weight,
            min_samples_leaf=3, max_depth=8,  # Example hyperparameters
            random_state=42, n_jobs=-1
        ))
    ])
    
    group_kfold = GroupKFold(n_splits=n_splits)
    all_y_true, all_y_pred, all_y_prob = [], [], []
    all_feature_importances_list = []
    cv_metric_keys = [
        'accuracy', 'auc', 'auc_pr_positive', 
        'sensitivity_positive', 'specificity_negative', 
        'precision_positive', 'f1_positive'
    ]
    cv_scores = {key: [] for key in cv_metric_keys}
    fold_errors = 0

    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups)):
        print(f"\n--- Fold {fold + 1}/{n_splits} for {model_name} ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if X_train.empty or X_test.empty or len(y_train.unique()) < 2:
            print(f"Warning: Fold {fold + 1} skipped: empty data or single class in y_train.")
            fold_errors += 1
            for key in cv_scores: cv_scores[key].append(np.nan)
            continue

        current_pipeline = ImbPipeline(steps=[
            (name, clone(estimator)) for name, estimator in pipeline.steps
        ])

        minority_count_train = y_train.value_counts().min()
        if 'smote' in current_pipeline.named_steps:
            smote_estimator = current_pipeline.named_steps['smote']
            current_k = smote_estimator.k_neighbors
            if minority_count_train <= current_k:
                new_k = max(1, minority_count_train - 1)
                if new_k == 0:
                    print(f"SMOTE adjustment: Minority count ({minority_count_train}) too low. Removing SMOTE.")
                    current_pipeline.steps = [
                        s_step for s_step in current_pipeline.steps if s_step[0] != 'smote'
                    ]
                else:
                    print(f"SMOTE adjustment: Minority count ({minority_count_train}) <= k ({current_k}). Setting k to {new_k}.")
                    current_pipeline.set_params(smote__k_neighbors=new_k)
        
        try:
            current_pipeline.fit(X_train, y_train)
            y_pred_fold = current_pipeline.predict(X_test)
            y_prob_fold = current_pipeline.predict_proba(X_test)[:, 1]
            
            classifier_step = current_pipeline.named_steps['classifier']
            if hasattr(classifier_step, 'feature_importances_'):
                importances = pd.Series(
                    classifier_step.feature_importances_, index=X_train.columns
                )
                all_feature_importances_list.append(importances)

        except Exception as pipe_e:
            print(f"Error in Fold {fold + 1} pipeline for {model_name}: {pipe_e}")
            traceback.print_exc()
            fold_errors += 1
            for key in cv_scores: cv_scores[key].append(np.nan)
            continue

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_fold)
        all_y_prob.extend(y_prob_fold)

        try: # Metric calculation for the fold
            accuracy_val = accuracy_score(y_test, y_pred_fold)
            roc_auc_val = roc_auc_score(y_test, y_prob_fold) if len(np.unique(y_test)) > 1 else np.nan
            
            cm_fold = confusion_matrix(y_test, y_pred_fold, labels=[0, 1])
            tn, fp, fn, tp = cm_fold.ravel()

            sens_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1_pos_val = (2 * prec_pos * sens_pos) / (prec_pos + sens_pos) if (prec_pos + sens_pos) > 0 else 0.0
            
            auc_pr_val = np.nan
            if len(np.unique(y_test)) > 1 and 1 in y_test and 0 in y_test :
                precision_pr_f, recall_pr_f, _ = precision_recall_curve(y_test, y_prob_fold, pos_label=1)
                auc_pr_val = auc(recall_pr_f, precision_pr_f)

            metrics_this_fold = [
                accuracy_val, roc_auc_val, auc_pr_val, 
                sens_pos, spec_neg, prec_pos, f1_pos_val
            ]
            for key, value in zip(cv_scores.keys(), metrics_this_fold):
                cv_scores[key].append(value)
        except Exception as metric_e:
            print(f"Error calculating metrics for Fold {fold + 1} ({model_name}): {metric_e}")
            traceback.print_exc()
            for key_err in cv_scores: cv_scores[key_err].append(np.nan)

    if fold_errors == n_splits or not all_y_true:
        print(f"\nERROR: {model_name} training failed completely or no results.")
        return None

    feature_importance_df = pd.DataFrame()
    if all_feature_importances_list:
        valid_importances = [s for s in all_feature_importances_list if isinstance(s, pd.Series) and not s.empty]
        if valid_importances:
            feature_importance_df = (
                pd.concat(valid_importances, axis=1).fillna(0).mean(axis=1)
                .sort_values(ascending=False).reset_index()
            )
            feature_importance_df.columns = ['Feature', 'Importance']
    
    # --- Plotting & Reporting Overall Results ---
    # (This part is very similar to train_baseline_vit_model, can be refactored into a helper)
    overall_roc_auc = np.nan
    if len(all_y_true) > 0 and len(np.unique(all_y_true)) > 1:
        overall_roc_auc = roc_auc_score(all_y_true, all_y_prob)
    
    overall_cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]) if len(all_y_true) > 0 else np.array([[0, 0], [0, 0]])
    target_names = ['Non-Progressor (0)', 'Progressor (1)']
    report_text = "No predictions generated."
    report_dict = {}
    if len(all_y_true) > 0:
        report_text = classification_report(
            all_y_true, all_y_pred, zero_division=0, labels=[0, 1], target_names=target_names
        )
        report_dict = classification_report(
            all_y_true, all_y_pred, output_dict=True, zero_division=0, labels=[0, 1]
        )

    auc_pr_score_overall = np.nan
    if len(all_y_true) > 0 and len(np.unique(all_y_true)) > 1 and 1 in all_y_true and 0 in all_y_true:
        precision_pr, recall_pr, _ = precision_recall_curve(all_y_true, all_y_prob, pos_label=1)
        auc_pr_score_overall = auc(recall_pr, precision_pr)
        plt.figure(figsize=(7, 6))
        plt.plot(recall_pr, precision_pr, lw=2.5, color='blueviolet', 
                 label=f'PR Curve (AUC-PR = {auc_pr_score_overall:.3f})')
        plt.xlabel('Recall (Progressor)')
        plt.ylabel('Precision (Progressor)')
        plt.title(f'Precision-Recall Curve (Progressor) - {model_name}')
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(7, 6))
    if not np.isnan(overall_roc_auc):
        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
        plt.plot(fpr, tpr, lw=2.5, color='darkorange', label=f'Overall ROC (AUC = {overall_roc_auc:.3f})')
    else:
        plt.text(0.5, 0.5, 'ROC AUC N/A', ha='center', va='center', fontsize=12, color='red')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Overall ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

    if not feature_importance_df.empty:
        plt.figure(figsize=(10, max(4, len(feature_importance_df) // 2))) # Adjusted height
        sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title(f'Feature Importances - {model_name}')
        plt.xlabel('Mean Importance (Gini)')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    print(f"\n--- {model_name}: Cross-Validation Summary ---")
    for metric_key, scores_list in cv_scores.items():
        valid_scores = [s for s in scores_list if not np.isnan(s)]
        mean_val = np.mean(valid_scores) if valid_scores else np.nan
        std_val = np.std(valid_scores) if valid_scores and len(valid_scores) > 1 else np.nan
        mean_std_str = f"{mean_val:.3f} ± {std_val:.3f}" if not np.isnan(mean_val) and not np.isnan(std_val) else "N/A"
        metric_name_fmt = metric_key.replace('_', ' ').capitalize()
        print(f"Mean {metric_name_fmt:<25}: {mean_std_str} ({len(valid_scores)}/{n_splits} valid folds)")
    
    print(f"\n--- {model_name}: Overall Performance (Aggregated over all test folds) ---")
    roc_auc_display = f"{overall_roc_auc:.3f}" if not np.isnan(overall_roc_auc) else "N/A"
    pr_auc_display = f"{auc_pr_score_overall:.3f}" if not np.isnan(auc_pr_score_overall) else "N/A"
    print(f"Overall ROC AUC              : {roc_auc_display}")
    print(f"Overall PR AUC (Progressor)  : {pr_auc_display}")
    if report_dict and 'accuracy' in report_dict:
        print(f"Overall Accuracy             : {report_dict['accuracy']:.3f}")
    print("Confusion Matrix (Overall):\n", overall_cm)
    print("Classification Report (Overall):\n", report_text)
    
    return {
        'model_name': model_name,
        'feature_importance': feature_importance_df,
        'overall_roc_auc': overall_roc_auc,
        'overall_auc_pr': auc_pr_score_overall,
        'overall_confusion_matrix': overall_cm,
        'overall_classification_report_dict': report_dict,
        'cv_scores': cv_scores
    }

def plot_subject_eye_predictions_over_time(
    df_with_predictions: pd.DataFrame,
    num_progressors: int = 3,
    num_non_progressors: int = 3
):
    """Plots example ViT prediction sequences over time for progressors and non-progressors."""
    if ('vit_prediction' not in df_with_predictions.columns or
        df_with_predictions['vit_prediction'].isnull().all()):
        print("ViT predictions column not available or all null. Cannot plot sequences.")
        return

    # Ensure 'Visit Number Int' exists and is numeric for plotting
    if 'Visit Number Int' not in df_with_predictions.columns:
        df_with_predictions['Visit Number Int'] = pd.to_numeric(
            df_with_predictions['Visit Number'], errors='coerce'
        )
    df_plot = df_with_predictions.dropna(subset=['Visit Number Int', 'vit_prediction', 'Progression Status'])

    progressors = df_plot[df_plot['Progression Status'] == 1]
    non_progressors = df_plot[df_plot['Progression Status'] == 0]

    unique_progressor_subj_eyes = progressors[['Subject Number', 'Laterality']].drop_duplicates()
    unique_non_progressor_subj_eyes = non_progressors[['Subject Number', 'Laterality']].drop_duplicates()

    print(f"Plotting examples from {len(unique_progressor_subj_eyes)} unique progressor subject-eyes "
          f"and {len(unique_non_progressor_subj_eyes)} unique non-progressor subject-eyes.")

    prog_subj_eyes_sample = pd.DataFrame()
    if not unique_progressor_subj_eyes.empty:
        prog_subj_eyes_sample = unique_progressor_subj_eyes.sample(
            n=min(num_progressors, len(unique_progressor_subj_eyes)), random_state=42
        )
        
    non_prog_subj_eyes_sample = pd.DataFrame()
    if not unique_non_progressor_subj_eyes.empty:
        non_prog_subj_eyes_sample = unique_non_progressor_subj_eyes.sample(
            n=min(num_non_progressors, len(unique_non_progressor_subj_eyes)), random_state=42
        )

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False, sharey=True)
    if not prog_subj_eyes_sample.empty or not non_prog_subj_eyes_sample.empty:
        plt.ylim(0, 1) # Assuming predictions are probabilities

    # Plot Progressor Examples
    if not prog_subj_eyes_sample.empty:
        for _, row in prog_subj_eyes_sample.iterrows():
            subj, lat = row['Subject Number'], row['Laterality']
            subject_data = progressors[
                (progressors['Subject Number'] == subj) & (progressors['Laterality'] == lat)
            ].sort_values('Visit Number Int')
            if not subject_data.empty:
                axes[0].plot(subject_data['Visit Number Int'], subject_data['vit_prediction'],
                             marker='o', linestyle='-', label=f'Prog Subj {subj}-{lat}')
    axes[0].set_title('Progressor Examples - ViT Predictions Over Time')
    axes[0].set_ylabel('ViT Predicted Probability (Progression)')
    if not prog_subj_eyes_sample.empty: # Only add legend and grid if plots were made
        axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axes[0].grid(True, linestyle='--', alpha=0.7)

    # Plot Non-Progressor Examples
    if not non_prog_subj_eyes_sample.empty:
        for _, row in non_prog_subj_eyes_sample.iterrows():
            subj, lat = row['Subject Number'], row['Laterality']
            subject_data = non_progressors[
                (non_progressors['Subject Number'] == subj) & (non_progressors['Laterality'] == lat)
            ].sort_values('Visit Number Int')
            if not subject_data.empty:
                axes[1].plot(subject_data['Visit Number Int'], subject_data['vit_prediction'],
                             marker='x', linestyle='--', label=f'Non-Prog Subj {subj}-{lat}')
    axes[1].set_title('Non-Progressor Examples - ViT Predictions Over Time')
    axes[1].set_xlabel('Visit Number (Numeric)')
    axes[1].set_ylabel('ViT Predicted Probability (Progression)')
    if not non_prog_subj_eyes_sample.empty: # Only add legend and grid if plots were made
        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for external legend
    plt.show()


class FocalLoss(nn.Module): # Copied from previous
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
        
class SequenceDataset(Dataset):
    def __init__(self, sequences: List[torch.Tensor], labels: List[int], subject_ids: List[Any]):
        self.sequences = sequences
        self.labels = labels
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.sequences[idx], self.labels[idx]


def collate_fn_pad(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    return padded_sequences, labels_tensor, lengths


class LSTMProgressionClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout_prob: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_x)
        hidden_combined = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        out = self.dropout(hidden_combined)
        out = self.fc(out)
        return out
    
def train_longitudinal_vit_lstm_model(
    df_with_features: pd.DataFrame,
    model_params: Dict,
    n_splits: int = 5,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    patience: int = 10,
    use_focal_loss: bool = False,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    use_weighted_sampler: bool = True,
    monitor_metric_early_stop: str = 'f1_positive'
) -> Optional[Dict[str, Any]]:
    print(f"\n===== Training Longitudinal ViT-LSTM Model (Embeddings) =====")
    required_cols = ['vit_features', 'Subject Number', 'Laterality', 'Progression Status', 'Visit Number']
    if not all(col in df_with_features.columns for col in required_cols):
        print(f"Error: LSTM DataFrame missing: {[c for c in required_cols if c not in df_with_features.columns]}")
        return None

    df_clean = df_with_features.dropna(subset=['vit_features', 'Progression Status'])
    if df_clean.empty:
        print("Error: LSTM DataFrame empty after NaNs drop.")
        return None

    df_clean.sort_values(['Subject Number', 'Laterality', 'Visit Number'], inplace=True)
    grouped = df_clean.groupby(['Subject Number', 'Laterality'])
    sequences, labels, subject_ids_for_kfold, unique_subject_laterality_ids = [], [], [], []

    for (subject, laterality), group_df in tqdm(grouped, desc="Preparing ViT-LSTM (Embedding) Sequences"):
        if group_df.empty or group_df['vit_features'].isnull().any():
            continue
        seq_features = np.vstack(group_df['vit_features'].values)
        sequences.append(torch.tensor(seq_features, dtype=torch.float32))
        labels.append(group_df['Progression Status'].iloc[0])
        subject_ids_for_kfold.append(subject)
        unique_subject_laterality_ids.append(f"{subject}_{laterality}")

    if not sequences:
        print("Error: No sequences for ViT-LSTM (Embeddings).")
        return None

    labels_series = pd.Series(labels)
    print(f"Prepared {len(sequences)} sequences. Class dist: \n{labels_series.value_counts(normalize=True)}")
    if len(labels_series.unique()) < 2:
        print("Error: Only one class in target.")
        return None

    group_kfold = GroupKFold(n_splits=n_splits)
    X_dummy_groups = np.array(unique_subject_laterality_ids)
    y_targets_np = np.array(labels)
    unique_subjects_count = len(np.unique(subject_ids_for_kfold))
    if unique_subjects_count < n_splits:
        n_splits = max(2, unique_subjects_count)
        print(f"Warning: Reducing n_splits to {n_splits}.")
        group_kfold = GroupKFold(n_splits=n_splits)
    if n_splits < 2:
        print("Error: Not enough unique subjects for CV.")
        return None

    fold_results_list = []
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X_dummy_groups, y_targets_np, groups=subject_ids_for_kfold)):
        print(f"\n--- ViT-LSTM (Embeddings) Fold {fold+1}/{n_splits} ---")
        train_seqs_fold = [sequences[i] for i in train_idx]
        train_lbls_fold = [labels[i] for i in train_idx]
        val_seqs_fold = [sequences[i] for i in val_idx]
        val_lbls_fold = [labels[i] for i in val_idx]
        if not train_seqs_fold or not val_seqs_fold or len(pd.Series(train_lbls_fold).unique()) < 2:
            print(f"Fold skip: empty/single class.")
            continue

        train_labels_series_fold = pd.Series(train_lbls_fold)
        print(f"  Fold Train Class Dist: \n{train_labels_series_fold.value_counts(normalize=True)}")
        train_dataset = SequenceDataset(train_seqs_fold, train_lbls_fold, [])
        val_dataset = SequenceDataset(val_seqs_fold, val_lbls_fold, [])
        sampler = None
        shuffle_train = True
        counts_fold = train_labels_series_fold.value_counts().sort_index()
        neg_c, pos_c = counts_fold.get(0, 0), counts_fold.get(1, 0)
        if use_weighted_sampler:
            print("Using WeightedRandomSampler.")
            class_sample_counts = train_labels_series_fold.value_counts().sort_index().values
            if len(class_sample_counts) == 2 and class_sample_counts[0] > 0 and class_sample_counts[1] > 0:
                weight = 1. / class_sample_counts
                samples_weight = torch.from_numpy(np.array([weight[t] for t in train_lbls_fold])).double()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                shuffle_train = False
            else:
                print("Warning: Cannot use WeightedRandomSampler in fold.")

        train_loader = PyTorchDataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            shuffle=shuffle_train, collate_fn=collate_fn_pad
        )
        val_loader = PyTorchDataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_pad
        )
        model = LSTMProgressionClassifier(**model_params).to(DEVICE)
        criterion = None
        if use_focal_loss and not use_weighted_sampler:
            print(f"Using Focal Loss alpha={focal_alpha}, gamma={focal_gamma}")
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            pos_w = None
            if not use_weighted_sampler:
                if neg_c > 0 and pos_c > 0:
                    pos_w = torch.tensor([neg_c / pos_c], dtype=torch.float).to(DEVICE)
                if pos_w is not None:
                    print(f"Using BCEWithLogitsLoss with pos_weight: {pos_w.item()}")
                else:
                    print("Using unweighted BCEWithLogitsLoss.")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=patience // 2 + 2, factor=0.2, verbose=False, min_lr=1e-7
        )
        best_val_metric_score = -float('inf') if monitor_metric_early_stop != 'loss' else float('inf')
        epochs_no_improve = 0
        print(f"Monitoring '{monitor_metric_early_stop}' for early stopping.")
        fold_y_true_best, fold_y_prob_best, fold_y_pred_best = [], [], []

        for epoch in range(num_epochs):  # Training Loop
            model.train()
            train_loss_epoch = 0.0
            for seq_b, lbl_b, len_b in train_loader:
                seq_b, lbl_b, len_b = seq_b.to(DEVICE), lbl_b.to(DEVICE), len_b
                optimizer.zero_grad()
                outputs = model(seq_b, len_b).squeeze(1)
                loss = criterion(outputs, lbl_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_epoch += loss.item() * seq_b.size(0)
            train_loss_epoch /= len(train_loader.dataset) if sampler is None else len(sampler)

            model.eval()
            val_loss_epoch = 0.0
            current_epoch_y_true_val, current_epoch_y_prob_val, current_epoch_y_pred_val = [], [], []
            with torch.no_grad():
                for seq_b, lbl_b, len_b in val_loader:
                    seq_b, lbl_b, len_b = seq_b.to(DEVICE), lbl_b.to(DEVICE), len_b
                    outputs = model(seq_b, len_b).squeeze(1)
                    loss_val_item = criterion(outputs, lbl_b)
                    val_loss_epoch += loss_val_item.item() * seq_b.size(0)
                    probs = torch.sigmoid(outputs)
                    current_epoch_y_true_val.extend(lbl_b.cpu().numpy())
                    current_epoch_y_prob_val.extend(probs.cpu().numpy())
                    current_epoch_y_pred_val.extend((probs > 0.5).int().cpu().numpy())
            val_loss_epoch /= len(val_loader.dataset)
            val_acc_ep = accuracy_score(current_epoch_y_true_val, current_epoch_y_pred_val)
            val_auc_ep = roc_auc_score(current_epoch_y_true_val, current_epoch_y_prob_val) if len(np.unique(current_epoch_y_true_val)) > 1 else 0.0
            report_val_epoch = classification_report(
                current_epoch_y_true_val, current_epoch_y_pred_val, output_dict=True, zero_division=0, labels=[0, 1]
            )
            f1_pos_ep = report_val_epoch.get('1', {}).get('f1-score', 0.0)
            auc_pr_pos_ep = np.nan
            if (
                len(np.unique(current_epoch_y_true_val)) > 1
                and 1 in current_epoch_y_true_val
                and 0 in current_epoch_y_true_val
            ):
                prec_pr_ep, rec_pr_ep, _ = precision_recall_curve(
                    current_epoch_y_true_val, current_epoch_y_prob_val, pos_label=1
                )
                auc_pr_pos_ep = auc(rec_pr_ep, prec_pr_ep)
            print(
                f"E {epoch+1}/{num_epochs} | TrL: {train_loss_epoch:.4f} | VL: {val_loss_epoch:.4f} | "
                f"VAcc: {val_acc_ep:.4f} | VAUC: {val_auc_ep:.4f} | VF1(P): {f1_pos_ep:.4f} | VPR-AUC(P): {auc_pr_pos_ep:.4f}"
            )
            metric_for_scheduler = val_loss_epoch
            current_metric_for_early_stop_val = 0.0
            if monitor_metric_early_stop == 'auc':
                current_metric_for_early_stop_val = val_auc_ep
            elif monitor_metric_early_stop == 'f1_positive':
                current_metric_for_early_stop_val = f1_pos_ep
            elif monitor_metric_early_stop == 'auc_pr_positive':
                current_metric_for_early_stop_val = auc_pr_pos_ep if not np.isnan(auc_pr_pos_ep) else 0.0
            elif monitor_metric_early_stop == 'loss':
                current_metric_for_early_stop_val = val_loss_epoch
            scheduler.step(metric_for_scheduler)
            improved = False
            if monitor_metric_early_stop == 'loss':
                if current_metric_for_early_stop_val < best_val_metric_score:
                    best_val_metric_score = current_metric_for_early_stop_val
                    improved = True
            else:
                if current_metric_for_early_stop_val > best_val_metric_score:
                    best_val_metric_score = current_metric_for_early_stop_val
                    improved = True
            if improved:
                epochs_no_improve = 0
                fold_y_true_best = current_epoch_y_true_val
                fold_y_prob_best = current_epoch_y_prob_val
                fold_y_pred_best = current_epoch_y_pred_val
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping. Best {monitor_metric_early_stop}: {best_val_metric_score:.4f}")
                break

        if fold_y_true_best:  # Store fold results
            cm_fold = confusion_matrix(fold_y_true_best, fold_y_pred_best, labels=[0, 1])
            tn, fp, fn, tp = (cm_fold.ravel() if cm_fold.shape == (2, 2) else (0, 0, 0, 0))
            precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_pos = (
                2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)
                if (precision_pos + recall_pos) > 0
                else 0
            )
            auc_pr_fold_val_final, final_roc_auc_fold = np.nan, np.nan
            if (
                len(np.unique(fold_y_true_best)) > 1
                and 1 in fold_y_true_best
                and 0 in fold_y_true_best
            ):
                precision_pr_fold_f, recall_pr_fold_f, _ = precision_recall_curve(
                    fold_y_true_best, fold_y_prob_best, pos_label=1
                )
                auc_pr_fold_val_final = auc(recall_pr_fold_f, precision_pr_fold_f)
                final_roc_auc_fold = roc_auc_score(fold_y_true_best, fold_y_prob_best)
            fold_results_list.append(
                {
                    'loss': best_val_metric_score if monitor_metric_early_stop == 'loss' else val_loss_epoch,
                    'accuracy': accuracy_score(fold_y_true_best, fold_y_pred_best),
                    'auc': final_roc_auc_fold,
                    'auc_pr_positive': auc_pr_fold_val_final,
                    'sensitivity_positive': recall_pos,
                    'specificity_negative': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'precision_positive': precision_pos,
                    'f1_positive': f1_pos,
                    'y_true': fold_y_true_best,
                    'y_prob': fold_y_prob_best,
                    'y_pred': fold_y_pred_best,
                }
            )

    # Aggregate and report LSTM results
    if not fold_results_list:
        print("\nViT-LSTM (Embeddings) training completed no valid fold results.")
        return None

    print("\n--- Longitudinal ViT-LSTM (Embeddings): CV Summary ---")
    metric_keys_to_agg = [
        'loss', 'accuracy', 'auc', 'auc_pr_positive',
        'sensitivity_positive', 'specificity_negative',
        'precision_positive', 'f1_positive'
    ]
    agg_metrics = {
        k: [res[k] for res in fold_results_list if k in metric_keys_to_agg and res.get(k) is not None]
        for k in metric_keys_to_agg
    }
    final_summary = {}
    for name, values in agg_metrics.items():
        valid_vals = [v for v in values if not np.isnan(v)]
        mean_val, std_val = (np.mean(valid_vals), np.std(valid_vals)) if valid_vals else (np.nan, np.nan)
        print(
            f"Mean {name.replace('_',' ').capitalize():<25}: {mean_val:.3f} ± {std_val:.3f}"
            if valid_vals else f"Mean {name.replace('_',' ').capitalize():<25}: N/A"
        )
        if valid_vals:
            final_summary[f'mean_{name}'] = mean_val

    all_folds_y_true = sum([res['y_true'] for res in fold_results_list if 'y_true' in res and res['y_true']], [])
    all_folds_y_prob = sum([res['y_prob'] for res in fold_results_list if 'y_prob' in res and res['y_prob']], [])
    all_folds_y_pred_agg = sum([res['y_pred'] for res in fold_results_list if 'y_pred' in res and res['y_pred']], [])

    if all_folds_y_true and all_folds_y_prob:
        overall_auc_agg, overall_auc_pr_agg = np.nan, np.nan
        if (
            len(np.unique(all_folds_y_true)) > 1
            and 1 in all_folds_y_true
            and 0 in all_folds_y_true
        ):
            overall_auc_agg = roc_auc_score(all_folds_y_true, all_folds_y_prob)
            precision_pr_overall, recall_pr_overall, _ = precision_recall_curve(
                all_folds_y_true, all_folds_y_prob, pos_label=1
            )
            overall_auc_pr_agg = auc(recall_pr_overall, precision_pr_overall)
            final_summary['overall_auc_aggregated'] = overall_auc_agg
            final_summary['overall_auc_pr_aggregated'] = overall_auc_pr_agg
            plt.figure(figsize=(7, 6))
            fpr, tpr, _ = roc_curve(all_folds_y_true, all_folds_y_prob)
            plt.plot(fpr, tpr, lw=2.5, color='darkgreen', label=f'Overall ROC (AUC = {overall_auc_agg:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('Overall ROC Curve - ViT-LSTM (Embeddings)')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.4)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(7, 6))
            plt.plot(
                recall_pr_overall, precision_pr_overall, lw=2.5, color='blueviolet',
                label=f'PR Curve (AUC-PR = {overall_auc_pr_agg:.3f})'
            )
            plt.xlabel('Recall (Progressor)')
            plt.ylabel('Precision (Progressor)')
            plt.title(f'Overall PR Curve (Prog) - ViT-LSTM (Embeddings)')
            plt.legend()
            plt.grid(alpha=0.4)
            plt.tight_layout()
            plt.show()
        else:
            print("Single/missing classes in aggregated y_true for ViT-LSTM (Embeddings).")

        overall_cm_agg = (
            confusion_matrix(all_folds_y_true, all_folds_y_pred_agg, labels=[0, 1])
            if all_folds_y_pred_agg else np.array([[0, 0], [0, 0]])
        )
        target_names_lstm = ['Non-Progressor (0)', 'Progressor (1)']
        cr_text_agg = (
            classification_report(
                all_folds_y_true, all_folds_y_pred_agg, output_dict=False, zero_division=0,
                labels=[0, 1], target_names=target_names_lstm
            ) if all_folds_y_pred_agg else "No predictions."
        )
        cr_dict_agg = (
            classification_report(
                all_folds_y_true, all_folds_y_pred_agg, output_dict=True, zero_division=0,
                labels=[0, 1]
            ) if all_folds_y_pred_agg else {}
        )
        final_summary['overall_cm_aggregated'] = overall_cm_agg
        final_summary['overall_cr_aggregated_dict'] = cr_dict_agg
        print("\n--- Longitudinal ViT-LSTM (Embeddings): Overall Performance ---")
        print(
            f"Overall Aggregated ROC AUC     : {overall_auc_agg if not np.isnan(overall_auc_agg) else 'N/A':.3f}"
        )
        print(
            f"Overall Aggregated PR AUC (Prog): {overall_auc_pr_agg if not np.isnan(overall_auc_pr_agg) else 'N/A':.3f}"
        )
        print(f"Overall Aggregated Accuracy  : {cr_dict_agg.get('accuracy', 0.0):.3f}")
        print("Overall Aggregated CM:\n", overall_cm_agg)
        print("Overall Aggregated Report:\n", cr_text_agg)
    else:
        print("Could not generate overall report for ViT-LSTM (Embeddings).")

    return final_summary

# %% ----- Main Execution Script -----
def main():
    # --- Configuration ---
    DATA_ROOT = "D:/glaucoma/data/raw/GRAPE" 
    VIT_MODEL_NAME_TIMM = "vit_base_patch16_224" 
    CUSTOM_VIT_PATH = r"D:\glaucoma\scripts\experiments\multi_arch_comparison_fixed25epoch_20250505_1724\vit_VFM_custom_e25_20250506_084200\checkpoints\vit_VFM_custom_e25_20250506_084200_best_model.pth"
    VIT_NUM_ORIGINAL_CLASSES_FOR_PREDICTION_MODEL = 1 
    VIT_EXPECTED_EMBEDDING_DIM = 768 
    FORCE_REEXTRACT_EMBEDDINGS = False 
    FORCE_REEXTRACT_PREDICTIONS = False 

    # --- 1. Load Data ---
    data_loader = GRAPEDataLoader(data_root=DATA_ROOT)
    merged_df_initial = data_loader.merge_data()
    if merged_df_initial.empty or 'Progression Status' not in merged_df_initial.columns: print("CRITICAL: Data loading failed."); return
    print(f"Loaded and merged data: {merged_df_initial.shape[0]} records.")
    stats = data_loader.get_progression_stats(merged_df_initial); 
    if stats and not stats['overall'].empty: print("\nProgression Status Distribution:"); print(stats['overall']); data_loader.plot_progression_distribution(merged_df_initial)

    # --- 2.a ViT Embedding Extraction ---
    df_with_embeddings = pd.DataFrame() # Initialize empty
    try:
        vit_embedding_extractor = ViTEmbeddingExtractor(VIT_MODEL_NAME_TIMM, CUSTOM_VIT_PATH).to(DEVICE)
        vit_image_transform = get_vit_image_transforms(VIT_MODEL_NAME_TIMM) 
        df_with_embeddings = extract_and_cache_all_embeddings_vit(
            merged_df_initial.copy(), vit_embedding_extractor, vit_image_transform, DEVICE,
            cache_file=VIT_EMBEDDING_CACHE_FILE, force_reextract=FORCE_REEXTRACT_EMBEDDINGS
        )
        if 'vit_features' not in df_with_embeddings.columns or df_with_embeddings['vit_features'].isnull().all():
             print("Warning: ViT Embedding extraction resulted in missing features.")
             df_with_embeddings = merged_df_initial.copy() # Reset if failed
    except Exception as e_embed_extract:
         print(f"Could not initialize or run ViTEmbeddingExtractor: {e_embed_extract}. Skipping models that use embeddings.")
         df_with_embeddings = merged_df_initial.copy() # Ensure df exists, but without features

    # --- 2.b ViT Prediction Extraction ---
    df_with_predictions_and_embeddings = df_with_embeddings.copy() # Start with whatever df_with_embeddings is
    try:
        # Ensure ViT image transforms are loaded if embedding extractor failed
        if 'vit_image_transform' not in locals(): 
             vit_image_transform = get_vit_image_transforms(VIT_MODEL_NAME_TIMM)

        vit_prediction_extractor = ViTPredictionExtractor(
            VIT_MODEL_NAME_TIMM, CUSTOM_VIT_PATH, VIT_NUM_ORIGINAL_CLASSES_FOR_PREDICTION_MODEL
        ).to(DEVICE)
        df_with_predictions_and_embeddings = extract_and_cache_all_predictions_vit(
            df_with_predictions_and_embeddings, vit_prediction_extractor, vit_image_transform, DEVICE,
            cache_file=VIT_PREDICTION_CACHE_FILE, force_reextract=FORCE_REEXTRACT_PREDICTIONS
        )
    except Exception as e_pred_extract:
        print(f"Could not initialize or run ViTPredictionExtractor: {e_pred_extract}. Skipping models that use raw predictions.")
        if 'vit_prediction' in df_with_predictions_and_embeddings.columns:
            df_with_predictions_and_embeddings['vit_prediction'] = np.nan # Ensure column exists but is NaN

    # Plot prediction sequences if predictions were successfully extracted
    if 'vit_prediction' in df_with_predictions_and_embeddings.columns and not df_with_predictions_and_embeddings['vit_prediction'].isnull().all():
         plot_subject_eye_predictions_over_time(df_with_predictions_and_embeddings.copy())
    else:
         print("Skipping prediction sequence plotting as predictions are missing.")

    # --- Prepare Final DataFrames for Models ---
    df_for_embedding_models = pd.DataFrame()
    if 'vit_features' in df_with_predictions_and_embeddings.columns:
         df_for_embedding_models = df_with_predictions_and_embeddings.dropna(subset=['vit_features']).copy()
    if df_for_embedding_models.empty: print("No data available for embedding-based models.")
    
    df_for_prediction_feature_models = pd.DataFrame()
    if 'vit_prediction' in df_with_predictions_and_embeddings.columns:
         df_for_prediction_feature_models = df_with_predictions_and_embeddings.dropna(subset=['vit_prediction']).copy()
    if df_for_prediction_feature_models.empty: print("No data available for prediction-feature-based models.")

    # --- MODEL 1: Baseline ViT-RF (using embeddings) ---
    if not df_for_embedding_models.empty:
        df_for_embedding_models['Visit Number Int'] = pd.to_numeric(df_for_embedding_models['Visit Number'], errors='coerce')
        baseline_df_vit_embeddings = df_for_embedding_models.loc[df_for_embedding_models.groupby(['Subject Number', 'Laterality'])['Visit Number Int'].idxmin()].copy()
        if not baseline_df_vit_embeddings.empty:
            print("\n--- Training Baseline ViT-RF (Embeddings) ---")
            train_baseline_vit_model(baseline_df_vit_embeddings, model_name="Baseline ViT-RF (Embeddings)", n_splits=3, select_features=True, selector_c=0.05, rf_n_estimators=200, rf_max_depth=12, rf_min_samples_leaf=3)
        else: print("\nSkipping Baseline ViT-RF (Embeddings) - no baseline data found.")
    else: print("\nSkipping Baseline ViT-RF (Embeddings) - no embedding data.")

    # --- MODEL 2: RF on Engineered ViT Prediction Features ---
    if not df_for_prediction_feature_models.empty:
        engineered_pred_features_df = engineer_longitudinal_prediction_features(df_for_prediction_feature_models)
        if not engineered_pred_features_df.empty:
            print("\n--- Training RF on ViT Prediction Features ---")
            train_rf_on_prediction_features(engineered_pred_features_df, model_name="RF on ViT Prediction Features", n_splits=3)
        else: print("\nSkipping RF on Prediction Features - failed to engineer features.")
    else: print("\nSkipping RF on Prediction Features - no prediction data.")

    # --- MODEL 3: Longitudinal ViT-LSTM (using embeddings) ---
    if not df_for_embedding_models.empty:
        print("\n--- Training Longitudinal ViT-LSTM (Embeddings) ---")
        longitudinal_df_vit_embeddings = df_for_embedding_models.copy()
        # Check if embeddings are valid before proceeding
        if longitudinal_df_vit_embeddings['vit_features'].isnull().all():
             print("Skipping ViT-LSTM (Embeddings) - Embeddings column contains only NaNs.")
        else:
            first_valid_embedding = longitudinal_df_vit_embeddings['vit_features'].dropna().iloc[0]
            # Ensure it's a numpy array for shape check
            if isinstance(first_valid_embedding, np.ndarray):
                vit_embedding_dim_actual = first_valid_embedding.shape[0]
                if vit_embedding_dim_actual != VIT_EXPECTED_EMBEDDING_DIM: print(f"WARNING: Actual ViT embedding dim ({vit_embedding_dim_actual}) != expected ({VIT_EXPECTED_EMBEDDING_DIM}).")
                lstm_model_params_emb = {'input_dim': vit_embedding_dim_actual, 'hidden_dim': 64, 'num_layers': 1, 'num_classes': 1, 'dropout_prob': 0.4 }
                train_longitudinal_vit_lstm_model(longitudinal_df_vit_embeddings, model_params=lstm_model_params_emb, n_splits=3, num_epochs=60, batch_size=8, learning_rate=5e-5, patience=15, use_focal_loss=False, use_weighted_sampler=True, monitor_metric_early_stop='f1_positive')
            else:
                 print(f"Skipping ViT-LSTM (Embeddings) - First valid embedding is not a numpy array (type: {type(first_valid_embedding)}). Problem during extraction?")
    else: print("\nSkipping Longitudinal ViT-LSTM (Embeddings) - no embedding data.")

    print("\n--- Script Finished ---")

if __name__ == '__main__':
    main()