# %%
# Standard libraries
import json
import os
import pickle
import random
import time
from datetime import datetime
import logging # Added
import glob # Added for CHAKSU loading
import hashlib # For potential future cache keying
import re # For sanitizing filenames

# Data manipulation and linear algebra
import numpy as np
import pandas as pd

# Machine Learning and Neural Networks
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
from torchvision.transforms import (Compose, Normalize, RandomAffine, RandomApply, RandomHorizontalFlip,
                                    RandomResizedCrop, RandomRotation, Resize, ToTensor)
from torch.utils.data import DataLoader, Dataset, Subset

# Image processing
from PIL import Image
import cv2
# import timm # Not strictly needed for ResNet-18 from torchvision

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm # Changed from tqdm.notebook

# --- Sklearn imports for classification part ---
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score # confusion_matrix already imported

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# %%
# --- Configuration ---
SEED = 42
BASE_DATA_DIR = r'D:\glaucoma\data\raw\SMDG-19' # Main dataset base
METADATA_FILE = os.path.join(BASE_DATA_DIR, 'metadata - standardized.csv') # Main metadata
IMAGE_DIR = os.path.join(BASE_DATA_DIR, 'full-fundus', 'full-fundus') # Main image directory

PAPILLA_PREFIX = 'PAPILA' # Used by assign_dataset_source

# --- Output Directories ---
# Make sure these are distinguished if you change model or major parameters
MODEL_NAME_TAG = "unet_multitask" # A tag for the model used
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR = f'output_{MODEL_NAME_TAG}_{RUN_TIMESTAMP}'

CHECKPOINT_DIR = os.path.join(BASE_OUTPUT_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, 'results') # Main dir for figures and summary CSVs
METRICS_CACHE_DIR = os.path.join(BASE_OUTPUT_DIR, 'cache') # Directory for metrics cache

# Segmentation Model and Metrics
MODEL_PATH = r'D:\glaucoma\multitask_training_logs\best_multitask_model_epoch_25.pth'
APPLY_SMOOTHING = False
SMOOTHING_FACTOR = 0.1
REQUIRED_METRICS = ['vcdr', 'hcdr', 'area_ratio']

# Main SMDG Data Sampling & Display
SAMPLE_SIZE = 100000 # Reduced for faster testing, adjust as needed
NUM_EXAMPLES_TO_SHOW = 2 # Reduced for brevity in output, adjust as needed

# Configuration for external data loading
BASE_DATA_ROOT_FOR_PYTHON = r"D:\glaucoma\data" # General root for external data if paths are relative
SMDG_META_CSV_RAW_FOR_EXT_LOAD = r"D:\glaucoma\data\raw\SMDG-19\metadata - standardized.csv"
SMDG_IMG_DIR_RAW_FOR_EXT_LOAD = r"D:\glaucoma\data\raw\SMDG-19\full-fundus\full-fundus"
CHAKSU_BASE_DIR_RAW = r"D:\glaucoma\data\raw\Chaksu\Train\Train\1.0_Original_Fundus_Images"
CHAKSU_DECISION_DIR_RAW = r"D:\glaucoma\data\raw\Chaksu\Train\Train\6.0_Glaucoma_Decision"
CHAKSU_META_DIR_RAW = r"D:\glaucoma\data\raw\Chaksu\Train\Train\6.0_Glaucoma_Decision\Majority"

RAW_DIR_NAME = 'raw' # For adjust_path_for_data_type
PROCESSED_DIR_NAME = 'processed' # For adjust_path_for_data_type

# Flags for evaluating external datasets
EVAL_PAPILLA_EXT = True
EVAL_OIAODIR_TEST_EXT = True
EVAL_CHAKSU_EXT = True
EXTERNAL_DATA_TYPE_TARGET = 'raw' # 'raw' or 'processed' - affects path adjustment

# --- Metrics Cache File ---
os.makedirs(METRICS_CACHE_DIR, exist_ok=True)
METRICS_CACHE_FILE = os.path.join(METRICS_CACHE_DIR, 'extracted_metrics_cache.csv')

# Create output directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# %%
# --- Add project directory to Python path (USER-SPECIFIC) ---
import sys
project_root = "D:/glaucoma" # USER: Make sure this is correct for your setup
if project_root not in sys.path:
    sys.path.append(project_root)
    logger.info(f"Appended {project_root} to sys.path")

# --- Import custom modules ---
try:
    from src.models.segmentation.unet import OpticDiscCupPredictor
    from src.features.metrics import GlaucomaMetrics
    from skimage.measure import regionprops
    from scipy.ndimage import binary_fill_holes
    logger.info("Successfully imported custom modules: OpticDiscCupPredictor, GlaucomaMetrics.")
except ImportError as e:
    logger.error(f"Error importing custom modules: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    logger.error("Please ensure 'project_root' is set correctly and custom modules are in the Python path.")
    raise

# %%
# --- Helper Function: sanitize_filename ---
def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be used as a filename."""
    name = name.replace(' ', '_') # Replace spaces
    name = re.sub(r'[^\w\-_.]', '', name) # Remove non-alphanumeric, underscore, hyphen, dot
    name = name[:200] # Limit length to avoid issues with max filename length
    return name

# %%
# --- Helper Function: assign_dataset_source (Provided by User) ---
def assign_dataset_source(name_series: pd.Series) -> pd.Series:
    """
    Assigns a 'dataset_source' label based on filename prefixes.
    Uses the global PAPILLA_PREFIX for PAPILA dataset.
    """
    labels = pd.Series("SMDG_Unknown", index=name_series.index, dtype=str)
    if name_series.empty:
        return labels

    name_series_lower = name_series.str.lower().fillna("")
    
    prefix_to_dataset_map = {
        "beh": "BEH", "crfo": "CRFO", "dr-hagis": "DR-HAGIS", 
        "drishti-gs1-test": "DRISHTI-GS1", "drishti-gs1-train": "DRISHTI-GS1", 
        "fives": "FIVES", "g1020": "G1020", "hrf": "HRF", 
        "jsiec": "JSIEC", "les-av": "LES-AV", 
        "oia-odir-test": "OIA-ODIR-test", "oia-odir-train": "OIA-ODIR-train", 
        "origa": "ORIGA", 
        "papila": PAPILLA_PREFIX, 
        "refuge1": "REFUGE1", "sjchoi86": "sjchoi86-HRF"
    }

    for prefix, dataset_name in prefix_to_dataset_map.items():
        mask = name_series_lower.str.startswith(prefix)
        labels.loc[mask] = dataset_name
    return labels

# %%
# --- Helper Functions for External Data Loading (Provided by User, with minor adaptations) ---
PAPILLA_DATASET_PREFIX_LOADER: str = "PAPILA" 

def adjust_path_for_data_type(original_path: str, data_type: str, base_data_root: str, raw_dir_name: str, processed_dir_name: str) -> str:
    if data_type == 'processed':
        norm_original_path = os.path.normpath(original_path)
        norm_raw_segment = os.path.normpath(os.path.join(base_data_root, raw_dir_name))
        norm_processed_segment = os.path.normpath(os.path.join(base_data_root, processed_dir_name))
        if norm_original_path.startswith(norm_raw_segment):
            processed_path = norm_original_path.replace(norm_raw_segment, norm_processed_segment, 1)
            logger.debug(f"Adjusted path for 'processed': {original_path} -> {processed_path}")
            return processed_path
        else:
            logger.warning(f"Path {original_path} does not start with expected raw segment '{norm_raw_segment}'. Cannot convert to 'processed'. Returning original.")
            return original_path
    elif data_type == 'raw':
        return original_path
    else:
        logger.warning(f"Unknown data_type '{data_type}' in adjust_path_for_data_type. Returning original path.")
        return original_path

def extract_base_chaksu(filename: str) -> str:
    if filename is None: return ""
    return os.path.splitext(os.path.basename(str(filename)))[0]

def _load_chaksu_data_for_evaluation(
    chaksu_base_dir_eval: str, 
    chaksu_decision_dir_raw: str,
    chaksu_metadata_dir_raw: str,
    data_type: str,
    ) -> pd.DataFrame:
    logger.info(f"Loading CHAKSU data (data_type context: {data_type}). Base image dir: {chaksu_base_dir_eval}")
    chaksu_rows = []
    camera_types = ["Bosch", "Forus", "Remidio"]
    camera_csv_map = {
        "Bosch": "Glaucoma_Decision_Comparison_Bosch_majority.csv",
        "Forus": "Glaucoma_Decision_Comparison_Forus_majority.csv",
        "Remidio": "Glaucoma_Decision_Comparison_Remidio_majority.csv",
    }

    def label_to_numeric(label: str) -> int | None:
        label_upper = str(label).strip().upper()
        if "NORMAL" in label_upper: return 0
        if "GLAUCOMA" in label_upper: return 1
        return None

    for camera in camera_types:
        img_dir_for_camera = os.path.join(chaksu_base_dir_eval, camera) 
        decision_csv = os.path.join(chaksu_decision_dir_raw, camera_csv_map[camera])
        metadata_csv = os.path.join(chaksu_metadata_dir_raw, f"{camera}.csv")

        if not os.path.exists(decision_csv):
            logger.warning(f"CHAKSU decision CSV not found: {decision_csv}. Skipping {camera}.")
            continue
        if not os.path.exists(img_dir_for_camera):
            logger.warning(f"CHAKSU image directory for camera not found: {img_dir_for_camera}. Skipping {camera}.")
            continue
        
        try:
            label_df = pd.read_csv(decision_csv)
            meta_df = pd.DataFrame() 
            if "Images" not in label_df.columns:
                logger.error(f"'Images' column missing in {decision_csv}. Skipping {camera}.")
                continue
            
            label_df["types"] = label_df["Majority Decision"].apply(label_to_numeric)
            label_df["base_name"] = label_df["Images"].apply(extract_base_chaksu)
            label_df = label_df.set_index("base_name")

            if os.path.exists(metadata_csv):
                meta_df_temp = pd.read_csv(metadata_csv)
                if not meta_df_temp.empty and "Images" in meta_df_temp.columns:
                    meta_df_temp["base_name"] = meta_df_temp["Images"].apply(extract_base_chaksu)
                    meta_df = meta_df_temp.set_index("base_name")
            
            img_files = []
            for ext in ["*.JPG", "*.jpg", "*.png", "*.PNG"]:
                img_files.extend(glob.glob(os.path.join(img_dir_for_camera, ext)))
            
            for img_path in img_files:
                base_filename = extract_base_chaksu(os.path.basename(img_path))
                if base_filename not in label_df.index: continue 
                
                label_info = label_df.loc[base_filename]
                label_value = label_info["types"]
                if pd.isna(label_value): continue

                metadata_values = {}
                if not meta_df.empty and base_filename in meta_df.index:
                    metadata_values = meta_df.loc[base_filename].to_dict()
                    metadata_values.pop("Images", None) 
                
                chaksu_rows.append({
                    "names": os.path.basename(img_path), 
                    "types": int(label_value),
                    "image_path": img_path, 
                    "dataset_source": f"CHAKSU-{camera}",
                    "camera": camera,
                    **metadata_values 
                })
        except Exception as e:
            logger.error(f"Error processing CHAKSU for camera {camera}: {e}", exc_info=True)

    df_chaksu = pd.DataFrame(chaksu_rows)
    if not df_chaksu.empty:
        df_chaksu.dropna(subset=['types', 'image_path'], inplace=True)
        df_chaksu['types'] = df_chaksu['types'].astype(int)
    logger.info(f"Loaded {len(df_chaksu)} CHAKSU samples (data_type context: {data_type}).")
    return df_chaksu

def load_external_test_data(
    smdg_metadata_file_raw: str, smdg_image_dir_raw: str,
    chaksu_base_dir_raw_images: str, chaksu_decision_dir_raw: str, chaksu_metadata_dir_raw: str,
    data_type: str, base_data_root: str, raw_dir_name_fs: str, processed_dir_name_fs: str,
    eval_papilla: bool, eval_oiaodir_test: bool, eval_chaksu: bool
    ) -> dict[str, pd.DataFrame]:
    logger.info(f"--- Loading External Test Data (target data_type: {data_type}) ---")
    test_data_map = {}

    if eval_papilla:
        if not os.path.exists(smdg_metadata_file_raw):
            logger.warning(f"PAPILLA metadata not found: {smdg_metadata_file_raw}. Skipping.")
        else:
            try:
                df_smdg_all_pap = pd.read_csv(smdg_metadata_file_raw)
                df_papilla = df_smdg_all_pap[df_smdg_all_pap['names'].str.startswith(PAPILLA_DATASET_PREFIX_LOADER, na=False)].copy()
                if not df_papilla.empty:
                    df_papilla['image_path_raw'] = df_papilla['names'].apply(lambda n: os.path.join(smdg_image_dir_raw, f"{n}.png"))
                    df_papilla['image_path'] = df_papilla['image_path_raw'].apply(lambda p: adjust_path_for_data_type(p, data_type, base_data_root, raw_dir_name_fs, processed_dir_name_fs))
                    df_papilla = df_papilla[df_papilla['image_path'].apply(lambda p: isinstance(p, str) and os.path.exists(p))]
                    df_papilla['dataset_source'] = PAPILLA_DATASET_PREFIX_LOADER
                    df_papilla.dropna(subset=['types', 'image_path'], inplace=True)
                    if not df_papilla.empty: df_papilla['types'] = df_papilla['types'].astype(int)
                    if not df_papilla.empty:
                        test_data_map["PAPILLA"] = df_papilla
                        logger.info(f"Loaded {len(df_papilla)} PAPILLA samples.")
                    else: logger.warning("PAPILLA DataFrame empty after processing.")
                else: logger.info("No PAPILLA samples in SMDG metadata.")
            except Exception as e: logger.error(f"Failed to load PAPILLA: {e}", exc_info=True)

    if eval_oiaodir_test:
        if not os.path.exists(smdg_metadata_file_raw):
            logger.warning(f"OIA-ODIR metadata not found: {smdg_metadata_file_raw}. Skipping.")
        else:
            try:
                df_smdg_all_oia = pd.read_csv(smdg_metadata_file_raw) # Reload or use df_smdg_all_pap if appropriate
                df_oia = df_smdg_all_oia[df_smdg_all_oia['names'].str.contains("oia-odir-test", case=False, na=False)].copy()
                if not df_oia.empty:
                    df_oia['image_path_raw'] = df_oia['names'].apply(lambda n: os.path.join(smdg_image_dir_raw, f"{n}.png"))
                    df_oia['image_path'] = df_oia['image_path_raw'].apply(lambda p: adjust_path_for_data_type(p, data_type, base_data_root, raw_dir_name_fs, processed_dir_name_fs))
                    df_oia = df_oia[df_oia['image_path'].apply(lambda p: isinstance(p, str) and os.path.exists(p))]
                    df_oia['dataset_source'] = 'OIA-ODIR-test'
                    df_oia.dropna(subset=['types', 'image_path'], inplace=True)
                    if not df_oia.empty: df_oia['types'] = df_oia['types'].astype(int)
                    if not df_oia.empty:
                        test_data_map["OIA-ODIR-test"] = df_oia
                        logger.info(f"Loaded {len(df_oia)} OIA-ODIR-test samples.")
                    else: logger.warning("OIA-ODIR-test DataFrame empty after processing.")
                else: logger.info("No OIA-ODIR-test samples in SMDG metadata.")
            except Exception as e: logger.error(f"Failed to load OIA-ODIR-test: {e}", exc_info=True)

    if eval_chaksu:
        chaksu_image_base_for_loader = chaksu_base_dir_raw_images
        if data_type == 'processed':
             chaksu_image_base_for_loader = adjust_path_for_data_type(chaksu_base_dir_raw_images, data_type, base_data_root, raw_dir_name_fs, processed_dir_name_fs)
             logger.info(f"CHAKSU image base path for loader (target: {data_type}): {chaksu_image_base_for_loader}")
        
        df_chaksu = _load_chaksu_data_for_evaluation(
            chaksu_base_dir_eval=chaksu_image_base_for_loader, 
            chaksu_decision_dir_raw=chaksu_decision_dir_raw,
            chaksu_metadata_dir_raw=chaksu_metadata_dir_raw,
            data_type=data_type
        )
        if df_chaksu is not None and not df_chaksu.empty:
            test_data_map["CHAKSU"] = df_chaksu
            logger.info(f"Processed CHAKSU data; {len(df_chaksu)} samples added.")
        else: logger.warning("CHAKSU loading resulted in empty DataFrame.")
            
    return test_data_map


# %%
# --- Metrics Caching Functions ---
def load_metrics_cache(cache_path: str) -> pd.DataFrame:
    if os.path.exists(cache_path):
        logger.info(f"Loading metrics cache from {cache_path}")
        try:
            # Specify dtypes for robustness, especially for numeric and ID columns
            dtype_spec = {metric: float for metric in REQUIRED_METRICS}
            dtype_spec.update({'image_path': str, 'name': str, 'dataset_source': str, 'type': float}) # Load 'type' as float initially
            
            df = pd.read_csv(cache_path, dtype=dtype_spec)
            
            # Ensure all required metric columns are present
            for metric in REQUIRED_METRICS:
                if metric not in df.columns:
                    df[metric] = np.nan
            
            # Standardize 'type' column if it exists and handle NaN before int conversion
            if 'type' in df.columns:
                # Convert non-NaN 'type' values to int, keep NaNs as is (if any, though should be rare for 'type')
                df['type'] = df['type'].apply(lambda x: int(x) if pd.notna(x) else x)
            else: # If 'type' column is missing, add it. This state implies data integrity issue for cached items.
                df['type'] = np.nan 

            # Drop rows where essential identifiers or all metrics are NaN, as they are not useful
            # df.dropna(subset=['image_path', 'name', 'type'], how='any', inplace=True) # Critical IDs
            # df.dropna(subset=REQUIRED_METRICS, how='all', inplace=True) # Must have at least one metric
            logger.info(f"Loaded {len(df)} records from cache.")
            return df
        except Exception as e:
            logger.error(f"Error loading cache file {cache_path}: {e}. Returning empty DataFrame.", exc_info=True)
            return pd.DataFrame(columns=['image_path', 'name', 'type', 'dataset_source'] + REQUIRED_METRICS)
    logger.info(f"Metrics cache {cache_path} not found. Will create a new one if new metrics are processed.")
    return pd.DataFrame(columns=['image_path', 'name', 'type', 'dataset_source'] + REQUIRED_METRICS)

def save_metrics_to_cache(df_to_save: pd.DataFrame, cache_path: str):
    if df_to_save.empty:
        logger.info("No metrics to save to cache.")
        return
    logger.info(f"Saving/Updating metrics cache to {cache_path} with {len(df_to_save)} records.")
    cols_to_save = ['image_path', 'name', 'type', 'dataset_source'] + REQUIRED_METRICS
    # Ensure all columns to save are present; if not, log warning but proceed with available ones
    present_cols_to_save = [col for col in cols_to_save if col in df_to_save.columns]
    if len(present_cols_to_save) != len(cols_to_save):
        missing_cols = set(cols_to_save) - set(present_cols_to_save)
        logger.warning(f"Cache saving: Missing expected columns {missing_cols}. Saving with available columns.")

    df_for_csv = df_to_save[present_cols_to_save].copy()
    try:
        df_for_csv.to_csv(cache_path, index=False)
        logger.info(f"Successfully saved cache to {cache_path}.")
    except Exception as e:
        logger.error(f"Error saving cache to {cache_path}: {e}", exc_info=True)

# %%
# --- Core Processing and Analysis Functions ---

def process_dataframe_to_extract_metrics(
    df_input: pd.DataFrame, 
    image_path_col: str, 
    name_col: str, 
    type_col: str,
    dataset_source_col: str,
    predictor_instance: OpticDiscCupPredictor, 
    metrics_calculator_instance: GlaucomaMetrics,
    desc_prefix: str = "Processing"
    ) -> pd.DataFrame:
    
    if df_input.empty:
        logger.warning(f"{desc_prefix}: Input DataFrame for metric extraction is empty.")
        return pd.DataFrame()
    if predictor_instance is None or metrics_calculator_instance is None:
        logger.error(f"{desc_prefix}: Predictor or Metrics Calculator not provided.")
        raise ValueError("Predictor and Metrics Calculator instances are required.")

    results_list = []
    logger.info(f"{desc_prefix}: Starting metric extraction for {len(df_input)} images...")

    for index, row in tqdm(df_input.iterrows(), total=len(df_input), desc=f"{desc_prefix} Images"):
        image_path = row[image_path_col]
        # Use .get for name_col and type_col as they might be 'name'/'type' from a re-processed df
        # vs 'names'/'types' from an initial df. The caller should ensure consistency.
        # For this function, we assume the columns exist as specified.
        image_name = row[name_col] 
        glaucoma_type = row[type_col]
        dataset_source = row[dataset_source_col]

        result_entry = {
            'name': image_name, # Standardized output column name
            'image_path': image_path,
            'type': glaucoma_type, # Standardized output column name
            'dataset_source': dataset_source,
            'disc_mask': None,
            'cup_mask': None
        }

        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}. Skipping {image_name}.")
                continue
            
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Failed to load image {image_path}. Skipping {image_name}.")
                continue

            disc_mask, cup_mask = predictor_instance.predict(
                img, refine_smooth=APPLY_SMOOTHING, smoothing_factor=SMOOTHING_FACTOR
            )
            result_entry['disc_mask'] = disc_mask
            result_entry['cup_mask'] = cup_mask

            metrics = metrics_calculator_instance.extract_metrics(disc_mask, cup_mask)
            if metrics and all(key in metrics for key in REQUIRED_METRICS):
                for key in REQUIRED_METRICS:
                    result_entry[key] = metrics[key]
                results_list.append(result_entry)
            else:
                missing_m = [key for key in REQUIRED_METRICS if key not in (metrics or {})]
                logger.warning(f"Could not calculate all required metrics ({missing_m}) for {image_name}. Metrics: {list(metrics.keys()) if metrics else 'None'}")

        except Exception as e:
            logger.error(f"Error processing {image_name} ({image_path}): {e}", exc_info=False)

    logger.info(f"{desc_prefix}: Finished. Metrics calculated for {len(results_list)} of {len(df_input)} images.")
    return pd.DataFrame(results_list)


def analyze_metrics_results(
    metrics_df: pd.DataFrame, 
    dataset_display_name: str, 
    results_dir_path: str, # For saving plots
    show_examples: bool = False, 
    num_examples_to_show_local: int = NUM_EXAMPLES_TO_SHOW
    ):
    
    if metrics_df.empty:
        logger.warning(f"No results to analyze for {dataset_display_name}. Skipping analysis.")
        return

    logger.info(f"\n--- Starting Analysis for: {dataset_display_name} ---")
    sanitized_dataset_name = sanitize_filename(dataset_display_name)
    
    if 'Glaucoma Status' not in metrics_df.columns and 'type' in metrics_df.columns:
        metrics_df['Glaucoma Status'] = metrics_df['type'].map({0: 'Non-Glaucoma', 1: 'Glaucoma'})
    elif 'type' not in metrics_df.columns:
        logger.error(f"'type' column missing in metrics_df for {dataset_display_name}. Cannot create 'Glaucoma Status'. Skipping parts of analysis.")
        return # Critical for most analyses

    # --- Perform ROC Analysis First to get optimal thresholds for distribution plots ---
    logger.info(f"\n--- Performing ROC Analysis for {dataset_display_name} ---")
    roc_summary = {} # Stores AUC and Optimal Threshold for each metric
    
    for metric_col in REQUIRED_METRICS:
        roc_summary[metric_col] = {"AUC": np.nan, "Optimal Threshold": np.nan, "Optimal TPR": np.nan, "Optimal FPR": np.nan}
        if metric_col not in metrics_df.columns or metrics_df[metric_col].isnull().all():
            logger.warning(f"Score column '{metric_col}' not found or all NaN for {dataset_display_name}. Skipping ROC.")
            continue
        if metrics_df['type'].nunique() < 2:
            logger.warning(f"Need >=2 classes in 'type' for ROC of '{metric_col}' in {dataset_display_name}. Skipping.")
            continue

        y_true = metrics_df['type'].values
        y_scores = metrics_df[metric_col].values
        valid_indices = ~np.isnan(y_scores) & ~pd.isna(y_true) # Ensure y_true is also not NaN
        y_true_clean = y_true[valid_indices].astype(int) # Ensure integer type for roc_curve
        y_scores_clean = y_scores[valid_indices]

        if len(np.unique(y_true_clean)) < 2 or len(y_true_clean) < 10:
            logger.warning(f"Not enough valid data/classes for ROC of '{metric_col}' in {dataset_display_name} after NaN removal. Samples: {len(y_true_clean)}, Classes: {np.unique(y_true_clean)}. Skipping.")
            continue

        fpr, tpr, thresholds = roc_curve(y_true_clean, y_scores_clean, pos_label=1)
        roc_auc_val = auc(fpr, tpr)
        roc_summary[metric_col]["AUC"] = roc_auc_val
        logger.info(f"ROC Analysis for {metric_col.upper()} ({dataset_display_name}): AUC = {roc_auc_val:.4f}")

        optimal_threshold, optimal_tpr_val, optimal_fpr_val = None, None, None
        if len(thresholds) > 1 and len(tpr) > 0 and len(fpr) > 0:
            # Youden's J statistic = Sensitivity + Specificity - 1 = TPR - FPR
            # Handle cases where thresholds might not be perfectly aligned or have NaNs/Infs
            finite_indices = np.isfinite(thresholds) & np.isfinite(tpr) & np.isfinite(fpr)
            if np.any(finite_indices):
                tpr_finite = tpr[finite_indices]
                fpr_finite = fpr[finite_indices]
                thresholds_finite = thresholds[finite_indices]
                
                if len(tpr_finite) > 0: # Ensure there are valid points
                    youden_j = tpr_finite - fpr_finite
                    optimal_idx_in_finite = np.argmax(youden_j)
                    optimal_idx = np.where(finite_indices)[0][optimal_idx_in_finite] # Map back to original index

                    optimal_threshold = thresholds[optimal_idx]
                    optimal_tpr_val = tpr[optimal_idx]
                    optimal_fpr_val = fpr[optimal_idx]
                    
                    roc_summary[metric_col]["Optimal Threshold"] = optimal_threshold
                    roc_summary[metric_col]["Optimal TPR"] = optimal_tpr_val
                    roc_summary[metric_col]["Optimal FPR"] = optimal_fpr_val
                    logger.info(f"  Optimal Threshold: {optimal_threshold:.4f} (Sensitivity: {optimal_tpr_val:.4f}, Specificity: {1-optimal_fpr_val:.4f})")
            else:
                 logger.warning(f"No finite points found for optimal threshold calculation for {metric_col} in {dataset_display_name}.")
        else:
            logger.warning(f"Not enough distinct thresholds/points for optimal threshold for {metric_col} in {dataset_display_name}.")

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
        if optimal_threshold is not None and optimal_fpr_val is not None and optimal_tpr_val is not None:
            plt.scatter(optimal_fpr_val, optimal_tpr_val, marker='o', color='red', s=100, zorder=5,
                        label=f'Optimal Thr={optimal_threshold:.3f}\nSens={optimal_tpr_val:.3f}, Spec={1-optimal_fpr_val:.3f}')
        plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)'); plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC Curve: {metric_col.upper()} Classification\nDataset: {dataset_display_name} (n={len(y_true_clean)})')
        plt.legend(loc="lower right"); plt.grid(True)
        fig_path = os.path.join(results_dir_path, f"{sanitized_dataset_name}_{metric_col}_roc.png")
        plt.savefig(fig_path, bbox_inches='tight'); plt.close()
        logger.info(f"Saved ROC plot to {fig_path}")

    logger.info(f"\n--- Summary of Optimal Thresholds & AUC for {dataset_display_name} ---")
    for metric, values in roc_summary.items():
        opt_thr_str = f"{values['Optimal Threshold']:.4f}" if values['Optimal Threshold'] is not None and not pd.isna(values['Optimal Threshold']) else "N/A"
        auc_str = f"{values['AUC']:.4f}" if not pd.isna(values['AUC']) else "N/A"
        logger.info(f"{metric.upper()}: Optimal Threshold = {opt_thr_str}, AUC = {auc_str}")

    # --- Visualize Examples (Optional) ---
    if show_examples:
        logger.info(f"Preparing {num_examples_to_show_local} random examples for {dataset_display_name}...")
        # Ensure masks and path exist, and type is valid for status mapping
        valid_examples_df = metrics_df.dropna(subset=['disc_mask', 'cup_mask', 'image_path', 'type'])
        num_to_show = min(num_examples_to_show_local, len(valid_examples_df))
        
        if num_to_show > 0:
            example_indices = random.sample(range(len(valid_examples_df)), num_to_show)
            plt.figure(figsize=(12, 5 * num_to_show)) # Adjusted for potentially more text
            plot_num = 1
            for series_idx in example_indices:
                example = valid_examples_df.iloc[series_idx]
                try:
                    img_orig = cv2.imread(example['image_path'])
                    if img_orig is None: 
                        logger.warning(f"Could not read example image {example['image_path']}.")
                        continue
                    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

                    disc_mask_vis = example['disc_mask']
                    cup_mask_vis = example['cup_mask']
                    vcdr_val = example.get('vcdr', float('nan')) 
                    hcdr_val = example.get('hcdr', float('nan'))
                    area_ratio_val = example.get('area_ratio', float('nan'))
                    glaucoma_status = example.get('Glaucoma Status', "Status N/A")

                    metrics_text = f"VCDR: {vcdr_val:.3f}\nHCDR: {hcdr_val:.3f}\nArea Ratio: {area_ratio_val:.3f}" if not any(pd.isna(v) for v in [vcdr_val, hcdr_val, area_ratio_val]) else "Metrics N/A"
                    
                    img_overlay = img_rgb.copy()
                    if disc_mask_vis is not None:
                        disc_contours, _ = cv2.findContours(disc_mask_vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img_overlay, disc_contours, -1, (0, 255, 0), 2)
                    if cup_mask_vis is not None:
                        cup_contours, _ = cv2.findContours(cup_mask_vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img_overlay, cup_contours, -1, (255, 0, 0), 2)

                    plt.subplot(num_to_show, 2, plot_num); plot_num += 1
                    plt.imshow(img_rgb)
                    plt.title(f"{example['name']}\nOriginal ({glaucoma_status})\nSource: {example.get('dataset_source', 'N/A')}", fontsize=10)
                    plt.axis('off')

                    plt.subplot(num_to_show, 2, plot_num); plot_num += 1
                    plt.imshow(img_overlay)
                    plt.title(f"Segmentation Overlay\n{metrics_text}", fontsize=10)
                    plt.axis('off')
                except Exception as e_vis:
                    logger.error(f"Error visualizing example {example.get('name', 'Unknown')}: {e_vis}")
                    if plot_num % 2 == 1: plot_num += 2 # Failed on original
                    else: plot_num += 1 # Failed on overlay
                    if plot_num > num_to_show * 2: break

            plt.suptitle(f"Segmentation Examples: {dataset_display_name}", fontsize=16, y=1.00)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            fig_path = os.path.join(results_dir_path, f"{sanitized_dataset_name}_segmentation_examples.png")
            plt.savefig(fig_path, bbox_inches='tight'); plt.close()
            logger.info(f"Saved example plot to {fig_path}")
        elif len(metrics_df) > 0:
             logger.warning(f"Not enough valid examples with masks/paths for {dataset_display_name}.")

    # --- Plot Metric Distributions ---
    logger.info(f"\n--- Plotting Metric Distributions for {dataset_display_name} ---")
    for metric_col_info in [('vcdr', "VCDR"), ('hcdr', "HCDR"), ('area_ratio', "Area Ratio")]:
        metric_col, metric_title_short = metric_col_info
        if metric_col not in metrics_df.columns or metrics_df[metric_col].isnull().all():
            logger.warning(f"Metric '{metric_col}' not found or all NaN for {dataset_display_name}. Skipping distribution.")
            continue
        if 'Glaucoma Status' not in metrics_df.columns or metrics_df['Glaucoma Status'].nunique() < 1:
            logger.warning(f"Not enough class diversity for '{metric_col}' in {dataset_display_name} for distribution plot. Skipping.")
            continue

        plt.figure(figsize=(10, 6))
        sns.histplot(data=metrics_df, x=metric_col, hue='Glaucoma Status', kde=True,
                     palette={'Non-Glaucoma': 'blue', 'Glaucoma': 'red'}, bins=30, element="step", stat="density", common_norm=False)
        
        optimal_thr_for_metric = roc_summary.get(metric_col, {}).get('Optimal Threshold')
        if optimal_thr_for_metric is not None and not pd.isna(optimal_thr_for_metric):
            plt.axvline(optimal_thr_for_metric, color='green', linestyle='--', linewidth=2, 
                        label=f'Optimal Thr: {optimal_thr_for_metric:.3f}')
            plt.legend() # Show legend if threshold line is added

        plt.title(f'Distribution of {metric_title_short}\nDataset: {dataset_display_name} (n={len(metrics_df[metric_col].dropna())})')
        plt.xlabel(f'Calculated {metric_col.upper()}'); plt.ylabel('Density')
        plt.grid(axis='y', linestyle='--')
        fig_path = os.path.join(results_dir_path, f"{sanitized_dataset_name}_{metric_col}_distribution.png")
        plt.savefig(fig_path, bbox_inches='tight'); plt.close()
        logger.info(f"Saved distribution plot to {fig_path}")

        logger.info(f"{metric_col.upper()} Statistics for {dataset_display_name}:")
        try:
            print(metrics_df.groupby('Glaucoma Status')[metric_col].describe())
        except KeyError:
             logger.warning(f"Could not compute stats for {metric_col} in {dataset_display_name}.")


# %%
# --- MAIN SCRIPT EXECUTION ---

# --- Load existing metrics cache ---
full_metrics_cache_df = load_metrics_cache(METRICS_CACHE_FILE)
all_processed_data_accumulator = [] # To collect all results for a final summary CSV

# --- 1. Load and Prepare Main SMDG Metadata ---
logger.info("Loading main SMDG metadata...")
try:
    df_all = pd.read_csv(METADATA_FILE)
    logger.info(f"Initial SMDG metadata rows: {len(df_all)}")
    if 'names' not in df_all.columns: raise ValueError("Metadata CSV must contain 'names'.")
    df_all.dropna(subset=['types'], inplace=True)
    df_all = df_all[df_all['types'].isin([0, 1])]
    df_all['types'] = df_all['types'].astype(int)
    logger.info(f"SMDG rows after label filtering: {len(df_all)}")
    df_all['image_path'] = df_all['names'].apply(lambda name: os.path.join(IMAGE_DIR, f"{name}.png"))
    df_all['dataset_source'] = assign_dataset_source(df_all['names'])
    logger.info(f"SMDG rows ready for sampling: {len(df_all)}")
except FileNotFoundError:
    logger.error(f"SMDG Metadata file {METADATA_FILE} not found. Exiting."); sys.exit(1)
except ValueError as ve:
    logger.error(f"ValueError in SMDG metadata prep: {ve}. Exiting."); sys.exit(1)

# --- 2. Stratified Sampling of Main SMDG Data ---
df_sampled = pd.DataFrame()
if not df_all.empty:
    actual_sample_size = min(SAMPLE_SIZE, len(df_all))
    if len(df_all) < SAMPLE_SIZE:
        logger.warning(f"Sample size ({SAMPLE_SIZE}) > available SMDG data ({len(df_all)}). Using all {len(df_all)} data.")
    if df_all['types'].nunique() < 2 or actual_sample_size == len(df_all) :
        logger.info("Performing random sampling (single class, or using all data).")
        df_sampled = df_all.sample(n=actual_sample_size, random_state=SEED) if actual_sample_size < len(df_all) else df_all.copy()
    else:
        try:
            _, df_sampled = train_test_split(
                df_all, test_size=actual_sample_size, stratify=df_all['types'], random_state=SEED
            ) # test_size here means the size of the sampled set
            logger.info(f"Sampled {len(df_sampled)} images from SMDG (stratified).")
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}. Performing random sampling.")
            df_sampled = df_all.sample(n=actual_sample_size, random_state=SEED)
else:
    logger.warning("df_all is empty, cannot sample SMDG data.")

# --- 3. Initialize Model and Metrics Calculator ---
logger.info("Initializing OpticDiscCupPredictor and GlaucomaMetrics...")
if not os.path.exists(MODEL_PATH):
     logger.error(f"Model weights {MODEL_PATH} not found. Exiting."); sys.exit(1)
try:
    predictor = OpticDiscCupPredictor(model_path=MODEL_PATH)
    metrics_calculator = GlaucomaMetrics()
    logger.info("Predictor and Metrics Calculator initialized.")
except Exception as e:
    logger.error(f"Error initializing predictor/metrics: {e}", exc_info=True); sys.exit(1)

# --- 4. Process Sampled SMDG Data (with Caching) and Analyze ---
results_df_main_smdg = pd.DataFrame()
if not df_sampled.empty:
    logger.info("\n--- Processing Main Sampled SMDG Dataset (with Cache Check) ---")
    df_input_current_run = df_sampled.copy() # Has 'names', 'types', 'dataset_source', 'image_path'

    # Merge with cache to find what's missing
    cache_subset_for_merge = pd.DataFrame()
    if not full_metrics_cache_df.empty:
        cache_subset_for_merge = full_metrics_cache_df[
            full_metrics_cache_df['image_path'].isin(df_input_current_run['image_path'])
        ][['image_path'] + REQUIRED_METRICS].copy()

    if not cache_subset_for_merge.empty:
        df_merged_with_cache = df_input_current_run.merge(cache_subset_for_merge, on='image_path', how='left')
    else:
        df_merged_with_cache = df_input_current_run.copy()
        for metric in REQUIRED_METRICS: df_merged_with_cache[metric] = np.nan
    
    condition_needs_processing = df_merged_with_cache[REQUIRED_METRICS].isnull().any(axis=1)
    df_to_process_freshly = df_merged_with_cache[condition_needs_processing].copy()
    # Keep original column names for process_dataframe_to_extract_metrics
    df_to_process_freshly = df_to_process_freshly[df_input_current_run.columns] 

    df_already_processed_from_cache = df_merged_with_cache[~condition_needs_processing].copy()
    # Add None for mask columns and rename for consistency with newly_processed_df
    df_already_processed_from_cache['disc_mask'] = None
    df_already_processed_from_cache['cup_mask'] = None
    df_already_processed_from_cache.rename(columns={'names': 'name', 'types': 'type'}, inplace=True)
    # Ensure 'dataset_source' and 'image_path' are there (should be from merge)

    newly_processed_df = pd.DataFrame()
    if not df_to_process_freshly.empty:
        logger.info(f"SMDG Sample: Found {len(df_already_processed_from_cache)} records in cache. Processing {len(df_to_process_freshly)} new/incomplete records.")
        newly_processed_df = process_dataframe_to_extract_metrics(
            df_input=df_to_process_freshly,
            image_path_col='image_path', name_col='names', type_col='types', dataset_source_col='dataset_source',
            predictor_instance=predictor, metrics_calculator_instance=metrics_calculator,
            desc_prefix="Main SMDG Sample Processing"
        )
        if not newly_processed_df.empty:
            new_cache_entries_df = newly_processed_df[['image_path', 'name', 'type', 'dataset_source'] + REQUIRED_METRICS].copy()
            full_metrics_cache_df = pd.concat([full_metrics_cache_df, new_cache_entries_df]).drop_duplicates(subset=['image_path'], keep='last')
            save_metrics_to_cache(full_metrics_cache_df, METRICS_CACHE_FILE)
    else:
        logger.info("SMDG Sample: All required records found in cache.")

    results_df_main_smdg = pd.concat([df_already_processed_from_cache, newly_processed_df], ignore_index=True)
    # Ensure 'type' is int if all notna
    if 'type' in results_df_main_smdg.columns and results_df_main_smdg['type'].notna().all():
        results_df_main_smdg['type'] = results_df_main_smdg['type'].astype(int)
    
    if not results_df_main_smdg.empty:
        all_processed_data_accumulator.append(results_df_main_smdg.drop(columns=['disc_mask', 'cup_mask'], errors='ignore'))
        analyze_metrics_results(results_df_main_smdg, "Full SMDG Sampled Dataset", RESULTS_DIR, show_examples=True, num_examples_to_show_local=NUM_EXAMPLES_TO_SHOW)
    else:
        logger.warning("No results for main SMDG sampled data after processing and cache check.")
else:
    logger.warning("df_sampled is empty. Skipping SMDG processing.")


# --- 5. Analyze SMDG Data per Dataset Source ---
if not results_df_main_smdg.empty and 'dataset_source' in results_df_main_smdg.columns:
    unique_sources_in_sample = results_df_main_smdg['dataset_source'].unique()
    logger.info(f"\n--- Analyzing SMDG Sampled Data by Individual Source ({len(unique_sources_in_sample)} sources) ---")
    for source_name in unique_sources_in_sample:
        if pd.isna(source_name) or source_name == "N/A" or source_name == "SMDG_Unknown":
            logger.info(f"Skipping analysis for source: {source_name}")
            continue
        
        df_source_subset = results_df_main_smdg[results_df_main_smdg['dataset_source'] == source_name].copy() # Use .copy()
        if not df_source_subset.empty:
            logger.info(f"\n-- Analyzing SMDG source: {source_name} ({len(df_source_subset)} images) --")
            analyze_metrics_results(df_source_subset, f"SMDG Source - {source_name}", RESULTS_DIR)
        else:
            logger.warning(f"No data for source '{source_name}' in SMDG results.")
else:
    logger.warning("results_df_main_smdg empty or 'dataset_source' missing. Skipping per-source analysis.")


# --- 6. Classification using Extracted Features from Main SMDG Sample ---
logger.info("\n--- Training Classifiers on Extracted Features (Main SMDG Sample) ---")
if results_df_main_smdg.empty or not all(col in results_df_main_smdg.columns for col in REQUIRED_METRICS + ['type']):
    logger.warning("SMDG results empty or missing columns for classification. Skipping.")
else:
    feature_cols = REQUIRED_METRICS
    target_col = 'type'
    
    X = results_df_main_smdg[feature_cols].copy()
    y = results_df_main_smdg[target_col].copy()

    nan_mask = X.isnull().any(axis=1) | y.isnull()
    if nan_mask.sum() > 0:
        logger.warning(f"Dropping {nan_mask.sum()} rows with NaNs in features/target for classification.")
        X = X[~nan_mask]; y = y[~nan_mask]
    
    if len(X) < 20 or y.nunique() < 2:
         logger.warning(f"Insufficient data for CV after NaN removal ({len(X)} samples, {y.nunique()} classes). Skipping classifier training.")
    else:
        y = y.astype(int) # Ensure target is int for classifiers
        classifiers = {
            "Logistic Regression": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=SEED, max_iter=1000, solver='liblinear'))]),
            "SVM": Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, random_state=SEED, C=1.0))]),
            "Random Forest": Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(random_state=SEED, n_estimators=100))])
        }
        N_SPLITS = min(10, len(X) // 2) if len(X) >= 4 else 2 # Adjust splits
        N_SPLITS = max(2, N_SPLITS) # Ensure at least 2 splits
        if len(X) < N_SPLITS * 2 and N_SPLITS > 2: # If not enough samples per fold for many splits
            N_SPLITS = max(2, len(np.unique(y)) if y.nunique() > 1 else 2, len(X)//10 if len(X)//10 > 1 else 2) # Ensure N_SPLITS is reasonable
            logger.warning(f"Adjusted N_SPLITS to {N_SPLITS} due to sample size {len(X)}.")


        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        cv_results_clf = {name: {'Accuracy': [], 'AUC': [], 'Precision': [], 'Recall': [], 'F1': []} for name in classifiers.keys()}
        tprs_clf = {name: [] for name in classifiers.keys()}; mean_fpr_clf = np.linspace(0, 1, 100)

        logger.info(f"Starting {N_SPLITS}-Fold CV for classifiers...")
        fold_loop = tqdm(enumerate(skf.split(X, y)), total=N_SPLITS, desc="CV Folds (Classifiers)")
        for fold, (train_idx, val_idx) in fold_loop:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            for name, pipeline in classifiers.items():
                try:
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_val)
                    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
                    cv_results_clf[name]['Accuracy'].append(accuracy_score(y_val, y_pred))
                    if len(np.unique(y_val)) > 1:
                        cv_results_clf[name]['AUC'].append(roc_auc_score(y_val, y_pred_proba))
                        fpr_f, tpr_f, _ = roc_curve(y_val, y_pred_proba); interp_tpr = np.interp(mean_fpr_clf, fpr_f, tpr_f)
                        interp_tpr[0] = 0.0; tprs_clf[name].append(interp_tpr)
                    else: cv_results_clf[name]['AUC'].append(np.nan); tprs_clf[name].append(np.full_like(mean_fpr_clf, np.nan))
                    cv_results_clf[name]['Precision'].append(precision_score(y_val, y_pred, zero_division=0))
                    cv_results_clf[name]['Recall'].append(recall_score(y_val, y_pred, zero_division=0))
                    cv_results_clf[name]['F1'].append(f1_score(y_val, y_pred, zero_division=0))
                except Exception as e_clf:
                    logger.error(f"Err Fold {fold+1} for {name}: {e_clf}")
                    for ml in cv_results_clf[name].values(): ml.append(np.nan)
                    if name in tprs_clf: tprs_clf[name].append(np.full_like(mean_fpr_clf, np.nan))
        
        summary_cv_clf = {name: {metric: f"{np.nanmean(vals):.3f}±{np.nanstd(vals):.3f}" if vals else "N/A" 
                                for metric, vals in cv_results_clf[name].items()} 
                          for name in classifiers.keys()}
        summary_df = pd.DataFrame.from_dict(summary_cv_clf, orient='index')
        logger.info(f"\nClassifier CV Results (Mean ± Std):\n{summary_df}")
        summary_df.to_csv(os.path.join(RESULTS_DIR, "classifier_cv_summary.csv"))

        plt.figure(figsize=(10, 8))
        colors_clf = plt.cm.get_cmap('viridis', len(classifiers))
        for i, name in enumerate(classifiers.keys()):
            valid_tprs = np.array([t for t in tprs_clf[name] if not np.all(np.isnan(t))])
            if len(valid_tprs) == 0: logger.warning(f"No valid TPRs for {name} ROC plot."); continue
            mean_tpr = np.mean(valid_tprs, axis=0); mean_tpr[-1] = 1.0
            auc_str = summary_df.loc[name, 'AUC'].split('±')[0] if 'AUC' in summary_df.columns and summary_df.loc[name, 'AUC'] != "N/A" else "N/A"
            plt.plot(mean_fpr_clf, mean_tpr, color=colors_clf(i), lw=2, label=f'{name} (AUC {auc_str})')
            std_tpr = np.std(valid_tprs, axis=0)
            plt.fill_between(mean_fpr_clf, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), color=colors_clf(i), alpha=0.2)
        plt.plot([0,1],[0,1], linestyle='--', lw=2, color='grey', label='Chance', alpha=0.8)
        plt.xlim([-0.05,1.05]); plt.ylim([-0.05,1.05])
        plt.xlabel('Mean FPR'); plt.ylabel('Mean TPR')
        plt.title(f'Avg ROC - Classifiers ({N_SPLITS}-Fold CV)\nFeatures: {", ".join(feature_cols)}')
        plt.legend(loc="lower right"); plt.grid(True)
        fig_path = os.path.join(RESULTS_DIR, f"classifiers_avg_roc_{sanitize_filename('_'.join(feature_cols))}.png")
        plt.savefig(fig_path, bbox_inches='tight'); plt.close()
        logger.info(f"Saved classifier ROC plot to {fig_path}")

# --- 7. Load and Process External Test Datasets ---
logger.info("\n--- Loading and Processing External Test Datasets ---")
try:
    external_test_sets_map = load_external_test_data(
        SMDG_META_CSV_RAW_FOR_EXT_LOAD, SMDG_IMG_DIR_RAW_FOR_EXT_LOAD,
        CHAKSU_BASE_DIR_RAW, CHAKSU_DECISION_DIR_RAW, CHAKSU_META_DIR_RAW,
        EXTERNAL_DATA_TYPE_TARGET, BASE_DATA_ROOT_FOR_PYTHON, RAW_DIR_NAME, PROCESSED_DIR_NAME,
        EVAL_PAPILLA_EXT, EVAL_OIAODIR_TEST_EXT, EVAL_CHAKSU_EXT
    )
except Exception as e_ext_load:
    logger.error(f"Fatal error loading external data: {e_ext_load}", exc_info=True)
    external_test_sets_map = {}

if not external_test_sets_map:
    logger.warning("No external test datasets loaded. Skipping their analysis.")
else:
    for dataset_name, df_ext_test_orig in external_test_sets_map.items():
        logger.info(f"\n--- Processing External Test Dataset: {dataset_name} ({len(df_ext_test_orig)} images) ---")
        if df_ext_test_orig.empty:
            logger.warning(f"DataFrame for '{dataset_name}' is empty. Skipping."); continue
        
        df_input_current_run_ext = df_ext_test_orig.copy() # Has 'names', 'types', 'dataset_source', 'image_path'
        if 'dataset_source' not in df_input_current_run_ext.columns: # Should be set by loader
            df_input_current_run_ext['dataset_source'] = dataset_name 

        # Cache check logic for external dataset
        cache_subset_for_merge_ext = pd.DataFrame()
        if not full_metrics_cache_df.empty:
            cache_subset_for_merge_ext = full_metrics_cache_df[
                full_metrics_cache_df['image_path'].isin(df_input_current_run_ext['image_path'])
            ][['image_path'] + REQUIRED_METRICS].copy()

        if not cache_subset_for_merge_ext.empty:
            df_merged_with_cache_ext = df_input_current_run_ext.merge(cache_subset_for_merge_ext, on='image_path', how='left')
        else:
            df_merged_with_cache_ext = df_input_current_run_ext.copy()
            for metric in REQUIRED_METRICS: df_merged_with_cache_ext[metric] = np.nan
        
        condition_needs_processing_ext = df_merged_with_cache_ext[REQUIRED_METRICS].isnull().any(axis=1)
        df_to_process_freshly_ext = df_merged_with_cache_ext[condition_needs_processing_ext].copy()
        df_to_process_freshly_ext = df_to_process_freshly_ext[df_input_current_run_ext.columns]

        df_already_processed_from_cache_ext = df_merged_with_cache_ext[~condition_needs_processing_ext].copy()
        df_already_processed_from_cache_ext['disc_mask'] = None
        df_already_processed_from_cache_ext['cup_mask'] = None
        df_already_processed_from_cache_ext.rename(columns={'names': 'name', 'types': 'type'}, inplace=True)

        newly_processed_df_ext = pd.DataFrame()
        if not df_to_process_freshly_ext.empty:
            logger.info(f"{dataset_name}: Found {len(df_already_processed_from_cache_ext)} records in cache. Processing {len(df_to_process_freshly_ext)} new/incomplete records.")
            newly_processed_df_ext = process_dataframe_to_extract_metrics(
                df_input=df_to_process_freshly_ext,
                image_path_col='image_path', name_col='names', type_col='types', dataset_source_col='dataset_source',
                predictor_instance=predictor, metrics_calculator_instance=metrics_calculator,
                desc_prefix=f"Ext Test: {dataset_name}"
            )
            if not newly_processed_df_ext.empty:
                new_cache_entries_df_ext = newly_processed_df_ext[['image_path', 'name', 'type', 'dataset_source'] + REQUIRED_METRICS].copy()
                full_metrics_cache_df = pd.concat([full_metrics_cache_df, new_cache_entries_df_ext]).drop_duplicates(subset=['image_path'], keep='last')
                save_metrics_to_cache(full_metrics_cache_df, METRICS_CACHE_FILE) # Save cache after each dataset
        else:
            logger.info(f"{dataset_name}: All required records found in cache.")

        results_df_ext_final = pd.concat([df_already_processed_from_cache_ext, newly_processed_df_ext], ignore_index=True)
        if 'type' in results_df_ext_final.columns and results_df_ext_final['type'].notna().all():
             results_df_ext_final['type'] = results_df_ext_final['type'].astype(int)

        if not results_df_ext_final.empty:
            all_processed_data_accumulator.append(results_df_ext_final.drop(columns=['disc_mask', 'cup_mask'], errors='ignore'))
            analyze_metrics_results(results_df_ext_final, f"External Test - {dataset_name}", RESULTS_DIR, show_examples=False) # show_examples=False for externals
        else:
            logger.warning(f"No results for external dataset {dataset_name} after processing and cache check.")

# --- 8. Save all accumulated metrics to a single CSV ---
if all_processed_data_accumulator:
    final_summary_df = pd.concat(all_processed_data_accumulator, ignore_index=True)
    # Drop duplicates that might arise if same image was in multiple processed DFs (e.g. full sample and a specific source)
    # Keep='first' is okay, or 'last' if later processing is preferred for some reason.
    final_summary_df.drop_duplicates(subset=['image_path'], keep='first', inplace=True) 
    
    summary_csv_path = os.path.join(RESULTS_DIR, "all_datasets_extracted_metrics_summary.csv")
    # Ensure standard columns are present
    final_cols = ['name', 'image_path', 'type', 'dataset_source'] + REQUIRED_METRICS
    final_summary_df = final_summary_df[[col for col in final_cols if col in final_summary_df.columns]]

    final_summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved summary of all processed metrics ({len(final_summary_df)} unique entries) to {summary_csv_path}")
else:
    logger.warning("No data was processed or accumulated. Final summary CSV not saved.")

logger.info("\n--- Script Execution Finished ---")