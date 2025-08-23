import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from scipy.interpolate import splprep, splev
import logging
import glob
import pandas as pd
import sys
from typing import Optional, Tuple, Union
import argparse

# Add the same sys.path as training script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from scripts.train_classification import (
        load_chaksu_data, load_airogs_data, load_and_split_data, 
        adjust_path_for_data_type, assign_dataset_source, 
        PAPILLA_PREFIX, RAW_DIR_NAME, PROCESSED_DIR_NAME
    )
    from src.utils.helpers import set_seed
except ImportError as e:
    print(f"Error importing from training script: {e}")
    sys.exit(1)

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- B-Spline Smoothing for Mask Refinement ---
def smooth_mask_with_bspline(mask: np.ndarray, smoothing_factor: float = 0.1, num_points: int = 200) -> Optional[np.ndarray]:
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    mask_for_contours = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 or mask.max() == 1 else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        logger.debug("No contours found for B-spline smoothing.")
        return np.zeros_like(mask, dtype=np.uint8)
    largest_contour = max(contours, key=cv2.contourArea)
    min_points_for_spline = 4
    if len(largest_contour) < min_points_for_spline:
        logger.debug(f"Contour too small for B-spline ({len(largest_contour)} points). Returning filled original contour.")
        smoothed_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(smoothed_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
        return smoothed_mask
    contour_points = largest_contour.squeeze()
    if contour_points.ndim == 1 and contour_points.shape[0] < 2 : # Check if it's a single point or not enough for x,y
        logger.debug("Contour points squeezed to insufficient dimensions. Returning filled original.")
        smoothed_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(smoothed_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
        return smoothed_mask
    elif contour_points.ndim == 1 and contour_points.shape[0] >= 2: # A line of points, can happen
         # This case might still be problematic for splprep with per=True if not enough distinct points
         # For simplicity, we'll let it try, but a more robust handling might be needed for such edge cases.
         pass


    try:
        x, y = contour_points[:, 0], contour_points[:, 1]
        # Ensure k is less than the number of data points if using splprep
        k_spline = min(3, len(x)-1)
        if k_spline < 1: # Not enough points for even a linear spline
            logger.debug(f"Not enough distinct points ({len(x)}) for spline of order {k_spline}. Returning filled original.")
            smoothed_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(smoothed_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
            return smoothed_mask

        s_val = smoothing_factor * len(x) if smoothing_factor is not None else None
        tck, u = splprep([x, y], s=s_val, per=True, k=k_spline, quiet=2)
        if tck is None:
            raise ValueError("splprep failed to compute spline representation.")
        u_new = np.linspace(u.min(), u.max(), num_points)
        x_smooth, y_smooth = splev(u_new, tck, der=0)
        smooth_contour_points = np.vstack((x_smooth, y_smooth)).T.astype(np.int32).reshape((-1, 1, 2))
        h, w = mask.shape
        smooth_contour_points[:,:,0] = np.clip(smooth_contour_points[:,:,0], 0, w-1)
        smooth_contour_points[:,:,1] = np.clip(smooth_contour_points[:,:,1], 0, h-1)
        smoothed_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.fillPoly(smoothed_mask, [smooth_contour_points], 1)
        return smoothed_mask
    except Exception as e:
        logger.error(f"B-spline smoothing error: {e}. Returning filled original contour.")
        smoothed_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(smoothed_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
        return smoothed_mask

# --- U-Net Model Definitions (DoubleConv, MultiTaskUNet) ---
class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # CORRECTED: Removed bias=False to match checkpoints trained with bias=True (default)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), # bias=True (default)
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), # bias=True (default)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class MultiTaskUNet(nn.Module):
    """
    Multi-task U-Net for simultaneous Optic Disc and Optic Cup segmentation.
    Shares the encoder and has two separate decoders.
    """
    def __init__(self, n_channels: int = 3, bilinear: bool = False):
        super().__init__()
        factor = 1 # U-Net with ConvTranspose doesn't halve channels like bilinear upsampling would.
                  # If you used bilinear=True and upsample layers, then factor would be 2.
                  # Based on your original DoubleConv usage, seems like direct channel counts.

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024 // factor)) # factor is 1 here

        # Decoder for Optic Cup
        # Concatenation input channels: upsampled_channels + skip_connection_channels
        self.up1_oc = nn.ConvTranspose2d(1024 // factor, 512 // factor, kernel_size=2, stride=2)
        self.up_conv1_oc = DoubleConv(512 // factor + 512, 512 // factor) # 512 (from x4) + 512 (from up1_oc)
        self.up2_oc = nn.ConvTranspose2d(512 // factor, 256 // factor, kernel_size=2, stride=2)
        self.up_conv2_oc = DoubleConv(256 // factor + 256, 256 // factor) # 256 (from x3) + 256 (from up2_oc)
        self.up3_oc = nn.ConvTranspose2d(256 // factor, 128 // factor, kernel_size=2, stride=2)
        self.up_conv3_oc = DoubleConv(128 // factor + 128, 128 // factor) # 128 (from x2) + 128 (from up3_oc)
        self.up4_oc = nn.ConvTranspose2d(128 // factor, 64, kernel_size=2, stride=2)
        self.up_conv4_oc = DoubleConv(64 + 64, 64) # 64 (from x1) + 64 (from up4_oc)
        self.outc_oc = nn.Conv2d(64, 1, kernel_size=1)

        # Decoder for Optic Disc (identical structure to OC decoder)
        self.up1_od = nn.ConvTranspose2d(1024 // factor, 512 // factor, kernel_size=2, stride=2)
        self.up_conv1_od = DoubleConv(512 // factor + 512, 512 // factor)
        self.up2_od = nn.ConvTranspose2d(512 // factor, 256 // factor, kernel_size=2, stride=2)
        self.up_conv2_od = DoubleConv(256 // factor + 256, 256 // factor)
        self.up3_od = nn.ConvTranspose2d(256 // factor, 128 // factor, kernel_size=2, stride=2)
        self.up_conv3_od = DoubleConv(128 // factor + 128, 128 // factor)
        self.up4_od = nn.ConvTranspose2d(128 // factor, 64, kernel_size=2, stride=2)
        self.up_conv4_od = DoubleConv(64 + 64, 64)
        self.outc_od = nn.Conv2d(64, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Optic Cup decoder
        oc = self.up1_oc(x5); oc = self.up_conv1_oc(torch.cat([x4, oc], dim=1))
        oc = self.up2_oc(oc); oc = self.up_conv2_oc(torch.cat([x3, oc], dim=1))
        oc = self.up3_oc(oc); oc = self.up_conv3_oc(torch.cat([x2, oc], dim=1))
        oc = self.up4_oc(oc); oc = self.up_conv4_oc(torch.cat([x1, oc], dim=1))
        oc_mask_prob = self.sigmoid(self.outc_oc(oc))

        # Optic Disc decoder
        od = self.up1_od(x5); od = self.up_conv1_od(torch.cat([x4, od], dim=1))
        od = self.up2_od(od); od = self.up_conv2_od(torch.cat([x3, od], dim=1))
        od = self.up3_od(od); od = self.up_conv3_od(torch.cat([x2, od], dim=1))
        od = self.up4_od(od); od = self.up_conv4_od(torch.cat([x1, od], dim=1))
        od_mask_prob = self.sigmoid(self.outc_od(od))

        return oc_mask_prob, od_mask_prob

# --- Segmentation Predictor Class ---
class OpticDiscCupPredictor:
    """Handles loading a trained U-Net model and performing OD/OC segmentation."""
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Segmentation predictor using device: {self.device}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = MultiTaskUNet(n_channels=3)
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load or initialize segmentation model from {model_path}: {e}")
            raise
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
        ])

    def _preprocess_input_image(self, image_input: Union[np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        original_size = None
        if isinstance(image_input, np.ndarray):
            original_size = (image_input.shape[1], image_input.shape[0])
            if image_input.ndim == 2: image_pil = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB))
            elif image_input.shape[2] == 4: image_pil = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB))
            elif image_input.shape[2] == 3: image_pil = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else: raise ValueError("Unsupported numpy array format.")
        elif isinstance(image_input, Image.Image):
            original_size = image_input.size
            image_pil = image_input.convert('RGB') if image_input.mode != 'RGB' else image_input
        else: raise TypeError("Input image must be a NumPy array or a PIL Image.")
        return self.transform(image_pil).unsqueeze(0).to(self.device), original_size

    def predict(self, image_input: Union[np.ndarray, Image.Image], threshold: float = 0.5,
                refine_smooth: bool = False, smoothing_factor: float = 0.1, num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            image_tensor, original_size = self._preprocess_input_image(image_input)
            original_w, original_h = original_size if original_size else (0, 0)
            cup_pred_prob, disc_pred_prob = self.model(image_tensor)
            cup_mask_at_model_res = (cup_pred_prob.squeeze().cpu().numpy() > threshold).astype(np.uint8)
            disc_mask_at_model_res = (disc_pred_prob.squeeze().cpu().numpy() > threshold).astype(np.uint8)
            if refine_smooth:
                smoothed_d = smooth_mask_with_bspline(disc_mask_at_model_res, smoothing_factor, num_points)
                if smoothed_d is not None: disc_mask_at_model_res = smoothed_d
                smoothed_c = smooth_mask_with_bspline(cup_mask_at_model_res, smoothing_factor, num_points)
                if smoothed_c is not None: cup_mask_at_model_res = smoothed_c
            cup_mask_constrained_at_model_res = cv2.bitwise_and(cup_mask_at_model_res, disc_mask_at_model_res)
            if original_h > 0 and original_w > 0:
                cup_mask_final = cv2.resize(cup_mask_constrained_at_model_res, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                disc_mask_final = cv2.resize(disc_mask_at_model_res, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            else:
                cup_mask_final = cup_mask_constrained_at_model_res
                disc_mask_final = disc_mask_at_model_res
                logger.warning("Original image dimensions not available for mask resizing. Using model output size.")
        return (disc_mask_final > 0).astype(np.uint8), (cup_mask_final > 0).astype(np.uint8)

# --- Optic Disc Properties Extraction ---
def get_optic_disc_properties(od_mask_binary: np.ndarray, min_contour_area_pixels: int = 100) -> Optional[Tuple[int, int, int, np.ndarray]]:
    contours, _ = cv2.findContours(od_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: logger.debug("No contours in OD mask during property extraction."); return None
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area_pixels]
    if not valid_contours: logger.debug(f"No OD contours > {min_contour_area_pixels}px area."); return None
    largest_contour = max(valid_contours, key=cv2.contourArea)
    (cx_float, cy_float), radius = cv2.minEnclosingCircle(largest_contour)
    center_x, center_y, diameter = int(cx_float), int(cy_float), int(radius * 2)
    if diameter == 0:
        logger.warning("OD diameter 0 from minEnclosingCircle. Fallback to bounding box.")
        x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(largest_contour)
        center_x, center_y, diameter = x_bb + w_bb // 2, y_bb + h_bb // 2, max(w_bb, h_bb)
        if diameter == 0: logger.error("Fallback OD diameter also 0."); return None
    return center_x, center_y, diameter, largest_contour

# --- Image Cropping Utility ---
def crop_around_center(image: np.ndarray, center_x: int, center_y: int, crop_dimension: int) -> np.ndarray:
    img_h, img_w = image.shape[:2]
    half_crop = crop_dimension // 2
    x1_ideal, y1_ideal = center_x - half_crop, center_y - half_crop
    x2_ideal, y2_ideal = x1_ideal + crop_dimension, y1_ideal + crop_dimension
    x1_img_slice, y1_img_slice = max(0, x1_ideal), max(0, y1_ideal)
    x2_img_slice, y2_img_slice = min(img_w, x2_ideal), min(img_h, y2_ideal)
    cropped_from_image = image[y1_img_slice:y2_img_slice, x1_img_slice:x2_img_slice]
    pad_top, pad_bottom = max(0, -y1_ideal), max(0, y2_ideal - img_h)
    pad_left, pad_right = max(0, -x1_ideal), max(0, x2_ideal - img_w)
    if any(p > 0 for p in [pad_top, pad_bottom, pad_left, pad_right]):
        padding_color = [0,0,0] if image.ndim == 3 and image.shape[2] == 3 else 0
        final_crop = cv2.copyMakeBorder(cropped_from_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=padding_color)
    else:
        final_crop = cropped_from_image
    if final_crop.shape[0] != crop_dimension or final_crop.shape[1] != crop_dimension:
         logger.debug(f"Final crop size {final_crop.shape[:2]} differs from target {crop_dimension}x{crop_dimension}. Resizing.")
         final_crop = cv2.resize(final_crop, (crop_dimension, crop_dimension), interpolation=cv2.INTER_AREA)
    return final_crop

# --- CLAHE (Contrast Limited Adaptive Histogram Equalization) ---
def apply_clahe_color(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8,8)) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def apply_color_normalization(image_bgr: np.ndarray, alpha: float = 4.0, beta: float = -4.0, 
                            sigma: float = 5.0, delta: float = 128.0) -> np.ndarray:
    """
    Apply color normalization using Gaussian filtering as described in Kaggle Diabetic Retinopathy Detection.
    Formula: Bc = α × P + β × Gauss(P, s) + δ
    
    Args:
        image_bgr: Input image in BGR format
        alpha: Scaling factor for original image (default: 4.0)
        beta: Scaling factor for Gaussian filtered image (default: -4.0)
        sigma: Standard deviation for Gaussian filter (default: 5.0)
        delta: Bias term (default: 128.0)
    
    Returns:
        Color normalized image in BGR format
    """
    # Convert to float32 for processing
    image_float = image_bgr.astype(np.float32)
    
    # Apply Gaussian filter
    gaussian_filtered = cv2.GaussianBlur(image_float, (0, 0), sigma)
    
    # Apply the formula: Bc = α × P + β × Gauss(P, s) + δ
    normalized = alpha * image_float + beta * gaussian_filtered + delta
    
    # Clip to valid range and convert back to uint8
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return normalized

def apply_enhanced_preprocessing(image_bgr: np.ndarray, 
                               clahe_clip_limit: float = 2.0, 
                               clahe_tile_grid_size: Tuple[int, int] = (8, 8),
                               color_norm_alpha: float = 4.0,
                               color_norm_beta: float = -4.0,
                               color_norm_sigma: float = 5.0,
                               color_norm_delta: float = 128.0) -> np.ndarray:
    """
    Apply enhanced preprocessing: CLAHE + Color Normalization
    
    Args:
        image_bgr: Input image in BGR format
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_grid_size: CLAHE tile grid size
        color_norm_*: Color normalization parameters
    
    Returns:
        Enhanced image in BGR format
    """
    # Step 1: Apply CLAHE (existing function)
    clahe_enhanced = apply_clahe_color(image_bgr, clahe_clip_limit, clahe_tile_grid_size)
    
    # Step 2: Apply Color Normalization
    color_normalized = apply_color_normalization(
        clahe_enhanced, 
        alpha=color_norm_alpha, 
        beta=color_norm_beta, 
        sigma=color_norm_sigma, 
        delta=color_norm_delta
    )
    
    return color_normalized

def create_training_config_for_preprocessing():
    """
    Create a mock config object that matches your training script's arguments.
    Adjust these values to match what you actually use for training.
    """
    config = argparse.Namespace()
    
    # Data configuration - MATCH YOUR TRAINING ARGS
    config.data_type = 'raw'  # or 'processed' - match what you use in training
    config.base_data_root = r'D:\glaucoma\data'
    config.seed = 42
    
    # SMDG-19 config
    config.smdg_metadata_file = r'D:\glaucoma\data\raw\SMDG-19\metadata - standardized.csv'
    config.smdg_image_dir = r'D:\glaucoma\data\raw\SMDG-19\full-fundus\full-fundus'
    
    # CHAKSU config - MATCH YOUR TRAINING SETTINGS
    config.use_chaksu = True  # Set to False if you disable CHAKSU in training
    config.chaksu_base_dir = r'D:\glaucoma\data\raw\Chaksu\Train\Train\1.0_Original_Fundus_Images'
    config.chaksu_decision_dir = r'D:\glaucoma\data\raw\Chaksu\Train\Train\6.0_Glaucoma_Decision'
    config.chaksu_metadata_dir = r'D:\glaucoma\data\raw\Chaksu\Train\Train\6.0_Glaucoma_Decision\Majority'
    
    # AIROGS config - MATCH YOUR TRAINING SETTINGS
    config.use_airogs = True  # Set to False if you disable AIROGS in training
    config.airogs_label_file = r'D:\glaucoma\data\raw\AIROGS\train_labels.csv'
    config.airogs_image_dir = r'D:\glaucoma\data\raw\AIROGS\img'
    config.airogs_num_rg_samples = 3000   # MATCH YOUR TRAINING VALUES
    config.airogs_num_nrg_samples = 3000  # MATCH YOUR TRAINING VALUES
    config.use_airogs_cache = True        # IMPORTANT: Use the same cache as training
    
    # Other required fields
    config.min_samples_per_group_for_stratify = 2
    config.save_data_samples = False
    config.num_data_samples_per_source = 0
    
    # Create a mock output_dir for the data loading functions
    config.output_dir = os.path.join(os.path.dirname(__file__), 'temp_preprocessing_output')
    os.makedirs(config.output_dir, exist_ok=True)
    
    return config

def get_training_data_paths(config):
    """
    Use the EXACT same data loading logic as your training script
    to get the precise list of images that would be used in training.
    """
    logger.info("Loading data using EXACT same logic as training script...")
    
    # Set the global variable needed by the training functions
    import scripts.train_classification as train_script
    train_script.BASE_DATA_DIR_CONFIG_KEY = config.base_data_root
    
    # Adjust paths for processed data if needed (same as training script)
    if config.data_type == 'processed':
        logger.info(f"Data type '{config.data_type}'. Adjusting image paths...")
        config.smdg_image_dir = adjust_path_for_data_type(config.smdg_image_dir, config.data_type)
        config.chaksu_base_dir = adjust_path_for_data_type(config.chaksu_base_dir, config.data_type)
        config.airogs_image_dir = adjust_path_for_data_type(config.airogs_image_dir, config.data_type)
    
    # Use the EXACT same data loading and splitting logic
    train_df, val_df, test_df = load_and_split_data(config)
    
    # Combine all dataframes to get ALL images that would be used
    all_training_dfs = []
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        if not df.empty:
            df['split'] = name  # Track which split each image belongs to
            all_training_dfs.append(df)
    
    if all_training_dfs:
        combined_df = pd.concat(all_training_dfs, ignore_index=True)
        logger.info(f"Found {len(combined_df)} total images that would be used in training")
        logger.info(f"Split distribution: {combined_df['split'].value_counts().to_dict()}")
        logger.info(f"Dataset source distribution: {combined_df['dataset_source'].value_counts().to_dict()}")
        return combined_df
    else:
        logger.warning("No training data found!")
        return pd.DataFrame()

def get_all_ood_data_paths(config):
    """
    Get all OOD/external test dataset paths that need preprocessing.
    This uses the same logic as your OOD evaluation script.
    """
    from src.data.external_loader import load_external_test_data
    from src.data.utils import adjust_path_for_data_type
    
    logger.info("Loading OOD/external test datasets for preprocessing...")
    
    # Set up paths for OOD datasets (same as your evaluation script)
    chaksu_base_dir_for_loader = os.path.join('raw', 'Chaksu', 'Train', 'Train', '1.0_Original_Fundus_Images')
    if config.data_type == 'processed':
        chaksu_base_dir_for_loader = adjust_path_for_data_type(
            relative_path_part=chaksu_base_dir_for_loader, 
            target_data_type='processed',
            base_data_dir_abs=config.base_data_root, 
            raw_subdir_name='raw',
            processed_subdir_name='processed'
        )
    
    # Load all external test datasets
    external_test_sets_map = load_external_test_data(
        smdg_metadata_file_raw=os.path.join('raw', 'SMDG-19', 'metadata - standardized.csv'),
        smdg_image_dir_raw=os.path.join('raw', 'SMDG-19', 'full-fundus', 'full-fundus'),
        chaksu_base_dir_eval=chaksu_base_dir_for_loader,
        chaksu_decision_dir_raw=os.path.join('raw', 'Chaksu', 'Train', 'Train', '6.0_Glaucoma_Decision'),
        chaksu_metadata_dir_raw=os.path.join('raw', 'Chaksu', 'Train', 'Train', '6.0_Glaucoma_Decision', 'Majority'),
        data_type=config.data_type,
        base_data_root=config.base_data_root,
        raw_dir_name='raw',
        processed_dir_name='processed',
        eval_papilla=True,
        eval_oiaodir_test=False,  # Set to True if you want to include this
        eval_chaksu=True,
        eval_acrima=True,        # NEW
        eval_hygd=True,          # NEW
        acrima_image_dir_raw=os.path.join('raw', 'ACRIMA', 'Database', 'Images'),
        hygd_image_dir_raw=os.path.join('raw', 'HYGD', 'HYGD', 'Images'),
        hygd_labels_file_raw=os.path.join('raw', 'HYGD', 'HYGD', 'Labels.csv')
    )
    
    # Convert to list format similar to training data
    ood_images_to_process = []
    
    for dataset_name, df_dataset in external_test_sets_map.items():
        if df_dataset.empty:
            continue
            
        logger.info(f"Found {len(df_dataset)} images in {dataset_name} for preprocessing")
        
        for _, row in df_dataset.iterrows():
            original_path = row['image_path']
            
            # Convert to raw path if needed (same logic as training data)
            if config.data_type == 'processed':
                raw_path = original_path.replace(f"{os.sep}processed{os.sep}", f"{os.sep}raw{os.sep}")
            else:
                raw_path = original_path
            
            # Target processed path
            target_processed_path = original_path if config.data_type == 'processed' else original_path.replace(f"{os.sep}raw{os.sep}", f"{os.sep}processed{os.sep}")
            
            if os.path.exists(raw_path):
                ood_images_to_process.append({
                    'raw_path': raw_path,
                    'target_processed_path': target_processed_path,
                    'split': 'ood_test',  # Mark as OOD test data
                    'dataset_source': dataset_name,
                    'label': row.get('types', row.get('label', -1))
                })
            else:
                logger.warning(f"OOD raw image not found: {raw_path}")
    
    logger.info(f"Found {len(ood_images_to_process)} OOD images to process")
    logger.info(f"OOD dataset distribution: {pd.Series([img['dataset_source'] for img in ood_images_to_process]).value_counts().to_dict()}")
    
    return ood_images_to_process

def main_preprocess_matched():
    """
    Main preprocessing function that processes BOTH training data AND OOD test data.
    """
    # --- Configuration ---
    SEGMENTATION_MODEL_PATH = r'D:\glaucoma\multitask_training_logs\best_multitask_model_epoch_25.pth'
    BASE_DATA_DIR = r'D:\glaucoma\data'
    RAW_SUBDIR_NAME = 'raw'
    PROCESSED_SUBDIR_NAME = 'processed'
    
    # Control whether to use OD detection and cropping
    USE_OD_DETECTION_AND_CROPPING = False  # Set to False to skip OD detection and cropping
    
    # Segmentation and cropping parameters
    PRIMARY_SEG_THRESHOLD = 0.5
    FALLBACK_SEG_THRESHOLD = 0.25
    CROP_FACTOR = 2.0
    MIN_CONTOUR_AREA_PIXELS_FOR_OD = 500
    MIN_EFFECTIVE_OD_DIAMETER_PIXELS = 80
    MIN_CROP_DIMENSION_ABSOLUTE_PIXELS = 256
    MAX_CROP_DIMENSION_FACTOR_OF_IMG_SHORT_SIDE = 0.95
    FALLBACK_CENTER_CROP_DIMENSION = 512

    # --- Get Training Configuration ---
    config = create_training_config_for_preprocessing()
    set_seed(config.seed)
    
    # --- Get Training Data Paths ---
    logger.info("=== PROCESSING TRAINING DATA ===")
    training_data_df = get_training_data_paths(config)
    
    # --- Get OOD Test Data Paths ---
    logger.info("=== PROCESSING OOD TEST DATA ===")
    ood_data_list = get_all_ood_data_paths(config)
    
    # --- Combine All Images to Process ---
    all_images_to_process = []
    
    # Add training data
    if not training_data_df.empty:
        for _, row in training_data_df.iterrows():
            original_path = row['image_path']
            
            if config.data_type == 'processed':
                raw_path = original_path.replace(f"{os.sep}{PROCESSED_SUBDIR_NAME}{os.sep}", f"{os.sep}{RAW_SUBDIR_NAME}{os.sep}")
            else:
                raw_path = original_path
            
            if os.path.exists(raw_path):
                all_images_to_process.append({
                    'raw_path': raw_path,
                    'target_processed_path': original_path if config.data_type == 'processed' else original_path.replace(f"{os.sep}{RAW_SUBDIR_NAME}{os.sep}", f"{os.sep}{PROCESSED_SUBDIR_NAME}{os.sep}"),
                    'split': row.get('split', 'unknown'),
                    'dataset_source': row.get('dataset_source', 'training'),
                    'label': row.get('types', -1),
                    'data_category': 'training'
                })
    
    # Add OOD data
    for ood_img in ood_data_list:
        ood_img['data_category'] = 'ood_test'
        all_images_to_process.append(ood_img)
    
    total_images = len(all_images_to_process)
    training_count = sum(1 for img in all_images_to_process if img['data_category'] == 'training')
    ood_count = sum(1 for img in all_images_to_process if img['data_category'] == 'ood_test')
    
    logger.info(f"=== TOTAL PREPROCESSING SUMMARY ===")
    logger.info(f"Total images to process: {total_images}")
    logger.info(f"  - Training data: {training_count}")
    logger.info(f"  - OOD test data: {ood_count}")
    
    if total_images == 0:
        logger.critical("No images found to process. Exiting.")
        return
    
    # --- Initialize Segmentation Predictor (only if needed) ---
    predictor = None
    if USE_OD_DETECTION_AND_CROPPING:
        try:
            predictor = OpticDiscCupPredictor(SEGMENTATION_MODEL_PATH)
            logger.info("OD detection and cropping enabled - segmentation predictor loaded")
        except Exception as e:
            logger.critical(f"Could not initialize segmentation predictor: {e}")
            return
    else:
        logger.info("OD detection and cropping disabled - applying only enhanced preprocessing")

    # --- Process Each Image ---
    already_processed_count = 0
    newly_processed_count = 0
    failed_to_process_count = 0
    od_primary_success_count = 0
    od_fallback_threshold_success_count = 0
    od_fallback_center_crop_used_count = 0
    
    # Track progress by category
    training_processed = 0
    ood_processed = 0
    
    # Progress tracking
    log_interval = max(1, total_images // 20)
    last_log_count = 0
    
    for i, img_info in enumerate(all_images_to_process):
        raw_image_path = img_info['raw_path']
        processed_save_path = img_info['target_processed_path']
        data_category = img_info['data_category']
        
        # Progress logging
        current_count = i + 1
        if (current_count - last_log_count >= log_interval) or (current_count == total_images):
            progress_pct = (current_count / total_images) * 100
            logger.info(f"Progress: {current_count}/{total_images} ({progress_pct:.1f}%) - "
                       f"Processed: {newly_processed_count} (Training: {training_processed}, OOD: {ood_processed}), "
                       f"Skipped: {already_processed_count}, Failed: {failed_to_process_count}")
            last_log_count = current_count
        
        try:
            # Check if already processed
            base_fn, ext_fn = os.path.splitext(os.path.basename(processed_save_path))
            
            if USE_OD_DETECTION_AND_CROPPING:
                # When using OD detection, check for various fallback suffixes
                potential_suffixed_paths = [
                    processed_save_path,
                    os.path.join(os.path.dirname(processed_save_path), f"{base_fn}_fb_LowThr{FALLBACK_SEG_THRESHOLD:.2f}".replace(".","p") + f"{ext_fn}"),
                    os.path.join(os.path.dirname(processed_save_path), f"{base_fn}_fb_CenterCrop{FALLBACK_CENTER_CROP_DIMENSION}{ext_fn}")
                ]
            else:
                # When not using OD detection, just use the normal path
                potential_suffixed_paths = [processed_save_path]
            
            if any(os.path.exists(p) for p in potential_suffixed_paths):
                already_processed_count += 1
                continue
            
            # Load image
            img_bgr = cv2.imread(raw_image_path)
            if img_bgr is None:
                logger.warning(f"Could not read: {os.path.basename(raw_image_path)}")
                failed_to_process_count += 1
                continue

            if USE_OD_DETECTION_AND_CROPPING:
                # OD detection and cropping logic
                od_cx, od_cy, od_diameter = None, None, None
                crop_dim = 0
                final_od_method_suffix = ""
                
                disc_mask, _ = predictor.predict(img_bgr, threshold=PRIMARY_SEG_THRESHOLD, refine_smooth=True, smoothing_factor=0.2)
                od_props = get_optic_disc_properties(disc_mask, min_contour_area_pixels=MIN_CONTOUR_AREA_PIXELS_FOR_OD)
                
                if od_props:
                    od_cx, od_cy, od_diameter, _ = od_props
                    od_primary_success_count += 1
                else:
                    disc_mask_fb, _ = predictor.predict(img_bgr, threshold=FALLBACK_SEG_THRESHOLD, refine_smooth=True, smoothing_factor=0.2)
                    od_props_fb = get_optic_disc_properties(disc_mask_fb, min_contour_area_pixels=MIN_CONTOUR_AREA_PIXELS_FOR_OD)
                    
                    if od_props_fb:
                        od_cx, od_cy, od_diameter, _ = od_props_fb
                        final_od_method_suffix = f"_fb_LowThr{FALLBACK_SEG_THRESHOLD:.2f}".replace(".","p")
                        od_fallback_threshold_success_count += 1
                    else:
                        img_h_orig, img_w_orig = img_bgr.shape[:2]
                        od_cx, od_cy, od_diameter = img_w_orig // 2, img_h_orig // 2, 0
                        crop_dim = max(FALLBACK_CENTER_CROP_DIMENSION, MIN_CROP_DIMENSION_ABSOLUTE_PIXELS)
                        final_od_method_suffix = f"_fb_CenterCrop{crop_dim}"
                        od_fallback_center_crop_used_count += 1
                
                # Calculate crop dimension and crop
                if od_diameter > 0:
                    eff_od_diam = max(od_diameter, MIN_EFFECTIVE_OD_DIAMETER_PIXELS)
                    target_crop_dim = int(CROP_FACTOR * eff_od_diam)
                    crop_dim = max(target_crop_dim, MIN_CROP_DIMENSION_ABSOLUTE_PIXELS)
                    img_h, img_w = img_bgr.shape[:2]
                    max_perm_crop = int(min(img_h, img_w) * MAX_CROP_DIMENSION_FACTOR_OF_IMG_SHORT_SIDE)
                    crop_dim = min(crop_dim, max_perm_crop)
                    crop_dim = max(crop_dim, MIN_CROP_DIMENSION_ABSOLUTE_PIXELS)
                elif crop_dim == 0:
                    logger.error(f"Failed to set crop_dim for {os.path.basename(raw_image_path)}")
                    failed_to_process_count += 1
                    continue
                
                cropped_od_region = crop_around_center(img_bgr, od_cx, od_cy, crop_dim)
                final_processed_image = apply_enhanced_preprocessing(cropped_od_region)
                
                # Save with appropriate suffix for OD detection methods
                base, ext = os.path.splitext(os.path.basename(processed_save_path))
                final_save_filename = f"{base}{final_od_method_suffix}{ext}"
                
            else:
                # Apply only enhanced preprocessing without cropping
                final_processed_image = apply_enhanced_preprocessing(img_bgr)
                
                # Use normal filename (no suffix)
                final_save_filename = os.path.basename(processed_save_path)
            
            # Save processed image
            processed_output_dir = os.path.dirname(processed_save_path)
            os.makedirs(processed_output_dir, exist_ok=True)
            final_save_path = os.path.join(processed_output_dir, final_save_filename)
            
            cv2.imwrite(final_save_path, final_processed_image)
            newly_processed_count += 1
            
            # Track by category
            if data_category == 'training':
                training_processed += 1
            else:
                ood_processed += 1
            
            # Debug logging
            if USE_OD_DETECTION_AND_CROPPING and final_od_method_suffix:
                logger.debug(f"Saved with OD method: {os.path.basename(final_save_path)}")
            else:
                logger.debug(f"Saved enhanced: {os.path.basename(final_save_path)}")
                
        except Exception as e_main_loop:
            logger.error(f"Error processing {os.path.basename(raw_image_path)}: {str(e_main_loop)[:100]}...")
            failed_to_process_count += 1

    # Final summary
    logger.info(f"\n=== PREPROCESSING COMPLETE ===")
    logger.info(f"Processing mode: {'OD Detection + Cropping + Enhancement' if USE_OD_DETECTION_AND_CROPPING else 'Enhancement Only'}")
    logger.info(f"Total images considered: {total_images}")
    logger.info(f"✓ Already processed (skipped): {already_processed_count}")
    logger.info(f"✓ Newly processed: {newly_processed_count}")
    logger.info(f"  - Training data processed: {training_processed}")
    logger.info(f"  - OOD test data processed: {ood_processed}")
    
    if USE_OD_DETECTION_AND_CROPPING and newly_processed_count > 0:
        logger.info(f"  - Primary OD detection: {od_primary_success_count}/{newly_processed_count} ({od_primary_success_count/newly_processed_count*100:.1f}%)")
        logger.info(f"  - Fallback low threshold: {od_fallback_threshold_success_count}/{newly_processed_count} ({od_fallback_threshold_success_count/newly_processed_count*100:.1f}%)")
        logger.info(f"  - Center crop fallback: {od_fallback_center_crop_used_count}/{newly_processed_count} ({od_fallback_center_crop_used_count/newly_processed_count*100:.1f}%)")
    
    logger.info(f"✗ Failed to process: {failed_to_process_count}")
    
    # Processing rate
    if newly_processed_count > 0:
        success_rate = (newly_processed_count / (newly_processed_count + failed_to_process_count)) * 100
        logger.info(f"Processing success rate: {success_rate:.1f}%")
    
    # Save comprehensive manifest
    processed_manifest_path = os.path.join(config.output_dir, "all_processed_images_manifest.csv")
    processed_manifest_data = []
    for info in all_images_to_process:
        processed_manifest_data.append({
            'raw_path': info['raw_path'],
            'processed_path': info['target_processed_path'],
            'split': info['split'],
            'dataset_source': info['dataset_source'],
            'label': info['label'],
            'data_category': info['data_category'],
            'processing_mode': 'od_crop_enhance' if USE_OD_DETECTION_AND_CROPPING else 'enhance_only'
        })
    
    pd.DataFrame(processed_manifest_data).to_csv(processed_manifest_path, index=False)
    logger.info(f"Comprehensive processing manifest saved to: {processed_manifest_path}")
    
    # Category breakdown
    manifest_df = pd.DataFrame(processed_manifest_data)
    logger.info(f"\nDataset breakdown:")
    logger.info(f"By category: {manifest_df['data_category'].value_counts().to_dict()}")
    logger.info(f"By dataset source: {manifest_df['dataset_source'].value_counts().to_dict()}")

if __name__ == '__main__':
    main_preprocess_matched()