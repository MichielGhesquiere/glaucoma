import os
import cv2
import numpy as np
import logging
from typing import Tuple
import matplotlib.pyplot as plt

# Import the functions we just added
from preprocessing import apply_color_normalization, apply_enhanced_preprocessing, apply_clahe_color

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_preprocessing_on_sample():
    """Test the enhanced preprocessing on 5 sample images"""
    
    # Configuration
    TEST_OUTPUT_DIR = r'D:\glaucoma\notebooks\test_enhanced_preprocessing_output'
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    
    # Find 5 sample raw images from different datasets
    sample_paths = []
    
    # Try to get samples from different sources
    base_dirs_to_search = [
        r'D:\glaucoma\data\raw\SMDG-19\full-fundus\full-fundus',
        r'D:\glaucoma\data\raw\Chaksu\Train\Train\1.0_Original_Fundus_Images',
        r'D:\glaucoma\data\raw\AIROGS\img'
    ]
    
    for base_dir in base_dirs_to_search:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        sample_paths.append(os.path.join(root, file))
                        if len(sample_paths) >= 5:
                            break
                if len(sample_paths) >= 5:
                    break
            if len(sample_paths) >= 5:
                break
    
    if len(sample_paths) < 5:
        logger.warning(f"Only found {len(sample_paths)} sample images")
    
    logger.info(f"Testing enhanced preprocessing on {len(sample_paths)} images:")
    for path in sample_paths:
        logger.info(f"  - {os.path.basename(path)}")
    
    # Process each sample image
    for i, img_path in enumerate(sample_paths):
        try:
            logger.info(f"Processing {i+1}/{len(sample_paths)}: {os.path.basename(img_path)}")
            
            # Load original image
            original_bgr = cv2.imread(img_path)
            if original_bgr is None:
                logger.error(f"Could not load {img_path}")
                continue
            
            # Resize for consistent comparison (optional)
            height, width = original_bgr.shape[:2]
            if max(height, width) > 1024:
                scale = 1024 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                original_bgr = cv2.resize(original_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Apply different enhancement steps
            clahe_only = apply_clahe_color(original_bgr)
            color_norm_only = apply_color_normalization(original_bgr)
            enhanced_full = apply_enhanced_preprocessing(original_bgr)
            
            # Save comparison images
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, f"{base_name}_01_original.jpg"), original_bgr)
            cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, f"{base_name}_02_clahe_only.jpg"), clahe_only)
            cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, f"{base_name}_03_color_norm_only.jpg"), color_norm_only)
            cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, f"{base_name}_04_enhanced_full.jpg"), enhanced_full)
            
            # Create a comparison grid
            create_comparison_grid(original_bgr, clahe_only, color_norm_only, enhanced_full, 
                                 os.path.join(TEST_OUTPUT_DIR, f"{base_name}_comparison.jpg"))
            
            logger.info(f"Successfully processed and saved: {base_name}")
            
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(img_path)}: {e}")
    
    logger.info(f"Test completed. Check results in: {TEST_OUTPUT_DIR}")

def create_comparison_grid(original, clahe_only, color_norm_only, enhanced_full, save_path):
    """Create a 2x2 comparison grid of the different preprocessing methods"""
    
    # Resize all images to the same size for comparison
    target_size = (512, 512)
    original_resized = cv2.resize(original, target_size)
    clahe_resized = cv2.resize(clahe_only, target_size)
    color_norm_resized = cv2.resize(color_norm_only, target_size)
    enhanced_resized = cv2.resize(enhanced_full, target_size)
    
    # Create 2x2 grid
    top_row = np.hstack([original_resized, clahe_resized])
    bottom_row = np.hstack([color_norm_resized, enhanced_resized])
    comparison_grid = np.vstack([top_row, bottom_row])
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2
    
    # Add labels to each quadrant
    cv2.putText(comparison_grid, "Original", (10, 30), font, font_scale, color, thickness)
    cv2.putText(comparison_grid, "CLAHE Only", (522, 30), font, font_scale, color, thickness)
    cv2.putText(comparison_grid, "Color Norm Only", (10, 542), font, font_scale, color, thickness)
    cv2.putText(comparison_grid, "CLAHE + Color Norm", (522, 542), font, font_scale, color, thickness)
    
    cv2.imwrite(save_path, comparison_grid)

if __name__ == '__main__':
    test_enhanced_preprocessing_on_sample()