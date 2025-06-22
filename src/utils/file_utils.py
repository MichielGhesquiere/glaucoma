import os
import re
import pandas as pd
import logging
from typing import List, Dict, Optional, Pattern
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def find_image_files_pattern(root_dir: str,
                             filename_pattern: Pattern,
                             group_names: List[str],
                             group_types: List[type] = None) -> Optional[pd.DataFrame]:
    """
    Walks through a directory and finds image files matching a regex pattern,
    extracting metadata from capture groups.

    Args:
        root_dir (str): The root directory to search within.
        filename_pattern (Pattern): A compiled regex pattern object (e.g., re.compile(...)).
                                    Must contain capture groups corresponding to group_names.
        group_names (List[str]): A list of names for the captured groups in the regex pattern.
                                 Must include 'Image Path'.
        group_types (List[type], optional): A list of types to cast the captured groups to.
                                            Must match the order of group_names (excluding 'Image Path').
                                            Defaults to str for all if None.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the extracted metadata and image paths,
                                or None if the directory doesn't exist or required inputs are invalid.
                                Returns an empty DataFrame if no files match.
    """
    if not os.path.isdir(root_dir):
        logging.error(f"Root directory not found or is not a directory: {root_dir}")
        return None
    if 'Image Path' not in group_names:
        logging.error("'Image Path' must be included in group_names.")
        return None
    if group_types is not None and len(group_types) != (len(group_names) - 1):
         logging.error(f"Length of group_types ({len(group_types)}) must match number of capture groups ({len(group_names) - 1}).")
         return None
    if filename_pattern.groups != (len(group_names) - 1):
        logging.error(f"Number of capture groups in regex ({filename_pattern.groups}) doesn't match expected number ({len(group_names) - 1}).")
        return None


    extracted_data = []
    logging.info(f"Scanning for images matching pattern in: {root_dir}")
    found_files = 0

    for current_root, _, files in os.walk(root_dir):
        for filename in files:
            match = filename_pattern.match(filename)
            if match:
                found_files += 1
                record = {}
                try:
                    # Extract captured groups
                    captured_values = match.groups()

                    # Populate record dictionary, applying type casting if specified
                    idx_type = 0
                    for i, name in enumerate(group_names):
                        if name == 'Image Path':
                             record[name] = os.path.join(current_root, filename)
                        else:
                             value = captured_values[idx_type]
                             if group_types:
                                 record[name] = group_types[idx_type](value)
                             else:
                                 record[name] = str(value) # Default to string
                             idx_type += 1

                    extracted_data.append(record)
                except (IndexError, ValueError, TypeError) as e:
                    logging.warning(f"Error processing file '{filename}' with pattern: {e}. Skipping.")
                except Exception as e:
                     logging.error(f"Unexpected error processing file '{filename}': {e}. Skipping.")


    if not extracted_data:
        logging.warning(f"No image files matching the pattern found in {root_dir}")
        return pd.DataFrame(columns=group_names) # Return empty DF with correct columns

    logging.info(f"Found {found_files} files matching the pattern.")
    image_df = pd.DataFrame(extracted_data)
    return image_df

# Add the path adjustment function (same as in training script)
def adjust_path_for_data_type(original_path: str, data_type: str) -> str:
    """
    Adjusts image paths between 'raw' and 'processed' data types.
    
    Args:
        original_path: Original image path
        data_type: 'raw' or 'processed'
    
    Returns:
        Adjusted path for the specified data type
    """
    if data_type == 'processed':
        # Convert raw -> processed
        if '/raw/' in original_path or '\\raw\\' in original_path:
            return original_path.replace('/raw/', '/processed/').replace('\\raw\\', '\\processed\\')
        else:
            # If path doesn't contain 'raw', try to infer
            parts = original_path.split(os.sep)
            if 'data' in parts:
                data_idx = parts.index('data')
                if data_idx + 1 < len(parts):
                    # Replace the part after 'data' with 'processed'
                    parts[data_idx + 1] = 'processed'
                    return os.sep.join(parts)
    elif data_type == 'raw':
        # Convert processed -> raw
        if '/processed/' in original_path or '\\processed\\' in original_path:
            return original_path.replace('/processed/', '/raw/').replace('\\processed\\', '\\raw\\')
        else:
            # If path doesn't contain 'processed', try to infer
            parts = original_path.split(os.sep)
            if 'data' in parts:
                data_idx = parts.index('data')
                if data_idx + 1 < len(parts):
                    # Replace the part after 'data' with 'raw'
                    parts[data_idx + 1] = 'raw'
                    return os.sep.join(parts)
    
    # If no conversion needed or couldn't convert, return original
    return original_path

def handle_processed_image_suffixes(image_path: str) -> str:
    """
    Handle processed images that may have suffixes from preprocessing.
    
    Args:
        image_path: Original image path
        
    Returns:
        Path to existing processed image (with suffix if needed)
    """
    if os.path.exists(image_path):
        return image_path
    
    # If original doesn't exist, try common preprocessing suffixes
    base_path, ext = os.path.splitext(image_path)
    
    # Common suffixes from preprocessing
    suffixes = [
        "_fb_LowThr0p25",  # Low threshold fallback
        "_fb_CenterCrop512",  # Center crop fallback
        "_fb_CenterCrop256",  # Different crop sizes
        "_fb_CenterCrop1024"
    ]
    
    for suffix in suffixes:
        candidate_path = f"{base_path}{suffix}{ext}"
        if os.path.exists(candidate_path):
            logger.debug(f"Found processed image with suffix: {os.path.basename(candidate_path)}")
            return candidate_path
    
    logger.warning(f"Could not find processed image for: {os.path.basename(image_path)}")
    return image_path  # Return original even if it doesn't exist

def find_best_checkpoint(checkpoint_dir: str) -> str | None:
    """
    Finds the best or latest model checkpoint in a directory.
    Priority: '*_best_model_epoch*.pth' (highest epoch),
              then '*_final_epoch*.pth' (highest epoch),
              then latest modified '*.pth'.
    """
    best_checkpoint, highest_epoch_best = None, -1
    for f_path in glob.glob(os.path.join(checkpoint_dir, "*_best_model_epoch*.pth")):
        try:
            epoch_num = int(os.path.basename(f_path).split('_epoch')[-1].split('.pth')[0])
            if epoch_num > highest_epoch_best:
                highest_epoch_best, best_checkpoint = epoch_num, f_path
        except ValueError: continue
    if best_checkpoint:
        logger.info(f"Found best model checkpoint (highest epoch): {best_checkpoint}")
        return best_checkpoint

    logger.warning(f"No '*_best_model_epoch*.pth' in {checkpoint_dir}. Looking for '*_final_epoch*.pth'.")
    final_checkpoint, highest_epoch_final = None, -1
    for f_path in glob.glob(os.path.join(checkpoint_dir, "*_final_epoch*.pth")):
        try:
            epoch_num = int(os.path.basename(f_path).split('_epoch')[-1].split('.pth')[0])
            if epoch_num > highest_epoch_final:
                highest_epoch_final, final_checkpoint = epoch_num, f_path
        except ValueError: continue
    if final_checkpoint:
        logger.info(f"Found final epoch checkpoint (highest epoch): {final_checkpoint}")
        return final_checkpoint

    logger.warning(f"No '*_final_epoch*.pth' in {checkpoint_dir}. Looking for the latest '*.pth'.")
    all_pth_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not all_pth_files:
        logger.error(f"No .pth files found in checkpoint directory: {checkpoint_dir}")
        return None
    latest_pth = max(all_pth_files, key=os.path.getmtime)
    logger.info(f"Found latest modified .pth file as fallback: {latest_pth}")
    return latest_pth
