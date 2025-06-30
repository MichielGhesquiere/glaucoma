"""
Path utilities for glaucoma classification project.

This module contains functions for adjusting paths between raw and processed data directories.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Global Configuration
PAPILLA_PREFIX = "PAPILA"
BASE_DATA_DIR_CONFIG_KEY = ""  # Set from args in main
RAW_DIR_NAME = "raw"
PROCESSED_DIR_NAME = "processed"


def adjust_path_for_data_type(current_path: str, data_type: str) -> str:
    """
    Adjusts file path from 'raw' to 'processed' data directory if data_type is 'processed'.
    """
    if data_type != "processed":
        return current_path

    norm_current_path = os.path.normpath(current_path)
    norm_base_data_dir = os.path.normpath(BASE_DATA_DIR_CONFIG_KEY)
    norm_base_raw_path = os.path.join(norm_base_data_dir, RAW_DIR_NAME)
    norm_base_processed_path = os.path.join(norm_base_data_dir, PROCESSED_DIR_NAME)

    # Case 1: Path starts with the base raw data directory
    if norm_current_path.lower().startswith(norm_base_raw_path.lower()):
        relative_part = os.path.relpath(norm_current_path, norm_base_raw_path)
        new_path = os.path.join(norm_base_processed_path, relative_part)
        logger.debug(f"Path adjustment (base replace): '{current_path}' -> '{new_path}'")
        return new_path

    # Case 2: 'raw' segment is somewhere in the path
    path_parts = norm_current_path.split(os.sep)
    try:
        raw_index = next(i for i, part in enumerate(path_parts) if part.lower() == RAW_DIR_NAME.lower())
        path_parts[raw_index] = PROCESSED_DIR_NAME
        new_path = os.sep.join(path_parts)
        logger.debug(f"Path adjustment (segment replace): '{current_path}' -> '{new_path}'")
        return new_path
    except StopIteration:
        # No 'raw' segment found
        logger.debug(f"No path adjustment needed for: '{current_path}'")
        return current_path


def set_base_data_dir(base_dir: str):
    """Set the global base data directory configuration."""
    global BASE_DATA_DIR_CONFIG_KEY
    BASE_DATA_DIR_CONFIG_KEY = base_dir
