import argparse
import os
import pandas as pd
import logging

# Make sure Python can find the src modules
import sys
# Add the project root to the Python path
# Adjust the number of '..' based on the script's location relative to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logging
from src.data.data_loading import (
    load_grape_metadata, find_grape_image_paths, merge_grape_data,
    load_smdg_metadata, load_airogs_metadata
)

def main(args):
    """Main function to run data preprocessing."""
    # Load configuration
    config = load_config(args.config)
    paths_config = config.get('paths', {})
    preprocess_config = config.get('preprocessing', {})

    # Setup logging
    log_dir = paths_config.get('log_dir', 'logs')
    setup_logging(log_level=config.get('log_level', 'INFO'), log_dir=log_dir, log_file='preprocess_data.log')
    logging.info("Starting data preprocessing script...")
    logging.info(f"Loaded configuration from: {args.config}")

    raw_data_dir = paths_config.get('raw_data_dir')
    processed_data_dir = paths_config.get('processed_data_dir')
    if not raw_data_dir or not processed_data_dir:
        logging.error("raw_data_dir or processed_data_dir not specified in config paths.")
        return
    os.makedirs(processed_data_dir, exist_ok=True)
    logging.info(f"Raw data directory: {raw_data_dir}")
    logging.info(f"Processed data directory: {processed_data_dir}")

    # --- Process GRAPE Dataset ---
    grape_config = preprocess_config.get('grape', {})
    if grape_config.get('process', False):
        logging.info("Processing GRAPE dataset...")
        try:
            metadata_path = os.path.join(raw_data_dir, grape_config.get('metadata_file', 'GRAPE/VF and clinical information.xlsx'))
            image_dir = os.path.join(raw_data_dir, grape_config.get('image_subdir', 'GRAPE/CFPs/CFPs'))
            output_file = os.path.join(processed_data_dir, grape_config.get('output_file', 'grape_merged.csv'))

            grape_meta_df = load_grape_metadata(metadata_path)
            grape_image_df = find_grape_image_paths(image_dir)
            grape_merged_df = merge_grape_data(
                grape_meta_df,
                grape_image_df,
                drop_missing_progression=grape_config.get('drop_missing_progression', True)
            )

            if grape_merged_df is not None and not grape_merged_df.empty:
                grape_merged_df.to_csv(output_file, index=False)
                logging.info(f"GRAPE processed data saved to: {output_file}")
            else:
                logging.warning("GRAPE processing resulted in an empty DataFrame. No file saved.")
        except Exception as e:
            logging.error(f"Error processing GRAPE dataset: {e}", exc_info=True)
    else:
        logging.info("Skipping GRAPE dataset processing (disabled in config).")

    # --- Process SMDG-19 Dataset ---
    smdg_config = preprocess_config.get('smdg', {})
    if smdg_config.get('process', False):
        logging.info("Processing SMDG-19 dataset...")
        try:
            metadata_path = os.path.join(raw_data_dir, smdg_config.get('metadata_file', 'SMDG-19/metadata - standardized.csv'))
            base_image_dir = os.path.join(raw_data_dir, smdg_config.get('base_subdir', 'SMDG-19'))
            output_file = os.path.join(processed_data_dir, smdg_config.get('output_file', 'smdg_cleaned.csv'))

            smdg_df = load_smdg_metadata(
                csv_path=metadata_path,
                image_dir=base_image_dir,
                # Allow overriding subfolders from config if needed
                fundus_subfolder=smdg_config.get('fundus_subfolder', 'full-fundus/full-fundus'),
                oc_subfolder=smdg_config.get('oc_subfolder', 'optic-cup/optic-cup'),
                od_subfolder=smdg_config.get('od_subfolder', 'optic-disc/optic-disc')
            )

            if smdg_df is not None and not smdg_df.empty:
                smdg_df.to_csv(output_file, index=False)
                logging.info(f"SMDG-19 processed data saved to: {output_file}")
            else:
                logging.warning("SMDG-19 processing resulted in an empty DataFrame. No file saved.")
        except Exception as e:
            logging.error(f"Error processing SMDG-19 dataset: {e}", exc_info=True)
    else:
        logging.info("Skipping SMDG-19 dataset processing (disabled in config).")

    # --- Process AIROGS Dataset ---
    airogs_config = preprocess_config.get('airogs', {})
    if airogs_config.get('process', False):
        logging.info("Processing AIROGS dataset...")
        try:
            metadata_path = os.path.join(raw_data_dir, airogs_config.get('metadata_file', 'AIROGS/train_labels.csv'))
            image_dir = os.path.join(raw_data_dir, airogs_config.get('image_subdir', 'AIROGS/img'))
            output_file = os.path.join(processed_data_dir, airogs_config.get('output_file', 'airogs_paths.csv'))

            airogs_df = load_airogs_metadata(
                csv_path=metadata_path,
                image_dir=image_dir
            )

            if airogs_df is not None and not airogs_df.empty:
                airogs_df.to_csv(output_file, index=False)
                logging.info(f"AIROGS processed data saved to: {output_file}")
            else:
                logging.warning("AIROGS processing resulted in an empty DataFrame. No file saved.")
        except Exception as e:
            logging.error(f"Error processing AIROGS dataset: {e}", exc_info=True)
    else:
        logging.info("Skipping AIROGS dataset processing (disabled in config).")

    logging.info("Data preprocessing script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Glaucoma Datasets")
    parser.add_argument('--config', type=str, required=True, help='Path to the main configuration YAML file.')
    args = parser.parse_args()
    main(args)