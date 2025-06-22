import argparse
import os
import pandas as pd
import torch
import logging

# Make sure Python can find the src modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logging
from src.models.segmentation.unet import MultiTaskUNet # Import model definition
# Import the feature building function and a placeholder preprocessor
from src.features.build_features import build_progression_features, preprocess_for_segmentation_model
from src.data.transforms import get_segmentation_transforms # Import transform used during *training*

def main(args):
    """Main function to extract features using a trained segmentation model."""
    # Load configuration
    config = load_config(args.config)
    paths_config = config.get('paths', {})
    feature_config = config.get('feature_extraction', {})
    seg_model_config = config.get('model', {}).get('segmentation', {}) # Need model params used for training

    # Setup logging
    log_dir = paths_config.get('log_dir', 'logs')
    setup_logging(log_level=config.get('log_level', 'INFO'), log_dir=log_dir, log_file='extract_features.log')
    logging.info("Starting feature extraction script...")
    logging.info(f"Loaded configuration from: {args.config}")

    # --- Setup ---
    processed_data_dir = paths_config.get('processed_data_dir')
    checkpoint_dir = paths_config.get('checkpoint_dir', 'checkpoints') # Where models are saved
    features_dir = paths_config.get('features_dir', os.path.join(processed_data_dir, 'features')) # Save features here
    os.makedirs(features_dir, exist_ok=True)

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # --- Load Data ---
    # Load the *longitudinal* data (e.g., merged GRAPE)
    data_file = feature_config.get('input_data_file', 'grape_merged.csv')
    data_path = os.path.join(processed_data_dir, data_file)
    logging.info(f"Loading longitudinal data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        # Check for required columns (adjust based on build_progression_features needs)
        required_cols = ['Subject Number', 'Laterality', 'Visit Number', 'Image Path', 'Progression Status']
        if not all(col in df.columns for col in required_cols):
             raise ValueError(f"Input data missing required columns: {required_cols}")
        logging.info(f"Loaded {len(df)} records for feature extraction.")
    except FileNotFoundError:
        logging.error(f"Input data file not found: {data_path}")
        return
    except Exception as e:
        logging.error(f"Error loading input data file {data_path}: {e}", exc_info=True)
        return

    # --- Load Trained Segmentation Model ---
    seg_model_name = seg_model_config.get('name', 'unet_segmentation')
    model_checkpoint_name = f"best_{seg_model_name}.pth" # Assumes trainer saves this way
    model_checkpoint_path = args.model_checkpoint if args.model_checkpoint else os.path.join(checkpoint_dir, model_checkpoint_name)

    if not os.path.exists(model_checkpoint_path):
        logging.error(f"Segmentation model checkpoint not found at: {model_checkpoint_path}")
        logging.error("Please provide the correct path using --model_checkpoint or ensure it exists in the default location.")
        return

    logging.info(f"Loading segmentation model from: {model_checkpoint_path}")
    try:
        # Initialize model architecture (must match the saved checkpoint)
        n_channels = seg_model_config.get('n_channels', 3)
        bilinear = seg_model_config.get('bilinear', True)
        model = MultiTaskUNet(n_channels=n_channels, bilinear=bilinear)

        checkpoint = torch.load(model_checkpoint_path, map_location=torch.device(device)) # Load to target device
        # Handle different checkpoint formats (dict vs raw state_dict)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval() # Set to evaluation mode
        logging.info("Segmentation model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading segmentation model: {e}", exc_info=True)
        return

    # --- Define Preprocessing for Feature Extraction ---
    # This should match the *validation* transform used during segmentation training
    # (usually no augmentation, just resize/ToTensor/Normalize)
    img_size = seg_model_config.get('img_size', 256)
    # IMPORTANT: Ensure this transform matches what build_progression_features expects
    # via preprocess_for_segmentation_model
    # If preprocess_for_segmentation_model handles PIL->Tensor->Norm, pass that transform
    segmentation_transform = get_segmentation_transforms(img_size=img_size) # Assuming ToTensor only for now
    # Add normalization within preprocess_fn if needed, or pass a combined transform
    logging.info(f"Using image size {img_size} for preprocessing.")


    # --- Extract Features ---
    logging.info("Starting feature extraction process...")
    try:
        # Call the function from src.features
        features_df = build_progression_features(
            df=df,
            segmentation_model=model,
            device=device,
            image_path_col='Image Path', # Adjust if column name differs
            group_by_cols=['Subject Number', 'Laterality'],
            sort_col='Visit Number',
            progression_status_col='Progression Status',
            preprocess_fn=preprocess_for_segmentation_model, # Pass the preprocessing function
            segmentation_transform=segmentation_transform, # Pass the transform it needs
            use_mock_segmentation=args.use_mock # Use mock data if flag is set
        )
    except Exception as e:
         logging.error(f"Error during build_progression_features: {e}", exc_info=True)
         return

    # --- Save Features ---
    if features_df is not None and not features_df.empty:
        output_filename = feature_config.get('output_file', 'progression_features.csv')
        output_path = os.path.join(features_dir, output_filename)
        features_df.to_csv(output_path, index=False)
        logging.info(f"Progression features saved to: {output_path} ({len(features_df)} sequences)")
    else:
        logging.warning("Feature extraction resulted in an empty DataFrame. No file saved.")

    logging.info("Feature extraction script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features using a trained segmentation model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the main configuration YAML file.')
    parser.add_argument('--model_checkpoint', type=str, help='Path to the trained segmentation model checkpoint (.pth). Overrides default path.')
    parser.add_argument('--device', type=str, help='Device to use (cuda or cpu). Overrides config if set.')
    parser.add_argument('--use_mock', action='store_true', help='Use mock segmentation instead of the model.')
    args = parser.parse_args()
    main(args)