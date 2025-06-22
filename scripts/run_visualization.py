import argparse
import os
import pandas as pd
import torch
import cv2 # Keep if needed by plotting functions
import joblib # For loading sklearn models if needed for viz
import logging
import matplotlib.pyplot as plt # Keep top-level import for plt.show() etc.

# Make sure Python can find the src modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logging
# Import all relevant plotting functions
from src.utils.plotting import (
    plot_training_history_classification,
    plot_training_history_segmentation,
    visualize_segmentation_predictions,
    visualize_subject_sequence_progression,
    visualize_glaucoma_metrics,
    plot_metrics_over_time,
    plot_roc_curve,
    plot_feature_importance,
    show_dataset_samples
)
# Import necessary data/model components if needed for specific visualizations
from src.data.datasets import GlaucomaClassificationDataset, FundusMultiTaskDataset, FundusSequenceDataset
from src.data.transforms import get_classification_val_transforms, get_segmentation_transforms, get_default_progression_transform
from src.models.segmentation.unet import MultiTaskUNet # Example if needed for segmentation func
from src.features.metrics import GlaucomaMetrics # Example if needed for metrics viz

# Placeholder for getting segmentation masks (adapt as needed)
def get_segmentation(model, device, transform):
    """Wrapper function to pass to visualize_subject_sequence_progression"""
    model.eval()
    def segment_func(image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                oc_pred, od_pred = model(img_tensor)
            oc_mask = (oc_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            od_mask = (od_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            return od_mask, oc_mask # Return disc, cup
        except Exception as e:
            logging.error(f"Error segmenting {image_path}: {e}")
            return None, None
    return segment_func


def main(args):
    """Main function to run specific visualizations."""
    # Load configuration
    config = load_config(args.config)
    paths_config = config.get('paths', {})
    viz_config = config.get('visualization', {})

    # Setup logging
    log_dir = paths_config.get('log_dir', 'logs')
    log_file = os.path.join(log_dir, f'visualization_{args.type}.log')
    setup_logging(log_level=config.get('log_level', 'INFO'), log_dir=log_dir, log_file=log_file)
    logging.info(f"Starting visualization script for type: {args.type}")
    logging.info(f"Loaded configuration from: {args.config}")

    # Setup output directory for plots
    results_dir = paths_config.get('results_dir', 'results')
    viz_output_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(viz_output_dir, exist_ok=True)
    logging.info(f"Saving visualization outputs to: {viz_output_dir}")

    # --- Select Visualization Type ---
    viz_type = args.type.lower()

    try:
        if viz_type == 'samples':
            # Visualize samples from a dataset
            dataset_name = args.dataset_name
            if not dataset_name:
                logging.error("--dataset_name is required for 'samples' visualization.")
                return
            logging.info(f"Visualizing samples for dataset: {dataset_name}")
            # Logic to load the correct dataset based on name
            # This requires more config structure or specific loading logic here
            processed_data_dir = paths_config.get('processed_data_dir')
            dataset_config = config.get('datasets', {}).get(dataset_name, {})
            data_path = os.path.join(processed_data_dir, dataset_config.get('processed_file'))
            data_type = dataset_config.get('type') # e.g., 'classification', 'segmentation'
            df = pd.read_csv(data_path)

            if data_type == 'classification':
                transform = get_classification_val_transforms() # Use simple transform
                dataset = GlaucomaClassificationDataset(df, transform=transform)
            elif data_type == 'segmentation':
                transform = get_segmentation_transforms()
                 # Adjust columns based on config/defaults
                img_col = dataset_config.get('image_col', 'fundus_path')
                oc_col = dataset_config.get('oc_mask_col', 'oc_mask_path')
                od_col = dataset_config.get('od_mask_col', 'od_mask_path')
                dataset = FundusMultiTaskDataset(df, image_col=img_col, oc_mask_col=oc_col, od_mask_col=od_col, transform=transform)
            # Add logic for progression dataset if needed
            else:
                 logging.error(f"Unsupported dataset type '{data_type}' for sample visualization.")
                 return

            save_path = os.path.join(viz_output_dir, f"{dataset_name}_samples.png")
            show_dataset_samples(dataset, num_images=args.num_samples, title=f"{dataset_name} Samples", save_path=save_path)

        elif viz_type == 'segmentation_preds':
            # Visualize segmentation predictions
            logging.info("Visualizing segmentation predictions...")
            device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
            seg_model_config = config.get('model', {}).get('segmentation', {})
            model_name = seg_model_config.get('name', 'unet_segmentation')
            checkpoint_dir = paths_config.get('checkpoint_dir', 'checkpoints')
            model_path = args.model_path if args.model_path else os.path.join(checkpoint_dir, f'best_{model_name}.pth')
            # Load model
            # ... (similar loading logic as in evaluate_segmentation) ...
            if not os.path.exists(model_path): raise FileNotFoundError(model_path)
            n_channels = seg_model_config.get('n_channels', 3)
            bilinear = seg_model_config.get('bilinear', True)
            model = MultiTaskUNet(n_channels=n_channels, bilinear=bilinear)
            checkpoint = torch.load(model_path, map_location=device)
            # ... load state dict ...
            if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
            else: model.load_state_dict(checkpoint)

            # Load validation/test data
            processed_data_dir = paths_config.get('processed_data_dir')
            data_file = viz_config.get('segmentation_data', 'smdg_cleaned.csv')
            df = pd.read_csv(os.path.join(processed_data_dir, data_file))
             # ... create FundusMultiTaskDataset and DataLoader (use val split or test split) ...
            img_size = seg_model_config.get('img_size', 256)
            transform = get_segmentation_transforms(img_size=img_size)
            # Use a small subset for visualization
            viz_df = df.sample(n=min(len(df), args.num_samples * 4), random_state=42) # Sample more to ensure batches
            viz_dataset = FundusMultiTaskDataset(viz_df.reset_index(drop=True), transform=transform)
            viz_loader = DataLoader(viz_dataset, batch_size=4, shuffle=False) # Small batch size


            save_path = os.path.join(viz_output_dir, f"{model_name}_predictions.png")
            visualize_segmentation_predictions(model, viz_loader, device, num_samples=args.num_samples, save_path=save_path)

        elif viz_type == 'progression_sequence':
             # Visualize sequence for a subject
             if not args.subject_id or not args.laterality:
                 logging.error("--subject_id and --laterality are required for 'progression_sequence' visualization.")
                 return
             logging.info(f"Visualizing progression sequence for Subject {args.subject_id}, Eye {args.laterality}")
             processed_data_dir = paths_config.get('processed_data_dir')
             data_file = viz_config.get('progression_data', 'grape_merged.csv')
             df = pd.read_csv(os.path.join(processed_data_dir, data_file))

             # Filter data for the subject/eye
             subject_data = df[(df['Subject Number'] == int(args.subject_id)) & (df['Laterality'] == args.laterality.upper())]
             subject_data = subject_data.sort_values('Visit Number')

             segmentation_func = None
             if args.segmentation_model:
                 # Load segmentation model if path provided
                 logging.info(f"Loading segmentation model for overlay: {args.segmentation_model}")
                 device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
                 seg_model_config = config.get('model', {}).get('segmentation', {}) # Assuming same model type
                 # ... (Load MultiTaskUNet model similar to above) ...
                 n_channels = seg_model_config.get('n_channels', 3)
                 bilinear = seg_model_config.get('bilinear', True)
                 seg_model = MultiTaskUNet(n_channels=n_channels, bilinear=bilinear)
                 checkpoint = torch.load(args.segmentation_model, map_location=device)
                 if 'model_state_dict' in checkpoint: seg_model.load_state_dict(checkpoint['model_state_dict'])
                 else: seg_model.load_state_dict(checkpoint)
                 seg_model.to(device)

                 img_size = seg_model_config.get('img_size', 256)
                 transform = get_segmentation_transforms(img_size=img_size) # Use basic transform
                 segmentation_func = get_segmentation(seg_model, device, transform) # Create the wrapper

             save_path = os.path.join(viz_output_dir, f"subject_{args.subject_id}_{args.laterality}_sequence.png")
             visualize_subject_sequence_progression(
                 df_subject=subject_data,
                 subject_id=args.subject_id,
                 laterality=args.laterality.upper(),
                 segmentation_func=segmentation_func,
                 save_path=save_path
             )

        # Add more visualization types as needed (e.g., 'metrics_over_time', 'roc_curve', 'feature_importance')
        # These would load results files (JSON, CSV) or model files and call the respective plotting functions.

        else:
            logging.error(f"Unsupported visualization type: {viz_type}")

    except FileNotFoundError as e:
         logging.error(f"File not found during visualization: {e}")
    except Exception as e:
        logging.error(f"An error occurred during visualization '{viz_type}': {e}", exc_info=True)


    logging.info(f"Visualization script for '{args.type}' finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Glaucoma Project Visualizations")
    parser.add_argument('--config', type=str, required=True, help='Path to the main configuration YAML file.')
    parser.add_argument('--type', type=str, required=True,
                        choices=['samples', 'segmentation_preds', 'progression_sequence', # Add more types here
                                 # 'metrics_over_time', 'roc_curve', 'feature_importance'
                                 ],
                        help='Type of visualization to generate.')
    # --- Args for specific visualizations ---
    parser.add_argument('--dataset_name', type=str, help="Name of the dataset section in config (for 'samples').")
    parser.add_argument('--num_samples', type=int, default=5, help="Number of samples/batches to visualize.")
    parser.add_argument('--model_path', type=str, help="Path to the model checkpoint (for 'segmentation_preds').")
    parser.add_argument('--subject_id', type=str, help="Subject ID (for 'progression_sequence').")
    parser.add_argument('--laterality', type=str, help="Eye laterality OD/OS (for 'progression_sequence').")
    parser.add_argument('--segmentation_model', type=str, help="Path to segmentation model for overlays (for 'progression_sequence').")
    # Add args for other plot types if needed (e.g., path to results JSON)
    parser.add_argument('--device', type=str, help='Device for PyTorch models (cuda or cpu).')

    args = parser.parse_args()
    main(args)