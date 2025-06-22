import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import cv2
import os
import logging
from PIL import Image # Needed for dataset sample visualization
from sklearn.metrics import roc_curve, auc
from typing import Dict, Any, Optional, Tuple
from ..models.segmentation.unet import OpticDiscCupPredictor 
from ..features.build_features import GlaucomaMetrics
import traceback
from collections import defaultdict
import random
from torchvision import transforms
import scipy.stats as stats



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sns.set_theme(style="whitegrid") # Set a default theme

# === Training History Plotting ===

logger = logging.getLogger(__name__)

def plot_training_history(history, save_path=None):
    """Plots training and validation loss and accuracy."""
    if not history or not history.get('train_loss'):
        logger.warning("No training history found to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    epochs_run = len(history['train_loss'])
    epochs = range(1, epochs_run + 1)

    # Plot losses
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
    if history.get('val_loss'):
        # Filter out potential None values
        val_loss_epochs = [i + 1 for i, v in enumerate(history['val_loss']) if v is not None]
        val_loss_values = [v for v in history['val_loss'] if v is not None]
        if val_loss_values:
            ax1.plot(val_loss_epochs, val_loss_values, 'ro-', label='Val Loss')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(epochs, history['train_acc'], 'bo-', label='Train Acc')
    if history.get('val_acc'):
        val_acc_epochs = [i + 1 for i, v in enumerate(history['val_acc']) if v is not None]
        val_acc_values = [v for v in history['val_acc'] if v is not None]
        if val_acc_values:
             ax2.plot(val_acc_epochs, val_acc_values, 'ro-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.set_title('Training and Validation Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving training plot to {save_path}: {e}")
    #plt.show()
    plt.close(fig) # Close the figure after showing/saving


def plot_training_history_segmentation(history: dict, save_path: str = None):
    """
    Plots training and validation losses for multi-task segmentation.

    Args:
        history (dict): Dictionary containing lists/arrays for keys:
                        'train_loss', 'val_loss', 'train_oc_loss', 'val_oc_loss',
                        'train_od_loss', 'val_od_loss'.
        save_path (str, optional): If provided, saves the plot to this path. Defaults to None.
    """
    required_keys = ['train_loss', 'val_loss', 'train_oc_loss', 'val_oc_loss', 'train_od_loss', 'val_od_loss']
    if not all(k in history for k in required_keys):
        logging.error("History dictionary is missing required keys for segmentation plotting.")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 12))

    # Plot combined losses
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history['train_loss'], 'o-', label='Combined Train Loss')
    plt.plot(epochs, history['val_loss'], 's-', label='Combined Val Loss')
    plt.ylabel('Combined Loss')
    plt.legend()
    plt.title('Combined Training and Validation Loss')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot task-specific losses
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history['train_oc_loss'], 'o-', label='OC Train Loss', color='tab:blue')
    plt.plot(epochs, history['val_oc_loss'], 's-', label='OC Val Loss', color='tab:cyan')
    plt.plot(epochs, history['train_od_loss'], 'o--', label='OD Train Loss', color='tab:orange')
    plt.plot(epochs, history['val_od_loss'], 's--', label='OD Val Loss', color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('Task-Specific Loss')
    plt.legend()
    plt.title('Task-Specific Training and Validation Loss (OC vs OD)')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Segmentation training history plot saved to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save segmentation training history plot to {save_path}: {e}")

    #plt.show()


# === Prediction Visualization ===

def visualize_segmentation_predictions(model: torch.nn.Module,
                                       dataloader: torch.utils.data.DataLoader,
                                       device: str,
                                       num_samples: int = 5,
                                       save_path: str = None):
    """
    Visualizes multi-task segmentation predictions against ground truth.

    Args:
        model: Trained segmentation model (e.g., MultiTaskUNet).
        dataloader: DataLoader providing batches of (image, oc_mask, od_mask).
        device (str): Device to run inference on ('cuda' or 'cpu').
        num_samples (int): Number of samples (batches) to visualize.
        save_path (str, optional): If provided, saves the plot. Defaults to None.
    """
    model.eval()
    model.to(device)
    samples_shown = 0
    num_cols = 5 # Image, GT OC, Pred OC, GT OD, Pred OD

    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols * 3, num_samples * 3))
    if num_samples == 1: # Handle case where axes is not a 2D array
         axes = axes[np.newaxis, :]

    fig.suptitle("Segmentation Predictions vs Ground Truth", fontsize=16)

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if i >= num_samples:
                break

            try:
                images, oc_masks, od_masks = batch_data
                if images is None: continue # Skip if batch is problematic

                images = images.to(device)
                # Masks might already be on device if dataloader pre-fetches
                oc_masks = oc_masks.to(device)
                od_masks = od_masks.to(device)

                oc_preds, od_preds = model(images)

                # Move tensors to CPU for visualization
                images_cpu = images.cpu()
                oc_masks_cpu = oc_masks.cpu()
                od_masks_cpu = od_masks.cpu()
                oc_preds_cpu = oc_preds.cpu()
                od_preds_cpu = od_preds.cpu()

                # Display the first image in the current batch
                img_display = images_cpu[0].permute(1, 2, 0).numpy()
                # Un-normalize if needed for display (requires knowing mean/std)
                # Assuming ToTensor scaled to [0, 1] - check your transforms
                img_display = np.clip(img_display, 0, 1)

                ax = axes[i, 0]
                ax.imshow(img_display)
                ax.set_title(f'Image {i+1}')
                ax.axis('off')

                ax = axes[i, 1]
                ax.imshow(oc_masks_cpu[0].squeeze().numpy(), cmap='gray')
                ax.set_title('GT Optic Cup')
                ax.axis('off')

                ax = axes[i, 2]
                ax.imshow(oc_preds_cpu[0].squeeze().numpy(), cmap='gray')
                ax.set_title('Pred Optic Cup')
                ax.axis('off')

                ax = axes[i, 3]
                ax.imshow(od_masks_cpu[0].squeeze().numpy(), cmap='gray')
                ax.set_title('GT Optic Disc')
                ax.axis('off')

                ax = axes[i, 4]
                ax.imshow(od_preds_cpu[0].squeeze().numpy(), cmap='gray')
                ax.set_title('Pred Optic Disc')
                ax.axis('off')

            except Exception as e:
                logging.error(f"Error visualizing batch {i}: {e}")
                # Optionally clear the row if error occurs
                for j in range(num_cols): axes[i, j].axis('off')
                axes[i, 0].set_title(f"Error Batch {i+1}")


    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Segmentation predictions plot saved to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save segmentation predictions plot to {save_path}: {e}")

    plt.show()


# === Progression / Longitudinal Visualization ===

def visualize_subject_sequence_with_metrics(
    subject_number: int,
    df: pd.DataFrame,
    laterality: str,
    cup_disc_model: Optional[OpticDiscCupPredictor] = None,
    smooth_predictions: bool = False,
) -> Optional[Tuple[plt.Figure, Optional[plt.Figure], Optional[Dict[str, Any]]]]:
    """
    Visualizes image sequence for a subject with segmentations and metrics plots.
    Uses tight_layout for better automatic spacing and title visibility.

    Args:
        subject_number: The subject ID to visualize.
        df: DataFrame containing the dataset metadata.
        laterality: The eye laterality ('OD' or 'OS') to filter by.
        cup_disc_model: An instantiated segmentation model predictor.

    Returns:
        A tuple containing: (fig_sequence, fig_metrics, progression_results)
        Returns None if visualization cannot be completed.
    """
    # --- Input Validation (same as before) ---
    if cup_disc_model is None: print("Error (visualize): No model provided."); return None
    if not isinstance(df, pd.DataFrame) or df.empty: print("Error (visualize): Invalid DataFrame."); return None
    required_cols = ['Subject Number', 'Laterality', 'Visit Number', 'Image Path', 'Progression Status']
    if not all(col in df.columns for col in required_cols): print(f"Error (visualize): Missing columns: {required_cols}."); return None

    try:
        # --- Data Preparation (same as before) ---
        subject_data = df[(df['Subject Number'] == subject_number) & (df['Laterality'] == laterality)].copy().sort_values('Visit Number')
        if subject_data.empty: print(f"Info (visualize): No data for subject {subject_number}, {laterality}."); return None

        # --- Initialization (same as before) ---
        images, zoomed_images, zoomed_segmentations = [], [], []
        visit_numbers = subject_data['Visit Number'].tolist()
        metrics_list = [{} for _ in range(len(subject_data))]
        processed_indices = []
        metrics_calculator = GlaucomaMetrics() # Assuming GlaucomaMetrics class is defined
        print(f"\nVisualizing Subject {subject_number} - {laterality} ({len(subject_data)} visits)...")

        # --- Process Each Visit (same logic as before) ---
        for i, (row_idx, row) in enumerate(subject_data.iterrows()):
            visit_num = row['Visit Number']
            try:
                # ... (image loading, segmentation, metrics extraction logic remains the same) ...
                image = cv2.imread(row['Image Path'])
                if image is None: print(f"  Warn (Visit {visit_num}): Failed load image."); continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                try: disc_mask, cup_mask = cup_disc_model.predict(image, refine_smooth=smooth_predictions)
                except Exception as se: print(f"  Warn (Visit {visit_num}): Seg error: {se}"); continue
                if disc_mask is None or cup_mask is None or np.sum(disc_mask) == 0: print(f"  Warn (Visit {visit_num}): Seg failed/empty."); continue

                metrics = metrics_calculator.extract_metrics(disc_mask, cup_mask)
                if not metrics: print(f"  Warn (Visit {visit_num}): Metrics failed."); continue

                metrics_list[i] = metrics; processed_indices.append(i); images.append(image_rgb)

                # Zoomed views (same logic)
                disc_props = metrics_calculator._get_main_region_props(disc_mask)
                if disc_props:
                    center=disc_props.centroid; radius=np.sqrt(disc_props.area/np.pi); padding=int(radius*1.5)
                    y_min=max(0,int(center[0]-padding)); y_max=min(image_rgb.shape[0],int(center[0]+padding))
                    x_min=max(0,int(center[1]-padding)); x_max=min(image_rgb.shape[1],int(center[1]+padding))
                    if y_min < y_max and x_min < x_max:
                        zoomed_raw = image_rgb[y_min:y_max, x_min:x_max].copy()
                        overlay = zoomed_raw.copy(); mask_vis = np.zeros_like(zoomed_raw)
                        mask_vis[disc_mask[y_min:y_max, x_min:x_max]>0]=[0,255,0]
                        mask_vis[cup_mask[y_min:y_max, x_min:x_max]>0]=[255,0,0]
                        zoomed_seg = cv2.addWeighted(overlay, 0.7, mask_vis, 0.3, 0)
                        zoomed_images.append(zoomed_raw); zoomed_segmentations.append(zoomed_seg)
                    else: zoomed_images.append(image_rgb.copy()); zoomed_segmentations.append(image_rgb.copy())
                else: zoomed_images.append(image_rgb.copy()); zoomed_segmentations.append(image_rgb.copy())


            except Exception as visit_e: print(f"  Error Visit {visit_num}: {visit_e}"); continue

        # --- Filter results & Check ---
        final_metrics = [metrics_list[i] for i in processed_indices]
        final_visits = [visit_numbers[i] for i in processed_indices]
        if not images: print("Info (visualize): No visits processed successfully."); return None

        # --- Plotting Sequence Figure ---
        num_valid_visits = len(images)
        # Adjust figsize: Tweak width factor if columns are still too far apart
        fig_width = max(8, 2.0 * num_valid_visits) # Smaller factor for less width
        fig_width = min(fig_width, 18) # Max width cap
        fig_height = 8
        fig_sequence, axes = plt.subplots(3, num_valid_visits, figsize=(fig_width, fig_height), squeeze=False)
        fig_sequence.set_facecolor('white')

        # Plot images onto axes
        for ax_idx, img, visit in zip(range(num_valid_visits), images, final_visits):
            axes[0, ax_idx].imshow(img)
            axes[0, ax_idx].set_title(f'Visit {visit}', pad=3, fontsize=9) # Smaller font
            axes[0, ax_idx].axis('off')
        axes[0, 0].set_ylabel('Original', fontsize=9)

        for ax_idx, img in enumerate(zoomed_images):
            axes[1, ax_idx].imshow(img); axes[1, ax_idx].axis('off')
        axes[1, 0].set_ylabel('Zoomed Region', fontsize=9)

        for ax_idx, img in enumerate(zoomed_segmentations):
            axes[2, ax_idx].imshow(img); axes[2, ax_idx].axis('off')
        axes[2, 0].set_ylabel('Segmentation', fontsize=9)

        # --- Progression Analysis & Metrics Plot (same as before) ---
        fig_metrics, progression_results, assessment_text = None, None, "N/A (< 2 visits)"
        if len(final_metrics) >= 2:
            progression_results = metrics_calculator.analyze_progression(final_metrics)
            if progression_results and 'error' not in progression_results:
                assessment = progression_results.get('progression_likely', False)
                assessment_text = "Progression detected" if assessment else "No progression detected"
                # Assuming plot_metrics_over_time is part of GlaucomaMetrics
                fig_metrics = metrics_calculator.plot_metrics_over_time(final_metrics, final_visits)
            elif progression_results: assessment_text = f"Prog. Failed: {progression_results.get('error', 'Unknown')}"
            else: assessment_text = "Prog. Failed (Unknown)"


        # --- Add Title and Adjust Layout for Sequence Figure ---
        gt_status = subject_data['Progression Status'].iloc[0]
        visits_str = ", ".join(map(str, final_visits))

        # Add the main title FIRST, then adjust layout
        title_text = (f'Subject {subject_number} - {laterality} (Visits: {visits_str})\n'
                      f'Clinical Status: {gt_status} - Model Assessment: {assessment_text}')
        print(f"  Title: {title_text}") # For debugging
        fig_sequence.suptitle(title_text, fontsize=11)

        # Use tight_layout AFTER adding the title.
        # The rect parameter leaves space: [left, bottom, right, top]
        # Reduce top slightly from 1.0 to guarantee space for the suptitle.
        try:
            fig_sequence.tight_layout(rect=[0, 0.03, 1, 0.93]) # Leave 7% space at the top
        except ValueError:
             # tight_layout can sometimes fail with very specific arrangements
             print("Warning: tight_layout failed, attempting subplots_adjust fallback.")
             fig_sequence.subplots_adjust(top=0.88, wspace=0.05, hspace=0.1)


        plt.show(fig_sequence)

        return fig_sequence, fig_metrics, progression_results

    except Exception as e:
        print(f"Critical Error visualizing subject {subject_number}, {laterality}: {e}")
        traceback.print_exc()
        if 'fig_sequence' in locals() and plt.fignum_exists(fig_sequence.number):
            plt.close(fig_sequence)
        return None

# === Sklearn Model Evaluation Plotting ===

def plot_roc_curve(y_true, y_prob, title: str = 'Receiver Operating Characteristic (ROC) Curve', save_path: str = None):
    """
    Plots the ROC curve and calculates the AUC.

    Args:
        y_true: True binary labels.
        y_prob: Probabilities of the positive class.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance Level')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"ROC curve plot saved to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save ROC curve plot to {save_path}: {e}")

    #plt.show()


def plot_feature_importance(feature_importance_df: pd.DataFrame,
                            top_n: int = 15,
                            title: str = 'Feature Importance',
                            save_path: str = None):
    """
    Plots feature importances from a trained model (e.g., RandomForest).

    Args:
        feature_importance_df (pd.DataFrame): DataFrame with columns 'Feature' and 'Importance',
                                             sorted by importance descending.
        top_n (int): Number of top features to display. Defaults to 15.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    if feature_importance_df is None or feature_importance_df.empty:
        logging.warning("Feature importance data is empty. Cannot plot.")
        return
    if not all(col in feature_importance_df.columns for col in ['Feature', 'Importance']):
         logging.error("Feature importance DataFrame missing 'Feature' or 'Importance' column.")
         return

    # Ensure sorted
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, max(6, top_n * 0.4))) # Adjust height based on N
    sns.barplot(data=feature_importance_df.head(top_n), y='Feature', x='Importance', palette='viridis')
    plt.title(f'{title} (Top {top_n})')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Name')
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Feature importance plot saved to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save feature importance plot to {save_path}: {e}")

    #plt.show()


# === General Utilities ===

def show_dataset_samples(dataset: torch.utils.data.Dataset, num_images: int = 5, title: str = "Dataset Samples", save_path: str = None):
    """
    Displays a few sample images and labels from a PyTorch Dataset.

    Args:
        dataset: The PyTorch dataset instance.
        num_images (int): Number of random samples to display. Defaults to 5.
        title (str): Title for the overall plot. Defaults to "Dataset Samples".
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    if len(dataset) == 0:
        logging.warning("Dataset is empty, cannot show samples.")
        return

    actual_num_images = min(num_images, len(dataset))
    indices = np.random.choice(len(dataset), actual_num_images, replace=False)

    fig, axes = plt.subplots(1, actual_num_images, figsize=(actual_num_images * 3, 4))
    if actual_num_images == 1: # Handle single image case
         axes = [axes]

    fig.suptitle(title, fontsize=14)

    for i, idx in enumerate(indices):
        try:
            # Attempt to get item - handle different dataset return types
            item = dataset[idx]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                image, label = item[0], item[1]
                # Handle label tensor/numpy array/scalar
                if isinstance(label, torch.Tensor): label = label.item()
                label_str = f'Label: {label}'
            elif isinstance(item, torch.Tensor): # Assume item is just the image
                image = item
                label_str = f'Index: {idx}'
            else:
                 logging.warning(f"Unexpected item format at index {idx}. Skipping.")
                 axes[i].set_title(f'Index {idx}\n(Error)')
                 axes[i].axis('off')
                 continue

            # Convert image tensor to numpy (H, W, C) for display
            if isinstance(image, torch.Tensor):
                 img_np = image.permute(1, 2, 0).cpu().numpy()
            elif isinstance(image, Image.Image): # Handle PIL Image
                 img_np = np.array(image)
            else:
                 logging.warning(f"Unexpected image type {type(image)} at index {idx}. Skipping.")
                 axes[i].set_title(f'Index {idx}\n(Format Error)')
                 axes[i].axis('off')
                 continue


            # Normalize image for display if needed (e.g., un-normalize or scale to 0-1)
            # Simple scaling assuming input is roughly standard range after transforms
            img_display = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6) # Add epsilon for stability
            img_display = np.clip(img_display, 0, 1)

            axes[i].imshow(img_display)
            axes[i].axis('off')
            axes[i].set_title(label_str)

        except Exception as e:
            logging.error(f"Error displaying sample at index {idx}: {e}")
            axes[i].set_title(f'Index {idx}\n(Error)')
            axes[i].axis('off')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Dataset samples plot saved to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save dataset samples plot to {save_path}: {e}")

    plt.show()

def show_samples(dataset, num_samples=5, title="Dataset Samples", mean=None, std=None):
    """
    Displays sample images from a dataset that returns (image, label, attributes_dict).
    """
    if dataset is None or len(dataset) == 0:
        logger.warning(f"Cannot show samples for {title}: Dataset is empty or None.")
        return

    # Determine normalization parameters (use defaults if not provided)
    # Using ImageNet defaults as fallback
    mean = mean if mean is not None else [0.485, 0.456, 0.406]
    std = std if std is not None else [0.229, 0.224, 0.225]

    # Ensure dataset length is sufficient before sampling
    actual_num_samples = min(num_samples, len(dataset))
    if actual_num_samples <= 0:
        logger.warning(f"Dataset for '{title}' has length 0 or less. Cannot show samples.")
        # No need to create a figure if no samples can be shown
        return

    # Create figure and axes only if samples can be shown
    fig, axes = plt.subplots(1, actual_num_samples, figsize=(actual_num_samples * 3, 4))
    # Ensure axes is always iterable, even if actual_num_samples is 1
    if actual_num_samples == 1:
        axes = [axes]

    indices = random.sample(range(len(dataset)), actual_num_samples)

    valid_samples_shown = 0
    plotted_indices = [] # Keep track of which axes are used successfully

    for i, idx in enumerate(indices):
        # Assign axis. i should always be less than actual_num_samples here.
        ax = axes[i]
        try:
            sample = dataset[idx]
            if sample is None:
                logger.debug(f"Skipping sample display for index {idx}: dataset[{idx}] returned None.")
                ax.set_title("Load Error")
                ax.axis('off')
                continue

            # --- *** SIMPLIFIED UNPACKING *** ---
            # Always expect 3 items: img, label, attributes_dict
            if not isinstance(sample, (list, tuple)) or len(sample) != 3:
                logger.error(f"Unexpected sample format at index {idx}. Expected 3 items (img, lbl, attrs), got {len(sample) if isinstance(sample, (list, tuple)) else type(sample)}. Skipping.")
                ax.set_title("Format Error")
                ax.axis('off')
                continue

            img, label, attributes_dict = sample
            # ------------------------------------

            # --- Basic Display Logic ---
            # Inverse normalize for display
            # Ensure img is a Tensor before normalization
            if not isinstance(img, torch.Tensor):
                 logger.error(f"Image at index {idx} is not a Tensor (type: {type(img)}). Skipping display.")
                 ax.set_title("Img Type Error")
                 ax.axis('off')
                 continue

            inv_normalize = transforms.Normalize(
                mean=[-m/s for m, s in zip(mean, std)],
                std=[1/s for s in std]
            )
            # Ensure img is on CPU before numpy conversion
            img_display = inv_normalize(img.cpu()).permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)

            ax.imshow(img_display)

            # --- Title Generation (Simplified) ---
            title_str = f"Label: {label}" # Basic title
            # Optional: Add specific attributes if they exist in the dict
            if isinstance(attributes_dict, dict):
                 age = attributes_dict.get('age')
                 eye = attributes_dict.get('eye')
                 camera = attributes_dict.get('camera')
                 if age is not None: title_str += f"\nAge: {age:.1f}" if isinstance(age, (float, int)) else f"\nAge: {age}"
                 if eye is not None: title_str += f", Eye: {'OD' if eye==0 else ('OS' if eye==1 else eye)}"
                 if camera is not None: title_str += f"\nCam: {camera}"
            ax.set_title(title_str)
            # --- End Title Generation ---

            ax.axis('off')
            valid_samples_shown += 1
            plotted_indices.append(i) # Mark this axis as used successfully

        except Exception as e:
            # Log the full traceback for debugging
            logger.error(f"Error processing or displaying sample at index {idx}: {e}", exc_info=True)
            ax.set_title("Display Error") # Show error on the specific subplot
            ax.axis('off')

    # --- Clean up unused axes ---
    if valid_samples_shown < actual_num_samples:
        logger.warning(f"Displayed {valid_samples_shown}/{actual_num_samples} requested samples for '{title}'.")
        all_axes_indices = set(range(actual_num_samples))
        used_axes_indices = set(plotted_indices)
        unused_axes_indices = all_axes_indices - used_axes_indices
        for unused_idx in unused_axes_indices:
            # Safety check if axes array was smaller than num_samples (shouldn't happen with check at start)
            if unused_idx < len(axes):
                # Keep axes with errors visible, hide only completely unused ones (those never touched)
                if axes[unused_idx].get_title() == "": # Check if title is still empty
                    try:
                        fig.delaxes(axes[unused_idx])
                    except Exception as del_e:
                        logger.warning(f"Could not delete unused axis {unused_idx}: {del_e}")

    if valid_samples_shown == 0:
        logger.warning(f"Could not display ANY valid samples for {title}.")
        plt.close(fig) # Close the figure if nothing useful was plotted
        return

    plt.suptitle(title)
    # Adjust layout based on potentially fewer axes if some had errors
    # Use valid_samples_shown for figsize calculation
    fig.set_size_inches(valid_samples_shown * 3, 4) if valid_samples_shown > 0 else None
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()


def plot_metric_comparison(metrics_dict, metric_names, results_dir, filename='metrics_comparison_bar_chart.png'):
    """Plots a bar chart comparing specified metrics across subgroups."""
    logger.info(f"Plotting metric comparison for: {metric_names}")

    # Filter out groups with None values for the metrics being plotted
    valid_keys = [k for k, v in metrics_dict.items() if v is not None and all(v.get(m) is not None for m in metric_names)]
    if not valid_keys:
        logger.warning("No valid data to plot for metric comparison bar chart.")
        return

    plot_data = {k: {m: metrics_dict[k][m] for m in metric_names} for k in valid_keys}
    df = pd.DataFrame(plot_data).T # Transpose, already selected metrics
    df = df.reindex(valid_keys) # Keep original order

    num_metrics = len(df.columns)
    num_groups = len(df)
    if num_groups == 0 or num_metrics == 0:
         logger.warning("No data available for metric comparison plot after filtering.")
         return

    bar_width = 0.8 / num_metrics
    index = np.arange(num_groups)

    fig, ax = plt.subplots(figsize=(max(8, num_groups * 1.5), 6))

    all_bars = []
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num_metrics))

    for i, metric in enumerate(df.columns):
        means = df[metric].astype(float) # Ensure numeric for plotting
        bar_positions = index + i * bar_width - (num_metrics - 1) * bar_width / 2
        bars = ax.bar(bar_positions, means, bar_width, label=metric, color=colors[i])
        all_bars.append(bars)

    # Add labels
    for bars in all_bars:
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8, rotation=0)

    ax.set_ylabel('Rate / Score')
    ax.set_title('Performance Metrics Comparison by Subgroup')
    ax.set_xticks(index)
    ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.legend(title="Metric", loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(1.05, df.max().max() * 1.15)) # Dynamic ylim with buffer

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plot_path = os.path.join(results_dir, filename)
    try:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        logger.info(f"Metric comparison bar chart saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save metric comparison bar chart: {e}")
    #plt.show()
    plt.close(fig)


def plot_tpr_fpr_parity(metrics_dict, results_dir, filename='tpr_fpr_parity_scatter.png'):
    """Plots a scatter plot of TPR vs FPR for each subgroup (Equalized Odds viz)."""
    logger.info("Plotting TPR vs FPR parity...")

    # Filter groups with valid TPR/FPR
    plot_data = {k: v for k, v in metrics_dict.items() if v is not None and v.get('TPR') is not None and v.get('FPR') is not None}
    if not plot_data:
        logger.warning("No valid data (TPR/FPR) to plot for parity scatter plot.")
        return

    labels = list(plot_data.keys())
    fprs = np.array([plot_data[k]['FPR'] for k in labels], dtype=float)
    tprs = np.array([plot_data[k]['TPR'] for k in labels], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(labels)))

    scatter = ax.scatter(fprs, tprs, s=100, alpha=0.7, c=colors, zorder=2)

    # Annotate points
    for i, label in enumerate(labels):
        ax.annotate(label, (fprs[i] + 0.01, tprs[i]), fontsize=9)

    # Ideal point and random guess line
    ax.scatter(0, 1, marker='*', s=150, color='gold', label='Ideal Point (0, 1)', zorder=3)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC=0.5)', alpha=0.5)

    # Highlight Overall point if present
    if 'Overall' in plot_data:
        try:
             overall_idx = labels.index('Overall')
             ax.scatter(fprs[overall_idx], tprs[overall_idx], marker='X', s=150, color='red', label='Overall Performance', zorder=3, edgecolors='black')
        except ValueError:
             pass # Overall not found

    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Equalized Odds Check: TPR vs FPR by Subgroup')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    plot_path = os.path.join(results_dir, filename)
    try:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        logger.info(f"TPR vs FPR parity plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save TPR vs FPR parity plot: {e}")
    #plt.show()
    plt.close(fig)


def plot_subgroup_roc_curves(results_df, subgroup_cols, subgroup_metrics, overall_roc_data, results_dir,
                                     filename='roc_curves_subgroups.png', min_samples_per_group=10): 
    """Plots ROC curves for the overall dataset and each specified subgroup (Revised based on user code)."""
    logger.info("Plotting subgroup ROC curves...")
    if results_df is None or results_df.empty:
        logger.warning("Cannot plot subgroup ROCs: Input DataFrame is empty or None.")
        return

    required_cols = ['label', 'probability'] + subgroup_cols
    if not all(col in results_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in results_df.columns]
        logger.error(f"Cannot plot subgroup ROCs: Missing required columns: {missing}")
        return
    if results_df[['label', 'probability']].isnull().any().any():
        logger.warning("NaN values found in 'label' or 'probability' columns. ROC calculation might be affected or fail.")
        # Optional: Handle NaNs, e.g., results_df = results_df.dropna(subset=['label', 'probability'])


    fig, ax = plt.subplots(figsize=(10, 8))

    # --- 1. Plot Overall ROC ---
    if overall_roc_data and isinstance(overall_roc_data.get('auc'), (float, np.number)) and \
       overall_roc_data.get('fpr') is not None and overall_roc_data.get('tpr') is not None:
        ax.plot(overall_roc_data['fpr'], overall_roc_data['tpr'], color='black', lw=3, linestyle='-',
                 label=f"Overall (AUC = {overall_roc_data['auc']:.3f})") # Added 'Overall' text
    else:
         logger.info("Overall ROC data not plotted (missing or invalid AUC/FPR/TPR).")

    # --- 2. Plot Subgroup ROCs ---
    # Count valid subgroups for color mapping
    num_subgroups_to_plot = 0
    subgroup_keys_to_plot = []
    for group_col in subgroup_cols:
         if group_col in results_df.columns:
             valid_groups = results_df[group_col].dropna().unique()
             for group_val in valid_groups:
                  group_key = f"{group_col}_{str(group_val)}"
                  sub_df = results_df.loc[results_df[group_col] == group_val]
                  if len(sub_df) >= min_samples_per_group: # Check min samples early
                      subgroup_keys_to_plot.append((group_col, group_val, group_key))
                      num_subgroups_to_plot += 1
                  elif not sub_df.empty:
                       logger.debug(f"Skipping ROC for {group_key}: Too few samples ({len(sub_df)} < {min_samples_per_group})")


    if num_subgroups_to_plot > 0:
        # colors = plt.cm.viridis(np.linspace(0, 1, num_subgroups_to_plot)) # Viridis is fine
        # Alternative palette:
        try:
             import seaborn as sns
             colors = sns.color_palette("husl", num_subgroups_to_plot)
        except ImportError:
             colors = plt.cm.viridis(np.linspace(0, 1, num_subgroups_to_plot))
    else:
        colors = []
        logger.warning("No subgroups found meeting criteria to plot ROC curves.")

    color_idx = 0
    plotted_count = 0
    for group_col, group_val, group_key in subgroup_keys_to_plot:
        mask = results_df[group_col] == group_val
        sub_df = results_df.loc[mask]

        # Already checked min samples, but double-check empty just in case
        if sub_df.empty: continue

        sub_labels = sub_df['label'].values
        sub_probs = sub_df['probability'].values

        # Check for NaNs within the subgroup (more specific check)
        if np.isnan(sub_labels).any() or np.isnan(sub_probs).any():
            logger.warning(f"Skipping ROC for {group_key}: Contains NaN values in labels or probabilities.")
            continue

        # Get pre-calculated AUC
        group_metrics = subgroup_metrics.get(group_key)
        auc_sub = group_metrics.get('AUC') if group_metrics and isinstance(group_metrics.get('AUC'), (float, np.number)) else None

        # --- THE CORRECT CHECK ---
        unique_labels = np.unique(sub_labels)
        if len(unique_labels) > 1: # Check if more than one class exists
            if auc_sub is not None: # Check if we have a valid pre-calculated AUC
                 try:
                     fpr_sub, tpr_sub, _ = roc_curve(sub_labels, sub_probs)
                     current_color = colors[color_idx % len(colors)] if colors else 'blue' # Fallback color
                     label_text = f"{group_key.replace('_', ' ')} (AUC = {auc_sub:.3f}, n={len(sub_df)})" # Nicer label
                     ax.plot(fpr_sub, tpr_sub, color=current_color, lw=1.5, linestyle='--', alpha=0.9,
                             label=label_text)
                     color_idx += 1
                     plotted_count += 1
                 except Exception as e:
                     logger.warning(f"Could not calculate or plot ROC for {group_key}: {e}")
            else:
                # Have both classes, but no valid pre-calculated AUC
                 logger.debug(f"Skipping ROC plot for {group_key}: Both classes present, but pre-calculated AUC is missing or invalid.")
                 # Optional: Calculate AUC here if desired: auc_sub = auc(fpr, tpr) # Need to calculate fpr,tpr first
        else:
            # Only one class present
            logger.warning(f"Skipping ROC plot for {group_key}: Only one class present (labels: {unique_labels}). Samples: {len(sub_df)}")

    # --- 3. Final Plot Adjustments ---
    if plotted_count > 0 or (overall_roc_data and overall_roc_data.get('auc') is not None): # Only add chance line if something was plotted
        ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle=':', label='Chance / Random Guess') # Added 'Chance'

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curves by Subgroup')
    # Adjust legend font size and position if needed
    ax.legend(loc="lower right", fontsize='small')
    ax.grid(True, alpha=0.5)

    plot_path = os.path.join(results_dir, filename)
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=150) # Added bbox_inches and dpi
        logger.info(f"Subgroup ROC plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save subgroup ROC plot: {e}")

    # Decide if you want to show the plot interactively
    # plt.show() # Uncomment this if you want the plot to pop up

    plt.close(fig) # Close the figure to free memory


def plot_subgroup_prevalence(results_df, subgroup_cols, results_dir, target_label=1, target_name='Glaucoma'):
    """Plots the prevalence (proportion) of the target label within each subgroup."""
    logger.info(f"Plotting '{target_name}' prevalence by subgroups...")
    if results_df is None or results_df.empty or 'label' not in results_df.columns:
        logger.warning("Cannot plot prevalence: DataFrame empty or 'label' column missing.")
        return

    for group_col in subgroup_cols:
        if group_col not in results_df.columns or results_df[group_col].isnull().all():
            logger.warning(f"Skipping prevalence plot for '{group_col}': column missing or all NaN.")
            continue

        try:
            # Filter out NaNs in the group column before grouping
            df_filtered = results_df.dropna(subset=[group_col])
            if df_filtered.empty:
                 logger.warning(f"No non-NaN data to plot prevalence for {group_col}.")
                 continue

            # Calculate prevalence for the target label
            prevalence = df_filtered.groupby(group_col)['label'].apply(lambda x: np.mean(x == target_label))
            counts = df_filtered[group_col].value_counts().sort_index()
            prevalence = prevalence.sort_index() # Ensure order matches counts

            if prevalence.empty:
                 logger.warning(f"No data to plot prevalence for {group_col} after grouping.")
                 continue

            fig, ax = plt.subplots(figsize=(max(6, len(prevalence) * 0.8), 5))
            # Ensure index is string for consistent plotting
            bar_labels = [str(idx) for idx in prevalence.index]
            bars = ax.bar(bar_labels, prevalence.values, color='skyblue')

            ax.bar_label(bars, fmt='%.3f', padding=3)
            ax.set_xlabel(group_col.replace('_', ' ').title())
            ax.set_ylabel(f'Prevalence of {target_name} (Label={target_label})')
            ax.set_title(f'{target_name} Prevalence by {group_col.replace("_", " ").title()}')
            ax.set_ylim(0, max(0.1, prevalence.max() * 1.15)) # Ensure some space even if prevalence is low
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha="right")

            # Add counts as text on bars
            for i, bar in enumerate(bars):
                yval = bar.get_height()
                count = counts.get(prevalence.index[i], 0) # Get count for this group
                ax.text(bar.get_x() + bar.get_width()/2.0, yval / 2, f'n={count}', va='center', ha='center', color='black', fontsize=9)


            plt.tight_layout()
            plot_path = os.path.join(results_dir, f'prevalence_by_{group_col}.png')
            try:
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                plt.savefig(plot_path)
                logger.info(f"Prevalence plot for {group_col} saved to {plot_path}")
            except Exception as e:
                logger.error(f"Failed to save prevalence plot for {group_col}: {e}")
            #plt.show()
            plt.close(fig)

        except Exception as e:
             logger.error(f"Failed to create prevalence plot for {group_col}: {e}", exc_info=True)


def plot_age_correlation(results_df, results_dir, filename='age_correlation_plot.png'):
    """Analyzes and visualizes the correlation between age and model predictions."""
    logger.info("Analyzing Age Correlation with Predictions...")
    if results_df is None or results_df.empty or not all(c in results_df.columns for c in ['age', 'probability', 'label']):
        logger.warning("Skipping age correlation: DataFrame empty or required columns missing ('age', 'probability', 'label').")
        return None

    # Prepare data, coercing types and dropping NaNs
    analysis_df = results_df[['age', 'probability', 'label']].copy()
    analysis_df['age'] = pd.to_numeric(analysis_df['age'], errors='coerce')
    analysis_df['probability'] = pd.to_numeric(analysis_df['probability'], errors='coerce')
    analysis_df.dropna(subset=['age', 'probability', 'label'], inplace=True) # Drop NaNs in all needed columns
    analysis_df['label'] = analysis_df['label'].astype(int) # Ensure label is integer

    if analysis_df.empty or len(analysis_df) < 2:
        logger.warning("Not enough valid data for age correlation analysis after cleaning.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot colored by true label
    label_map = {0: 'Normal', 1: 'Glaucoma'} # Assuming 0=Normal, 1=Glaucoma
    colors = {0: 'blue', 1: 'red'}
    for label_val, label_name in label_map.items():
        mask = analysis_df['label'] == label_val
        if mask.any():
            ax.scatter(analysis_df.loc[mask, 'age'], analysis_df.loc[mask, 'probability'],
                       alpha=0.5, label=f'True {label_name}', s=25, c=colors[label_val])

    # Trend line calculation and plotting
    trend_slope, trend_intercept, correlation, p_value, sig_stars = None, None, None, None, 'N/A'
    try:
        # Check for sufficient variance before fitting/correlating
        if analysis_df['age'].nunique() > 1 and analysis_df['probability'].nunique() > 1:
            z = np.polyfit(analysis_df['age'], analysis_df['probability'], 1)
            p = np.poly1d(z)
            trend_slope, trend_intercept = z[0], z[1]
            age_range = np.linspace(analysis_df['age'].min(), analysis_df['age'].max(), 100)
            ax.plot(age_range, p(age_range), "k--", alpha=0.8, label=f'Trend (Slope={trend_slope:.3f})')

            correlation, p_value = stats.pearsonr(analysis_df['age'], analysis_df['probability'])
            if p_value < 0.001: sig_stars = '***'
            elif p_value < 0.01: sig_stars = '**'
            elif p_value < 0.05: sig_stars = '*'
            else: sig_stars = 'ns'
            corr_title_part = f'Correlation: {correlation:.3f}{sig_stars} (p={p_value:.2e})'
        else:
             logger.warning("Insufficient variance in age or probability for trend line or correlation.")
             corr_title_part = "Correlation: N/A (Insufficient Variance)"

    except Exception as e:
        logger.error(f"Could not calculate trend line or correlation for age: {e}")
        corr_title_part = "Correlation: Error"

    # Final plot setup
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Predicted Probability of Glaucoma')
    ax.set_title(f'Age vs. Predicted Probability\n{corr_title_part}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(results_dir, filename)
    try:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        logger.info(f"Age correlation plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save age correlation plot: {e}")
    # plt.show()
    plt.close(fig)

    # Statistical summary dictionary
    stats_summary = {
        'correlation': correlation if correlation is not None else None,
        'p_value': p_value if p_value is not None else None,
        'trend_slope': trend_slope if trend_slope is not None else None,
        'trend_intercept': trend_intercept if trend_intercept is not None else None,
        'significance': sig_stars
    }

    # Log stats clearly
    logger.info("Age Correlation Statistics:")
    logger.info(f"  Pearson Correlation: {stats_summary['correlation'] if stats_summary['correlation'] is not None else 'N/A'}")
    logger.info(f"  P-value: {stats_summary['p_value'] if stats_summary['p_value'] is not None else 'N/A'} ({stats_summary['significance']})")
    logger.info(f"  Trend Slope: {stats_summary['trend_slope'] if stats_summary['trend_slope'] is not None else 'N/A'}")


    # Decade stats (optional, can be removed if not needed)
    try:
        analysis_df['age_decade'] = (analysis_df['age'] // 10) * 10
        decade_stats_agg = analysis_df.groupby('age_decade').agg(
             Mean_Probability=('probability', 'mean'),
             Std_Probability=('probability', 'std'),
             N=('probability', 'count'),
             Glaucoma_Prevalence=('label', lambda x: np.mean(x == 1)) # Prevalence of label 1
        ).round(4)
        logger.info("Prediction Statistics by Decade:")
        logger.info(f"\n{decade_stats_agg.to_string()}")
        stats_summary['decade_stats'] = decade_stats_agg.to_dict('index')
    except Exception as decade_e:
        logger.error(f"Error calculating decade statistics: {decade_e}")
        stats_summary['decade_stats'] = None

    return stats_summary