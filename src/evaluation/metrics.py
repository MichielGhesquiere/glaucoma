import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)
import logging
from typing import Dict, Any, Union, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# Segmentation Metrics
# ==============================================================================

def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculates the Dice Coefficient (F1 Score for segmentation) for binary masks.

    Args:
        y_true (np.ndarray): Ground truth binary mask (0 or 1). Flattened or original shape.
        y_pred (np.ndarray): Predicted binary mask (0 or 1). Must have the same shape as y_true.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: The Dice Coefficient. Returns NaN if both masks are empty.
    """
    y_true_f = y_true.flatten().astype(np.float64)
    y_pred_f = y_pred.flatten().astype(np.float64)

    intersection = np.sum(y_true_f * y_pred_f)
    pred_sum = np.sum(y_pred_f)
    true_sum = np.sum(y_true_f)

    # Handle case where both masks are empty
    if true_sum == 0 and pred_sum == 0:
        # Conventionally, Dice is 1 if both are empty (perfect match of nothing)
        # Or return NaN if this case should be handled differently downstream
        return 1.0 # Or return np.nan

    dice = (2. * intersection + smooth) / (true_sum + pred_sum + smooth)
    return float(dice)


def iou_score(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculates the Intersection over Union (IoU) or Jaccard Index for binary masks.

    Args:
        y_true (np.ndarray): Ground truth binary mask (0 or 1). Flattened or original shape.
        y_pred (np.ndarray): Predicted binary mask (0 or 1). Must have the same shape as y_true.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: The IoU score. Returns NaN if both masks are empty.
    """
    y_true_f = y_true.flatten().astype(np.float64)
    y_pred_f = y_pred.flatten().astype(np.float64)

    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection

    # Handle case where both masks are empty (union is 0)
    if union < smooth: # If union is effectively zero
        # Conventionally, IoU is 1 if both are empty
        return 1.0 # Or return np.nan

    iou = (intersection + smooth) / (union + smooth)
    return float(iou)

def pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the pixel-wise accuracy for binary masks.

    Args:
        y_true (np.ndarray): Ground truth binary mask (0 or 1).
        y_pred (np.ndarray): Predicted binary mask (0 or 1).

    Returns:
        float: Pixel accuracy.
    """
    if y_true.size == 0:
        return 1.0 # Or NaN? Define behavior for empty input
    correct_pixels = np.sum(y_true == y_pred)
    total_pixels = y_true.size
    return float(correct_pixels / total_pixels)


def calculate_segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> Dict[str, float]:
    """
    Calculates common segmentation metrics (Dice, IoU, Pixel Accuracy) for a single mask pair.

    Args:
        y_true (np.ndarray): Ground truth binary mask (0 or 1).
        y_pred (np.ndarray): Predicted binary mask (0 or 1). Thresholding should be applied beforehand.
        smooth (float): Smoothing factor for Dice and IoU.

    Returns:
        Dict[str, float]: Dictionary containing 'dice', 'iou', 'pixel_accuracy'.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Input shapes must match. Got {y_true.shape} and {y_pred.shape}")

    # Ensure inputs are binary
    y_true_bin = (y_true > 0.5).astype(np.uint8)
    y_pred_bin = (y_pred > 0.5).astype(np.uint8)

    dice = dice_coefficient(y_true_bin, y_pred_bin, smooth)
    iou = iou_score(y_true_bin, y_pred_bin, smooth)
    acc = pixel_accuracy(y_true_bin, y_pred_bin)

    return {
        "dice": dice,
        "iou": iou,
        "pixel_accuracy": acc,
    }

def calculate_multi_task_segmentation_metrics(
    pred_oc: np.ndarray, gt_oc: np.ndarray,
    pred_od: np.ndarray, gt_od: np.ndarray,
    smooth: float = 1e-6) -> Dict[str, float]:
    """
    Calculates segmentation metrics for both Optic Cup (OC) and Optic Disc (OD).

    Args:
        pred_oc (np.ndarray): Predicted OC mask (binary).
        gt_oc (np.ndarray): Ground truth OC mask (binary).
        pred_od (np.ndarray): Predicted OD mask (binary).
        gt_od (np.ndarray): Ground truth OD mask (binary).
        smooth (float): Smoothing factor.

    Returns:
        Dict[str, float]: Dictionary containing metrics for OC and OD (e.g., 'oc_dice', 'od_dice', ...).
    """
    metrics = {}
    oc_metrics = calculate_segmentation_metrics(gt_oc, pred_oc, smooth)
    od_metrics = calculate_segmentation_metrics(gt_od, pred_od, smooth)

    for key, value in oc_metrics.items():
        metrics[f"oc_{key}"] = value
    for key, value in od_metrics.items():
        metrics[f"od_{key}"] = value

    return metrics


# ==============================================================================
# Classification Metrics
# ==============================================================================

def calculate_rates(cm):
    """Calculates TPR, FPR, TNR, FNR from a 2x2 confusion matrix."""
    if not isinstance(cm, np.ndarray) or cm.shape != (2, 2):
        logger.warning(f"Received confusion matrix of unexpected shape/type {cm.shape if isinstance(cm, np.ndarray) else type(cm)}. Cannot calculate rates. CM: {cm}")
        return {'TPR': None, 'FPR': None, 'TNR': None, 'FNR': None}

    tn, fp, fn, tp = cm.ravel()
    # Denominators
    actual_pos = tp + fn
    actual_neg = tn + fp

    tpr = tp / actual_pos if actual_pos > 0 else 0.0 # Sensitivity, Recall
    fpr = fp / actual_neg if actual_neg > 0 else 0.0 # Fall-out
    tnr = tn / actual_neg if actual_neg > 0 else 0.0 # Specificity
    fnr = fn / actual_pos if actual_pos > 0 else 0.0 # Miss rate

    return {'TPR': tpr, 'FPR': fpr, 'TNR': tnr, 'FNR': fnr}


def calculate_metric(metric_name, labels, probs, preds_at_threshold):
    """
    Helper to calculate a specific metric given true labels, probabilities,
    and binary predictions at a chosen threshold.
    """
    if labels is None or len(labels) == 0:
        return None

    metric_name_upper = metric_name.upper()

    try:
        unique_labels = np.unique(labels)

        if metric_name_upper == 'AUC':
            if len(unique_labels) > 1 and probs is not None:
                fpr, tpr, _ = roc_curve(labels, probs)
                return auc(fpr, tpr)
            else:
                logger.debug("Cannot compute AUC: only one class present or probabilities missing.")
                return None
        elif metric_name_upper == 'ACCURACY':
             if preds_at_threshold is not None:
                 return accuracy_score(labels, preds_at_threshold)
             else:
                 logger.debug("Cannot compute Accuracy: predictions missing.")
                 return None
        elif metric_name_upper in ['TPR', 'FPR', 'TNR', 'FNR']:
             if preds_at_threshold is None:
                  logger.debug(f"Cannot compute {metric_name_upper}: predictions missing.")
                  return None
             # Ensure CM is calculated correctly even if only one class is predicted
             cm = confusion_matrix(labels, preds_at_threshold, labels=[0, 1])
             rates = calculate_rates(cm)
             return rates.get(metric_name_upper, None)
        else:
            logger.warning(f"Unsupported metric '{metric_name}' requested.")
            return None
    except ValueError as ve:
         logger.debug(f"ValueError calculating {metric_name}: {ve}")
         return None
    except Exception as e:
        logger.error(f"Unexpected error calculating {metric_name}: {e}", exc_info=True)
        return None

def calculate_sensitivity_at_specificity(fpr: np.ndarray, tpr: np.ndarray, target_spec: float = 0.95) -> float:
    """Calculate sensitivity at a specific specificity level."""
    # Specificity = 1 - FPR, so target FPR = 1 - target_spec
    target_fpr = 1 - target_spec
    
    # Find the closest FPR to our target
    idx = np.argmin(np.abs(fpr - target_fpr))
    return tpr[idx]

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Expected Calibration Error (ECE) and return calibration plot data.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins for calibration
        
    Returns:
        Tuple of (ece_score, bin_boundaries, bin_accuracies, bin_confidences, bin_counts)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    ece = 0.0
    total_samples = len(y_true)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.sum() / total_samples
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence for this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Add to ECE calculation
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)
    
    return ece, bin_boundaries, np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)

def calculate_metrics_from_predictions(df_predictions, dataset_name: str) -> dict:
    """
    Calculate basic metrics from predictions DataFrame.
    
    Args:
        df_predictions: DataFrame with 'label' and 'probability_class1' columns
        dataset_name: Name for logging
        
    Returns:
        Dictionary with metrics
    """
    if df_predictions.empty:
        logger.warning(f"Empty predictions DataFrame for {dataset_name}")
        return {}
    
    metrics = {}
    
    # Overall metrics
    if len(np.unique(df_predictions["label"])) < 2:
        logger.warning(f"Overall {dataset_name} has only one class. AUC cannot be computed.")
        accuracy = accuracy_score(df_predictions["label"], df_predictions["probability_class1"] > 0.5) if len(df_predictions["label"]) > 0 else np.nan
        auc_value = np.nan
        fpr_overall, tpr_overall = None, None
    else:
        accuracy = accuracy_score(df_predictions["label"], df_predictions["probability_class1"] > 0.5)
        fpr_overall, tpr_overall, _ = roc_curve(df_predictions["label"], df_predictions["probability_class1"])
        auc_value = auc(fpr_overall, tpr_overall)
    
    metrics["overall"] = {
        "accuracy": accuracy,
        "auc": auc_value,
        "num_samples": len(df_predictions),
        "fpr": fpr_overall,
        "tpr": tpr_overall
    }
    
    return metrics