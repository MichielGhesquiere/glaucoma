"""
Calibration-related functions for model evaluation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from .metrics import calculate_ece

logger = logging.getLogger(__name__)

def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, 
                          output_dir: str, n_bins: int = 10, epoch_str: str = ""):
    """
    Plot calibration curve and save as PNG.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        model_name: Name for plot title and filename
        output_dir: Directory to save plot
        n_bins: Number of bins for calibration
        epoch_str: Epoch string for filename
    """
    ece, bin_boundaries, bin_accuracies, bin_confidences, bin_counts = calculate_ece(y_true, y_prob, n_bins)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calibration plot
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    # Only plot bins with samples
    mask = bin_counts > 0
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')
    ax1.plot(bin_confidences[mask], bin_accuracies[mask], 'o-', 
             color='red', markersize=8, linewidth=2, label=f'{model_name}')
    
    # Add confidence intervals or bin counts as text
    for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
        if count > 0:
            ax1.annotate(f'n={count}', (conf, acc), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title(f'Calibration Plot - {model_name}\nECE = {ece:.4f}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Histogram of predicted probabilities
    ax2.hist(y_prob, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Distribution of Predicted Probabilities\n{model_name}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    epoch_suffix = f"_epoch{epoch_str}" if epoch_str and epoch_str != "Ensemble" else ""
    filename = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_calibration{epoch_suffix}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Calibration plot for {model_name} saved to: {filepath}")
    return ece

def calculate_ece_from_dataframe(df_predictions: pd.DataFrame, model_name: str, 
                                output_dir: str, epoch_str: str = "", n_bins: int = 10) -> dict:
    """
    Calculate ECE and create calibration plots from prediction DataFrame.
    
    Args:
        df_predictions: DataFrame with 'label' and 'probability_class1' columns
        model_name: Model name for plots and logging
        output_dir: Directory to save plots
        epoch_str: Epoch string for filenames
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with ECE results
    """
    if df_predictions.empty:
        logger.warning(f"Empty predictions DataFrame for {model_name}. Skipping ECE calculation.")
        return {}
    
    y_true = df_predictions['label'].values
    y_prob = df_predictions['probability_class1'].values
    
    # Calculate overall ECE
    overall_ece = plot_calibration_curve(y_true, y_prob, model_name, output_dir, n_bins, epoch_str)
    
    ece_results = {
        'overall_ece': overall_ece,
        'n_samples': len(y_true),
        'per_source_ece': {}
    }
    
    # Calculate per-source ECE if dataset_source is available
    if 'dataset_source' in df_predictions.columns:
        for source_name, source_group in df_predictions.groupby('dataset_source'):
            source_name = str(source_name)
            if source_name.lower() in ['nan', 'unknown_source', 'none'] or len(source_group) < 20:
                continue
                
            if len(source_group['label'].unique()) >= 2:  # Need both classes
                source_y_true = source_group['label'].values
                source_y_prob = source_group['probability_class1'].values
                
                try:
                    source_ece, _, _, _, _ = calculate_ece(source_y_true, source_y_prob, n_bins)
                    ece_results['per_source_ece'][source_name] = {
                        'ece': source_ece,
                        'n_samples': len(source_group)
                    }
                    logger.info(f"ECE for {model_name} on {source_name}: {source_ece:.4f} (N={len(source_group)})")
                except Exception as e:
                    logger.warning(f"Could not calculate ECE for {model_name} on source {source_name}: {e}")
    
    logger.info(f"Overall ECE for {model_name}: {overall_ece:.4f} (N={len(y_true)})")
    return ece_results