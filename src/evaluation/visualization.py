"""
Visualization functions for model evaluation.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score
import logging
from ..utils.helpers import NpEncoder
from .metrics import calculate_sensitivity_at_specificity, calculate_ece

logger = logging.getLogger(__name__)

def plot_roc_curves(df_predictions: pd.DataFrame, dataset_name: str, results_dir: str, 
                   epoch_str: str, plot_per_source: bool = True, min_samples_per_source: int = 50):
    """
    Plot ROC curves for overall and per-source performance.
    """
    if df_predictions.empty:
        logger.warning(f"Empty predictions for {dataset_name}")
        return None
    
    # Calculate overall metrics
    if len(np.unique(df_predictions["label"])) < 2:
        logger.warning(f"Only one class in {dataset_name}")
        return None
    
    fpr_overall, tpr_overall, _ = roc_curve(df_predictions["label"], df_predictions["probability_class1"])
    auc_overall = auc(fpr_overall, tpr_overall)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    plt.plot(fpr_overall, tpr_overall, color='black', lw=2.5, 
            label=f'Overall {dataset_name} (AUC={auc_overall:.3f})')
    
    # Per-source ROC curves
    if plot_per_source and 'dataset_source' in df_predictions.columns:
        unique_sources = sorted([s for s in df_predictions["dataset_source"].astype(str).unique() 
                               if s.lower() not in ['nan', 'unknown_source', 'none']])
        if unique_sources:
            cmap_name = 'tab10' if len(unique_sources) <= 10 else 'tab20'
            colors = plt.colormaps.get_cmap(cmap_name).resampled(len(unique_sources))
            
            for i, src_name in enumerate(unique_sources):
                src_df = df_predictions[df_predictions["dataset_source"].astype(str) == src_name]
                if len(src_df) < min_samples_per_source or len(src_df["label"].unique()) < 2:
                    continue
                    
                fpr_s, tpr_s, _ = roc_curve(src_df["label"], src_df["probability_class1"])
                auc_s = auc(fpr_s, tpr_s)
                plt.plot(fpr_s, tpr_s, color=colors(i), lw=1.5, linestyle='--',
                        label=f'{src_name} (AUC={auc_s:.3f}, N={len(src_df)})')
    
    plt.plot([0, 1], [0, 1], 'k:', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plot_title_suffix = f" (Epoch {epoch_str})" if "Ensemble" not in dataset_name else ""
    plt.title(f'{dataset_name} ROC Curves{plot_title_suffix}')
    plt.legend(loc="lower right", fontsize='medium')
    plt.grid(True, alpha=0.7)
    
    # Save plot
    roc_plot_filename_suffix = f"_epoch{epoch_str}" if "Ensemble" not in dataset_name else ""
    roc_plot_filename = f'{dataset_name.lower().replace(" ", "_")}_roc{roc_plot_filename_suffix}.png'
    plt.savefig(os.path.join(results_dir, roc_plot_filename))
    plt.close()
    
    logger.info(f"ROC plot saved: {roc_plot_filename}")
    return {'fpr': fpr_overall, 'tpr': tpr_overall, 'auc': auc_overall}

def plot_combined_roc_with_ensemble(all_models_roc_data: dict, ensemble_roc_data: dict | None, 
                                   output_dir: str, timestamp_str: str):
    """Enhanced ROC plotting function that includes ensemble curves."""
    logger.info("\n--- Generating Combined ROC Plot with Ensemble ---")
    plt.figure(figsize=(13, 11))
    
    # Plot individual models
    num_models = len(all_models_roc_data)
    colors_cmap = plt.cm.get_cmap('tab10' if num_models <= 10 else 'viridis', num_models if num_models > 0 else 1)
    sorted_roc_data = sorted(all_models_roc_data.items(), 
                           key=lambda item: item[1]['auc'] if not np.isnan(item[1]['auc']) else -1, 
                           reverse=True)

    for i, (model_name, roc_data) in enumerate(sorted_roc_data):
        if roc_data and roc_data['fpr'] is not None and roc_data['tpr'] is not None:
            auc_val = roc_data['auc']
            label_str = f'{model_name} (AUC={auc_val:.3f})' if not np.isnan(auc_val) else f'{model_name} (AUC=N/A)'
            plt.plot(roc_data['fpr'], roc_data['tpr'], lw=2.5, label=label_str, 
                    color=colors_cmap(i), alpha=0.8)

    # Plot ensemble if available
    if ensemble_roc_data:
        plt.plot(ensemble_roc_data['fpr'], ensemble_roc_data['tpr'], 
                color='black', lw=4, linestyle='--',
                label=f'Ensemble (AUC={ensemble_roc_data["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k:', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('Combined ROC Curves - In-Distribution Test Sets (Individual Models + Ensemble)', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize='large', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    combined_roc_filename = f"all_models_id_test_combined_roc_with_ensemble_{timestamp_str}.png"
    combined_roc_filepath = os.path.join(output_dir, combined_roc_filename)
    try:
        plt.savefig(combined_roc_filepath, bbox_inches='tight', dpi=150)
        logger.info(f"Combined ROC plot saved: {combined_roc_filepath}")
    except Exception as e: 
        logger.error(f"Failed to save combined ROC plot: {e}", exc_info=True)
    plt.close()

def create_summary_tables(all_models_roc_data: dict, ensemble_roc_data: dict | None, 
                         all_individual_model_prediction_dfs: list[pd.DataFrame],
                         model_display_names_for_dfs: list[str],
                         ensemble_df: pd.DataFrame | None,
                         all_model_metrics: dict,
                         output_dir: str, min_samples_for_source: int = 50):
    """Creates and saves comprehensive summary tables including ECE."""
    logger.info("Creating comprehensive summary tables with ECE...")
    
    # Table 1: Overall Performance Summary (AUC, Sensitivity, ECE)
    summary_data = []
    
    # Collect individual model data
    for model_name, roc_data in all_models_roc_data.items():
        auc_val = roc_data['auc']
        sens_95_spec = calculate_sensitivity_at_specificity(roc_data['fpr'], roc_data['tpr'], 0.95)
        
        # Get ECE from model metrics
        ece_val = None
        if model_name in all_model_metrics and 'overall' in all_model_metrics[model_name]:
            ece_val = all_model_metrics[model_name]['overall'].get('ece')
        
        summary_data.append({
            'Model': model_name,
            'Dataset': 'ID-Test',
            'AUC': auc_val,
            'Sens@95%Spec': sens_95_spec,
            'ECE': ece_val if ece_val is not None else np.nan
        })
    
    # Add ensemble data if available
    if ensemble_roc_data:
        auc_val = ensemble_roc_data['auc']
        sens_95_spec = calculate_sensitivity_at_specificity(ensemble_roc_data['fpr'], ensemble_roc_data['tpr'], 0.95)
        
        # Calculate ECE for ensemble
        ece_val = None
        if ensemble_df is not None and not ensemble_df.empty:
            y_true = ensemble_df['label'].values
            y_prob = ensemble_df['ensemble_probability_class1'].values
            ece_val, _, _, _, _ = calculate_ece(y_true, y_prob)
        
        summary_data.append({
            'Model': 'Ensemble',
            'Dataset': 'ID-Test',
            'AUC': auc_val,
            'Sens@95%Spec': sens_95_spec,
            'ECE': ece_val if ece_val is not None else np.nan
        })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        
        # Create figure for overall summary table
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        
        models = sorted([m for m in df_summary['Model'].unique() if m != 'Ensemble']) + \
                (['Ensemble'] if ensemble_roc_data else [])
        
        # AUC Table
        auc_table_data = []
        for model in models:
            model_data = df_summary[df_summary['Model'] == model]
            if not model_data.empty:
                auc_table_data.append([model, f"{model_data['AUC'].iloc[0]:.3f}"])
            else:
                auc_table_data.append([model, "N/A"])
        
        _create_table(ax1, auc_table_data, ['Model', 'AUC'], 
                     'AUC Values - In-Distribution Test Set', ensemble_roc_data is not None)
        
        # Sensitivity at 95% Specificity Table
        sens_table_data = []
        for model in models:
            model_data = df_summary[df_summary['Model'] == model]
            if not model_data.empty:
                sens_table_data.append([model, f"{model_data['Sens@95%Spec'].iloc[0]:.3f}"])
            else:
                sens_table_data.append([model, "N/A"])
        
        _create_table(ax2, sens_table_data, ['Model', 'Sens@95%Spec'], 
                     'Sensitivity at 95% Specificity - In-Distribution Test Set', 
                     ensemble_roc_data is not None)
        
        # ECE Table
        ece_table_data = []
        for model in models:
            model_data = df_summary[df_summary['Model'] == model]
            if not model_data.empty and not pd.isna(model_data['ECE'].iloc[0]):
                ece_table_data.append([model, f"{model_data['ECE'].iloc[0]:.4f}"])
            else:
                ece_table_data.append([model, "N/A"])
        
        _create_table(ax3, ece_table_data, ['Model', 'ECE'], 
                     'Expected Calibration Error - In-Distribution Test Set', 
                     ensemble_roc_data is not None)
        
        plt.tight_layout()
        overall_summary_path = os.path.join(output_dir, 'id_overall_performance_summary_table_with_ece.png')
        plt.savefig(overall_summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Overall performance summary table saved: {overall_summary_path}")

def _create_table(ax, data, col_labels, title, has_ensemble):
    """Helper function to create styled tables."""
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style ensemble row if exists
    if has_ensemble:
        ensemble_row = len(data)  # Last row
        for i in range(len(col_labels)):
            table[(ensemble_row, i)].set_facecolor('#FFE0B2')
            table[(ensemble_row, i)].set_text_props(weight='bold')