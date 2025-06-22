import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
import logging
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report

# Import functions from sibling modules
from .metrics import calculate_rates, calculate_metric
from .fairness import analyze_fairness, analyze_significance # Assuming analyze_fairness does not inherently calculate EO
from ..utils.plotting import (plot_metric_comparison, plot_tpr_fpr_parity,
                     plot_subgroup_roc_curves, plot_subgroup_prevalence,
                     plot_age_correlation)
from ..utils.helpers import NpEncoder # Import from utils
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)

def collect_predictions(model, loader, device):
    """
    Collects predictions, probabilities, labels, and sensitive attributes from a DataLoader.

    Args:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): DataLoader providing batches of (images, labels, attributes_dict).
        device (torch.device): The device to run inference on.

    Returns:
        pd.DataFrame: DataFrame containing 'label', 'probability', 'prediction',
                      and columns for each sensitive attribute collected.
                      Returns None if an error occurs or no data is processed.
    """
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    sensitive_attributes_names = []
    if hasattr(loader, 'dataset') and hasattr(loader.dataset, 'sensitive_attributes'):
        sensitive_attributes_names = loader.dataset.sensitive_attributes
        if sensitive_attributes_names is None:
             sensitive_attributes_names = []
    else:
        logger.warning("Could not retrieve sensitive_attributes list from loader.dataset. Attributes will not be collected.")

    collected_attributes = {attr: [] for attr in sensitive_attributes_names}

    device_type = device.type
    USE_AMP = torch.cuda.is_available()
    amp_enabled = USE_AMP and (device_type == 'cuda')
    amp_dtype = torch.float16 if device_type == 'cuda' else None

    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Collecting Predictions"):
            if batch_data is None:
                logger.warning("Skipping None batch potentially returned by safe_collate.")
                continue

            if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 3:
                 logger.error(f"Unexpected batch data format. Expected 3 items (img, lbl, attr_dict), got {len(batch_data) if isinstance(batch_data, (list, tuple)) else type(batch_data)}. Skipping batch.")
                 continue

            inputs, labels, attributes_batch = batch_data
            inputs = inputs.to(device)
            labels_np = labels.cpu().numpy()

            try:
                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                    outputs = model(inputs)

                probabilities = F.softmax(outputs.float(), dim=1)[:, 1].cpu().numpy()
                predictions = (probabilities >= 0.5).astype(int)

                all_labels.extend(labels_np)
                all_probs.extend(probabilities)
                all_preds.extend(predictions)

                if isinstance(attributes_batch, dict):
                    for attr_name in sensitive_attributes_names:
                        if attr_name in attributes_batch:
                            attr_values = attributes_batch[attr_name]
                            if isinstance(attr_values, torch.Tensor):
                                collected_attributes[attr_name].extend(attr_values.cpu().numpy())
                            elif isinstance(attr_values, (list, tuple)):
                                collected_attributes[attr_name].extend(attr_values)
                            else:
                                logger.warning(f"Unexpected type for batched attribute '{attr_name}': {type(attr_values)}. Attempting to extend.")
                                try:
                                    collected_attributes[attr_name].extend(attr_values)
                                except TypeError:
                                     logger.error(f"Could not extend attribute list for '{attr_name}' with non-iterable type {type(attr_values)}")
                                     collected_attributes[attr_name].extend([np.nan] * len(labels_np))
                        else:
                            logger.warning(f"Attribute '{attr_name}' expected but not found in attributes_batch dict for this batch. Appending NaNs.")
                            num_in_batch = len(labels_np)
                            collected_attributes[attr_name].extend([np.nan] * num_in_batch)
                else:
                     logger.error(f"Expected attributes_batch to be a dict, but got {type(attributes_batch)}. Attributes not collected for this batch.")
                     num_in_batch = len(labels_np)
                     for attr_name in sensitive_attributes_names:
                          collected_attributes[attr_name].extend([np.nan] * num_in_batch)

            except Exception as e:
                logger.error(f"Error processing batch during prediction collection: {e}", exc_info=True)
                continue

    if not all_labels:
        logger.error("No predictions were collected. Check input loader, model, and batch processing steps.")
        return None

    results_df = pd.DataFrame({
        'label': all_labels,
        'probability': all_probs,
        'prediction': all_preds,
    })

    for attr_name, attr_list in collected_attributes.items():
        if len(attr_list) == len(results_df):
            results_df[attr_name] = attr_list
        else:
            logger.warning(f"Length mismatch for attribute '{attr_name}' ({len(attr_list)}) vs DataFrame ({len(results_df)}). Skipping column.")

    logger.info(f"Collected predictions and attributes for {len(results_df)} samples.")
    return results_df


def analyze_overall_performance(results_df, results_dir=None):
    logger.info("Analyzing Overall Performance...")
    if results_df is None or results_df.empty or 'label' not in results_df or 'probability' not in results_df:
        logger.error("Cannot analyze overall performance: DataFrame is invalid or missing required columns.")
        return None, 0.5, None, None

    all_labels = results_df['label'].values
    all_probs = results_df['probability'].values

    overall_metrics = {}
    optimal_threshold = 0.5
    overall_roc_data = {'fpr': None, 'tpr': None, 'auc': None}

    try:
        unique_labels = np.unique(all_labels)
        if len(unique_labels) < 2:
            logger.warning("Only one class present in overall labels. Cannot calculate ROC AUC or optimal threshold via Youden's J.")
            overall_roc_data['auc'] = None
        else:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)
            overall_roc_data = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
            logger.info(f"Overall ROC AUC: {roc_auc:.4f}")

            valid_indices = np.where(np.isfinite(tpr) & np.isfinite(fpr))[0]
            if len(valid_indices) > 0:
                 youden_j = tpr[valid_indices] - fpr[valid_indices]
                 optimal_idx_relative = np.argmax(youden_j)
                 optimal_idx_absolute = valid_indices[optimal_idx_relative]
                 optimal_fpr = fpr[optimal_idx_absolute]
                 optimal_tpr = tpr[optimal_idx_absolute]
                 
                 optimal_threshold_candidate = thresholds[optimal_idx_absolute] if optimal_idx_absolute < len(thresholds) else 0.5
                 
                 if np.isfinite(optimal_threshold_candidate):
                     optimal_threshold = np.clip(optimal_threshold_candidate, 0.0, 1.0)
                 else:
                     logger.warning("Optimal threshold candidate non-finite, using default.")
                     optimal_threshold = 0.5

                 logger.info(f"Optimal Threshold (Max Youden's J) = {optimal_threshold:.4f}")
                 logger.info(f"  At this point: TPR ~ {optimal_tpr:.4f}, FPR ~ {optimal_fpr:.4f}")
            else:
                 logger.warning("No valid points for Youden's J. Using default threshold.")
                 optimal_threshold = 0.5

    except Exception as e:
        logger.error(f"Error during overall ROC analysis: {e}", exc_info=True)
        overall_roc_data['auc'] = None
        optimal_threshold = 0.5

    logger.info(f"Calculating Overall Performance at Threshold = {optimal_threshold:.4f}")
    all_preds_at_threshold = (all_probs >= optimal_threshold).astype(int)
    try:
        cm_overall = confusion_matrix(all_labels, all_preds_at_threshold, labels=[0, 1])
        rates_overall = calculate_rates(cm_overall)
        accuracy_overall = accuracy_score(all_labels, all_preds_at_threshold)

        overall_metrics = {
            'Accuracy': accuracy_overall,
            'AUC': overall_roc_data.get('auc'),
            **rates_overall
        }
        log_metrics = {k: (f"{v:.4f}" if v is not None else "N/A") for k, v in overall_metrics.items()}
        logger.info(f"Overall Metrics: {log_metrics}")
        logger.info(f"Overall Confusion Matrix (TN, FP, FN, TP): {cm_overall.ravel().tolist()}")

        logger.info("Overall Classification Report (at threshold):")
        try:
             report = classification_report(all_labels, all_preds_at_threshold, target_names=['Normal (0)', 'Glaucoma (1)'], output_dict=False, zero_division=0)
             for line in report.split('\n'): logger.info(f"  {line}")
        except Exception as report_e:
             logger.error(f"Could not generate classification report: {report_e}")

    except Exception as e:
        logger.error(f"Error calculating overall performance metrics at threshold: {e}", exc_info=True)
        overall_metrics = {'Accuracy': None, 'AUC': overall_roc_data.get('auc'), 'TPR': None, 'FPR': None, 'TNR': None, 'FNR': None}

    return overall_metrics, optimal_threshold, all_preds_at_threshold, overall_roc_data


def prepare_subgroup_data(results_df_raw, age_bins, age_labels, eye_map):
    if results_df_raw is None or results_df_raw.empty:
        logger.error("Input DataFrame for subgroup preparation is empty.")
        return None

    results_df = results_df_raw.copy()
    logger.info("Preparing Subgroup Data...")

    if 'age' in results_df.columns:
        results_df['age_numeric'] = pd.to_numeric(results_df['age'], errors='coerce')
        if results_df['age_numeric'].notna().any():
            results_df['age_group'] = pd.cut(results_df['age_numeric'], bins=age_bins, labels=age_labels, right=False, include_lowest=True)
            logger.info(f"Age groups created using bins {age_bins} -> labels {age_labels}.")
            logger.info(f"Age group distribution:\n{results_df['age_group'].value_counts(dropna=False).to_string()}")
        else:
            logger.warning("No valid numeric 'age' values found. Setting 'age_group' to NA.")
            results_df['age_group'] = pd.NA
    else:
        logger.warning("'age' column not found. Cannot create age groups.")
        results_df['age_group'] = pd.NA

    if 'eye' in results_df.columns:
        eye_col = results_df['eye']
        results_df['eye_group'] = pd.NA

        numeric_mask = pd.to_numeric(eye_col, errors='coerce').isin(eye_map.keys())
        results_df.loc[numeric_mask, 'eye_group'] = pd.to_numeric(eye_col[numeric_mask]).map(eye_map)

        string_mask = eye_col.astype(str).str.upper().isin(eye_map.values()) # Check against mapped values too
        results_df.loc[string_mask, 'eye_group'] = eye_col[string_mask].astype(str).str.upper()
        
        # Check against keys for strings if map keys are 'OD', 'OS'
        string_key_mask = eye_col.astype(str).str.upper().isin(eye_map.keys())
        results_df.loc[string_key_mask & results_df['eye_group'].isna(), 'eye_group'] = eye_col[string_key_mask].astype(str).str.upper().map(eye_map)


        unmapped_mask = results_df['eye_group'].isna() & eye_col.notna()
        unmapped_count = unmapped_mask.sum()
        if unmapped_count > 0:
             unmapped_vals = eye_col[unmapped_mask].unique()
             logger.warning(f"{unmapped_count} 'eye' entries could not be mapped. Original values found: {unmapped_vals}")

        logger.info(f"Eye group distribution:\n{results_df['eye_group'].value_counts(dropna=False).to_string()}")
    else:
        logger.warning("'eye' column not found. Cannot create eye groups.")
        results_df['eye_group'] = pd.NA

    return results_df


def _calculate_single_group_metrics(labels, probs, preds_at_threshold):
    metrics = {'Accuracy': None, 'AUC': None, 'TPR': None, 'FPR': None, 'TNR': None, 'FNR': None, 'n_samples': len(labels)}
    if labels is None or len(labels) == 0:
        return metrics
    try:
        if preds_at_threshold is not None:
            metrics['Accuracy'] = calculate_metric('Accuracy', labels, None, preds_at_threshold)
            cm = confusion_matrix(labels, preds_at_threshold, labels=[0, 1])
            rates = calculate_rates(cm)
            metrics.update(rates)
        if probs is not None:
            metrics['AUC'] = calculate_metric('AUC', labels, probs, None)
    except Exception as e:
        logger.error(f"Error calculating metrics for subgroup: {e}", exc_info=True)
        metrics = {k: v if k == 'n_samples' else None for k, v in metrics.items()}
    return metrics


def calculate_subgroup_metrics(results_df, subgroup_cols, optimal_threshold):
    logger.info(f"Calculating Subgroup Performance (at Threshold = {optimal_threshold:.4f})")
    if results_df is None or results_df.empty:
        logger.error("Cannot calculate subgroup metrics: Input DataFrame is empty or None.")
        return {}
    if 'probability' not in results_df.columns or 'label' not in results_df.columns:
        logger.error("Cannot calculate subgroup metrics: DataFrame missing 'probability' or 'label'.")
        return {}

    if 'prediction_thresh' not in results_df.columns:
         results_df['prediction_thresh'] = (results_df['probability'] >= optimal_threshold).astype(int)

    subgroup_metrics_all = {}
    for group_col in subgroup_cols:
        if group_col not in results_df.columns:
            logger.warning(f"Subgroup column '{group_col}' not found in results. Skipping.")
            continue

        logger.info(f"Analyzing by subgroup: {group_col}")
        grouped = results_df.groupby(group_col, dropna=False)

        for group_val, sub_df in grouped:
            group_key_suffix = str(group_val) if pd.notna(group_val) else "NaN"
            group_key = f"{group_col}_{group_key_suffix}"

            sub_labels = sub_df['label'].values
            sub_probs = sub_df['probability'].values
            sub_preds = sub_df['prediction_thresh'].values

            logger.info(f"  Processing Group: {group_key} (n={len(sub_labels)})")
            if len(sub_labels) > 0:
                metrics = _calculate_single_group_metrics(sub_labels, sub_probs, sub_preds)
                subgroup_metrics_all[group_key] = metrics
                log_metrics = {k: (f"{v:.4f}" if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items() if k != 'n_samples'}
                logger.info(f"    Metrics - {log_metrics}")
            else:
                 logger.info(f"    Skipping group {group_key}: No samples.")
                 subgroup_metrics_all[group_key] = {'n_samples': 0}
    return subgroup_metrics_all


def run_evaluation(model, test_loader, device, results_dir,
                   experiment_name="eval",
                   subgroup_cols=None, # These are RAW attribute names from evaluate_ood.py
                   age_bins=None, age_labels=None, eye_map=None,
                   metrics_to_compare=None,
                   n_bootstraps=0, # Default to 0 as significance is removed
                   alpha=0.05,     # Kept for now, might be used by other analyses if any
                   n_calibration_bins=10,
                   min_samples_for_calibration=30): # Default updated
    logger = logging.getLogger(__name__) # Use module-level logger
    logger.info(f"=== Starting Evaluation Pipeline: {experiment_name} ===")
    eval_results_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(eval_results_dir, exist_ok=True)
    logger.info(f"Evaluation outputs will be saved to: {eval_results_dir}")

    # Default subgroup_cols here refers to RAW attribute names for which binning might be applied
    if subgroup_cols is None: subgroup_cols = ['age', 'eye'] # Example raw attributes
    if age_bins is None: age_bins = [0, 65, np.inf]
    if age_labels is None: age_labels = ['<65', '>=65']
    if eye_map is None: eye_map = {'OD':'OD', 'OS':'OS', 'Right':'OD', 'Left':'OS', 'R':'OD', 'L':'OS', 0:'OD', 1:'OS'}
    if metrics_to_compare is None: metrics_to_compare = ['Accuracy', 'AUC', 'TPR', 'FPR', 'PPV']

    results_df_raw = collect_predictions(model, test_loader, device)
    if results_df_raw is None:
        logger.critical("Evaluation stopped: Failed to collect predictions.")
        return None, None
    results_df_raw.to_csv(os.path.join(eval_results_dir, 'raw_predictions.csv'), index=False)
    logger.info(f"Raw predictions saved to {os.path.join(eval_results_dir, 'raw_predictions.csv')}")

    overall_metrics, optimal_threshold, _, overall_roc_data = \
        analyze_overall_performance(results_df_raw, eval_results_dir)
    if overall_metrics is None: overall_metrics = {} # Ensure it's a dict
    if optimal_threshold is None:
        logger.warning("Failed to determine optimal threshold. Using default 0.5 for subsequent steps.")
        optimal_threshold = 0.5

    results_df = prepare_subgroup_data(results_df_raw, age_bins, age_labels, eye_map)
    if results_df is None:
         logger.critical("Evaluation stopped: Failed to prepare subgroup data.")
         return None, None
    results_df['prediction_thresh'] = (results_df['probability'] >= optimal_threshold).astype(int)
    results_df.to_csv(os.path.join(eval_results_dir, 'predictions_with_subgroups.csv'), index=False)
    logger.info(f"Predictions with subgroups saved to {os.path.join(eval_results_dir, 'predictions_with_subgroups.csv')}")

    # --- Determine actual columns to use for subgroup processing (binned if available) ---
    cols_for_subgroup_processing = []
    if subgroup_cols: # subgroup_cols are raw attribute names like ['age', 'eye', 'camera']
        for raw_col_name in subgroup_cols:
            binned_col_name = f"{raw_col_name}_group"
            if binned_col_name in results_df.columns:
                cols_for_subgroup_processing.append(binned_col_name)
                logger.info(f"For raw attribute '{raw_col_name}', using binned column '{binned_col_name}' for subgroup processing.")
            elif raw_col_name in results_df.columns:
                cols_for_subgroup_processing.append(raw_col_name)
                logger.info(f"For raw attribute '{raw_col_name}', using raw column '{raw_col_name}' for subgroup processing.")
            else:
                logger.warning(f"Raw subgroup column '{raw_col_name}' and its potential binned version not found. Skipping '{raw_col_name}' for subgroup processing.")
    
    if not cols_for_subgroup_processing: # Fallback if list is empty
        default_processed_cols = ['age_group', 'eye_group'] # Example default binned/processed names
        cols_for_subgroup_processing = [c for c in default_processed_cols if c in results_df.columns]
        logger.info(f"Using default processed columns for subgroup analysis: {cols_for_subgroup_processing}")
    logger.info(f"Final columns for subgroup processing: {cols_for_subgroup_processing}")
    # --- End column determination ---

    logger.info("Analyzing Model Calibration...")
    calibration_results = None # Initialize
    if 'label' not in results_df.columns or 'probability' not in results_df.columns:
         logger.error("Cannot perform calibration analysis: 'label' or 'probability' missing from results_df.")
    else:
         calibration_results = analyze_calibration(
             results_df,
             subgroup_cols=cols_for_subgroup_processing, # Use the processed list
             n_bins=n_calibration_bins,
             min_samples_per_group=min_samples_for_calibration
         )
         if calibration_results: # Save if results were generated
             calib_path = os.path.join(eval_results_dir, 'calibration_analysis.json')
             try:
                 with open(calib_path, 'w') as f:
                     json.dump(calibration_results, f, indent=4, cls=NpEncoder)
                 logger.info(f"Calibration analysis results saved to {calib_path}")
             except Exception as e:
                 logger.error(f"Failed to save calibration analysis results: {e}", exc_info=True)

    logger.info("Calculating Subgroup Performance Metrics...")
    subgroup_metrics_calculated = calculate_subgroup_metrics(results_df, cols_for_subgroup_processing, optimal_threshold)

    logger.info("Analyzing Fairness (Descriptive)...")
    fairness_summary = analyze_fairness(subgroup_metrics_calculated)

    # Significance analysis and plotting are removed

    logger.info("Generating Visualizations (excluding calibration)...")
    all_metrics_for_plotting = {'Overall': overall_metrics, **subgroup_metrics_calculated}
    # We will add calibration_results to the summary, but not plot it here per model/dataset

    plot_metric_comparison(all_metrics_for_plotting, metrics_to_compare, eval_results_dir)
    plot_tpr_fpr_parity(all_metrics_for_plotting, eval_results_dir)
    plot_subgroup_roc_curves(results_df, cols_for_subgroup_processing, subgroup_metrics_calculated, overall_roc_data, eval_results_dir)
    plot_subgroup_prevalence(results_df, cols_for_subgroup_processing, eval_results_dir)
    age_corr_results = plot_age_correlation(results_df, eval_results_dir)

    # No per-model/dataset calibration plot here
    # The plot_calibration_curves function itself can be moved to src.evaluation.plotting if not already there
    # and called from evaluate_ood.py at the end.

    logger.info("Saving Final Evaluation Summary...")
    evaluation_summary = {
        'experiment_name': experiment_name,
        'optimal_threshold': float(optimal_threshold) if optimal_threshold is not None else None,
        'overall_metrics': overall_metrics,
        'subgroup_metrics': subgroup_metrics_calculated,
        'fairness_summary_descriptive': fairness_summary,
        'significance_analysis': {}, # Explicitly empty as it's removed
        'calibration_analysis': calibration_results if calibration_results else {}, # Store the data
        'age_correlation': age_corr_results if age_corr_results else {}
    }

    summary_path = os.path.join(eval_results_dir, 'evaluation_summary.json')
    try:
        with open(summary_path, 'w') as f:
            json.dump(evaluation_summary, f, indent=4, cls=NpEncoder)
        logger.info(f"Evaluation summary saved to {summary_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation summary: {e}", exc_info=True)

    logger.info(f"=== Evaluation Pipeline Finished for {experiment_name} ===")
    # Return the detailed metrics dictionary (which now includes calibration) and predictions_df
    return evaluation_summary, results_df # Return the full summary and df_predictions


def plot_significance_results(significance_results, results_dir, filename="significance_comparison_plot.png"):
    if not significance_results:
        logging.warning("No significance results found to plot.")
        return

    comparisons = list(significance_results.keys())
    if not comparisons:
        logging.warning("Significance results dictionary is empty.")
        return

    all_metrics = set()
    for comp_results in significance_results.values():
        all_metrics.update(comp_results.keys())
    # Ensure a consistent order -- REMOVED Equalized Odds special handling
    metric_order = sorted(list(all_metrics))

    n_comparisons = len(comparisons)
    n_metrics_max = len(metric_order)

    colors = sns.color_palette("viridis", n_metrics_max)
    metric_color_map = {metric: colors[i] for i, metric in enumerate(metric_order)}

    fig, axes = plt.subplots(n_comparisons, 1, figsize=(10, 5 * n_comparisons), sharex=False)
    if n_comparisons == 1: axes = [axes]

    for i, (comparison_name, comp_results) in enumerate(significance_results.items()):
        ax = axes[i]
        plot_data = []
        valid_metrics_in_comp = [m for m in metric_order if m in comp_results]

        for metric in valid_metrics_in_comp:
            stats = comp_results[metric]
            if all(k in stats for k in ['observed_diff', 'diff_ci_low', 'diff_ci_high', 'p_value']) and \
               all(isinstance(stats[k], (int, float)) and not np.isnan(stats[k]) for k in ['observed_diff', 'diff_ci_low', 'diff_ci_high', 'p_value']):
                plot_data.append({
                    'metric': metric,
                    'observed_diff': stats['observed_diff'],
                    'ci_low': stats['diff_ci_low'],
                    'ci_high': stats['diff_ci_high'],
                    'p_value': stats['p_value']
                })
            else:
                logging.warning(f"Skipping metric '{metric}' in comparison '{comparison_name}' due to missing/invalid data.")

        if not plot_data:
            ax.set_title(f"{comparison_name.replace('_', ' ')}\n(No valid data to plot)")
            ax.text(0.5, 0.5, "No plottable data", ha='center', va='center', transform=ax.transAxes)
            continue

        df_plot = pd.DataFrame(plot_data)
        df_plot = df_plot.set_index('metric').reindex(valid_metrics_in_comp).reset_index()

        x = np.arange(len(df_plot))
        bar_colors = [metric_color_map.get(m, 'grey') for m in df_plot['metric']]

        lower_err = np.maximum(0, df_plot['observed_diff'] - df_plot['ci_low'])
        upper_err = np.maximum(0, df_plot['ci_high'] - df_plot['observed_diff'])
        errors = [lower_err.values, upper_err.values]

        bars = ax.bar(x, df_plot['observed_diff'], yerr=errors,
                      capsize=7, color=bar_colors, edgecolor='black', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(df_plot['metric'], rotation=45, ha='right')
        ax.set_ylabel('Observed Difference')
        ax.set_title(f"Metric Differences: {comparison_name.replace('_', ' ')}")
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='y', linestyle=':', alpha=0.7)

        for j, bar in enumerate(bars):
            p_val = df_plot.loc[j, 'p_value']
            height = bar.get_height()
            ci_high = df_plot.loc[j, 'ci_high']
            ci_low = df_plot.loc[j, 'ci_low']
            text_y = ci_high + 0.03 * abs(ci_high) if height >= 0 else ci_low - 0.05 * abs(ci_low)
            va_align = 'bottom' if height >= 0 else 'top'

            if p_val < 0.001: stars = '***'
            elif p_val < 0.01: stars = '**'
            elif p_val < 0.05: stars = '*'
            else: stars = ''

            if stars:
                 ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + np.sign(height)*0.01, stars,
                         ha='center', va=va_align, fontsize=12, color='red', fontweight='bold')

    plt.suptitle('Significance of Metric Differences Between Subgroups', y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 1])

    save_path = os.path.join(results_dir, filename)
    try:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Significance comparison plot saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save significance plot: {e}", exc_info=True)
    plt.close(fig)


def _calculate_calibration_metrics(y_true, y_prob, n_bins=10, strategy='uniform'):
    metrics = {
        'brier_score': None, 'ece': None, 'mce': None,
        'calibration_curve_data': {'prob_true': [], 'prob_pred': [], 'bin_counts': []},
        'n_samples': len(y_true), 'n_positive': int(np.sum(y_true)), 'error': None
    }
    if len(y_true) == 0: # Check for empty explicitly
        metrics['error'] = "No samples."
        return metrics
    if len(np.unique(y_true)) < 2:
        # If only one class, still report Brier score if possible, but ECE/MCE not meaningful
        if metrics['n_samples'] > 0:
            try:
                metrics['brier_score'] = brier_score_loss(y_true, y_prob)
            except Exception as e_brier:
                metrics['error'] = f"Error calculating Brier score for single class: {e_brier}"
        metrics['error'] = metrics.get('error', "") + " Only one class present, ECE/MCE not calculated."
        return metrics # Return early for single class after attempting Brier

    try:
        metrics['brier_score'] = brier_score_loss(y_true, y_prob)
        # calibration_curve can raise ValueError if y_prob contains non-finite values
        # or if y_true is not binary. We assume y_true is binary by this point.
        if np.any(~np.isfinite(y_prob)):
            metrics['error'] = "Non-finite values found in probabilities."
            return metrics

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)

        # Recalculate bin counts and ECE/MCE manually for more control and to handle empty bins better.
        if strategy == 'uniform': bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
        elif strategy == 'quantile':
             quantiles = np.linspace(0, 1, n_bins + 1)
             # Handle cases where y_prob might have insufficient unique values for quantiles
             if len(np.unique(y_prob)) < n_bins +1 :
                  bins = np.linspace(0., 1. + 1e-8, n_bins + 1) # Fallback to uniform
                  logger.debug(f"Falling back to uniform binning for calibration due to insufficient unique probabilities for quantile strategy (n_unique_probs={len(np.unique(y_prob))}, n_bins={n_bins}).")
             else:
                  bins = np.percentile(y_prob, quantiles * 100); bins[0]=0.; bins[-1]=1. + 1e-8 # Ensure last bin includes 1.0
        else: raise ValueError(f"Unsupported binning strategy: {strategy}")

        binids = np.digitize(y_prob, bins[1:-1], right=False) # Ensure bins[1:-1] is valid
        
        bin_counts = np.bincount(binids, minlength=n_bins)
        bin_sum_probs = np.bincount(binids, weights=y_prob, minlength=n_bins)
        bin_true_pos = np.bincount(binids, weights=y_true, minlength=n_bins)
        
        non_empty_bin_mask = bin_counts > 0
        
        # Use prob_pred from sklearn's calibration_curve as it handles empty bins correctly for plotting points
        metrics['calibration_curve_data']['prob_true'] = prob_true.tolist()
        metrics['calibration_curve_data']['prob_pred'] = prob_pred.tolist()
        # For bin_counts, use actual counts of non-empty bins that correspond to prob_true/prob_pred
        metrics['calibration_curve_data']['bin_counts'] = bin_counts[non_empty_bin_mask].tolist()


        if np.sum(non_empty_bin_mask) > 0: # If there's at least one non-empty bin
            actual_prob_true_manual = np.divide(bin_true_pos[non_empty_bin_mask], bin_counts[non_empty_bin_mask],
                                     out=np.zeros_like(bin_true_pos[non_empty_bin_mask], dtype=float), where=bin_counts[non_empty_bin_mask]>0)
            actual_prob_pred_manual = np.divide(bin_sum_probs[non_empty_bin_mask], bin_counts[non_empty_bin_mask],
                                     out=np.zeros_like(bin_sum_probs[non_empty_bin_mask], dtype=float), where=bin_counts[non_empty_bin_mask]>0)
            
            bin_weights = bin_counts[non_empty_bin_mask] / metrics['n_samples']
            bin_abs_diff = np.abs(actual_prob_true_manual - actual_prob_pred_manual)
            metrics['ece'] = np.sum(bin_weights * bin_abs_diff)
            metrics['mce'] = np.max(bin_abs_diff)
        else:
            metrics['ece'], metrics['mce'] = None, None
            metrics['error'] = metrics.get('error', "") + " All bins were empty."

    except ValueError as e: metrics['error'] = f"ValueError in calibration: {e}"
    except Exception as e: metrics['error'] = f"General error in calibration: {e}"
    
    for key in ['brier_score', 'ece', 'mce']:
        if metrics[key] is not None and isinstance(metrics[key], (np.float32, np.float64)):
             metrics[key] = float(metrics[key]) # Ensure JSON serializable
    return metrics


def analyze_calibration(results_df, subgroup_cols, n_bins=10, min_samples_per_group=30):
    logger = logging.getLogger(__name__) # Use module-level logger
    calibration_results = {}
    if 'label' not in results_df or 'probability' not in results_df:
        logger.error("Calibration analysis requires 'label' and 'probability' columns.")
        return calibration_results

    y_true_all = results_df['label'].values
    y_prob_all = results_df['probability'].values

    logger.info(f"Analyzing overall calibration ({len(y_true_all)} samples)...")
    if len(y_true_all) < min_samples_per_group:
        logger.warning(f"Skipping overall calibration: Insufficient samples ({len(y_true_all)} < {min_samples_per_group})")
        calibration_results['Overall'] = {
            'brier_score': None, 'ece': None, 'mce': None, 'calibration_curve_data': None,
            'n_samples': len(y_true_all), 'n_positive': int(np.sum(y_true_all)),
            'error': f"Insufficient samples ({len(y_true_all)} < {min_samples_per_group})"
        }
    else:
        calibration_results['Overall'] = _calculate_calibration_metrics(y_true_all, y_prob_all, n_bins)
        if calibration_results['Overall'].get('error'):
            logger.warning(f"Overall calibration processing issue: {calibration_results['Overall']['error']}")

    # `subgroup_cols` now contains the actual column names to iterate over (e.g., 'age_group', 'eye_group', 'camera')
    for col_to_group_by in subgroup_cols:
        if col_to_group_by not in results_df.columns:
            logger.warning(f"Subgroup column '{col_to_group_by}' for calibration not found in results_df. Skipping.")
            continue

        unique_group_values = results_df[col_to_group_by].unique()

        for group_val_raw in unique_group_values:
            group_val_str = str(group_val_raw) if pd.notna(group_val_raw) else "NaN"
            group_key_for_calib_dict = f"{col_to_group_by}_{group_val_str}"

            # Skip if 'Overall' to avoid reprocessing, though unique_groups_in_col shouldn't yield 'Overall'
            if group_key_for_calib_dict == 'Overall': continue 
            # Also skip if already processed (e.g. if subgroup_cols had redundant ways to get same group)
            if group_key_for_calib_dict in calibration_results: continue


            logger.info(f"Analyzing calibration for subgroup: {group_key_for_calib_dict} (from column {col_to_group_by})")

            if pd.isna(group_val_raw):
                subgroup_df = results_df[results_df[col_to_group_by].isna()]
            else:
                subgroup_df = results_df[results_df[col_to_group_by] == group_val_raw]
            
            y_true_sub = subgroup_df['label'].values
            y_prob_sub = subgroup_df['probability'].values

            if len(y_true_sub) < min_samples_per_group:
                logger.warning(f"Skipping calibration for subgroup '{group_key_for_calib_dict}': "
                               f"Insufficient samples ({len(y_true_sub)} < {min_samples_per_group})")
                calibration_results[group_key_for_calib_dict] = {
                    'brier_score': None, 'ece': None, 'mce': None, 'calibration_curve_data': None,
                    'n_samples': len(y_true_sub), 'n_positive': int(np.sum(y_true_sub)),
                    'error': f"Insufficient samples ({len(y_true_sub)} < {min_samples_per_group})"
                }
                continue

            current_group_calib_metrics = _calculate_calibration_metrics(y_true_sub, y_prob_sub, n_bins)
            calibration_results[group_key_for_calib_dict] = current_group_calib_metrics
            if current_group_calib_metrics.get('error'):
                 logger.warning(f"Calibration processing issue for subgroup '{group_key_for_calib_dict}': {current_group_calib_metrics['error']}")

    logger.info("Calibration analysis finished.")
    return calibration_results