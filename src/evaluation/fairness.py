import numpy as np
import pandas as pd
from sklearn.utils import resample
from scipy.stats import percentileofscore
from itertools import combinations
import json
import os
import logging
from sklearn.metrics import confusion_matrix

# Import necessary functions from sibling modules
from .metrics import calculate_metric, calculate_rates # Assuming metrics.py is in the same directory
from ..utils.helpers import NpEncoder # Import NpEncoder from utils

logger = logging.getLogger(__name__)

def analyze_fairness(subgroup_metrics):
    """
    Calculates descriptive fairness metrics (e.g., max differences)
    based on pre-calculated subgroup performance.

    Args:
        subgroup_metrics (dict): Dict where keys are subgroup names (e.g., 'age_group_<65')
                                 and values are dicts of metrics {'TPR': x, 'FPR': y, ...}.
                                 Should *not* include the 'Overall' metrics here.

    Returns:
        dict: Dictionary containing fairness summary statistics.
    """
    logger.info("Analyzing Descriptive Fairness Metrics...")
    fairness_summary = {}

    # Identify groups with valid TPR/FPR for Equalized Odds checks
    valid_eo_groups = {
        k: v for k, v in subgroup_metrics.items()
        if v is not None and v.get('TPR') is not None and v.get('FPR') is not None
    }

    if len(valid_eo_groups) < 2:
        logger.warning("Insufficient valid subgroup data (<2 groups with TPR/FPR) to calculate fairness differences.")
        fairness_summary.update({
            'max_tpr_difference': None, 'max_fpr_difference': None,
            'min_tpr': None, 'max_fpr': None,
            'min_tpr_group': None, 'max_fpr_group': None
        })
    else:
        tprs = np.array([valid_eo_groups[k]['TPR'] for k in valid_eo_groups])
        fprs = np.array([valid_eo_groups[k]['FPR'] for k in valid_eo_groups])
        group_keys_eo = list(valid_eo_groups.keys())

        max_tpr_diff = np.ptp(tprs) # Peak-to-peak difference (max - min)
        max_fpr_diff = np.ptp(fprs)
        min_tpr = np.min(tprs)
        max_fpr = np.max(fprs) # Max FPR is often the worst case
        min_tpr_idx = np.argmin(tprs)
        max_fpr_idx = np.argmax(fprs)
        min_tpr_group = group_keys_eo[min_tpr_idx]
        max_fpr_group = group_keys_eo[max_fpr_idx]

        logger.info("Equalized Odds Check (Descriptive):")
        logger.info(f"  Max TPR Difference: {max_tpr_diff:.4f} (Range: [{np.min(tprs):.4f} - {np.max(tprs):.4f}])")
        logger.info(f"  Max FPR Difference: {max_fpr_diff:.4f} (Range: [{np.min(fprs):.4f} - {np.max(fprs):.4f}])")

        fairness_summary.update({
            'max_tpr_difference': float(max_tpr_diff),
            'max_fpr_difference': float(max_fpr_diff),
            'min_tpr': float(min_tpr),
            'max_fpr': float(max_fpr), # Report max FPR
            'min_tpr_group': min_tpr_group,
            'max_fpr_group': max_fpr_group,
        })

    # Min-Max Fairness across other metrics (using all subgroups where metric is valid)
    all_valid_groups = {k:v for k,v in subgroup_metrics.items() if v is not None}
    group_keys_all = list(all_valid_groups.keys())

    def get_min_metric(metric_name):
         valid_metrics = [(k, g[metric_name]) for k, g in all_valid_groups.items() if g.get(metric_name) is not None]
         if not valid_metrics: return None, None
         min_val = min(m for _, m in valid_metrics)
         min_group = min(valid_metrics, key=lambda item: item[1])[0]
         return min_val, min_group

    min_auc, min_auc_group = get_min_metric('AUC')
    min_acc, min_acc_group = get_min_metric('Accuracy')

    logger.info("Min-Max Fairness (Worst-Case Performance):")
    logger.info(f"  Minimum TPR: {fairness_summary.get('min_tpr', 'N/A'):.4f} (Group: {fairness_summary.get('min_tpr_group', 'N/A')})")
    logger.info(f"  Maximum FPR: {fairness_summary.get('max_fpr', 'N/A'):.4f} (Group: {fairness_summary.get('max_fpr_group', 'N/A')})")
    logger.info(f"  Minimum AUC: {(f'{min_auc:.4f}' if min_auc is not None else 'N/A')} (Group: {min_auc_group or 'N/A'})")
    logger.info(f"  Minimum Accuracy: {(f'{min_acc:.4f}' if min_acc is not None else 'N/A')} (Group: {min_acc_group or 'N/A'})")

    fairness_summary.update({
        'min_auc': float(min_auc) if min_auc is not None else None,
        'min_accuracy': float(min_acc) if min_acc is not None else None,
        'min_auc_group': min_auc_group,
        'min_acc_group': min_acc_group,
    })

    return fairness_summary


def bootstrap_metric_difference(df, group_col, group1_val, group2_val,
                                metric_name, optimal_threshold,
                                n_bootstraps=1000, alpha=0.05):
    """ Calculates bootstrap CI and p-value for metric difference between two groups. """
    logger.debug(f"Bootstrapping difference for {metric_name} between {group_col}={group1_val} and {group_col}={group2_val}")

    # Ensure the group values exist and filter data
    if group1_val not in df[group_col].unique() or group2_val not in df[group_col].unique():
         logger.warning(f"One or both group values ('{group1_val}', '{group2_val}') not found in column '{group_col}' for bootstrapping.")
         return None
    group1_df = df[df[group_col] == group1_val]
    group2_df = df[df[group_col] == group2_val]

    if group1_df.empty or group2_df.empty:
        logger.warning(f"Cannot bootstrap: Group '{group1_val}' or '{group2_val}' in '{group_col}' is empty.")
        return None
    # Warning for small sample sizes
    min_n_warn = 10
    if len(group1_df) < min_n_warn or len(group2_df) < min_n_warn:
         logger.warning(f"Small sample size for bootstrapping {metric_name} ({group_col}): n1={len(group1_df)}, n2={len(group2_df)}. Results may be unstable.")

    bootstrap_diffs = []
    skipped_iters = 0

    # Pre-calculate predictions for original groups
    preds1_orig = (group1_df['probability'].values >= optimal_threshold).astype(int)
    preds2_orig = (group2_df['probability'].values >= optimal_threshold).astype(int)
    metric1_orig = calculate_metric(metric_name, group1_df['label'].values, group1_df['probability'].values, preds1_orig)
    metric2_orig = calculate_metric(metric_name, group2_df['label'].values, group2_df['probability'].values, preds2_orig)

    observed_diff = None
    if metric1_orig is not None and metric2_orig is not None:
        observed_diff = metric1_orig - metric2_orig
    else:
        logger.warning(f"Could not calculate observed difference for {metric_name} between {group1_val} and {group2_val}.")
        # Proceed with bootstrapping CI, but observed diff will be None

    # Bootstrapping loop
    for _ in range(n_bootstraps):
        try:
            # Stratified resampling (within each group)
            sample1 = resample(group1_df, replace=True, n_samples=len(group1_df))
            sample2 = resample(group2_df, replace=True, n_samples=len(group2_df))
        except ValueError as ve:
             logger.warning(f"Resampling error during bootstrap iter: {ve}")
             skipped_iters += 1
             continue

        preds1_boot = (sample1['probability'].values >= optimal_threshold).astype(int)
        preds2_boot = (sample2['probability'].values >= optimal_threshold).astype(int)

        metric1_boot = calculate_metric(metric_name, sample1['label'].values, sample1['probability'].values, preds1_boot)
        metric2_boot = calculate_metric(metric_name, sample2['label'].values, sample2['probability'].values, preds2_boot)

        if metric1_boot is not None and metric2_boot is not None:
            bootstrap_diffs.append(metric1_boot - metric2_boot)
        else:
            skipped_iters += 1

    valid_bootstraps = n_bootstraps - skipped_iters
    if valid_bootstraps < n_bootstraps * 0.5: # Warning if less than half the iterations were valid
         logger.warning(f"High skip rate ({skipped_iters}/{n_bootstraps}) in bootstrap for {metric_name} ({group_col}={group1_val} vs {group2_val}). CI/p-value may be unreliable.")

    if not bootstrap_diffs:
        logger.error(f"Could not calculate any valid bootstrap differences for {metric_name} between {group1_val} and {group2_val}.")
        return {'metric': metric_name, 'group1': group1_val, 'group2': group2_val,
                'observed_diff': observed_diff, 'diff_ci_low': None, 'diff_ci_high': None,
                'p_value': None, 'n_bootstraps': 0, 'alpha': alpha, 'error': 'No valid bootstrap samples'}

    # Calculate percentile CI
    lower_perc = (alpha / 2) * 100
    upper_perc = (1 - alpha / 2) * 100
    lower_bound = np.percentile(bootstrap_diffs, lower_perc)
    upper_bound = np.percentile(bootstrap_diffs, upper_perc)

    # Calculate two-sided p-value (approximate)
    # P(D >= |d_obs|) if d_obs > 0 or P(D <= -|d_obs|) if d_obs < 0, under H0 (diff=0)
    # Simpler approximation: 2 * min(P(D >= 0), P(D <= 0))
    pct_below_zero = percentileofscore(bootstrap_diffs, 0.0, kind='strict') # % strictly below 0
    pct_above_zero = percentileofscore(bootstrap_diffs, 0.0, kind='weak') # % <= 0 -> (100 - %) is % > 0
    pct_above_zero = 100.0 - pct_above_zero # Convert to % strictly above 0

    # Handle edge cases where all diffs are on one side of 0
    if pct_below_zero == 0 or pct_above_zero == 0:
         p_value = 1.0 / (valid_bootstraps + 1) # Smallest possible p-value estimate
    else:
         p_value = 2 * min(pct_below_zero / 100.0, pct_above_zero / 100.0)


    return {
        'metric': metric_name,
        'group1': group1_val,
        'group2': group2_val,
        'observed_diff': observed_diff,
        'diff_ci_low': lower_bound,
        'diff_ci_high': upper_bound,
        'p_value': p_value,
        'n_bootstraps': valid_bootstraps,
        'alpha': alpha
    }


def analyze_significance(results_df, subgroup_cols, subgroup_metrics, optimal_threshold, results_dir, n_bootstraps=1000, alpha=0.05):
    """ Performs statistical tests comparing metrics between subgroup pairs using bootstrapping. """
    logger.info("Analyzing Statistical Significance of Metric Differences via Bootstrapping...")
    if results_df is None or results_df.empty:
        logger.error("Cannot perform significance testing: Input DataFrame empty.")
        return {}

    significance_results = {}
    # Focus on key fairness-related metrics
    metrics_to_test = ['AUC', 'Accuracy', 'TPR', 'FPR']

    for group_col in subgroup_cols:
        if group_col not in results_df.columns:
            logger.warning(f"Skipping significance tests for missing column: {group_col}")
            continue

        # Ensure group column has categorical or string type for unique()
        try:
             groups = results_df[group_col].astype('category').cat.categories.tolist()
             # groups = results_df[group_col].dropna().unique() # Alternative if direct unique works
        except Exception as e:
             logger.error(f"Could not get unique groups for column '{group_col}': {e}. Skipping.")
             continue

        if len(groups) < 2:
            logger.info(f"Skipping significance tests for '{group_col}': needs at least 2 groups (found {len(groups)}).")
            continue

        logger.info(f"\n--- Comparing pairs within: {group_col} ---")
        for group1_val, group2_val in combinations(groups, 2):
            comparison_key = f"{group_col}:{group1_val}_vs_{group2_val}"
            significance_results[comparison_key] = {}
            logger.info(f"  Comparing: {group1_val} vs {group2_val}")

            for metric in metrics_to_test:
                # Check if metric was calculable for both groups in the original analysis
                metric1_exists = subgroup_metrics.get(f"{group_col}_{group1_val}", {}).get(metric) is not None
                metric2_exists = subgroup_metrics.get(f"{group_col}_{group2_val}", {}).get(metric) is not None

                if not (metric1_exists and metric2_exists):
                     logger.debug(f"    Skipping {metric}: Metric not originally available for both groups.")
                     continue

                try:
                    stat_res = bootstrap_metric_difference(
                        results_df, group_col, group1_val, group2_val,
                        metric, optimal_threshold, n_bootstraps, alpha
                    )
                    if stat_res:
                        significance_results[comparison_key][metric] = stat_res
                        obs_diff = stat_res['observed_diff']
                        ci_low = stat_res['diff_ci_low']
                        ci_high = stat_res['diff_ci_high']
                        p_val = stat_res['p_value']
                        n_bs = stat_res['n_bootstraps']

                        # Determine significance based on CI containing 0
                        is_significant = not (ci_low <= 0 <= ci_high) if ci_low is not None and ci_high is not None else False
                        sig_marker_ci = "*" if is_significant else "" # Mark if CI excludes 0

                        # P-value significance marker
                        if p_val is None:
                            sig_marker_p = "n/a"
                        else:
                            sig_marker_p = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < alpha else "ns"

                        # Log concise summary
                        obs_diff_str = f"{obs_diff:.4f}" if obs_diff is not None else "N/A"
                        ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]" if ci_low is not None else "[N/A, N/A]"
                        p_val_str = f"{p_val:.4f}" if p_val is not None else "N/A"

                        logger.info(f"    {metric+':':<10} Obs Diff={obs_diff_str}, {int((1-alpha)*100)}% CI={ci_str}{sig_marker_ci}, p={p_val_str} {sig_marker_p} (n={n_bs})")

                    else:
                         logger.warning(f"    Bootstrapping failed for {metric}.")
                         significance_results[comparison_key][metric] = {'error': 'Bootstrapping returned None'}


                except Exception as e:
                    logger.error(f"    Error during bootstrap comparison for {metric}: {e}", exc_info=True)
                    significance_results[comparison_key][metric] = {'error': str(e)}

    # Save significance results
    sig_path = os.path.join(results_dir, 'significance_analysis.json')
    try:
        os.makedirs(os.path.dirname(sig_path), exist_ok=True)
        with open(sig_path, 'w') as f:
            # Use NpEncoder to handle potential numpy types in results
            json.dump(significance_results, f, indent=4, cls=NpEncoder)
        logger.info(f"Significance analysis results saved to {sig_path}")
    except Exception as e:
        logger.error(f"Failed to save significance analysis results: {e}", exc_info=True)

    return significance_results


def calculate_fnr_fpr_for_subgroup(df_subgroup: pd.DataFrame, label_col: str, 
                                   pred_label_col: str, positive_label_value: int = 1):
    """Calculates FNR and FPR for a given subgroup DataFrame."""
    y_true_sub = df_subgroup[label_col]
    y_pred_sub = df_subgroup[pred_label_col]

    if len(y_true_sub) == 0:
        # logger.debug(f"Subgroup for FNR/FPR calculation is empty.")
        return np.nan, np.nan

    # Ensure labels are binary (0 and 1) for confusion_matrix
    # where 1 corresponds to the positive_label_value
    y_true_binary = (y_true_sub == positive_label_value).astype(int)
    y_pred_binary = (y_pred_sub == positive_label_value).astype(int)

    # Use labels=[0, 1] to ensure consistent order of TN, FP, FN, TP
    # even if one class is not present in the subgroup's predictions/labels.
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]) 
    tn, fp, fn, tp = cm.ravel()

    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    
    return fnr, fpr

def calculate_underdiagnosis_disparities(
    df_predictions: pd.DataFrame, 
    subgroup_definitions: dict,
    label_col: str = 'types', 
    pred_label_col: str = 'predicted_label',
    positive_label_value: int = 1
    ) -> dict:
    """
    Calculates FNR and FPR disparities between specified subgroups.

    Args:
        df_predictions: DataFrame with true labels, predicted labels, and subgroup columns.
        subgroup_definitions: Dict defining comparisons. Example:
            {
                "age": {"col": "age_group", "favored": "<65", "disfavored": ">=65"},
                "eye": {"col": "eye_group", "favored": "OD", "disfavored": "OS"}
            }
        label_col: Name of the true label column.
        pred_label_col: Name of the predicted label column (binarized).
        positive_label_value: The value in label_col representing the positive class.

    Returns:
        A dictionary containing FNR/FPR for each group and their disparities.
    """
    disparities_results = {}
    
    if pred_label_col not in df_predictions.columns:
        logger.error(f"Predicted label column '{pred_label_col}' not found in df_predictions. Cannot calculate disparities.")
        return disparities_results
    if label_col not in df_predictions.columns:
        logger.error(f"True label column '{label_col}' not found in df_predictions. Cannot calculate disparities.")
        return disparities_results

    for attr_key, definition in subgroup_definitions.items():
        subgroup_col_name = definition['col']
        favored_value = str(definition['favored']) # Ensure string for comparison
        disfavored_value = str(definition['disfavored'])

        if subgroup_col_name not in df_predictions.columns:
            logger.warning(f"Subgroup column '{subgroup_col_name}' for attribute '{attr_key}' "
                           "not found in predictions DataFrame. Skipping this disparity.")
            continue

        # Ensure the subgroup column is string type for consistent filtering
        df_predictions_copy = df_predictions.copy()
        try:
            df_predictions_copy[subgroup_col_name] = df_predictions_copy[subgroup_col_name].astype(str)
        except Exception as e:
            logger.error(f"Could not convert subgroup column '{subgroup_col_name}' to string: {e}. Skipping disparity for '{attr_key}'.")
            continue
            
        unique_values_in_col = df_predictions_copy[subgroup_col_name].unique()

        if favored_value not in unique_values_in_col:
            logger.warning(f"Favored group value '{favored_value}' for attribute '{attr_key}' "
                           f"(column '{subgroup_col_name}') not present in data. Unique values: {unique_values_in_col}. Skipping.")
            continue
        if disfavored_value not in unique_values_in_col:
            logger.warning(f"Disfavored group value '{disfavored_value}' for attribute '{attr_key}' "
                           f"(column '{subgroup_col_name}') not present in data. Unique values: {unique_values_in_col}. Skipping.")
            continue

        df_favored_subgroup = df_predictions_copy[df_predictions_copy[subgroup_col_name] == favored_value]
        df_disfavored_subgroup = df_predictions_copy[df_predictions_copy[subgroup_col_name] == disfavored_value]

        fnr_favored, fpr_favored = calculate_fnr_fpr_for_subgroup(
            df_favored_subgroup, label_col, pred_label_col, positive_label_value
        )
        fnr_disfavored, fpr_disfavored = calculate_fnr_fpr_for_subgroup(
            df_disfavored_subgroup, label_col, pred_label_col, positive_label_value
        )
        
        # Store individual group metrics
        disparities_results[f'FNR_{attr_key}_{favored_value}'] = fnr_favored
        disparities_results[f'FPR_{attr_key}_{favored_value}'] = fpr_favored
        disparities_results[f'FNR_{attr_key}_{disfavored_value}'] = fnr_disfavored
        disparities_results[f'FPR_{attr_key}_{disfavored_value}'] = fpr_disfavored
        disparities_results[f'COUNT_{attr_key}_{favored_value}'] = len(df_favored_subgroup)
        disparities_results[f'COUNT_{attr_key}_{disfavored_value}'] = len(df_disfavored_subgroup)


        # Calculate and store disparities (Disfavored - Favored)
        fnr_disp_key = f'FNR_disparity_{attr_key}_{disfavored_value}_vs_{favored_value}'
        if not np.isnan(fnr_disfavored) and not np.isnan(fnr_favored):
            disparities_results[fnr_disp_key] = fnr_disfavored - fnr_favored
        else:
            disparities_results[fnr_disp_key] = np.nan
            logger.debug(f"Could not calculate FNR disparity for {attr_key} due to NaN in subgroup FNRs.")
        
        fpr_disp_key = f'FPR_disparity_{attr_key}_{disfavored_value}_vs_{favored_value}'
        if not np.isnan(fpr_disfavored) and not np.isnan(fpr_favored):
            disparities_results[fpr_disp_key] = fpr_disfavored - fpr_favored
        else:
            disparities_results[fpr_disp_key] = np.nan
            logger.debug(f"Could not calculate FPR disparity for {attr_key} due to NaN in subgroup FPRs.")
            
    return disparities_results