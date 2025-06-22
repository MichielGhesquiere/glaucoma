# src/evaluation/plotting.py
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

def plot_tsne_by_attribute(features: np.ndarray, 
                           true_labels: np.ndarray, # Renamed for clarity
                           metadata_df: pd.DataFrame, 
                           attribute_to_color_by: str,
                           output_dir: str, 
                           filename_prefix: str,
                           perplexity: float = 30.0, 
                           n_iter: int = 1000, 
                           learning_rate: str = 'auto', # Changed to str to match sklearn 'auto'
                           max_samples_plot: int = 2000, 
                           min_samples_per_group: int = 20):
    """
    Generates and saves a t-SNE plot colored by a specified attribute,
    with marker shapes for true labels.
    """
    if features is None or true_labels is None or metadata_df is None or metadata_df.empty:
        logger.warning(f"Skipping t-SNE plot for '{attribute_to_color_by}': features, true_labels, or metadata is missing/empty.")
        return
    if attribute_to_color_by not in metadata_df.columns:
        logger.warning(f"Skipping t-SNE plot: Attribute '{attribute_to_color_by}' not found in metadata_df columns: {metadata_df.columns.tolist()}.")
        return
    if len(features) != len(true_labels) or len(features) != len(metadata_df):
        logger.warning(f"Skipping t-SNE plot for '{attribute_to_color_by}': Length mismatch between features ({len(features)}), "
                       f"true_labels ({len(true_labels)}), and metadata_df ({len(metadata_df)}).")
        return

    # --- Data Preparation ---
    # Create a working DataFrame from metadata, ensuring original index is preserved if possible
    # This df will be filtered.
    df_analysis = metadata_df.copy() 
    df_analysis['true_label_for_shape'] = true_labels # Store true labels for marker shapes

    # Filter by groups of the coloring attribute that have enough samples
    group_counts = df_analysis[attribute_to_color_by].value_counts()
    groups_to_keep = group_counts[group_counts >= min_samples_per_group].index
    
    if groups_to_keep.empty:
        logger.warning(f"No groups for attribute '{attribute_to_color_by}' meet min_samples_per_group ({min_samples_per_group}). Skipping t-SNE plot.")
        return
        
    df_filtered = df_analysis[df_analysis[attribute_to_color_by].isin(groups_to_keep)].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Get the indices from the filtered DataFrame to select corresponding features and labels
    # This assumes that metadata_df (and thus df_analysis) had an index that aligns with the original order of features and true_labels.
    # If metadata_df was created from scratch and rows were reordered, this needs careful handling.
    # Assuming metadata_df.index corresponds to the original 0..N-1 indexing of features and true_labels.
    indices_for_plot = df_filtered.index.values 

    features_for_plot = features[indices_for_plot]
    true_labels_for_plot = true_labels[indices_for_plot] # Use original true_labels array with filtered indices
    coloring_attribute_values = df_filtered[attribute_to_color_by].astype(str).tolist() # From the filtered df

    if len(features_for_plot) == 0:
        logger.warning(f"No samples remaining after filtering for attribute '{attribute_to_color_by}'. Skipping t-SNE plot.")
        return

    # --- Subsampling for t-SNE input ---
    if len(features_for_plot) > max_samples_plot:
        logger.info(f"Subsampling {max_samples_plot} from {len(features_for_plot)} for t-SNE plot by '{attribute_to_color_by}'.")
        sample_indices = np.random.choice(len(features_for_plot), max_samples_plot, replace=False)
        features_tsne_input = features_for_plot[sample_indices]
        true_labels_tsne_input = true_labels_for_plot[sample_indices]
        coloring_attr_tsne_input = [coloring_attribute_values[i] for i in sample_indices]
    else:
        features_tsne_input = features_for_plot
        true_labels_tsne_input = true_labels_for_plot
        coloring_attr_tsne_input = coloring_attribute_values
    
    if len(features_tsne_input) < 5 : 
        logger.warning(f"Too few samples ({len(features_tsne_input)}) after filtering/subsampling for t-SNE by '{attribute_to_color_by}'. Skipping plot.")
        return

    # --- t-SNE Computation ---
    actual_perplexity = float(perplexity)
    # Perplexity must be less than n_samples. Also, n_samples should be at least 3 for default init='pca' with n_components=2.
    if len(features_tsne_input) <= actual_perplexity or len(features_tsne_input) < 3:
        old_perplexity = actual_perplexity
        actual_perplexity = max(1.0, float(len(features_tsne_input) - 2.0)) 
        if len(features_tsne_input) < 3:
             logger.warning(f"Cannot run t-SNE: Need at least 3 samples for perplexity {actual_perplexity}, got {len(features_tsne_input)}. Skipping plot for '{attribute_to_color_by}'.")
             return
        logger.warning(f"Perplexity ({old_perplexity}) too high for n_samples ({len(features_tsne_input)}). Adjusted to {actual_perplexity} for t-SNE plot by '{attribute_to_color_by}'.")

    # Ensure learning_rate is valid for sklearn
    lr_tsne = 'auto' if isinstance(learning_rate, str) and learning_rate.lower() == 'auto' else float(learning_rate)

    tsne_model = TSNE(n_components=2, perplexity=actual_perplexity, n_iter=int(n_iter),
                      learning_rate=lr_tsne, random_state=42, init='pca', n_jobs=-1)
    try:
        embeddings = tsne_model.fit_transform(features_tsne_input)
    except Exception as e:
        logger.error(f"t-SNE fitting failed for attribute '{attribute_to_color_by}': {e}", exc_info=True)
        return

    # --- Plotting ---
    df_embedded = pd.DataFrame({
        'tsne1': embeddings[:,0],
        'tsne2': embeddings[:,1],
        'true_label': true_labels_tsne_input,      # These are the true binary labels (0 or 1)
        'color_attribute': coloring_attr_tsne_input # Values of the attribute used for coloring
    })
    
    plt.figure(figsize=(16,12)) # Consider adjusting based on number of legend items
    
    unique_color_values = sorted(df_embedded['color_attribute'].unique())
    num_unique_colors = len(unique_color_values)
    palette = sns.color_palette("tab20" if num_unique_colors > 10 else "tab10", n_colors=num_unique_colors)
    color_map = {val: palette[i % len(palette)] for i, val in enumerate(unique_color_values)}
    
    markers_dict = {0: "o", 1: "X"} 
    label_names = {0: "Normal", 1: "Glaucoma"} # Or use args.class_names if available

    # Plot points, ensuring legend items are created correctly
    for color_attr_val in unique_color_values:
        # Subset for the current value of the coloring attribute
        subset_by_color_attr = df_embedded[df_embedded['color_attribute'] == color_attr_val]
        
        for true_label_val, marker_style in markers_dict.items():
            # Further subset by true label
            final_subset = subset_by_color_attr[subset_by_color_attr['true_label'] == true_label_val]
            if not final_subset.empty:
                plt.scatter(final_subset['tsne1'], final_subset['tsne2'],
                            label=f"{color_attr_val} - {label_names.get(true_label_val, true_label_val)} (N={len(final_subset)})",
                            alpha=0.7, 
                            color=color_map[color_attr_val], # Color by the 'color_attribute'
                            marker=marker_style,             # Marker by the 'true_label'
                            s=60) # Adjust size as needed
    
    plt.title(f't-SNE of Embeddings by "{attribute_to_color_by}" (Markers: True Label)\nModel: {filename_prefix} (N_plot={len(features_tsne_input)})', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    
    # Adjust legend position and size
    # If many legend items, consider placing it outside or reducing font size.
    legend_title = f"{attribute_to_color_by} - True Label (Count)"
    if num_unique_colors * len(markers_dict) > 15 : # If many legend entries
        plt.legend(title=legend_title, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', ncol= (2 if num_unique_colors * len(markers_dict) > 30 else 1) )
        plt.tight_layout(rect=[0, 0, 0.80, 0.97]) # Adjust rect for external legend
    else:
        plt.legend(title=legend_title, loc='best', fontsize='medium')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.grid(True, alpha=0.3, linestyle=':')
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{filename_prefix}_tsne_by_{attribute_to_color_by}.png")
    try:
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        logger.info(f"t-SNE plot by '{attribute_to_color_by}' saved to: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save t-SNE plot by '{attribute_to_color_by}': {e}", exc_info=True)
    plt.close()


def plot_all_roc_curves(all_roc_data: dict, output_dir: str):
    """
    Plots ROC curves for multiple models on the same dataset.
    all_roc_data = {
        dataset_name1: {model_id1: {'fpr': ..., 'tpr': ..., 'auc': ...}, model_id2: ...},
        dataset_name2: ...
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    for dataset_name, models_roc_data in all_roc_data.items():
        if not models_roc_data:
            logger.warning(f"No ROC data available to plot for dataset: {dataset_name}")
            continue
        
        plt.figure(figsize=(10, 8))
        for model_id, roc_info in models_roc_data.items():
            if 'fpr' in roc_info and 'tpr' in roc_info and 'auc' in roc_info:
                plt.plot(roc_info['fpr'], roc_info['tpr'], lw=2,
                         label=f"{model_id} (AUC = {roc_info['auc']:.3f})")
            else:
                logger.warning(f"Missing 'fpr', 'tpr', or 'auc' key for model '{model_id}' on dataset '{dataset_name}'. Skipping its curve.")

        plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--') # Diagonal reference line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05]) # Slight margin at the top
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title(f'Receiver Operating Characteristic (ROC) Curves\nDataset: {dataset_name}', fontsize=14)
        plt.legend(loc="lower right", fontsize="small", title="Model (AUC)")
        plt.grid(True, linestyle=':', alpha=0.6)
        
        roc_plot_path = os.path.join(output_dir, f"roc_curves_comparison_{dataset_name}.png")
        try:
            plt.savefig(roc_plot_path, bbox_inches='tight')
            logger.info(f"Saved combined ROC plot for dataset '{dataset_name}' to: {roc_plot_path}")
        except Exception as e:
            logger.error(f"Failed to save combined ROC plot for dataset '{dataset_name}': {e}", exc_info=True)
        plt.close()


def plot_all_disparities(all_disparity_data: dict, output_dir: str):
    """
    Plots bar charts for FNR/FPR disparities across models for each dataset and attribute.
    all_disparity_data = {
        dataset_name1: {
            model_id1: {
                'FNR_disparity_age_groupB_vs_groupA': 0.1, 
                'FPR_disparity_age_groupB_vs_groupA': -0.05, ...
            },
            model_id2: {...}
        }, ...
    }
    """
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, models_disparity_metrics in all_disparity_data.items():
        if not models_disparity_metrics:
            logger.warning(f"No disparity data available to plot for dataset: {dataset_name}")
            continue

        # --- Reformat data for easier plotting with Seaborn ---
        plot_data_list = []
        for model_id, metrics_dict in models_disparity_metrics.items():
            for metric_key, disparity_value in metrics_dict.items():
                if 'disparity' in metric_key.lower() and not pd.isna(disparity_value):
                    try:
                        # Expecting key like: 'FNR_disparity_attribute_disfavoredVal_vs_favoredVal'
                        parts = metric_key.split('_')
                        metric_type = parts[0] # FNR or FPR
                        # parts[1] is 'disparity'
                        attribute_name = parts[2]
                        comparison_detail = "_".join(parts[3:]) # e.g., groupB_vs_groupA
                        
                        plot_data_list.append({
                            'Model': model_id,
                            'MetricType': metric_type,
                            'Attribute': attribute_name,
                            'Comparison': comparison_detail,
                            'DisparityValue': disparity_value
                        })
                    except IndexError:
                        logger.warning(f"Could not parse disparity metric key: '{metric_key}'. Skipping.")
        
        if not plot_data_list:
            logger.info(f"No valid disparity values to plot for dataset: {dataset_name} after parsing.")
            continue

        df_plot = pd.DataFrame(plot_data_list)

        # --- Create a plot for each (Attribute, MetricType) combination ---
        for (attribute, metric_type), group_data in df_plot.groupby(['Attribute', 'MetricType']):
            if group_data.empty:
                continue
            
            plt.figure(figsize=(max(10, 0.7 * len(group_data['Model'].unique())), 7)) # Adjust width by num models
            
            # If multiple comparisons exist for the same attribute (e.g. age_A_vs_B and age_C_vs_D)
            # hue can be used. For now, assuming one comparison type per (attribute, metric_type)
            # based on how subgroup_definitions are typically structured.
            # If hue is needed, ensure 'Comparison' provides distinct, meaningful labels.
            
            # Sort models for consistent plotting order if desired (e.g., alphabetically)
            # model_order = sorted(group_data['Model'].unique())
            # sns.barplot(data=group_data, x='Model', y='DisparityValue', order=model_order, ...)

            ax = sns.barplot(data=group_data, x='Model', y='DisparityValue', 
                             hue='Comparison', # Use hue if multiple comparison types for this attr/metric
                             palette="viridis", dodge= (group_data['Comparison'].nunique() > 1) ) # Dodge if hue is active

            # Get the comparison string for the title (assuming it's mostly uniform for this group)
            # If varied, the legend will clarify.
            title_comparison_str = group_data['Comparison'].iloc[0] if group_data['Comparison'].nunique() == 1 else "Multiple (see legend)"

            plt.title(f'{metric_type} Disparity for {attribute.capitalize()} ({title_comparison_str})\nDataset: {dataset_name}', fontsize=14)
            plt.ylabel(f'{metric_type} Disparity Value', fontsize=12)
            plt.xlabel('Model', fontsize=12)
            plt.xticks(rotation=35, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            plt.axhline(0, color='black', lw=1.2, linestyle='--') # Zero disparity line
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add value annotations on bars
            for p in ax.patches:
                 ax.annotate(format(p.get_height(), '.3f'), # 3 decimal places
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha = 'center', va = 'center', 
                               xytext = (0, 9), 
                               textcoords = 'offset points', fontsize=8, color='dimgray')

            if group_data['Comparison'].nunique() > 1:
                plt.legend(title='Comparison Group', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
            else:
                ax.get_legend().remove() # Remove legend if not needed (only one comparison type)
                plt.tight_layout()

            plot_filename = f"disparity_chart_{dataset_name}_{attribute}_{metric_type}.png"
            disparity_plot_path = os.path.join(output_dir, plot_filename)
            try:
                plt.savefig(disparity_plot_path, bbox_inches='tight')
                logger.info(f"Saved disparity bar chart: {disparity_plot_path}")
            except Exception as e:
                logger.error(f"Failed to save disparity bar chart {disparity_plot_path}: {e}", exc_info=True)
            plt.close()


def plot_calibration_curves(calibration_results, results_dir, filename="calibration_curves.png"):
    logger = logging.getLogger(__name__)
    if not calibration_results: logger.warning("No calibration results to plot."); return
    n_groups = len(calibration_results)
    if n_groups == 0: logger.warning("Calibration results empty."); return

    colors = sns.color_palette("husl", n_groups)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plot_lines = 0
    for i, (group_name, results) in enumerate(calibration_results.items()):
        if results.get('calibration_curve_data') and \
           results['calibration_curve_data'].get('prob_true') and \
           results['calibration_curve_data'].get('prob_pred') and \
           not results.get('error'):
            prob_true = results['calibration_curve_data']['prob_true']
            prob_pred = results['calibration_curve_data']['prob_pred']
            label_text = f"{group_name.replace('_', ' ')} " \
                         f"(Brier={results.get('brier_score', 'N/A'):.3f}, " \
                         f"ECE={results.get('ece', 'N/A'):.3f})"
            ax.plot(prob_pred, prob_true, "s-", label=label_text, color=colors[i], lw=2, alpha=0.8)
            plot_lines += 1
        else:
            logger.warning(f"Skipping plotting calibration for '{group_name}' due to missing data/error.")
    if plot_lines == 0:
         logger.warning("No valid calibration curves plotted.")
         ax.text(0.5, 0.5, "No valid calibration data", ha='center', va='center', transform=ax.transAxes)
    else:
        ax.set_xlabel("Mean predicted probability (per bin)"); ax.set_ylabel("Fraction of positives (per bin)")
        ax.set_ylim([-0.05, 1.05]); ax.legend(loc="lower right", fontsize='small')
        ax.set_title('Model Calibration Curves (Reliability Diagram)'); ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_path = os.path.join(results_dir, filename)
    try: plt.savefig(save_path, bbox_inches='tight'); logger.info(f"Calibration plot saved: {save_path}")
    except Exception as e: logger.error(f"Failed to save calibration plot: {e}", exc_info=True)
    plt.close(fig)


def plot_all_calibration_curves(all_calibration_data, output_dir, filename="all_models_calibration_curves.png"):
    """
    Plots combined calibration curves for multiple models/datasets.
    Focuses on the 'Overall' calibration for each entry.

    Args:
        all_calibration_data (dict):
            Key: Test dataset name (e.g., "PAPILLA")
            Value: Dict where:
                Key: Model identifier (e.g., "model_A_checkpoint1")
                Value: The 'Overall' part of calibration_results from analyze_calibration.
                       e.g., {'brier_score': 0.1, 'ece': 0.05, 'calibration_curve_data': {...}, ...}
        output_dir (str): Directory to save the plot.
        filename (str): Name of the plot file.
    """
    logger.info("Generating combined calibration plot across models/datasets...")
    num_plots = 0
    for dataset_name, model_data in all_calibration_data.items():
        num_plots += len(model_data)

    if num_plots == 0:
        logger.warning("No calibration data provided for combined plot.")
        return

    # Dynamically create subplots: aim for a somewhat square layout
    # Or, if too many, consider multiple figures or scrolling. For now, one figure.
    # This example plots each model-dataset 'Overall' curve on its own subplot for clarity.
    # If you want them overlaid, the logic would be different (more complex legend).
    
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))
    
    if num_plots == 0 : # Handle no data case
        logger.warning("No valid calibration data to plot.")
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        ax.text(0.5, 0.5, "No calibration data available", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        try:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Empty calibration plot placeholder saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save empty calibration plot: {e}", exc_info=True)
        plt.close(fig)
        return

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes_flat = axes.flatten()
    plot_idx = 0
    
    for dataset_name, models_calib_data in all_calibration_data.items():
        for model_id, calib_data_overall in models_calib_data.items():
            if plot_idx >= len(axes_flat):
                logger.warning("Ran out of subplots for combined calibration. Some data not plotted.")
                break

            ax = axes_flat[plot_idx]
            ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", lw=1)
            
            plot_title = f"{model_id} on {dataset_name}"
            
            if calib_data_overall and isinstance(calib_data_overall, dict) and \
               calib_data_overall.get('calibration_curve_data') and \
               calib_data_overall['calibration_curve_data'].get('prob_true') and \
               calib_data_overall['calibration_curve_data'].get('prob_pred') and \
               not calib_data_overall.get('error'):
                
                prob_true = calib_data_overall['calibration_curve_data']['prob_true']
                prob_pred = calib_data_overall['calibration_curve_data']['prob_pred']
                
                brier = calib_data_overall.get('brier_score')
                ece = calib_data_overall.get('ece')
                
                label_text = f"Model (Brier={brier:.3f}, ECE={ece:.3f})" if brier is not None and ece is not None \
                             else "Model (Metrics N/A)"

                ax.plot(prob_pred, prob_true, "s-", label=label_text, lw=2, alpha=0.8)
                ax.set_title(plot_title, fontsize=10)
                ax.set_xlabel("Mean predicted probability", fontsize=8)
                ax.set_ylabel("Fraction of positives", fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=7)
                ax.legend(loc="lower right", fontsize='x-small')
                ax.grid(True, linestyle='--', alpha=0.6)
            else:
                ax.text(0.5, 0.5, "Data N/A", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(plot_title + "\n(Data N/A)", fontsize=10)
            
            plot_idx += 1
        if plot_idx >= len(axes_flat): break
            
    # Hide any unused subplots
    for i in range(plot_idx, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect to make space for suptitle
    fig.suptitle("Overall Model Calibration Curves", fontsize=16, fontweight='bold')
    
    save_path = os.path.join(output_dir, filename)
    try:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Combined calibration plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save combined calibration plot: {e}", exc_info=True)
    plt.close(fig)