import os
import json
import glob
import argparse
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # For better color palettes
import sys
import warnings
import re # For cleaning subgroup names
import math # For checking NaN

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- Configuration ---
SUMMARY_FILENAME = "external_evaluation_summary.json"
CONSOLIDATED_JSON_OUTPUT = "consolidated_summary.json"
CONSOLIDATED_CSV_OUTPUT = "consolidated_summary_table.csv"
SUMMARY_PLOT_OUTPUT = "summary_auc_comparison.png"
# Define which subgroup plots to generate: { 'dataset_key_in_json': 'MetricToCompare' }
# Focus only on AUC
SUBGROUP_PLOTS_TO_GENERATE = {
    "external_chaksu": "AUC",
    "external_papilla": "AUC",
    "external_oia-odir-test": "AUC"
}


def parse_model_dir_name(dir_name):
    """
    Parses the model directory name to extract model type and tag.
    Improved logic to handle different ViT backbones and tags better.
    """
    parts = dir_name.split('_')
    if len(parts) < 3:
        logger.warning(f"Could not parse model directory name '{dir_name}' into >= 3 parts. Using fallback.")
        return parts[0] if parts else 'Unknown', '_'.join(parts[1:]) if len(parts) > 1 else 'Unknown'

    model_type = parts[0] # e.g., resnet50, vit

    # Try to identify known tags or indicators
    known_tags = ["ResNet50_timm_e25", "ViTBase_timm_e25", "VFM_custom_e25", "DINOv2_e25"]
    timestamp_part = parts[-1] # Assume last part is timestamp

    # Reconstruct potential tag by removing model type and timestamp
    potential_tag = '_'.join(parts[1:-1])

    # Check if reconstructed tag matches a known tag
    if potential_tag in known_tags:
         tag = potential_tag
         # Refine model_type for ViT variants if needed based on tag or first part
         if model_type == 'vit' and 'DINOv2' in tag: model_type = 'dinov2_vitb14'
         elif model_type == 'vit' and 'VFM' in tag: model_type = 'vfm_vitb16'
         elif model_type == 'vit' and 'ViTBase' in tag: model_type = 'vit_base_timm'
    else:
         tag = potential_tag if potential_tag else "UnknownTag"
         # Infer model type refinement based on first part if tag wasn't matched
         if model_type == 'vit' and any(p in dir_name for p in ['DINOv2', 'dinov2']): model_type = 'dinov2_vitb14'
         elif model_type == 'vit' and any(p in dir_name for p in ['VFM']): model_type = 'vfm_vitb16'
         elif model_type == 'vit' and any(p in dir_name for p in ['ViTBase']): model_type = 'vit_base_timm'
         # Add more elif conditions here if needed for other models
         logger.warning(f"Using fallback tag '{tag}' and potentially inferred model type '{model_type}' for directory '{dir_name}'")

    # Ensure the tag doesn't accidentally include the timestamp if parsing failed
    # Check if the last part really looks like a timestamp
    if timestamp_part.isdigit() and len(timestamp_part) >= 8:
        potential_tag_check = '_'.join(parts[1:-1])
        if potential_tag_check in known_tags:
             tag = potential_tag_check
        # If tag still ends with timestamp, remove it (handles cases where known tag check failed but timestamp is last)
        elif tag.endswith(timestamp_part) and len(tag) > len(timestamp_part):
             tag = tag[:-len(timestamp_part)-1] # Remove timestamp and trailing underscore
    else:
        # If last part isn't a clear timestamp, assume it's part of the tag
        tag = '_'.join(parts[1:])


    return model_type, tag


def find_latest_evaluation_dir(model_run_dir):
    """Finds the most recent 'ext_eval_*' directory within a model run directory, including 'external_evaluations' subdir."""
    external_eval_dir = Path(model_run_dir) / "external_evaluations"
    if external_eval_dir.is_dir():
        eval_dirs = list(external_eval_dir.glob('ext_eval_*'))
    else:
        eval_dirs = list(Path(model_run_dir).glob('ext_eval_*'))  # fallback to old behavior

    if not eval_dirs:
        return None
    eval_dirs.sort(key=lambda p: p.name, reverse=True)
    return eval_dirs[0]

def load_and_aggregate_results(base_experiment_dir):
    """Finds, parses, and aggregates results from all summary files."""
    all_results = []
    base_path = Path(base_experiment_dir)
    if not base_path.is_dir():
        logger.error(f"Base directory not found: {base_experiment_dir}")
        return None

    logger.info(f"Scanning for model run directories in: {base_path}")
    model_run_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not model_run_dirs:
         logger.warning(f"No model run subdirectories found directly under {base_path}")
         return None

    for model_dir in model_run_dirs:
        logger.info(f"Processing model directory: {model_dir.name}")
        latest_eval_dir = find_latest_evaluation_dir(model_dir)
        if latest_eval_dir:
            summary_file_path = latest_eval_dir / SUMMARY_FILENAME
            if summary_file_path.is_file():
                logger.info(f"  Found latest summary file: {summary_file_path}")
                try:
                    with open(summary_file_path, 'r') as f: data = json.load(f)
                    model_type, experiment_tag = parse_model_dir_name(model_dir.name)
                    all_results.append({
                        "model_type": model_type, "experiment_tag": experiment_tag,
                        "model_run_dir": model_dir.name, "evaluation_dir": latest_eval_dir.name,
                        "metrics": data })
                except Exception as e: logger.error(f"  Error processing file {summary_file_path}: {e}", exc_info=True)
            else: logger.warning(f"  Summary file '{SUMMARY_FILENAME}' not found in latest evaluation dir: {latest_eval_dir}")
        else: logger.warning(f"  No 'evaluation_*' directories found in: {model_dir}")
    return all_results

def create_summary_table(aggregated_results):
    """Creates a pandas DataFrame summarizing key overall metrics."""
    if not aggregated_results:
        logger.warning("No aggregated results to create summary table from.")
        return None
    table_data = []
    dataset_keys_to_extract = ["validation_reconstructed_overall", "papilla_test", "oiaodir_test", "chaksu_test"]
    for entry in aggregated_results:
        row = {"Model Type": entry["model_type"], "Experiment Tag": entry["experiment_tag"]}
        metrics = entry.get("metrics", {})
        total_samples_map = {} # To store calculated N for overall plot legend

        for actual_key in dataset_keys_to_extract:
            dataset_key_clean = actual_key.replace('_test','').replace('_reconstructed','').replace('_overall','')
            auc, acc, n_total = pd.NA, pd.NA, 0 # Default to NA and 0

            if actual_key in metrics:
                dataset_metrics = metrics[actual_key]
                if isinstance(dataset_metrics, dict):
                    # --- Calculate Total N by summing subgroup Ns ---
                    current_n_total = 0
                    if actual_key.endswith("_test"): # Only for test sets with subgroups
                        for sub_key, sub_vals in dataset_metrics.items():
                            if sub_key != "Overall" and isinstance(sub_vals, dict):
                                n = sub_vals.get("n_samples", sub_vals.get("n"))
                                if n is not None and not (isinstance(n, float) and math.isnan(n)):
                                    current_n_total += int(n)
                        n_total = current_n_total if current_n_total > 0 else pd.NA
                    # Try to get N for validation (might be in per_source)
                    elif actual_key == "validation_reconstructed_overall":
                        val_per_source = metrics.get("validation_reconstructed_per_source", {})
                        current_n_total = sum(int(v.get('n', 0)) for v in val_per_source.values() if isinstance(v, dict) and v.get('n') is not None)
                        n_total = current_n_total if current_n_total > 0 else pd.NA

                    total_samples_map[dataset_key_clean] = n_total # Store calculated N

                    # --- Extract Overall Metrics ---
                    overall_data = dataset_metrics.get("Overall") if isinstance(dataset_metrics.get("Overall"), dict) else dataset_metrics
                    auc = overall_data.get("AUC", overall_data.get("auc"))
                    acc = overall_data.get("Accuracy", overall_data.get("accuracy"))

                else: logger.warning(f"Unexpected data type for key '{actual_key}' in {entry['model_run_dir']}: {type(dataset_metrics)}")

            row[f"{dataset_key_clean}_AUC"] = auc
            row[f"{dataset_key_clean}_Acc"] = acc
            row[f"{dataset_key_clean}_N"] = n_total # Store N in the row (optional, useful for CSV)

        row["_total_samples_map"] = total_samples_map # Store the calculated Ns temporarily
        table_data.append(row)

    if not table_data: return None
    df = pd.DataFrame(table_data)
    return df

def create_subgroup_table(aggregated_results, dataset_key, metric):
    """Creates a DataFrame for a specific dataset's subgroup metrics, including sample count."""
    if not aggregated_results: return None
    table_data = []
    for entry in aggregated_results:
        metrics = entry.get("metrics", {})
        dataset_metrics = metrics.get(dataset_key) # e.g., "chaksu_test"
        if not isinstance(dataset_metrics, dict): continue # Skip if dataset not found or not dict

        # --- Include Overall metric in the subgroup table ---
        for subgroup_key, subgroup_values in dataset_metrics.items():
            # Process 'Overall' and other subgroup dicts
            if not isinstance(subgroup_values, dict): continue

            metric_val = subgroup_values.get(metric) # Get the desired metric (e.g., "AUC")
            n_samples = None
            if subgroup_key == "Overall":
                # Calculate N for Overall by summing subgroups for this dataset/model run
                n_samples_calc = sum(int(v.get("n_samples", v.get("n", 0)))
                                  for k, v in dataset_metrics.items()
                                  if k != "Overall" and isinstance(v, dict) and v.get("n_samples", v.get("n")) is not None)
                n_samples = n_samples_calc if n_samples_calc > 0 else None
                clean_subgroup_name = "Overall" # Use specific name
            else:
                # Get N for specific subgroup
                n_samples = subgroup_values.get("n_samples", subgroup_values.get("n"))
                # Clean up subgroup name (e.g., "camera_Bosch" -> "Bosch")
                clean_subgroup_name = re.sub(r'^(camera_|age_group_|eye_group_)', '', subgroup_key)

            if metric_val is not None:
                table_data.append({
                    "Model Type": entry["model_type"],
                    "Experiment Tag": entry["experiment_tag"],
                    "Model Run ID": f"{entry['model_type']} ({entry['experiment_tag']})", # For plotting x-axis
                    "Subgroup": clean_subgroup_name,
                    metric: metric_val, # Column name will be the metric itself (e.g., AUC)
                    "N Samples": n_samples if n_samples is not None else 0 # Store sample count
                })

    if not table_data:
        logger.warning(f"No subgroup or overall data found for dataset '{dataset_key}' and metric '{metric}'.")
        return None
    df = pd.DataFrame(table_data)
    return df


def create_summary_plot(summary_df, output_dir):
    """Creates a bar chart comparing Overall AUCs with sample counts in legend."""
    if summary_df is None or summary_df.empty: return
    
    # Columns needed: Model ID, AUC cols, N cols
    auc_cols = [col for col in summary_df.columns if col.endswith('_AUC')]
    n_cols = [col for col in summary_df.columns if col.endswith('_N') and not col.startswith('_')] # Get N cols, ignore temp map
    
    if not auc_cols:
         logger.warning("No overall AUC columns found for plotting.")
         return
         
    plot_df = summary_df[["Model Type", "Experiment Tag"] + auc_cols + n_cols].copy()
    
    # Create Model Run ID
    plot_df['Model Run ID'] = plot_df['Model Type'] + ' (' + plot_df['Experiment Tag'] + ')'
    
    # --- Prepare data for melting - Combine AUC and N ---
    plot_data_melt = []
    for index, row in plot_df.iterrows():
        model_run_id = row['Model Run ID']
        for auc_col in auc_cols:
            dataset_name = auc_col.replace('_AUC', '')
            n_col = f"{dataset_name}_N"
            auc_val = row[auc_col]
            n_val = row.get(n_col, pd.NA) # Get N, default to NA if column somehow missing
            
            if pd.notna(auc_val): # Only include if AUC is valid
                # Create legend label with N
                n_str = f" (n={int(n_val)})" if pd.notna(n_val) else " (n=?) "
                dataset_label = f"{dataset_name.replace('_',' ').title()}{n_str}"
                
                plot_data_melt.append({
                    'Model Run ID': model_run_id,
                    'Dataset Label': dataset_label, # Use this for HUE
                    'AUC': auc_val
                })

    if not plot_data_melt:
        logger.warning("No valid overall AUC data to plot after processing for legend.")
        return
        
    df_melted = pd.DataFrame(plot_data_melt)

    # --- Plotting ---
    plt.figure(figsize=(14, 8))
    n_colors = df_melted['Dataset Label'].nunique()
    try: palette = sns.color_palette("Paired", n_colors=n_colors)
    except: palette = sns.color_palette("viridis", n_colors=n_colors)

    # Sort hue order alphabetically by the dataset name part
    hue_order = sorted(df_melted['Dataset Label'].unique(), key=lambda x: x.split(' (n=')[0])
    
    barplot = sns.barplot(data=df_melted, x='Model Run ID', y='AUC', hue='Dataset Label', # Use combined label for hue
                          palette=palette, width=0.8, hue_order=hue_order)

    for container in barplot.containers:
        try: barplot.bar_label(container, fmt='%.3f', label_type='edge', fontsize=8, padding=2)
        except AttributeError:
             for bar in container:
                 height = bar.get_height();
                 if height > 0: barplot.text(bar.get_x() + bar.get_width()/2., height+0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.title('Overall AUC Comparison Across Models and Datasets', fontsize=16, pad=20)
    plt.xlabel('Model Run (Type + Tag)', fontsize=12)
    plt.ylabel('Area Under ROC Curve (AUC)', fontsize=12)
    plt.xticks(rotation=25, ha='right', fontsize=9); plt.yticks(fontsize=10)
    plt.ylim(0.5, 1.02);
    # Update legend title
    plt.legend(title='Dataset (N Samples)', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.6); plt.tight_layout(rect=[0, 0.03, 0.85, 0.97]) # Adjusted layout
    plot_path = Path(output_dir) / SUMMARY_PLOT_OUTPUT
    try:
        plt.savefig(plot_path, bbox_inches='tight', dpi=150); logger.info(f"Summary plot saved to: {plot_path}")
    except Exception as e: logger.error(f"Failed to save summary plot: {e}")
    plt.close()


def create_subgroup_plot(subgroup_df, dataset_name, metric_name, output_dir):
    """Creates a bar chart comparing subgroup metrics (including Overall) for a specific dataset."""
    if subgroup_df is None or subgroup_df.empty:
        logger.warning(f"No subgroup data available for {dataset_name} {metric_name} plot.")
        return

    # --- Add Sample Count to Subgroup Label for Legend ---
    subgroup_df['Subgroup (N)'] = subgroup_df.apply(
        lambda row: f"{row['Subgroup']} (n={int(row['N Samples'])})" if pd.notna(row['N Samples']) and row['N Samples'] > 0 else f"{row['Subgroup']} (n=?)",
        axis=1
    )
    # ---

    clean_dataset_name = dataset_name.replace('_test','').replace('_reconstructed','').replace('_overall','')
    plt.figure(figsize=(14, 8))
    n_colors = subgroup_df['Subgroup (N)'].nunique()
    try: palette = sns.color_palette("tab10", n_colors=n_colors)
    except: palette = sns.color_palette("viridis", n_colors=n_colors)

    # Define order: Put 'Overall' first, then sort others alphabetically
    unique_subgroups_n = subgroup_df['Subgroup (N)'].unique()
    hue_order = sorted([sg for sg in unique_subgroups_n if 'Overall' in sg], reverse=True) + \
                sorted([sg for sg in unique_subgroups_n if 'Overall' not in sg], key=lambda x: x.split(' (n=')[0])

    barplot = sns.barplot(data=subgroup_df, x='Model Run ID', y=metric_name, hue='Subgroup (N)',
                          palette=palette, width=0.8, hue_order=hue_order)

    for container in barplot.containers:
        try: barplot.bar_label(container, fmt='%.3f', label_type='edge', fontsize=8, padding=2)
        except AttributeError:
             for bar in container:
                 height = bar.get_height()
                 if height > 0: barplot.text(bar.get_x() + bar.get_width() / 2., height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.title(f'Subgroup {metric_name} Comparison for {clean_dataset_name.capitalize()} Dataset', fontsize=16, pad=20)
    plt.xlabel('Model Run (Type + Tag)', fontsize=12)
    plt.ylabel(f'{metric_name}', fontsize=12)
    plt.xticks(rotation=25, ha='right', fontsize=9); plt.yticks(fontsize=10)

    min_val = subgroup_df[metric_name].min()
    max_val = subgroup_df[metric_name].max()
    y_bottom = max(0, min_val - 0.1 * (max_val - min_val)) if pd.notna(min_val) else 0
    y_top = max_val + 0.1 * (max_val - min_val) if pd.notna(max_val) else 1.05
    if metric_name == "AUC":
        y_bottom = max(0.4, y_bottom)
        y_top = min(1.02, y_top)
    plt.ylim(y_bottom, y_top)

    plt.legend(title='Subgroup (N Samples)', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.6); plt.tight_layout(rect=[0, 0.03, 0.85, 0.97])
    plot_filename = f"{clean_dataset_name}_subgroup_{metric_name.lower()}_comparison.png"
    plot_path = Path(output_dir) / plot_filename
    try:
        plt.savefig(plot_path, bbox_inches='tight', dpi=150); logger.info(f"Subgroup plot saved to: {plot_path}")
    except Exception as e: logger.error(f"Failed to save subgroup plot: {e}")
    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='seaborn')
    parser = argparse.ArgumentParser(description="Consolidate evaluation results from a multi-model experiment.")
    parser.add_argument("base_experiment_dir", type=str, help="Path to the base directory containing model run subdirectories.")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Directory to save consolidated files (defaults to base_experiment_dir).")
    args = parser.parse_args()

    output_directory = Path(args.output_dir) if args.output_dir else Path(args.base_experiment_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    aggregated_data = load_and_aggregate_results(args.base_experiment_dir)

    if aggregated_data:
        json_output_path = output_directory / CONSOLIDATED_JSON_OUTPUT
        try:
            with open(json_output_path, 'w') as f: json.dump(aggregated_data, f, indent=4)
            logger.info(f"Consolidated JSON summary saved to: {json_output_path}")
        except Exception as e: logger.error(f"Failed to save consolidated JSON: {e}")

        summary_df = create_summary_table(aggregated_data)
        if summary_df is not None and not summary_df.empty:
            metric_cols = [col for col in summary_df.columns if col.endswith(('_AUC', '_Acc'))]
            if not summary_df[metric_cols].isnull().all().all():
                logger.info("\n--- Summary Table (Overall Metrics) ---")
                print(summary_df.drop(columns=[col for col in summary_df if col.endswith('_N')], errors='ignore').to_string(index=False, float_format="%.4f")) # Don't print N cols
                csv_output_path = output_directory / CONSOLIDATED_CSV_OUTPUT
                try:
                    summary_df.to_csv(csv_output_path, index=False, float_format="%.4f")
                    logger.info(f"Consolidated summary table saved to CSV: {csv_output_path}")
                except Exception as e: logger.error(f"Failed to save summary CSV: {e}")
                create_summary_plot(summary_df, output_directory) # Pass the df containing calculated N
            else: logger.warning("Overall summary DataFrame created but contains only NA metric values. Skipping print/save/plot.")
        else: logger.warning("Overall summary DataFrame could not be created or is empty.")

        logger.info("\n--- Generating Subgroup Plots (AUC only) ---")
        plots_generated = set()
        for dataset_key, metric_name in SUBGROUP_PLOTS_TO_GENERATE.items():
            if metric_name != "AUC": continue # Skip non-AUC plots based on new requirement
            plot_tuple = (dataset_key, metric_name)
            if plot_tuple in plots_generated: continue
            logger.info(f"Generating {metric_name} subgroup plot for dataset: {dataset_key}")
            subgroup_df = create_subgroup_table(aggregated_data, dataset_key, metric_name)
            if subgroup_df is not None and not subgroup_df.empty:
                 create_subgroup_plot(subgroup_df, dataset_key, metric_name, output_directory)
                 plots_generated.add(plot_tuple)
            else: logger.warning(f"No subgroup data found or table failed for {dataset_key} ({metric_name}), skipping plot.")
    else:
        logger.error("Failed to aggregate any results. Please check the base directory structure and file names.")
    logger.info("--- Consolidation Script Finished ---")