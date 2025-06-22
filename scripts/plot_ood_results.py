import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

# Seaborn style for nicer plots
sns.set_theme(style="whitegrid")

def plot_metric_bars(data_dict, metric_key, title, ylabel, output_dir, filename, lower_is_better=False, is_percentage=False):
    """
    Generates a bar plot for a given metric across models and datasets.
    Handles nested dictionaries for datasets within models.
    """
    plot_data = []
    for model_name, datasets in data_dict.items():
        for dataset_name, metrics in datasets.items():
            # Traverse to find the metric
            value = metrics
            # Handle potentially nested metric_key like "overall_metrics.AUC"
            keys = metric_key.split('.')
            valid_metric = True
            for k_part in keys:
                if isinstance(value, dict) and k_part in value:
                    value = value[k_part]
                else:
                    # Check if the key exists at the current level (for non-nested like Max TPR Diff)
                    if k_part in metrics: # e.g. fairness_summary_descriptive.max_tpr_difference
                        value = metrics[k_part]
                        break # Found it directly
                    else:
                        valid_metric = False
                        break
            
            if valid_metric and value is not None and not (isinstance(value, float) and np.isnan(value)):
                if is_percentage and isinstance(value, (int, float)):
                    value *= 100
                plot_data.append({
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Value": value
                })
            else:
                print(f"Warning: Metric '{metric_key}' not found or NaN for {model_name} on {dataset_name}.")


    if not plot_data:
        print(f"No data to plot for metric: {metric_key}")
        return

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 7))
    # Create a combined 'Model-Dataset' for unique hue/x-axis if needed, or plot datasets separately
    
    # If we want to group by model and have datasets as separate bars within each model group
    # This requires a bit more specific handling if datasets are not uniform across models.
    # For simplicity, let's make Model the hue and Dataset on x-axis, or vice-versa.
    
    # Let's try: X-axis = Dataset, Hue = Model
    # Or X-axis = Model, Hue = Dataset (better if many datasets, few models)

    # If only one dataset type is usually present per top-level dict (e.g. Papilla or Chaksu)
    # we can simplify. For metrics like "Max TPR Diff on PAPILLA", dataset is fixed.

    # General case:
    if df_plot['Dataset'].nunique() > 1: # Multiple datasets, group by dataset then model
        g = sns.catplot(x="Dataset", y="Value", hue="Model", data=df_plot, kind="bar", palette="muted", legend_out=True, height=6, aspect=1.5)
        g.despine(left=True)
        g.set_axis_labels("Dataset", ylabel)
        plt.title(title, fontsize=16)
        # Add value annotations
        for ax in g.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.3f'), 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'center', 
                           xytext = (0, 9), 
                           textcoords = 'offset points', fontsize=9)
    else: # Single dataset type implicitly, or we are plotting a summary metric that's per model
        sns.barplot(x="Model", y="Value", data=df_plot, palette="viridis")
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(ylabel, fontsize=14)
        plt.xlabel("Model", fontsize=14)
        plt.title(title, fontsize=16)
        # Add value annotations
        ax = plt.gca()
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.3f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points', fontsize=9)


    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"Saved plot: {filename}")


def plot_grouped_metric_bars(data_dict, dataset_focus, subgroup_category_key, metric_key, title, ylabel, output_dir, filename, is_percentage=False):
    """
    Generates a grouped bar plot for a metric across subgroups for each model, focused on a specific dataset.
    E.g., Metric: TPR, Subgroup Category: camera (Bosch, Forus, Remidio) for CHAKSU dataset.
    """
    plot_data = []
    for model_name, datasets_data in data_dict.items():
        if dataset_focus in datasets_data:
            dataset_metrics = datasets_data[dataset_focus]
            if "subgroup_metrics" in dataset_metrics:
                for subgroup_name, subgroup_data in dataset_metrics["subgroup_metrics"].items():
                    # Check if the subgroup_name is part of the category we're interested in
                    # e.g., if subgroup_category_key is 'camera', subgroup_name could be 'camera_Bosch'
                    if subgroup_category_key in subgroup_name: # Simple check, assumes format like 'category_Value'
                        metric_value = subgroup_data.get(metric_key)
                        if metric_value is not None and not (isinstance(metric_value, float) and np.isnan(metric_value)):
                            if is_percentage:
                                metric_value *= 100
                            plot_data.append({
                                "Model": model_name,
                                "Subgroup": subgroup_name.replace(f"{subgroup_category_key}_", ""), # Clean up subgroup name
                                "Value": metric_value
                            })
                        else:
                             print(f"Warning: Metric '{metric_key}' missing/NaN for {model_name}, {dataset_focus}, {subgroup_name}")


    if not plot_data:
        print(f"No data to plot for grouped metric: {metric_key} on {dataset_focus} by {subgroup_category_key}")
        return

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(14, 8))
    g = sns.catplot(
        data=df_plot, x="Model", y="Value", hue="Subgroup",
        kind="bar", palette="tab10", legend_out=True, height=7, aspect=1.8
    )
    g.despine(left=True)
    g.set_axis_labels("Model", ylabel)
    g.ax.tick_params(axis='x', rotation=30) # Rotate x-axis labels if model names are long
    plt.title(title, fontsize=16)
    
    # Add value annotations (can be tricky for catplot with multiple levels)
    for ax in g.axes.ravel():
        for c in ax.containers: # type: ignore
            labels = [f'{v.get_height():.3f}' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge', fontsize=8, padding=3)


    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"Saved grouped plot: {filename}")


def main(args):
    if not os.path.exists(args.json_results_path):
        print(f"Error: Results JSON file not found at {args.json_results_path}")
        return

    with open(args.json_results_path, 'r') as f:
        results_data = json.load(f)

    output_directory = args.output_dir if args.output_dir else os.path.join(os.path.dirname(args.json_results_path), "summary_plots")
    os.makedirs(output_directory, exist_ok=True)

    # --- Plot 1: Overall AUC on PAPILLA & CHAKSU per Model ---
    # Filter data for PAPILLA and CHAKSU only for this plot
    auc_plot_data = {}
    for model, datasets in results_data.items():
        auc_plot_data[model] = {}
        if "PAPILLA" in datasets:
            auc_plot_data[model]["PAPILLA"] = datasets["PAPILLA"]
        if "CHAKSU" in datasets:
            auc_plot_data[model]["CHAKSU"] = datasets["CHAKSU"]
    
    if any(auc_plot_data.values()): # Check if there's any data to plot
        plot_metric_bars(
            data_dict=auc_plot_data,
            metric_key="overall_metrics.AUC",
            title="Overall AUC on OOD Datasets (PAPILLA & CHAKSU)",
            ylabel="Area Under ROC Curve (AUC)",
            output_dir=output_directory,
            filename="overall_auc_papilla_chaksu.png"
        )

    # --- Plot 2: AUC by Camera Type (CHAKSU) per Model ---
    chaksu_data_for_plot = {model: {"CHAKSU": data.get("CHAKSU", {})} for model, data in results_data.items() if "CHAKSU" in data}
    if any(d.get("CHAKSU") for d in chaksu_data_for_plot.values()): # Check if CHAKSU data exists
        plot_grouped_metric_bars(
            data_dict=chaksu_data_for_plot,
            dataset_focus="CHAKSU",
            subgroup_category_key="camera", # Looks for "camera_Bosch", "camera_Forus", etc.
            metric_key="AUC",
            title="AUC by Camera Type (CHAKSU Dataset)",
            ylabel="Area Under ROC Curve (AUC)",
            output_dir=output_directory,
            filename="auc_by_camera_chaksu.png"
        )
    
    # --- Plot 3: TPR by Age Group (PAPILLA) per Model ---
    papilla_data_for_plot = {model: {"PAPILLA": data.get("PAPILLA", {})} for model, data in results_data.items() if "PAPILLA" in data}
    if any(d.get("PAPILLA") for d in papilla_data_for_plot.values()):
        plot_grouped_metric_bars(
            data_dict=papilla_data_for_plot,
            dataset_focus="PAPILLA",
            subgroup_category_key="age_group", # Looks for "age_group_<65", "age_group_>=65"
            metric_key="TPR", # True Positive Rate at optimal threshold
            title="True Positive Rate (Sensitivity) by Age Group (PAPILLA Dataset)",
            ylabel="True Positive Rate (TPR)",
            output_dir=output_directory,
            filename="tpr_by_age_papilla.png",
            is_percentage=True
        )

    # --- Plot 4: TPR by Camera Type (CHAKSU) per Model ---
    if any(d.get("CHAKSU") for d in chaksu_data_for_plot.values()): # Re-use chaksu_data_for_plot
        plot_grouped_metric_bars(
            data_dict=chaksu_data_for_plot,
            dataset_focus="CHAKSU",
            subgroup_category_key="camera",
            metric_key="TPR",
            title="True Positive Rate (Sensitivity) by Camera Type (CHAKSU Dataset)",
            ylabel="True Positive Rate (TPR)",
            output_dir=output_directory,
            filename="tpr_by_camera_chaksu.png",
            is_percentage=True
        )

    # --- Plot 5 & 6: Max TPR Difference (Fairness Summary) ---
    # PAPILLA
    max_tpr_diff_papilla_data = {}
    for model, datasets in results_data.items():
        if "PAPILLA" in datasets and "fairness_summary_descriptive" in datasets["PAPILLA"]:
            max_tpr_diff_papilla_data[model] = {"PAPILLA": datasets["PAPILLA"]["fairness_summary_descriptive"]}
    if max_tpr_diff_papilla_data:
        plot_metric_bars(
            data_dict=max_tpr_diff_papilla_data,
            metric_key="max_tpr_difference", # Directly access from the fairness_summary_descriptive
            title="Max TPR Difference (Underdiagnosis Disparity Proxy) - PAPILLA",
            ylabel="Max Difference in TPR across subgroups",
            output_dir=output_directory,
            filename="max_tpr_difference_papilla.png",
            is_percentage=True
        )
    # CHAKSU
    max_tpr_diff_chaksu_data = {}
    for model, datasets in results_data.items():
        if "CHAKSU" in datasets and "fairness_summary_descriptive" in datasets["CHAKSU"]:
            max_tpr_diff_chaksu_data[model] = {"CHAKSU": datasets["CHAKSU"]["fairness_summary_descriptive"]}
    if max_tpr_diff_chaksu_data:
        plot_metric_bars(
            data_dict=max_tpr_diff_chaksu_data,
            metric_key="max_tpr_difference",
            title="Max TPR Difference (Underdiagnosis Disparity Proxy) - CHAKSU",
            ylabel="Max Difference in TPR across subgroups",
            output_dir=output_directory,
            filename="max_tpr_difference_chaksu.png",
            is_percentage=True
        )
        
    # --- Plot 7: Overall ECE on PAPILLA & CHAKSU per Model ---
    ece_plot_data = {}
    for model, datasets in results_data.items():
        ece_plot_data[model] = {}
        for dataset_name in ["PAPILLA", "CHAKSU"]:
            if dataset_name in datasets and \
               datasets[dataset_name].get("calibration_analysis", {}).get("Overall", {}).get("ece") is not None:
                ece_plot_data[model][dataset_name] = datasets[dataset_name]["calibration_analysis"]["Overall"]
    
    if any(ece_plot_data.values()):
        plot_metric_bars(
            data_dict=ece_plot_data,
            metric_key="ece", # Directly access ECE from the "Overall" calibration
            title="Overall Expected Calibration Error (ECE) on OOD Datasets",
            ylabel="Expected Calibration Error (ECE)",
            output_dir=output_directory,
            filename="overall_ece_papilla_chaksu.png",
            lower_is_better=True
        )

    print(f"All summary plots saved to: {output_directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary plots from OOD evaluation JSON results.")
    parser.add_argument("json_results_path", type=str, help="Path to the comprehensive OOD evaluation summary JSON file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the plots. Defaults to a 'summary_plots' subdir next to the JSON file.")
    
    args = parser.parse_args()
    main(args)