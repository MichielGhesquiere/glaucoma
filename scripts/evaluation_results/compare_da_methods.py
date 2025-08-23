import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import re

def parse_auc_with_ci(auc_str):
    """Parse AUC string with 95% CI into AUC, lower CI, upper CI"""
    try:
        # Pattern: 0.873 (0.854-0.891)
        match = re.match(r'(\d+\.\d+)\s*\((\d+\.\d+)-(\d+\.\d+)\)', auc_str)
        if match:
            auc = float(match.group(1))
            ci_lower = float(match.group(2))
            ci_upper = float(match.group(3))
            return auc, ci_lower, ci_upper
        else:
            # If no CI found, try to parse just the AUC
            auc = float(auc_str)
            return auc, None, None
    except:
        return None, None, None

def find_evaluation_files(script_dir):
    """Dynamically find all evaluation summary tables in multisource directories"""
    
    evaluation_files = {}
    
    # Look for directories with "multisource" in their name
    for item in script_dir.iterdir():
        if item.is_dir() and "multisource" in item.name.lower():
            # Look for evaluation summary table files
            potential_files = [
                item / "evaluation_summary_table.csv",
                item / "evaluation_summary_table_TTA.csv"
            ]
            
            for file_path in potential_files:
                if file_path.exists():
                    method_name = item.name
                    evaluation_files[method_name] = file_path
                    print(f"Found evaluation file: {method_name} -> {file_path}")
                    break
    
    return evaluation_files

def load_and_process_multiple_data(evaluation_files):
    """Load and process evaluation data from multiple CSV files"""
    
    all_dataframes = []
    
    for method_name, file_path in evaluation_files.items():
        try:
            df = pd.read_csv(file_path)
            
            # Determine method label based on directory name
            if "tta" in method_name.lower():
                df['Method'] = f'VFM + TTA ({method_name})'
            else:
                df['Method'] = f'VFM ({method_name})'
            
            all_dataframes.append(df)
            print(f"Loaded {len(df)} rows from {method_name}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not all_dataframes:
        raise ValueError("No evaluation files could be loaded!")
    
    # Combine all datasets
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Parse AUC with confidence intervals
    auc_data = []
    for _, row in combined_df.iterrows():
        auc, ci_lower, ci_upper = parse_auc_with_ci(row['AUC (95% CI)'])
        auc_data.append({
            'Test_Dataset': row['Test Dataset'],
            'Method': row['Method'],
            'AUC': auc,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'N': row['N'],
            'Sensitivity': row['Sensitivity'],
            'Specificity': row['Specificity'],
            'Accuracy': row['Accuracy']
        })
    
    processed_df = pd.DataFrame(auc_data)
    
    # Calculate error bars for plotting
    processed_df['Error_Lower'] = processed_df['AUC'] - processed_df['CI_Lower']
    processed_df['Error_Upper'] = processed_df['CI_Upper'] - processed_df['AUC']
    
    return processed_df

def create_comparison_plot(data_df, output_path):
    """Create horizontal bar chart comparing AUC values with 95% CI"""
    
    # Get unique datasets and methods
    datasets = sorted(data_df['Test_Dataset'].unique())
    methods = sorted(data_df['Method'].unique())
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, len(datasets) * 0.6 + 2))
    
    # Color scheme - generate colors for all methods
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    color_map = {method: colors[i] for i, method in enumerate(methods)}
    
    # Width of bars
    bar_height = 0.8 / len(methods)
    
    # Y positions for datasets
    y_positions = np.arange(len(datasets))
    
    # Plot data for each method
    for i, method in enumerate(methods):
        method_data = data_df[data_df['Method'] == method]
        
        aucs = []
        error_lower = []
        error_upper = []
        
        for dataset in datasets:
            dataset_data = method_data[method_data['Test_Dataset'] == dataset]
            if not dataset_data.empty:
                aucs.append(dataset_data['AUC'].iloc[0])
                error_lower.append(dataset_data['Error_Lower'].iloc[0])
                error_upper.append(dataset_data['Error_Upper'].iloc[0])
            else:
                aucs.append(np.nan)
                error_lower.append(0)
                error_upper.append(0)
        
        # Create error bars array
        errors = [error_lower, error_upper]
        
        # Plot horizontal bars
        bars = ax.barh(y_positions + i * bar_height, aucs, bar_height,
                      label=method, color=color_map[method], alpha=0.8,
                      xerr=errors, capsize=3, error_kw={'linewidth': 1.5})
        
        # Add AUC values as text on bars (only for non-NaN values)
        for j, (bar, auc) in enumerate(zip(bars, aucs)):
            if not np.isnan(auc):
                ax.text(auc + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{auc:.3f}', va='center', fontsize=8, fontweight='bold')
    
    # Customize the plot
    ax.set_yticks(y_positions + bar_height * (len(methods) - 1) / 2)
    ax.set_yticklabels(datasets)
    ax.set_xlabel('AUC (Area Under Curve)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Dataset', fontsize=12, fontweight='bold')
    ax.set_title('AUC Comparison Across Different Methods\nwith 95% Confidence Intervals', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis limits
    ax.set_xlim(0.4, 1.0)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # Invert y-axis to have first dataset at top
    ax.invert_yaxis()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_improvement_analysis(data_df, output_path):
    """Create analysis of AUC improvements from baseline to other methods"""
    
    methods = sorted(data_df['Method'].unique())
    
    # Try to identify baseline method (assume it's the one without TTA)
    baseline_method = None
    for method in methods:
        if "tta" not in method.lower():
            baseline_method = method
            break
    
    if baseline_method is None:
        print("Warning: Could not identify baseline method, using first method as baseline")
        baseline_method = methods[0]
    
    print(f"Using '{baseline_method}' as baseline for improvement analysis")
    
    # Pivot data to compare methods
    pivot_df = data_df.pivot_table(index='Test_Dataset', columns='Method', values='AUC')
    
    # Calculate improvements relative to baseline
    improvement_columns = []
    for method in methods:
        if method != baseline_method and method in pivot_df.columns:
            improvement_col = f'{method}_Improvement'
            percent_col = f'{method}_Improvement_Percent'
            
            pivot_df[improvement_col] = pivot_df[method] - pivot_df[baseline_method]
            pivot_df[percent_col] = (pivot_df[improvement_col] / pivot_df[baseline_method]) * 100
            improvement_columns.append((method, improvement_col, percent_col))
    
    if not improvement_columns:
        print("No methods found for improvement comparison")
        return pivot_df
    
    # Create improvement plot
    n_methods = len(improvement_columns)
    fig, axes = plt.subplots(n_methods, 2, figsize=(16, 6 * n_methods))
    
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (method, imp_col, pct_col) in enumerate(improvement_columns):
        ax1, ax2 = axes[idx, 0], axes[idx, 1]
        
        # Sort by improvement for this method
        method_pivot = pivot_df.sort_values(imp_col, ascending=False)
        datasets = method_pivot.index
        improvements = method_pivot[imp_col]
        percentages = method_pivot[pct_col]
        
        colors = ['green' if x > 0 else 'red' for x in improvements]
        
        # Plot 1: Absolute improvement
        bars1 = ax1.barh(range(len(datasets)), improvements, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(datasets)))
        ax1.set_yticklabels(datasets)
        ax1.set_xlabel('AUC Improvement', fontweight='bold')
        ax1.set_title(f'Absolute AUC Improvement\n{method} vs {baseline_method}', fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add improvement values as text
        for i, (bar, imp) in enumerate(zip(bars1, improvements)):
            if not np.isnan(imp):
                ax1.text(imp + (0.001 if imp >= 0 else -0.001), bar.get_y() + bar.get_height()/2, 
                        f'{imp:.3f}', va='center', ha='left' if imp >= 0 else 'right', 
                        fontsize=8, fontweight='bold')
        
        # Plot 2: Percentage improvement
        bars2 = ax2.barh(range(len(datasets)), percentages, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(datasets)))
        ax2.set_yticklabels(datasets)
        ax2.set_xlabel('Percentage Improvement (%)', fontweight='bold')
        ax2.set_title(f'Relative AUC Improvement\n{method} vs {baseline_method}', fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add percentage values as text
        for i, (bar, pct) in enumerate(zip(bars2, percentages)):
            if not np.isnan(pct):
                ax2.text(pct + (0.1 if pct >= 0 else -0.1), bar.get_y() + bar.get_height()/2, 
                        f'{pct:.1f}%', va='center', ha='left' if pct >= 0 else 'right', 
                        fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return pivot_df

def generate_summary_statistics(data_df):
    """Generate summary statistics for the comparison"""
    
    # Basic statistics by method
    summary_stats = data_df.groupby('Method').agg({
        'AUC': ['mean', 'std', 'min', 'max', 'count']
    }).round(3)
    
    print("=== SUMMARY STATISTICS ===")
    print(summary_stats)
    print()
    
    # Calculate improvements relative to baseline
    methods = sorted(data_df['Method'].unique())
    
    # Try to identify baseline method
    baseline_method = None
    for method in methods:
        if "tta" not in method.lower():
            baseline_method = method
            break
    
    if baseline_method is None:
        baseline_method = methods[0]
    
    print(f"=== IMPROVEMENT ANALYSIS (relative to {baseline_method}) ===")
    
    # Calculate improvements for each non-baseline method
    pivot_df = data_df.pivot_table(index='Test_Dataset', columns='Method', values='AUC')
    
    for method in methods:
        if method != baseline_method and method in pivot_df.columns:
            improvements = pivot_df[method] - pivot_df[baseline_method]
            improvements = improvements.dropna()
            
            if len(improvements) > 0:
                print(f"\n{method} vs {baseline_method}:")
                print(f"  Mean AUC improvement: {improvements.mean():.3f}")
                print(f"  Std AUC improvement: {improvements.std():.3f}")
                print(f"  Datasets with improvement: {(improvements > 0).sum()}/{len(improvements)}")
                print(f"  Best improvement: {improvements.max():.3f} ({improvements.idxmax()})")
                print(f"  Worst change: {improvements.min():.3f} ({improvements.idxmin()})")
    
    print()
    
    return summary_stats, pivot_df

def main():
    """Main function to run the comparison analysis"""
    
    # File paths
    script_dir = Path(__file__).parent
    
    print("Searching for evaluation files in multisource directories...")
    evaluation_files = find_evaluation_files(script_dir)
    
    if not evaluation_files:
        print("No evaluation files found in multisource directories!")
        print("Expected directory names containing 'multisource' with evaluation_summary_table.csv files")
        return
    
    print(f"\nFound {len(evaluation_files)} evaluation files:")
    for method, path in evaluation_files.items():
        print(f"  {method}: {path.name}")
    
    print("\nLoading and processing data...")
    data_df = load_and_process_multiple_data(evaluation_files)
    
    print("\nGenerating summary statistics...")
    summary_stats, pivot_df = generate_summary_statistics(data_df)
    
    print("Creating AUC comparison plot...")
    comparison_plot_path = script_dir / "auc_comparison_multisource_methods.png"
    create_comparison_plot(data_df, comparison_plot_path)
    
    print("Creating improvement analysis plot...")
    improvement_plot_path = script_dir / "auc_improvement_analysis_multisource.png"
    improvement_df = create_improvement_analysis(data_df, improvement_plot_path)
    
    # Save improvement analysis to CSV
    improvement_csv_path = script_dir / "auc_improvement_analysis_multisource.csv"
    improvement_df.to_csv(improvement_csv_path)
    
    print(f"\nAnalysis complete!")
    print(f"- Comparison plot saved: {comparison_plot_path}")
    print(f"- Improvement plot saved: {improvement_plot_path}")
    print(f"- Improvement data saved: {improvement_csv_path}")

if __name__ == "__main__":
    main()
