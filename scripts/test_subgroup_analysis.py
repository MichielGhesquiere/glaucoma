#!/usr/bin/env python3
"""
Test script for subgroup analysis functionality.

This script creates mock data to test the subgroup analysis features
without requiring actual model weights or datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

def create_mock_data():
    """Create mock evaluation data with demographics for testing."""
    np.random.seed(42)
    
    # Create mock dataset with demographics
    n_samples = 500
    
    # Generate age data (realistic distribution)
    ages = np.random.normal(65, 15, n_samples)
    ages = np.clip(ages, 20, 90).astype(int)
    
    # Generate sex data (roughly balanced)
    sexes = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    
    # Generate labels with some demographic bias
    labels = np.zeros(n_samples)
    for i in range(n_samples):
        base_prob = 0.3
        # Older patients slightly more likely to have glaucoma
        if ages[i] >= 65:
            base_prob += 0.1
        # Sex difference
        if sexes[i] == 'Male':
            base_prob += 0.05
        
        labels[i] = np.random.binomial(1, base_prob)
    
    # Generate model scores (with realistic performance)
    scores = np.random.beta(2, 2, n_samples)  # Base scores
    
    # Make scores somewhat predictive
    for i in range(n_samples):
        if labels[i] == 1:  # Glaucoma cases
            scores[i] = scores[i] * 0.7 + 0.3  # Shift towards higher scores
        else:  # Normal cases
            scores[i] = scores[i] * 0.7  # Keep lower scores
    
    scores = np.clip(scores, 0.01, 0.99)
    
    return {
        'age': ages,
        'sex': sexes,
        'labels': labels.astype(int),
        'scores': scores
    }

def bootstrap_auc(y_true, y_scores, n_bootstraps=100, random_state=42):
    """Simplified bootstrap AUC calculation."""
    from sklearn.metrics import roc_auc_score
    
    np.random.seed(random_state)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    return np.mean(sorted_scores), confidence_lower, confidence_upper

def calculate_youden_threshold(fpr, tpr, thresholds):
    """Calculate Youden's J statistic and optimal threshold."""
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_j = j_scores[optimal_idx]
    return optimal_threshold, optimal_j, optimal_idx

def calculate_subgroup_metrics(y_true, y_scores, subgroup_labels):
    """Calculate metrics for each subgroup."""
    from sklearn.metrics import roc_auc_score, roc_curve
    
    results = []
    
    # Overall metrics first
    overall_auc = roc_auc_score(y_true, y_scores)
    overall_fpr, overall_tpr, overall_thresholds = roc_curve(y_true, y_scores)
    overall_threshold, _, _ = calculate_youden_threshold(overall_fpr, overall_tpr, overall_thresholds)
    overall_pred = (y_scores >= overall_threshold).astype(int)
    
    # Calculate confusion matrix for overall
    overall_tn = np.sum((y_true == 0) & (overall_pred == 0))
    overall_fp = np.sum((y_true == 0) & (overall_pred == 1))
    overall_fn = np.sum((y_true == 1) & (overall_pred == 0))
    overall_tp = np.sum((y_true == 1) & (overall_pred == 1))
    
    overall_sens = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_spec = overall_tn / (overall_tn + overall_fp) if (overall_tn + overall_fp) > 0 else 0
    
    # Bootstrap confidence intervals for overall AUC
    overall_auc_mean, overall_ci_lower, overall_ci_upper = bootstrap_auc(y_true, y_scores)
    
    results.append({
        'subgroup': 'Overall',
        'n_total': len(y_true),
        'n_positive': int(np.sum(y_true)),
        'n_negative': int(len(y_true) - np.sum(y_true)),
        'auc': overall_auc,
        'auc_ci_lower': overall_ci_lower,
        'auc_ci_upper': overall_ci_upper,
        'sensitivity': overall_sens,
        'specificity': overall_spec,
        'threshold': overall_threshold
    })
    
    # Subgroup metrics
    unique_groups = np.unique(subgroup_labels)
    for group in unique_groups:
        if pd.isna(group):
            continue
            
        mask = subgroup_labels == group
        if np.sum(mask) < 10:  # Skip groups with too few samples
            continue
            
        group_y_true = y_true[mask]
        group_y_scores = y_scores[mask]
        
        # Check if we have both classes in the subgroup
        if len(np.unique(group_y_true)) < 2:
            continue
            
        try:
            group_auc = roc_auc_score(group_y_true, group_y_scores)
            group_fpr, group_tpr, group_thresholds = roc_curve(group_y_true, group_y_scores)
            group_threshold, _, _ = calculate_youden_threshold(group_fpr, group_tpr, group_thresholds)
            group_pred = (group_y_scores >= group_threshold).astype(int)
            
            # Calculate confusion matrix for subgroup
            group_tn = np.sum((group_y_true == 0) & (group_pred == 0))
            group_fp = np.sum((group_y_true == 0) & (group_pred == 1))
            group_fn = np.sum((group_y_true == 1) & (group_pred == 0))
            group_tp = np.sum((group_y_true == 1) & (group_pred == 1))
            
            group_sens = group_tp / (group_tp + group_fn) if (group_tp + group_fn) > 0 else 0
            group_spec = group_tn / (group_tn + group_fp) if (group_tn + group_fp) > 0 else 0
            
            # Bootstrap confidence intervals for subgroup AUC
            group_auc_mean, group_ci_lower, group_ci_upper = bootstrap_auc(group_y_true, group_y_scores)
            
            results.append({
                'subgroup': group,
                'n_total': len(group_y_true),
                'n_positive': int(np.sum(group_y_true)),
                'n_negative': int(len(group_y_true) - np.sum(group_y_true)),
                'auc': group_auc,
                'auc_ci_lower': group_ci_lower,
                'auc_ci_upper': group_ci_upper,
                'sensitivity': group_sens,
                'specificity': group_spec,
                'threshold': group_threshold
            })
            
        except Exception as e:
            print(f"Could not calculate metrics for subgroup {group}: {e}")
            continue
    
    return pd.DataFrame(results)

def create_age_groups(ages, threshold=65):
    """Create age groups based on threshold."""
    age_groups = np.array(['Unknown'] * len(ages))
    valid_ages = pd.to_numeric(ages, errors='coerce')
    
    age_groups[valid_ages < threshold] = f'Age < {threshold}'
    age_groups[valid_ages >= threshold] = f'Age ≥ {threshold}'
    
    return age_groups

def plot_subgroup_analysis_test(subgroup_results_age, subgroup_results_sex):
    """Create test subgroup analysis plot."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    
    # Define colors for different metrics
    colors = {
        'Overall': 'black',
        'Male': '#D946EF',
        'Female': '#D946EF',
        'Age < 65': '#10B981',
        'Age ≥ 65': '#10B981'
    }
    
    # Define markers
    markers = {
        'Overall': 'o',
        'Male': 's',
        'Female': 's',
        'Age < 65': '^',
        'Age ≥ 65': '^'
    }
    
    # Combine all results for plotting
    all_results = []
    if subgroup_results_age is not None:
        age_results = subgroup_results_age.copy()
        age_results['category'] = 'Age'
        all_results.append(age_results)
    if subgroup_results_sex is not None:
        sex_results = subgroup_results_sex.copy()
        sex_results['category'] = 'Sex'
        all_results.append(sex_results)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Plot 1: AUC
    ax1 = axes[0]
    y_pos = 0
    
    for category in ['Overall', 'Sex', 'Age']:
        if category == 'Overall':
            overall_row = combined_results[combined_results['subgroup'] == 'Overall'].iloc[0]
            ax1.scatter([overall_row['auc']], [y_pos], 
                      color=colors['Overall'], marker=markers['Overall'], s=100, zorder=3)
            ax1.plot([overall_row['auc_ci_lower'], overall_row['auc_ci_upper']], 
                    [y_pos, y_pos], color=colors['Overall'], linewidth=2, zorder=2)
            ax1.text(-0.05, y_pos, f"Overall (n = {overall_row['n_total']})", 
                    ha='right', va='center', transform=ax1.get_yaxis_transform())
            y_pos += 1
        else:
            category_data = combined_results[
                (combined_results['category'] == category) & 
                (combined_results['subgroup'] != 'Overall')
            ]
            
            if len(category_data) == 0:
                continue
                
            # Add category header
            ax1.text(-0.05, y_pos, category, ha='right', va='center', 
                    transform=ax1.get_yaxis_transform(), weight='bold')
            y_pos += 0.5
            
            for _, row in category_data.iterrows():
                color = colors.get(row['subgroup'], 'gray')
                marker = markers.get(row['subgroup'], 'o')
                
                ax1.scatter([row['auc']], [y_pos], 
                          color=color, marker=marker, s=100, zorder=3)
                ax1.plot([row['auc_ci_lower'], row['auc_ci_upper']], 
                        [y_pos, y_pos], color=color, linewidth=2, zorder=2)
                ax1.text(-0.05, y_pos, f"  {row['subgroup']} (n = {row['n_total']})", 
                        ha='right', va='center', transform=ax1.get_yaxis_transform())
                y_pos += 1
            
            y_pos += 0.5
    
    ax1.set_xlim(0.5, 1.0)
    ax1.set_ylim(-0.5, y_pos)
    ax1.set_xlabel('Subgroup\nAUROC')
    ax1.set_title('AUROC by Subgroup')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # Plot 2: Sensitivity
    ax2 = axes[1]
    y_pos = 0
    
    for category in ['Overall', 'Sex', 'Age']:
        if category == 'Overall':
            overall_row = combined_results[combined_results['subgroup'] == 'Overall'].iloc[0]
            ax2.scatter([overall_row['sensitivity']], [y_pos], 
                      color=colors['Overall'], marker=markers['Overall'], s=100, zorder=3)
            y_pos += 1
        else:
            category_data = combined_results[
                (combined_results['category'] == category) & 
                (combined_results['subgroup'] != 'Overall')
            ]
            
            if len(category_data) == 0:
                continue
                
            y_pos += 0.5
            
            for _, row in category_data.iterrows():
                color = colors.get(row['subgroup'], 'gray')
                marker = markers.get(row['subgroup'], 'o')
                
                ax2.scatter([row['sensitivity']], [y_pos], 
                          color=color, marker=marker, s=100, zorder=3)
                y_pos += 1
            
            y_pos += 0.5
    
    ax2.set_xlim(0.5, 1.0)
    ax2.set_ylim(-0.5, y_pos)
    ax2.set_xlabel('Subgroup\nsensitivity')
    ax2.set_title('Sensitivity by Subgroup')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # Plot 3: Specificity
    ax3 = axes[2]
    y_pos = 0
    
    for category in ['Overall', 'Sex', 'Age']:
        if category == 'Overall':
            overall_row = combined_results[combined_results['subgroup'] == 'Overall'].iloc[0]
            ax3.scatter([overall_row['specificity']], [y_pos], 
                      color=colors['Overall'], marker=markers['Overall'], s=100, zorder=3)
            y_pos += 1
        else:
            category_data = combined_results[
                (combined_results['category'] == category) & 
                (combined_results['subgroup'] != 'Overall')
            ]
            
            if len(category_data) == 0:
                continue
                
            y_pos += 0.5
            
            for _, row in category_data.iterrows():
                color = colors.get(row['subgroup'], 'gray')
                marker = markers.get(row['subgroup'], 'o')
                
                ax3.scatter([row['specificity']], [y_pos], 
                          color=color, marker=marker, s=100, zorder=3)
                y_pos += 1
            
            y_pos += 0.5
    
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(-0.5, y_pos)
    ax3.set_xlabel('Subgroup\nspecificity')
    ax3.set_title('Specificity by Subgroup')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    
    # Plot 4: Sample counts
    ax4 = axes[3]
    y_pos = 0
    
    for category in ['Overall', 'Sex', 'Age']:
        if category == 'Overall':
            overall_row = combined_results[combined_results['subgroup'] == 'Overall'].iloc[0]
            ax4.barh(y_pos, overall_row['n_negative'], color='#10B981', height=0.6)
            ax4.barh(y_pos, overall_row['n_positive'], left=overall_row['n_negative'], 
                    color='#D946EF', height=0.6)
            y_pos += 1
        else:
            category_data = combined_results[
                (combined_results['category'] == category) & 
                (combined_results['subgroup'] != 'Overall')
            ]
            
            if len(category_data) == 0:
                continue
                
            y_pos += 0.5
            
            for _, row in category_data.iterrows():
                ax4.barh(y_pos, row['n_negative'], color='#10B981', height=0.6)
                ax4.barh(y_pos, row['n_positive'], left=row['n_negative'], 
                       color='#D946EF', height=0.6)
                y_pos += 1
            
            y_pos += 0.5
    
    ax4.set_ylim(-0.5, y_pos)
    ax4.set_xlabel('Number of Non-Glaucoma (green) and\nGlaucoma (purple) in each subgroup')
    ax4.set_title('Sample Distribution')
    ax4.invert_yaxis()
    
    plt.suptitle('Test Subgroup Analysis: Mock Dataset', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def main():
    """Run the test."""
    print("Creating mock data for subgroup analysis test...")
    
    # Create mock data
    data = create_mock_data()
    
    print(f"Mock dataset created:")
    print(f"  - Total samples: {len(data['labels'])}")
    print(f"  - Glaucoma cases: {np.sum(data['labels'])}")
    print(f"  - Age range: {data['age'].min()}-{data['age'].max()}")
    print(f"  - Sex distribution: {pd.Series(data['sex']).value_counts().to_dict()}")
    
    # Create age groups
    age_groups = create_age_groups(data['age'])
    
    # Calculate subgroup metrics
    print("\nCalculating age-based subgroup metrics...")
    subgroup_results_age = calculate_subgroup_metrics(
        data['labels'], data['scores'], age_groups
    )
    
    print("\nCalculating sex-based subgroup metrics...")
    subgroup_results_sex = calculate_subgroup_metrics(
        data['labels'], data['scores'], data['sex']
    )
    
    # Print results
    print("\nAge-based subgroup results:")
    print(subgroup_results_age[['subgroup', 'n_total', 'n_positive', 'auc', 'sensitivity', 'specificity']].round(3))
    
    print("\nSex-based subgroup results:")
    print(subgroup_results_sex[['subgroup', 'n_total', 'n_positive', 'auc', 'sensitivity', 'specificity']].round(3))
    
    # Create plot
    print("\nGenerating subgroup analysis plot...")
    plot_subgroup_analysis_test(subgroup_results_age, subgroup_results_sex)
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
