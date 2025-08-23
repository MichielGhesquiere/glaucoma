# Single-Task vs Multi-Task Learning Comparison

## Overview

I've successfully adapted your script to answer the research question: **"Does adding vCDR regression improve binary glaucoma classification performance?"**

## Key Changes Made

### 1. New Model Classes
- **`SingleTaskModel`**: Classification-only model (binary glaucoma detection)
- **`MultiTaskModel`**: Original multi-task model (classification + vCDR regression)

### 2. New Loss Functions
- **`SingleTaskLoss`**: BCE loss for classification only
- **`MultiTaskLoss`**: Combined classification + regression loss (original)

### 3. Unified Trainer
- **`TaskTrainer`**: Handles both single-task and multi-task training
- Adapts behavior based on `task_mode` parameter
- Separate training histories and plots for each mode

### 4. Comparison Training Function
- **`train_single_fold_comparison()`**: Trains both models on same data splits
- Direct performance comparison with statistical analysis
- Saves separate results for each task mode

### 5. New Command Line Arguments
- **`--compare_tasks`**: Enable comparison mode
- All other arguments remain the same

## Usage Examples

### 1. Basic Comparison (ResNet18)
```bash
python train_multitask_classification_regression.py \
    --vcdr_csv path/to/your/labels.csv \
    --compare_tasks
```

### 2. Comparison with VFM Backbone
```bash
python train_multitask_classification_regression.py \
    --vcdr_csv path/to/your/labels.csv \
    --compare_tasks \
    --backbone vfm
```

### 3. Force Retrain Both Models
```bash
python train_multitask_classification_regression.py \
    --vcdr_csv path/to/your/labels.csv \
    --compare_tasks \
    --force_retrain
```

### 4. Original Multi-Task Only (No Comparison)
```bash
python train_multitask_classification_regression.py \
    --vcdr_csv path/to/your/labels.csv
```

## Output Structure

When using `--compare_tasks`, the script creates:

```
multitask_results/
├── fold_DATASET1_resnet18_comparison/
│   ├── singletask/
│   │   ├── best_model_resnet18_singletask.pth
│   │   ├── predictions_resnet18_singletask.csv
│   │   └── training_history_resnet18_singletask.png
│   ├── multitask/
│   │   ├── best_model_resnet18_multitask.pth
│   │   ├── predictions_resnet18_multitask.csv
│   │   └── training_history_resnet18_multitask.png
│   ├── results_resnet18_singletask.json
│   ├── results_resnet18_multitask.json
│   └── comparison_resnet18.json
├── fold_DATASET2_resnet18_comparison/
│   └── ... (same structure)
├── singletask_summary.csv
├── multitask_summary.csv
├── comparison_summary.csv
└── aggregated_comparison_results.json
```

## Comparison Metrics

The script will output detailed comparisons:

### Console Output Example:
```
CLASSIFICATION PERFORMANCE COMPARISON:
==================================================
Metric       Single-Task     Multi-Task      Mean Δ       % Improved  
----------------------------------------------------------------------
ACCURACY     0.8234          0.8367          +0.0133      75.0%
PRECISION    0.7891          0.8123          +0.0232      87.5%
RECALL       0.8012          0.8089          +0.0077      62.5%
F1           0.7951          0.8106          +0.0155      75.0%
AUC          0.8456          0.8623          +0.0167      87.5%

DETAILED IMPROVEMENT STATISTICS:
==================================================

ACCURACY:
  Mean improvement: +0.0133 (+1.6%)
  Std improvement:  ±0.0089 (±1.1%)
  Datasets improved: 6/8 (75.0%)
  Datasets degraded: 2/8 (25.0%)
```

## Research Insights

### Expected Outcomes:

1. **Positive Case**: Multi-task learning helps
   - vCDR regression provides useful inductive bias
   - Shared representations improve classification
   - Better feature learning from fundus images

2. **Negative Case**: Multi-task learning hurts
   - Task interference between classification and regression
   - vCDR task is too difficult and distracts from classification
   - Loss balancing issues

3. **Neutral Case**: No significant difference
   - Tasks are orthogonal
   - Classification task dominates
   - Shared backbone already captures relevant features

### Realistic Expectations:

Multi-task learning **often helps** in medical imaging because:
- Both tasks require understanding of optic disc structure
- vCDR measurement requires detailed anatomical understanding
- Classification benefits from this detailed feature extraction
- Regularization effect from additional supervision

## File Structure Changes

The key files modified:
- `train_multitask_classification_regression.py` - Main script with comparison functionality
- All original functionality preserved with `--compare_tasks` flag

## Performance Considerations

- **Training Time**: ~2x longer (trains both models)
- **Storage**: ~2x storage for models and results
- **Memory**: Same GPU memory usage (models trained sequentially)
- **Evaluation**: Comprehensive statistical comparison

## Statistical Validity

The script provides:
- Mean improvement across datasets
- Standard deviation of improvements
- Percentage of datasets showing improvement
- Detailed per-dataset breakdowns
- Confidence intervals for differences

This gives you statistically sound evidence for whether multi-task learning helps your specific glaucoma classification task.
