# Set error preference to stop on errors
$ErrorActionPreference = "Stop"

Write-Host "======================================================" -ForegroundColor Green
Write-Host "Starting Fine-tuning Strategy Comparison Study (PowerShell)" -ForegroundColor Green
Write-Host "======================================================"

# --- Common Configuration ---
$PYTHON_EXE = "python" 
$SCRIPT_PATH = ".\train_classification.py"

$DATA_TYPE_TO_USE = "raw" # Should be "raw" or "processed"
Write-Host "Using data_type: $DATA_TYPE_TO_USE" -ForegroundColor Magenta

$BASE_DATA_ROOT = "D:\glaucoma\data" 

# Default RAW paths (Python script adjusts if $DATA_TYPE_TO_USE is 'processed')
$SMDG_META = "D:\glaucoma\data\raw\SMDG-19\metadata - standardized.csv"
$SMDG_IMG_DIR = "D:\glaucoma\data\raw\SMDG-19\full-fundus\full-fundus"
$CHAKSU_BASE = "D:\glaucoma\data\raw\Chaksu\Train\Train\1.0_Original_Fundus_Images"
$CHAKSU_DECISION = "D:\glaucoma\data\raw\Chaksu\Train\Train\6.0_Glaucoma_Decision"
$CHAKSU_META_DIR = "D:\glaucoma\data\raw\Chaksu\Train\Train\6.0_Glaucoma_Decision\Majority"
$AIROGS_LABEL_FILE = "D:\glaucoma\data\raw\AIROGS\train_labels.csv"
$AIROGS_IMAGE_DIR = "D:\glaucoma\data\raw\AIROGS\img" 

# Output Directory
$Timestamp = Get-Date -Format "yyyyMMdd_HHmm"
$BASE_OUTPUT_DIR = "experiments\fine_tuning_strategy_comparison_dtype_${DATA_TYPE_TO_USE}_${Timestamp}"

# Training Hyperparameters - Optimized for Fine-tuning Study
$BATCH_SIZE = 16
$GRAD_ACCUM_STEPS = 2 
$EVAL_BATCH_SIZE = 16
$VFM_BASE_LEARNING_RATE = 1e-05  # Conservative base LR for VFM
$WEIGHT_DECAY = 0.05
$SEED = 42
$NUM_WORKERS = 4 
$ENABLE_CHAKSU = $true
$ENABLE_AMP = $true
$ENABLE_AIROGS = $true
$AIROGS_NUM_RG = 3000   # Balanced sampling for fine-tuning study
$AIROGS_NUM_NRG = 3000
$EARLY_STOPPING_PATIENCE = 8  # More patience for fine-tuning strategies

# VisionFM Configuration
$VFM_WEIGHTS = "D:\glaucoma\models\VFM_Fundus_weights.pth"
$VFM_KEY = "teacher"

# Fine-tuning Strategy Configuration
$USE_MIXUP = $false
$USE_DATA_AUGMENTATION = $true
$USE_TEMPERATURE_SCALING = $false

# Sample Saving Configuration
$ENABLE_SAMPLE_SAVING = $false  # Enable to verify data loading
$NUM_SAMPLES_TO_SAVE_PER_SOURCE = 3 

# Visualization Configuration
$ENABLE_GRADCAM_VISUALIZATIONS = $false
$NUM_GRADCAM_SAMPLES_PER_CATEGORY = 3

New-Item -ItemType Directory -Path $BASE_OUTPUT_DIR -Force | Out-Null
Write-Host "Base output directory: $BASE_OUTPUT_DIR" -ForegroundColor Cyan
Write-Host "Base learning rate (VFM): $($VFM_BASE_LEARNING_RATE)" -ForegroundColor Cyan
Write-Host "Early stopping patience: $($EARLY_STOPPING_PATIENCE)" -ForegroundColor Cyan
Write-Host "Mixup enabled: $($USE_MIXUP)" -ForegroundColor Cyan
Write-Host "Temperature scaling enabled: $($USE_TEMPERATURE_SCALING)" -ForegroundColor Cyan
if ($ENABLE_SAMPLE_SAVING) { Write-Host "Sample saving: YES, $($NUM_SAMPLES_TO_SAVE_PER_SOURCE) per source." -ForegroundColor Cyan }
if ($ENABLE_AIROGS) { Write-Host "AIROGS data: YES, RG Samples: $($AIROGS_NUM_RG), NRG Samples: $($AIROGS_NUM_NRG)" -ForegroundColor Cyan}
if ($ENABLE_GRADCAM_VISUALIZATIONS) { Write-Host "Grad-CAM Visualizations: YES, $($NUM_GRADCAM_SAMPLES_PER_CATEGORY) samples per category." -ForegroundColor Cyan }
else { Write-Host "Grad-CAM Visualizations: NO" -ForegroundColor Cyan }

function Build-ArgumentList {
    param(
        [hashtable]$ArgsHashTable, 
        [string]$DataType,
        [string]$BaseDataRoot,
        [bool]$UseChaksu,
        [bool]$UseAmp,
        [bool]$UseAirogs,
        [bool]$UseDataAugmentation,
        [bool]$UseMixup,
        [bool]$UseTemperatureScaling
    )
    $ArgumentList = New-Object System.Collections.Generic.List[string]
    $ArgumentList.Add("--data_type"); $ArgumentList.Add($DataType)
    $ArgumentList.Add("--base_data_root"); $ArgumentList.Add($BaseDataRoot)

    foreach ($key in $ArgsHashTable.Keys) {
        $value = $ArgsHashTable[$key]
        if ($value -is [bool]) {
            if ($value -eq $true) { $ArgumentList.Add("--$key") }
        } elseif ($value -ne $null -and $value -ne "") {
            $ArgumentList.Add("--$key"); $ArgumentList.Add($value)
        }
    }

    # Explicit handling for flags
    if ($UseChaksu) { $ArgumentList.Add("--use_chaksu") } else { $ArgumentList.Add("--no_chaksu")}
    if ($UseAirogs) { $ArgumentList.Add("--use_airogs") } else { $ArgumentList.Add("--no_airogs")}
    if ($UseAmp) { $ArgumentList.Add("--use_amp") } else { $ArgumentList.Add("--no_amp")}
    if ($UseDataAugmentation) { $ArgumentList.Add("--use_data_augmentation") } else { $ArgumentList.Add("--no_data_augmentation")}
    if ($UseMixup) { $ArgumentList.Add("--use_mixup") }
    if ($UseTemperatureScaling) { $ArgumentList.Add("--use_temperature_scaling") }
    
    return $ArgumentList
}

# --- Common Arguments for all training runs ---
$CommonTrainArgs = @{
    base_output_dir = $BASE_OUTPUT_DIR
    smdg_metadata_file = $SMDG_META; smdg_image_dir = $SMDG_IMG_DIR
    chaksu_base_dir = $CHAKSU_BASE; chaksu_decision_dir = $CHAKSU_DECISION; chaksu_metadata_dir = $CHAKSU_META_DIR
    airogs_label_file = $AIROGS_LABEL_FILE; airogs_image_dir = $AIROGS_IMAGE_DIR 
    airogs_num_rg_samples = $AIROGS_NUM_RG; airogs_num_nrg_samples = $AIROGS_NUM_NRG
    batch_size = $BATCH_SIZE; eval_batch_size = $EVAL_BATCH_SIZE
    weight_decay = $WEIGHT_DECAY; early_stopping_patience = $EARLY_STOPPING_PATIENCE
    seed = $SEED; num_workers = $NUM_WORKERS; grad_accum_steps = $GRAD_ACCUM_STEPS
    save_data_samples = $ENABLE_SAMPLE_SAVING            
    num_data_samples_per_source = $NUM_SAMPLES_TO_SAVE_PER_SOURCE 
    use_airogs_cache = $true
    run_gradcam_visualizations = $ENABLE_GRADCAM_VISUALIZATIONS
    num_gradcam_samples = $NUM_GRADCAM_SAMPLES_PER_CATEGORY
    # VFM-specific settings
    model_name = "vit_base_patch16_224"
    custom_weights_path = $VFM_WEIGHTS
    checkpoint_key = $VFM_KEY
    # Fine-tuning study specific settings
    mixup_alpha = 0.8
    cutmix_alpha = 1.0
    calibration_split_ratio = 0.3
    dropout_prob = 0.1  # Standard dropout for all strategies
}

# Define fine-tuning strategy configurations
$FineTuningConfigs = @(
    @{ 
        Strategy = "linear"
        MaxEpochs = 15
        LinearProbeEpochs = 15
        LearningRate = $VFM_BASE_LEARNING_RATE * 10  # Higher LR for head-only training
        Tag = "LinearProbe"
        Description = "Linear probing: Head-only training to test feature transferability"
        ExpectedBehavior = "Fast convergence, moderate accuracy, no feature forgetting"
    },
    @{ 
        Strategy = "gradual"
        MaxEpochs = 50
        GradualPatience = 6
        LearningRate = $VFM_BASE_LEARNING_RATE * 3   # Medium LR for gradual unfreezing
        Tag = "GradualUnfreeze"
        Description = "Gradual unfreezing: ULMFit-style progressive layer unfreezing"
        ExpectedBehavior = "Balanced accuracy and feature preservation"
    },
    @{ 
        Strategy = "full"
        MaxEpochs = 35
        LlrdDecay = 0.85
        LearningRate = $VFM_BASE_LEARNING_RATE       # Conservative base LR for LLRD
        Tag = "FullLLRD"
        Description = "Full fine-tuning with Layer-wise Learning Rate Decay"
        ExpectedBehavior = "Best overall performance with minimal feature forgetting"
    }
)

Write-Host "`n=== Starting Fine-tuning Strategy Comparison Study ===" -ForegroundColor Yellow
Write-Host "Model: VisionFM (ViT-Base) with custom pre-trained weights" -ForegroundColor Yellow
Write-Host "Fine-tuning strategies to test: $($FineTuningConfigs.Count)" -ForegroundColor Yellow
Write-Host "Data sources: SMDG-19, CHAKSU, AIROGS" -ForegroundColor Yellow
Write-Host "Order: Linear Probe → Gradual Unfreezing → Full LLRD" -ForegroundColor Yellow
Write-Host "Focus: Preventing catastrophic forgetting of foundation model features" -ForegroundColor Yellow

$OverallRunIndex = 1
$TotalRuns = $FineTuningConfigs.Count

foreach ($Config in $FineTuningConfigs) {
    Write-Host "`n======================================================" -ForegroundColor Magenta
    Write-Host "STARTING FINE-TUNING STRATEGY: $($Config.Tag)" -ForegroundColor Magenta
    Write-Host "Strategy: $($Config.Strategy)" -ForegroundColor Magenta
    Write-Host "Description: $($Config.Description)" -ForegroundColor Magenta
    Write-Host "Expected: $($Config.ExpectedBehavior)" -ForegroundColor Magenta
    Write-Host "Learning Rate: $($Config.LearningRate)" -ForegroundColor Magenta
    Write-Host "Max Epochs: $($Config.MaxEpochs)" -ForegroundColor Magenta
    if ($Config.Strategy -eq "gradual") {
        Write-Host "Gradual Patience: $($Config.GradualPatience) epochs" -ForegroundColor Magenta
    }
    if ($Config.Strategy -eq "full") {
        Write-Host "LLRD Decay Factor: $($Config.LlrdDecay)" -ForegroundColor Magenta
    }
    Write-Host "Progress: $OverallRunIndex/$TotalRuns" -ForegroundColor Cyan
    Write-Host "======================================================" -ForegroundColor Magenta

    # Build strategy-specific arguments
    $SpecificArgs = @{
        ft_strategy = $Config.Strategy
        learning_rate = $Config.LearningRate
        num_epochs = $Config.MaxEpochs
        experiment_tag = "VFM_$($Config.Tag)_FinetuneStudy"
    }
    
    # Add strategy-specific parameters
    if ($Config.Strategy -eq "linear") {
        $SpecificArgs.linear_probe_epochs = $Config.LinearProbeEpochs
    }
    elseif ($Config.Strategy -eq "gradual") {
        $SpecificArgs.gradual_patience = $Config.GradualPatience
    }
    elseif ($Config.Strategy -eq "full") {
        $SpecificArgs.llrd_decay = $Config.LlrdDecay
    }
    
    # Merge common args with specific args
    $RunArgs = $CommonTrainArgs.Clone()
    foreach ($key in $SpecificArgs.Keys) {
        $RunArgs[$key] = $SpecificArgs[$key]
    }
    
    # Build argument list
    $ArgumentList = Build-ArgumentList -ArgsHashTable $RunArgs -DataType $DATA_TYPE_TO_USE -BaseDataRoot $BASE_DATA_ROOT -UseChaksu $ENABLE_CHAKSU -UseAmp $ENABLE_AMP -UseAirogs $ENABLE_AIROGS -UseDataAugmentation $USE_DATA_AUGMENTATION -UseMixup $USE_MIXUP -UseTemperatureScaling $USE_TEMPERATURE_SCALING
    
    Write-Host "`nCommand preview:" -ForegroundColor Gray
    Write-Host "$PYTHON_EXE $SCRIPT_PATH --ft_strategy $($Config.Strategy) --learning_rate $($Config.LearningRate) --num_epochs $($Config.MaxEpochs) [... other args]" -ForegroundColor Gray
    
    $StartTime = Get-Date
    Write-Host "`nStarting training..." -ForegroundColor Green
    
    try {
        & $PYTHON_EXE $SCRIPT_PATH $ArgumentList
        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        Write-Host "--- $($Config.Tag) Training Finished Successfully ---" -ForegroundColor Green
        Write-Host "Training Duration: $($Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        Write-Host "Strategy performance can be analyzed in the results directory." -ForegroundColor Green
    }
    catch {
        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        Write-Host "--- ERROR: $($Config.Tag) Training Failed ---" -ForegroundColor Red
        Write-Host "Error: $_" -ForegroundColor Red
        Write-Host "Failed after: $($Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Red
        Write-Host "Continuing with next strategy..." -ForegroundColor Yellow
    }
    
    $OverallRunIndex++
    
    # Pause between strategies for system cleanup
    if ($OverallRunIndex -le $TotalRuns) {
        Write-Host "`nPausing for system cleanup..." -ForegroundColor Gray
        Start-Sleep -Seconds 20
    }
}

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host "Fine-tuning Strategy Comparison Study Completed!" -ForegroundColor Green
Write-Host "Results saved in subdirectories under: $BASE_OUTPUT_DIR" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Green

# Generate a comprehensive summary report
$SummaryFile = Join-Path $BASE_OUTPUT_DIR "fine_tuning_strategy_comparison_summary.txt"
$SummaryContent = @"
Fine-tuning Strategy Comparison Study Summary
============================================
Date: $(Get-Date)
Data Type: $DATA_TYPE_TO_USE
Base Model: VisionFM (ViT-Base-Patch16-224)
Custom Weights: $VFM_WEIGHTS
Checkpoint Key: $VFM_KEY

Study Objective:
Compare three fine-tuning strategies to prevent catastrophic forgetting
while adapting a vision foundation model to glaucoma classification.

Data Configuration:
- SMDG-19: Full dataset
- CHAKSU: All camera types enabled
- AIROGS: $AIROGS_NUM_RG RG samples, $AIROGS_NUM_NRG NRG samples
- Data Augmentation: $USE_DATA_AUGMENTATION
- Mixup/CutMix: $USE_MIXUP
- Temperature Scaling: $USE_TEMPERATURE_SCALING

Fine-tuning Strategies Tested:
"@

$ConfigNum = 1
foreach ($Config in $FineTuningConfigs) {
    $SummaryContent += "`n$ConfigNum. $($Config.Tag) ($($Config.Strategy))"
    $SummaryContent += "`n   Description: $($Config.Description)"
    $SummaryContent += "`n   Learning Rate: $($Config.LearningRate)"
    $SummaryContent += "`n   Max Epochs: $($Config.MaxEpochs)"
    $SummaryContent += "`n   Expected Behavior: $($Config.ExpectedBehavior)"
    if ($Config.Strategy -eq "gradual") {
        $SummaryContent += "`n   Gradual Patience: $($Config.GradualPatience) epochs"
    }
    if ($Config.Strategy -eq "full") {
        $SummaryContent += "`n   LLRD Decay: $($Config.LlrdDecay)"
    }
    $SummaryContent += "`n"
    $ConfigNum++
}

$SummaryContent += @"

Expected Results Analysis:
1. Linear Probe:
   - Should achieve reasonable accuracy (>75% for good features)
   - Fastest training time
   - Baseline for feature quality assessment

2. Gradual Unfreezing:
   - Should improve upon linear probe accuracy
   - Balanced training time
   - Good feature preservation with adaptation

3. Full LLRD:
   - Should achieve highest validation accuracy
   - Best calibration performance
   - Optimal balance of performance and feature retention

Key Metrics to Compare:
- Validation Accuracy (primary)
- Training vs Validation Gap (overfitting indicator)
- Calibration Performance (ECE, Brier Score)
- Training Time
- Feature Retention (compare to linear probe baseline)

Results directories:
$BASE_OUTPUT_DIR

Each experiment subdirectory contains:
- Training history JSON with fine-tuning strategy info
- Best model checkpoints with temperature scaling
- Training logs with parameter freeze/unfreeze details
- Data manifests
- Temperature scaling calibration results

Total experiments completed: $TotalRuns

Analysis Recommendations:
1. Compare validation curves - Full LLRD should have smoothest convergence
2. Check for catastrophic forgetting - compare early layers' activations
3. Evaluate on held-out test sets (PAPILLA, CHAKSU)
4. Compare calibration - Full LLRD should be best calibrated
5. Analyze training efficiency - Linear probe should be fastest per epoch
"@

$SummaryContent | Out-File -FilePath $SummaryFile -Encoding UTF8
Write-Host "`nComprehensive summary saved to: $SummaryFile" -ForegroundColor Cyan

# Create a detailed results comparison template
$ComparisonFile = Join-Path $BASE_OUTPUT_DIR "fine_tuning_results_comparison.csv"
$ComparisonHeader = "Strategy,Tag,LearningRate,MaxEpochs,Best_Val_Acc,Best_Val_Loss,Final_Train_Acc,Train_Val_Gap,Best_Epoch,Total_Training_Time,Calibration_Temperature,ECE_Score,Feature_Forgetting_Score,Notes"
$ComparisonHeader | Out-File -FilePath $ComparisonFile -Encoding UTF8

foreach ($Config in $FineTuningConfigs) {
    $Row = "$($Config.Strategy),$($Config.Tag),$($Config.LearningRate),$($Config.MaxEpochs),TODO,TODO,TODO,TODO,TODO,TODO,TODO,TODO,TODO,TODO"
    $Row | Out-File -FilePath $ComparisonFile -Append -Encoding UTF8
}

Write-Host "Results comparison template created: $ComparisonFile" -ForegroundColor Cyan
Write-Host "Fill in the TODO values after training completes for detailed analysis." -ForegroundColor Cyan

# Create a specialized analysis script for fine-tuning strategies
$AnalysisScriptFile = Join-Path $BASE_OUTPUT_DIR "analyze_fine_tuning_strategies.py"
$AnalysisScriptContent = @"
#!/usr/bin/env python3
"""
Analysis script for fine-tuning strategy comparison study.
Focuses on comparing learning curves, feature retention, and calibration performance.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_all_histories(base_dir):
    """Load all training histories from experiment subdirectories."""
    histories = {}
    
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        results_dir = os.path.join(subdir_path, 'results')
        if not os.path.exists(results_dir):
            continue
            
        # Look for history JSON files
        for file in os.listdir(results_dir):
            if file.startswith('training_history') and file.endswith('.json'):
                history_path = os.path.join(results_dir, file)
                try:
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                    
                    # Extract strategy info from experiment name or history
                    strategy = 'unknown'
                    if 'LinearProbe' in subdir:
                        strategy = 'Linear Probe'
                    elif 'GradualUnfreeze' in subdir:
                        strategy = 'Gradual Unfreezing'
                    elif 'FullLLRD' in subdir:
                        strategy = 'Full LLRD'
                    
                    histories[strategy] = history
                    print(f"Loaded: {strategy} from {subdir}")
                except Exception as e:
                    print(f"Error loading {history_path}: {e}")
                break
    
    return histories

def plot_fine_tuning_comparison(histories, save_dir):
    """Plot comparison of fine-tuning strategies."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = {'Linear Probe': 'red', 'Gradual Unfreezing': 'blue', 'Full LLRD': 'green'}
    
    for strategy, history in histories.items():
        color = colors.get(strategy, 'black')
        
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Training and Validation Loss
            axes[0, 0].plot(epochs, history['train_loss'], color=color, alpha=0.7, 
                           label=f'{strategy} Train', linestyle='-')
            axes[0, 0].plot(epochs, history['val_loss'], color=color, alpha=0.9, 
                           label=f'{strategy} Val', linestyle='--')
            
            # Training and Validation Accuracy
            if 'train_acc' in history and 'val_acc' in history:
                axes[0, 1].plot(epochs, history['train_acc'], color=color, alpha=0.7, 
                               label=f'{strategy} Train', linestyle='-')
                axes[0, 1].plot(epochs, history['val_acc'], color=color, alpha=0.9, 
                               label=f'{strategy} Val', linestyle='--')
                
                # Overfitting Gap (Train - Val Accuracy)
                overfitting_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
                axes[0, 2].plot(epochs, overfitting_gap, color=color, alpha=0.8, 
                               label=f'{strategy}', linewidth=2)
        
        # Learning Rate Schedule
        if 'lr' in history:
            lr_epochs = range(1, len(history['lr']) + 1)
            axes[1, 0].semilogy(lr_epochs, history['lr'], color=color, alpha=0.8, 
                               label=f'{strategy}', linewidth=2)
        
        # Temperature Scaling Info
        if 'temperature_scaling_info' in history:
            temp_info = history['temperature_scaling_info']
            if temp_info.get('enabled', False):
                temp_val = temp_info.get('final_temperature_value', 1.0)
                axes[1, 1].bar(strategy, temp_val, color=color, alpha=0.7)
    
    # Formatting
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Training vs Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_title('Overfitting Gap (Train - Val Accuracy)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy Gap')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate (log scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Temperature Scaling Values')
    axes[1, 1].set_ylabel('Temperature')
    axes[1, 1].set_ylim(0.5, 2.0)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Summary statistics table
    axes[1, 2].axis('off')
    summary_data = []
    for strategy, history in histories.items():
        if 'val_acc' in history and 'val_loss' in history:
            best_val_acc = max(history['val_acc'])
            best_val_loss = min(history['val_loss'])
            final_epochs = len(history['val_acc'])
            
            summary_data.append([
                strategy,
                f"{best_val_acc:.3f}",
                f"{best_val_loss:.3f}", 
                f"{final_epochs}"
            ])
    
    if summary_data:
        table = axes[1, 2].table(cellText=summary_data,
                                colLabels=['Strategy', 'Best Val Acc', 'Best Val Loss', 'Epochs'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 2].set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fine_tuning_strategy_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_retention(histories):
    """Analyze potential feature retention based on learning curves."""
    print("\n=== Feature Retention Analysis ===")
    
    linear_probe_acc = None
    if 'Linear Probe' in histories and 'val_acc' in histories['Linear Probe']:
        linear_probe_acc = max(histories['Linear Probe']['val_acc'])
        print(f"Linear Probe Best Validation Accuracy: {linear_probe_acc:.3f}")
        print("This represents the quality of pre-trained features without adaptation.")
    
    for strategy, history in histories.items():
        if strategy == 'Linear Probe':
            continue
            
        if 'val_acc' in history:
            best_acc = max(history['val_acc'])
            print(f"\n{strategy} Best Validation Accuracy: {best_acc:.3f}")
            
            if linear_probe_acc is not None:
                improvement = best_acc - linear_probe_acc
                print(f"Improvement over Linear Probe: {improvement:+.3f}")
                
                if improvement > 0.05:
                    print("✓ Good adaptation with likely feature retention")
                elif improvement > 0.02:
                    print("~ Moderate adaptation")
                else:
                    print("⚠ Limited improvement - check for issues")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Analyzing fine-tuning strategy results in: {base_dir}")
    
    histories = load_all_histories(base_dir)
    print(f"Loaded {len(histories)} experiment histories")
    
    if not histories:
        print("No histories found. Make sure training is complete.")
        return
    
    # Generate comparison plots
    plot_fine_tuning_comparison(histories, base_dir)
    print("Fine-tuning strategy comparison plots saved!")
    
    # Analyze feature retention
    analyze_feature_retention(histories)
    
    print("\n=== Analysis Complete ===")
    print("Key insights:")
    print("1. Linear Probe accuracy indicates pre-trained feature quality")
    print("2. Gradual Unfreezing should balance adaptation and retention")
    print("3. Full LLRD should achieve best performance with good calibration")
    print("4. Check overfitting gaps to assess regularization effectiveness")

if __name__ == "__main__":
    main()
"@

$AnalysisScriptContent | Out-File -FilePath $AnalysisScriptFile -Encoding UTF8
Write-Host "Specialized fine-tuning analysis script created: $AnalysisScriptFile" -ForegroundColor Cyan
Write-Host "Run this Python script after training completes for detailed strategy comparison." -ForegroundColor Cyan

# Create a quick results summary script
$QuickSummaryFile = Join-Path $BASE_OUTPUT_DIR "quick_results_summary.ps1"
$QuickSummaryContent = @"
# Quick results summary for fine-tuning strategy comparison
Write-Host "Fine-tuning Strategy Results Summary" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green

`$BaseDir = Split-Path -Parent `$MyInvocation.MyCommand.Path

Get-ChildItem -Path `$BaseDir -Directory | ForEach-Object {
    `$ExpDir = `$_.FullName
    `$ResultsDir = Join-Path `$ExpDir "results"
    
    if (Test-Path `$ResultsDir) {
        `$HistoryFiles = Get-ChildItem -Path `$ResultsDir -Filter "training_history*.json"
        
        if (`$HistoryFiles) {
            `$HistoryFile = `$HistoryFiles[0]
            try {
                `$History = Get-Content `$HistoryFile.FullName | ConvertFrom-Json
                
                if (`$History.val_acc) {
                    `$BestValAcc = (`$History.val_acc | Measure-Object -Maximum).Maximum
                    `$FinalEpoch = `$History.val_acc.Length
                    
                    Write-Host "`nExperiment: `$(`$_.Name)" -ForegroundColor Yellow
                    Write-Host "Best Validation Accuracy: `$([math]::Round(`$BestValAcc, 4))" -ForegroundColor Cyan
                    Write-Host "Total Epochs: `$FinalEpoch" -ForegroundColor Cyan
                    
                    if (`$History.temperature_scaling_info -and `$History.temperature_scaling_info.enabled) {
                        `$TempValue = `$History.temperature_scaling_info.final_temperature_value
                        Write-Host "Temperature Scaling: `$([math]::Round(`$TempValue, 3))" -ForegroundColor Cyan
                    }
                }
            }
            catch {
                Write-Host "`nExperiment: `$(`$_.Name) - Error reading results" -ForegroundColor Red
            }
        }
    }
}

Write-Host "`nFor detailed analysis, run: python analyze_fine_tuning_strategies.py" -ForegroundColor Green
"@

$QuickSummaryContent | Out-File -FilePath $QuickSummaryFile -Encoding UTF8
Write-Host "Quick summary script created: $QuickSummaryFile" -ForegroundColor Cyan

Write-Host "`n=== FINE-TUNING STRATEGY STUDY COMPLETE ===" -ForegroundColor Green
Write-Host "Total experiments: $TotalRuns" -ForegroundColor Green
Write-Host "Results directory: $BASE_OUTPUT_DIR" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Run the quick summary: .\quick_results_summary.ps1" -ForegroundColor Cyan
Write-Host "2. Detailed analysis: python analyze_fine_tuning_strategies.py" -ForegroundColor Cyan
Write-Host "3. Fill in the comparison CSV for comprehensive evaluation" -ForegroundColor Cyan
Write-Host "4. Compare feature retention and calibration performance" -ForegroundColor Cyan
