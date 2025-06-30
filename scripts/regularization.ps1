# Set error preference to stop on errors
$ErrorActionPreference = "Stop"

Write-Host "======================================================" -ForegroundColor Green
Write-Host "Starting Multi-Model Regularization Study (PowerShell)" -ForegroundColor Green
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
$BASE_OUTPUT_DIR = "experiments\multi_model_regularization_study_dtype_${DATA_TYPE_TO_USE}"

# Training Hyperparameters
$NUM_EPOCHS = 60 
$PATIENCE = 5 
$BATCH_SIZE = 16
$GRAD_ACCUM_STEPS = 2 
$EVAL_BATCH_SIZE = 16
$RESNET_LEARNING_RATE = 3e-04  # Higher LR for ResNet50
$VFM_LEARNING_RATE = 1e-05     # Lower LR for VisionFM
$WEIGHT_DECAY = 0.05
$SEED = 42
$NUM_WORKERS = 4 
$ENABLE_CHAKSU = $true
$ENABLE_AMP = $true
$ENABLE_AIROGS = $true
$AIROGS_NUM_RG = 3000
$AIROGS_NUM_NRG = 3000

# VisionFM Configuration
$VFM_WEIGHTS = "D:\glaucoma\models\VFM_Fundus_weights.pth"
$VFM_KEY = "teacher"

# Sample Saving Configuration
$ENABLE_SAMPLE_SAVING = $false 
$NUM_SAMPLES_TO_SAVE_PER_SOURCE = 5 

# Visualization Configuration
$ENABLE_GRADCAM_VISUALIZATIONS = $false
$NUM_GRADCAM_SAMPLES_PER_CATEGORY = 3

New-Item -ItemType Directory -Path $BASE_OUTPUT_DIR -Force | Out-Null
Write-Host "Base output directory: $BASE_OUTPUT_DIR" -ForegroundColor Cyan
Write-Host "Training for $($NUM_EPOCHS) epochs with seed $($SEED)." -ForegroundColor Cyan
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
        [bool]$UseDataAugmentation
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
    
    return $ArgumentList
}

# --- Common Arguments for all training runs ---
$CommonTrainArgs = @{
    base_output_dir = $BASE_OUTPUT_DIR
    smdg_metadata_file = $SMDG_META; smdg_image_dir = $SMDG_IMG_DIR
    chaksu_base_dir = $CHAKSU_BASE; chaksu_decision_dir = $CHAKSU_DECISION; chaksu_metadata_dir = $CHAKSU_META_DIR
    airogs_label_file = $AIROGS_LABEL_FILE; airogs_image_dir = $AIROGS_IMAGE_DIR 
    airogs_num_rg_samples = $AIROGS_NUM_RG; airogs_num_nrg_samples = $AIROGS_NUM_NRG
    num_epochs = $NUM_EPOCHS; batch_size = $BATCH_SIZE; eval_batch_size = $EVAL_BATCH_SIZE
    weight_decay = $WEIGHT_DECAY; early_stopping_patience = $PATIENCE
    seed = $SEED; num_workers = $NUM_WORKERS; grad_accum_steps = $GRAD_ACCUM_STEPS
    save_data_samples = $ENABLE_SAMPLE_SAVING            
    num_data_samples_per_source = $NUM_SAMPLES_TO_SAVE_PER_SOURCE 
    use_airogs_cache = $true
    run_gradcam_visualizations = $ENABLE_GRADCAM_VISUALIZATIONS
    num_gradcam_samples = $NUM_GRADCAM_SAMPLES_PER_CATEGORY
}

# Define model configurations
$ModelConfigs = @(
    @{ 
        Name = "ResNet50"; 
        ModelName = "resnet50"; 
        LearningRate = $RESNET_LEARNING_RATE; 
        UseTimm = $true;
        CustomWeights = $null;
        CheckpointKey = $null;
        Tag = "ResNet50"
    },
    @{ 
        Name = "VisionFM"; 
        ModelName = "vit_base_patch16_224"; 
        LearningRate = $VFM_LEARNING_RATE; 
        UseTimm = $false;
        CustomWeights = $VFM_WEIGHTS;
        CheckpointKey = $VFM_KEY;
        Tag = "VFM"
    }
)

# Define regularization configurations in progressive order (baseline to strongest)
$RegularizationConfigs = @(
    @{ DataAug = $false; Dropout = 0.0; Tag = "Baseline_NoDataAug_NoDropout"; Description = "Baseline: No regularization" },
    @{ DataAug = $true;  Dropout = 0.0; Tag = "DataAug_NoDropout"; Description = "Data Augmentation only" },
    @{ DataAug = $true;  Dropout = 0.3; Tag = "DataAug_Dropout0.3"; Description = "Data Aug + Dropout 0.3" },
    @{ DataAug = $true;  Dropout = 0.5; Tag = "DataAug_Dropout0.5"; Description = "Data Aug + Dropout 0.5 (strongest)" }
)

Write-Host "`n=== Starting Multi-Model Progressive Regularization Study ===" -ForegroundColor Yellow
Write-Host "Models to test: $($ModelConfigs.Count)" -ForegroundColor Yellow
Write-Host "Regularization configurations per model: $($RegularizationConfigs.Count)" -ForegroundColor Yellow
Write-Host "Total training runs: $($ModelConfigs.Count * $RegularizationConfigs.Count)" -ForegroundColor Yellow
Write-Host "Order: Baseline → Data Aug → Data Aug + Dropout 0.3 → Data Aug + Dropout 0.5" -ForegroundColor Yellow

$OverallRunIndex = 1
$TotalRuns = $ModelConfigs.Count * $RegularizationConfigs.Count

foreach ($ModelConfig in $ModelConfigs) {
    Write-Host "`n======================================================" -ForegroundColor Magenta
    Write-Host "STARTING MODEL: $($ModelConfig.Name)" -ForegroundColor Magenta
    Write-Host "Model Architecture: $($ModelConfig.ModelName)" -ForegroundColor Magenta
    Write-Host "Learning Rate: $($ModelConfig.LearningRate)" -ForegroundColor Magenta
    if ($ModelConfig.CustomWeights) {
        Write-Host "Custom Weights: $($ModelConfig.CustomWeights)" -ForegroundColor Magenta
        Write-Host "Checkpoint Key: $($ModelConfig.CheckpointKey)" -ForegroundColor Magenta
    } else {
        Write-Host "Using TIMM Pretrained: $($ModelConfig.UseTimm)" -ForegroundColor Magenta
    }
    Write-Host "======================================================" -ForegroundColor Magenta

    $ConfigIndex = 1
    foreach ($Config in $RegularizationConfigs) {
        Write-Host "`n----------------------------------------" -ForegroundColor Yellow
        Write-Host "Overall Progress: $OverallRunIndex/$TotalRuns" -ForegroundColor Cyan
        Write-Host "Model: $($ModelConfig.Name) | Configuration $ConfigIndex/$($RegularizationConfigs.Count): $($Config.Tag)" -ForegroundColor Yellow
        Write-Host "$($Config.Description)" -ForegroundColor Yellow
        Write-Host "Data Augmentation: $($Config.DataAug), Dropout: $($Config.Dropout)" -ForegroundColor Cyan
        
        # Build model-specific arguments
        $SpecificArgs = @{
            model_name = $ModelConfig.ModelName
            learning_rate = $ModelConfig.LearningRate
            dropout_prob = $Config.Dropout
            experiment_tag = "$($ModelConfig.Tag)_$($Config.Tag)"
        }
        
        # Add model-specific configuration
        if ($ModelConfig.UseTimm) {
            $SpecificArgs.use_timm_pretrained_if_no_custom = $true
        }
        
        if ($ModelConfig.CustomWeights) {
            $SpecificArgs.custom_weights_path = $ModelConfig.CustomWeights
            $SpecificArgs.checkpoint_key = $ModelConfig.CheckpointKey
        }
        
        # Merge common args with specific args
        $RunArgs = $CommonTrainArgs.Clone()
        $RunArgs.GetEnumerator() | ForEach-Object { $SpecificArgs[$_.Name] = $_.Value }
        
        # Build argument list
        $ArgumentList = Build-ArgumentList -ArgsHashTable $SpecificArgs -DataType $DATA_TYPE_TO_USE -BaseDataRoot $BASE_DATA_ROOT -UseChaksu $ENABLE_CHAKSU -UseAmp $ENABLE_AMP -UseAirogs $ENABLE_AIROGS -UseDataAugmentation $Config.DataAug
        
        Write-Host "Command: $PYTHON_EXE $SCRIPT_PATH $($ArgumentList -join ' ')" -ForegroundColor Gray
        
        $StartTime = Get-Date
        try {
            & $PYTHON_EXE $SCRIPT_PATH $ArgumentList
            $EndTime = Get-Date
            $Duration = $EndTime - $StartTime
            Write-Host "--- $($ModelConfig.Name) - $($Config.Tag) Training Finished Successfully ---" -ForegroundColor Green
            Write-Host "Training Duration: $($Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        }
        catch {
            $EndTime = Get-Date
            $Duration = $EndTime - $StartTime
            Write-Host "--- ERROR: $($ModelConfig.Name) - $($Config.Tag) Training Failed ---" -ForegroundColor Red
            Write-Host "Error: $_" -ForegroundColor Red
            Write-Host "Failed after: $($Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Red
            # Continue with next configuration instead of stopping
        }
        
        $ConfigIndex++
        $OverallRunIndex++
        
        # Small pause between runs to allow system cleanup
        if ($OverallRunIndex -le $TotalRuns) {
            Write-Host "Pausing for system cleanup..." -ForegroundColor Gray
            Start-Sleep -Seconds 15
        }
    }
    
    Write-Host "`n--- Completed all configurations for $($ModelConfig.Name) ---" -ForegroundColor Green
}

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host "Multi-Model Progressive Regularization Study Completed!" -ForegroundColor Green
Write-Host "Results saved in subdirectories under: $BASE_OUTPUT_DIR" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Green

# Generate a comprehensive summary report
$SummaryFile = Join-Path $BASE_OUTPUT_DIR "multi_model_regularization_study_summary.txt"
$SummaryContent = @"
Multi-Model Progressive Regularization Study Summary
===================================================
Date: $(Get-Date)
Data Type: $DATA_TYPE_TO_USE
Total Epochs per Configuration: $NUM_EPOCHS
Weight Decay: $WEIGHT_DECAY
Early Stopping Patience: $PATIENCE
Batch Size: $BATCH_SIZE

Study Design:
This study progressively increases regularization strength across multiple models 
to understand the individual and cumulative effects of different regularization 
techniques on different architectures.

Models Tested:
"@

foreach ($ModelConfig in $ModelConfigs) {
    $SummaryContent += "`n- $($ModelConfig.Name) ($($ModelConfig.ModelName))"
    $SummaryContent += "`n  Learning Rate: $($ModelConfig.LearningRate)"
    if ($ModelConfig.CustomWeights) {
        $SummaryContent += "`n  Custom Weights: $($ModelConfig.CustomWeights)"
        $SummaryContent += "`n  Checkpoint Key: $($ModelConfig.CheckpointKey)"
    } else {
        $SummaryContent += "`n  TIMM Pretrained: $($ModelConfig.UseTimm)"
    }
    $SummaryContent += "`n"
}

$SummaryContent += "`nRegularization Configurations (applied to each model):`n"

$ConfigNum = 1
foreach ($Config in $RegularizationConfigs) {
    $SummaryContent += "`n$ConfigNum. $($Config.Tag)"
    $SummaryContent += "`n   Description: $($Config.Description)"
    $SummaryContent += "`n   Data Augmentation: $($Config.DataAug)"
    $SummaryContent += "`n   Dropout: $($Config.Dropout)"
    $SummaryContent += "`n"
    $ConfigNum++
}

$SummaryContent += @"

Expected Analysis:
- Compare training/validation curves across configurations for each model
- Compare regularization sensitivity between ResNet-50 and VisionFM
- Identify optimal regularization strength per model architecture
- Observe how different architectures respond to regularization
- Determine if pretrained vs custom weights affect regularization needs

Results directories:
$BASE_OUTPUT_DIR

Each experiment subdirectory contains:
- Training history JSON
- Best model checkpoints
- Training logs
- Data manifests

Total training runs completed: $TotalRuns
"@

$SummaryContent | Out-File -FilePath $SummaryFile -Encoding UTF8
Write-Host "`nComprehensive summary saved to: $SummaryFile" -ForegroundColor Cyan

# Also create a detailed results comparison template
$ComparisonFile = Join-Path $BASE_OUTPUT_DIR "results_comparison_template.csv"
$ComparisonHeader = "Model,Architecture,LearningRate,Configuration,Data_Aug,Dropout,Best_Val_Acc,Best_Val_Loss,Final_Train_Acc,Final_Train_Loss,Overfitting_Gap,Training_Time,Notes"
$ComparisonHeader | Out-File -FilePath $ComparisonFile -Encoding UTF8

foreach ($ModelConfig in $ModelConfigs) {
    foreach ($Config in $RegularizationConfigs) {
        $Row = "$($ModelConfig.Name),$($ModelConfig.ModelName),$($ModelConfig.LearningRate),$($Config.Tag),$($Config.DataAug),$($Config.Dropout),TODO,TODO,TODO,TODO,TODO,TODO,TODO"
        $Row | Out-File -FilePath $ComparisonFile -Append -Encoding UTF8
    }
}

Write-Host "Results comparison template created: $ComparisonFile" -ForegroundColor Cyan
Write-Host "Fill in the TODO values after training completes for easy comparison." -ForegroundColor Cyan

# Create a quick analysis script template
$AnalysisScriptFile = Join-Path $BASE_OUTPUT_DIR "analyze_results.py"
$AnalysisScriptContent = @"
#!/usr/bin/env python3
"""
Quick analysis script template for multi-model regularization study results.
Run this after all training is complete to generate comparison plots.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
                    histories[subdir] = history
                    print(f"Loaded: {subdir}")
                except Exception as e:
                    print(f"Error loading {history_path}: {e}")
                break
    
    return histories

def plot_learning_curves(histories, save_dir):
    """Plot learning curves for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for exp_name, history in histories.items():
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Determine model and config from experiment name
            model_type = 'ResNet50' if 'ResNet50' in exp_name else 'VFM'
            color = 'blue' if model_type == 'ResNet50' else 'red'
            
            # Plot training loss
            axes[0, 0].plot(epochs, history['train_loss'], color=color, alpha=0.7, label=exp_name)
            # Plot validation loss  
            axes[0, 1].plot(epochs, history['val_loss'], color=color, alpha=0.7, label=exp_name)
            
            if 'train_acc' in history and 'val_acc' in history:
                # Plot training accuracy
                axes[1, 0].plot(epochs, history['train_acc'], color=color, alpha=0.7, label=exp_name)
                # Plot validation accuracy
                axes[1, 1].plot(epochs, history['val_acc'], color=color, alpha=0.7, label=exp_name)
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Analyzing results in: {base_dir}")
    
    histories = load_all_histories(base_dir)
    print(f"Loaded {len(histories)} experiment histories")
    
    if histories:
        plot_learning_curves(histories, base_dir)
        print("Analysis plots saved!")
    else:
        print("No histories found. Make sure training is complete.")
"@

$AnalysisScriptContent | Out-File -FilePath $AnalysisScriptFile -Encoding UTF8
Write-Host "Analysis script template created: $AnalysisScriptFile" -ForegroundColor Cyan
Write-Host "Run this Python script after training completes to generate comparison plots." -ForegroundColor Cyan

Write-Host "`n=== STUDY COMPLETE ===" -ForegroundColor Green
Write-Host "Total experiments run: $TotalRuns" -ForegroundColor Green
Write-Host "Check the results comparison CSV and run the analysis script for insights!" -ForegroundColor Green