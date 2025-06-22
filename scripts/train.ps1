# Set error preference to stop on errors
$ErrorActionPreference = "Stop"

Write-Host "======================================================" -ForegroundColor Green
Write-Host "Starting Multi-Architecture Training Script (PowerShell)" -ForegroundColor Green
Write-Host "======================================================"

# --- Common Configuration ---
$PYTHON_EXE = "python" 
$SCRIPT_PATH = ".\train_classification.py" # Assuming train_classification.py is the updated Python script

$DATA_TYPE_TO_USE = "processed" # Should be "raw" or "processed"
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
$BASE_OUTPUT_DIR = "experiments\multi_arch_comparison_dtype_${DATA_TYPE_TO_USE}_${Timestamp}"

# Training Hyperparameters
$NUM_EPOCHS = 60 
$PATIENCE = 10 
$BATCH_SIZE = 16
$GRAD_ACCUM_STEPS = 4 
$EVAL_BATCH_SIZE = 16
$LEARNING_RATE = 1e-05 
$RESNET_LEARNING_RATE = 3e-04 # Higher LR for ResNet50
$WEIGHT_DECAY = 0.05
$SEED = 42
$NUM_WORKERS = 4 
$ENABLE_CHAKSU = $true
$ENABLE_AMP = $true
$ENABLE_AIROGS = $true
$AIROGS_NUM_RG = 3000
$AIROGS_NUM_NRG = 3000

# Sample Saving Configuration
$ENABLE_SAMPLE_SAVING = $false 
$NUM_SAMPLES_TO_SAVE_PER_SOURCE = 5 

# Visualization Configuration (NEW)
$ENABLE_GRADCAM_VISUALIZATIONS = $false # Or $false to disable by default
$NUM_GRADCAM_SAMPLES_PER_CATEGORY = 3  # Default number of samples

New-Item -ItemType Directory -Path $BASE_OUTPUT_DIR -Force | Out-Null
Write-Host "Base output directory: $BASE_OUTPUT_DIR" -ForegroundColor Cyan
Write-Host "Training for $($NUM_EPOCHS) epochs with seed $($SEED)." -ForegroundColor Cyan
if ($ENABLE_SAMPLE_SAVING) { Write-Host "Sample saving: YES, $($NUM_SAMPLES_TO_SAVE_PER_SOURCE) per source." -ForegroundColor Cyan }
if ($ENABLE_AIROGS) { Write-Host "AIROGS data: YES, RG Samples: $($AIROGS_NUM_RG), NRG Samples: $($AIROGS_NUM_NRG)" -ForegroundColor Cyan}
# NEW informational message for Grad-CAM
if ($ENABLE_GRADCAM_VISUALIZATIONS) { Write-Host "Grad-CAM Visualizations: YES, $($NUM_GRADCAM_SAMPLES_PER_CATEGORY) samples per category." -ForegroundColor Cyan }
else { Write-Host "Grad-CAM Visualizations: NO" -ForegroundColor Cyan }


function Build-ArgumentList {
    param(
        [hashtable]$ArgsHashTable, 
        [string]$DataType,
        [string]$BaseDataRoot,
        [bool]$UseChaksu,
        [bool]$UseAmp,
        [bool]$UseAirogs
    )
    $ArgumentList = New-Object System.Collections.Generic.List[string]
    $ArgumentList.Add("--data_type"); $ArgumentList.Add($DataType)
    $ArgumentList.Add("--base_data_root"); $ArgumentList.Add($BaseDataRoot)

    foreach ($key in $ArgsHashTable.Keys) {
        $value = $ArgsHashTable[$key]
        if ($value -is [bool]) {
            if ($value -eq $true) { $ArgumentList.Add("--$key") }
            # For boolean flags that need --no_ prefix when false, handle them explicitly below
            # or modify your Python script to accept --key false
        } elseif ($value -ne $null -and $value -ne "") {
            $ArgumentList.Add("--$key"); $ArgumentList.Add($value)
        }
    }

    # Explicit handling for flags that might need a --no_ prefix if false
    # (Your Python script's argparse already handles this for some flags if dest is set)
    if ($UseChaksu) { $ArgumentList.Add("--use_chaksu") } else { $ArgumentList.Add("--no_chaksu")}
    if ($UseAirogs) { $ArgumentList.Add("--use_airogs") } else { $ArgumentList.Add("--no_airogs")}
    if ($UseAmp) { $ArgumentList.Add("--use_amp") } else { $ArgumentList.Add("--no_amp")}
    
    return $ArgumentList
}

# --- Common Arguments for most training runs ---
$CommonTrainArgs = @{
    base_output_dir = $BASE_OUTPUT_DIR
    smdg_metadata_file = $SMDG_META; smdg_image_dir = $SMDG_IMG_DIR
    chaksu_base_dir = $CHAKSU_BASE; chaksu_decision_dir = $CHAKSU_DECISION; chaksu_metadata_dir = $CHAKSU_META_DIR
    airogs_label_file = $AIROGS_LABEL_FILE; airogs_image_dir = $AIROGS_IMAGE_DIR 
    airogs_num_rg_samples = $AIROGS_NUM_RG; airogs_num_nrg_samples = $AIROGS_NUM_NRG
    num_epochs = $NUM_EPOCHS; batch_size = $BATCH_SIZE; eval_batch_size = $EVAL_BATCH_SIZE
    learning_rate = $LEARNING_RATE; weight_decay = $WEIGHT_DECAY; early_stopping_patience = $PATIENCE
    seed = $SEED; num_workers = $NUM_WORKERS; grad_accum_steps = $GRAD_ACCUM_STEPS
    save_data_samples = $ENABLE_SAMPLE_SAVING            
    num_data_samples_per_source = $NUM_SAMPLES_TO_SAVE_PER_SOURCE 
    use_airogs_cache = $true
    # --- NEW Grad-CAM Args ---
    run_gradcam_visualizations = $ENABLE_GRADCAM_VISUALIZATIONS # This will add --run_gradcam_visualizations if $true
    num_gradcam_samples = $NUM_GRADCAM_SAMPLES_PER_CATEGORY
}

## --- Run 4: VisionFM (Custom Weights) ---
#Write-Host "`n----------------------------------------" -ForegroundColor Yellow
#Write-Host "Training VisionFM (Custom Weights)..." -ForegroundColor Yellow
#$VFM_WEIGHTS = "D:\glaucoma\models\VFM_Fundus_weights.pth"
#$VFM_KEY = "teacher"
#$VFMSpecificArgs = @{
#    model_name = "vit_base_patch16_224"; custom_weights_path = $VFM_WEIGHTS; checkpoint_key = $VFM_KEY
#    dropout_prob = 0.3; experiment_tag = "VFM_custom"
#}
#$VFMArgs = $CommonTrainArgs.Clone()
#$VFMArgs.GetEnumerator() | ForEach-Object { $VFMSpecificArgs[$_.Name] = $_.Value }
#$VFMArgumentList = Build-ArgumentList -ArgsHashTable $VFMSpecificArgs -DataType $DATA_TYPE_TO_USE -BaseDataRoot $BASE_DATA_ROOT -UseChaksu $ENABLE_CHAKSU -UseAmp $ENABLE_AMP -UseAirogs $ENABLE_AIROGS
#Write-Host "Command: $PYTHON_EXE $SCRIPT_PATH $($VFMArgumentList -join ' ')"
#& $PYTHON_EXE $SCRIPT_PATH $VFMArgumentList
#Write-Host "--- VisionFM Training Finished ---" -ForegroundColor Green

## --- Run 3: ConvNeXt-Base (Timm Pretrained) ---
#Write-Host "`n----------------------------------------" -ForegroundColor Yellow
#Write-Host "Training ConvNeXt-Base (Timm Pretrained)..." -ForegroundColor Yellow
#$ConvNeXtSpecificArgs = @{
#    model_name = "convnext_base" 
#    use_timm_pretrained_if_no_custom = $true 
#    dropout_prob = 0.3 
#    experiment_tag = "ConvNeXtBase_timm"
#}
#$ConvNeXtArgs = $CommonTrainArgs.Clone()
#$ConvNeXtArgs.GetEnumerator() | ForEach-Object { $ConvNeXtSpecificArgs[$_.Name] = $_.Value }
#$ConvNeXtArgumentList = Build-ArgumentList -ArgsHashTable $ConvNeXtSpecificArgs -DataType $DATA_TYPE_TO_USE -BaseDataRoot $BASE_DATA_ROOT -UseChaksu $ENABLE_CHAKSU -UseAmp $ENABLE_AMP -UseAirogs $ENABLE_AIROGS
#Write-Host "Command: $PYTHON_EXE $SCRIPT_PATH $($ConvNeXtArgumentList -join ' ')"
#& $PYTHON_EXE $SCRIPT_PATH $ConvNeXtArgumentList
#Write-Host "--- ConvNeXt-Base Training Finished ---" -ForegroundColor Green
## --- Run 1: ResNet50 (Timm Pretrained) - Uses HIGHER LR 0.0003 ---
#Write-Host "`n----------------------------------------" -ForegroundColor Yellow
#Write-Host "Training ResNet50 (Timm Pretrained) with LR=$RESNET_LEARNING_RATE..." -ForegroundColor Yellow
#$ResNetSpecificArgs = @{
#    model_name = "resnet50"
#    use_timm_pretrained_if_no_custom = $true 
#    dropout_prob = 0.3
#    experiment_tag = "ResNet50_timm"
#    learning_rate = $RESNET_LEARNING_RATE  
#}
#$ResNetArgs = $CommonTrainArgs.Clone()
#$ResNetArgs.GetEnumerator() | ForEach-Object { $ResNetSpecificArgs[$_.Name] = $_.Value }
#$ResNetArgumentList = Build-ArgumentList -ArgsHashTable $ResNetSpecificArgs -DataType $DATA_TYPE_TO_USE -BaseDataRoot $BASE_DATA_ROOT -UseChaksu $ENABLE_CHAKSU -UseAmp $ENABLE_AMP -UseAirogs $ENABLE_AIROGS
#Write-Host "Command: $PYTHON_EXE $SCRIPT_PATH $($ResNetArgumentList -join ' ')"
#& $PYTHON_EXE $SCRIPT_PATH $ResNetArgumentList
#Write-Host "--- ResNet50 Training Finished ---" -ForegroundColor Green
#
## --- Run 2: ViT-Base (Timm Pretrained) ---
# Write-Host "`n----------------------------------------" -ForegroundColor Yellow
# Write-Host "Training ViT-Base (Timm Pretrained)..." -ForegroundColor Yellow
# $ViTSpecificArgs = @{
#     model_name = "vit_base_patch16_224"
#     use_timm_pretrained_if_no_custom = $true
#     dropout_prob = 0.3
#     experiment_tag = "ViTBase_timm"
# }
# $ViTArgs = $CommonTrainArgs.Clone()
# $ViTArgs.GetEnumerator() | ForEach-Object { $ViTSpecificArgs[$_.Name] = $_.Value }
# $ViTArgumentList = Build-ArgumentList -ArgsHashTable $ViTSpecificArgs -DataType $DATA_TYPE_TO_USE -BaseDataRoot $BASE_DATA_ROOT -UseChaksu $ENABLE_CHAKSU -UseAmp $ENABLE_AMP -UseAirogs $ENABLE_AIROGS
# Write-Host "Command: $PYTHON_EXE $SCRIPT_PATH $($ViTArgumentList -join ' ')"
# & $PYTHON_EXE $SCRIPT_PATH $ViTArgumentList
# Write-Host "--- ViT-Base Training Finished ---" -ForegroundColor Green
#
# # --- Run 5: DINOv2 ViT-B/14 (Custom Weights) ---
#Write-Host "`n----------------------------------------" -ForegroundColor Yellow
#Write-Host "Training DINOv2 ViT-B/14 (Custom Weights)..." -ForegroundColor Yellow
#$DINOV2_WEIGHTS = "D:\glaucoma\models\dinov2_vitb14_reg4_pretrain.pth"
#$DINOV2_KEY = "teacher" 
#$DINOV2SpecificArgs = @{
#    model_name = "dinov2_vitb14"; custom_weights_path = $DINOV2_WEIGHTS; checkpoint_key = $DINOV2_KEY 
#    dropout_prob = 0.3; experiment_tag = "DINOv2"
#}
#$DINOV2Args = $CommonTrainArgs.Clone()
#$DINOV2Args.GetEnumerator() | ForEach-Object { $DINOV2SpecificArgs[$_.Name] = $_.Value }
#$DINOV2ArgumentList = Build-ArgumentList -ArgsHashTable $DINOV2SpecificArgs -DataType $DATA_TYPE_TO_USE -BaseDataRoot $BASE_DATA_ROOT -UseChaksu $ENABLE_CHAKSU -UseAmp $ENABLE_AMP -UseAirogs $ENABLE_AIROGS
#Write-Host "Command: $PYTHON_EXE $SCRIPT_PATH $($DINOV2ArgumentList -join ' ')"
#& $PYTHON_EXE $SCRIPT_PATH $DINOV2ArgumentList
#Write-Host "--- DINOv2 ViT-B/14 Training Finished ---" -ForegroundColor Green



# --- Run 6: Regression-to-Classification Fine-tuning ---
Write-Host "`n----------------------------------------" -ForegroundColor Yellow
Write-Host "Training ViT-Base (Regression-to-Classification)..." -ForegroundColor Yellow
$REGRESSION_WEIGHTS = "D:\glaucoma\experiments\VCDR_Regression\VFM_ViTBase_CustomWeights_Regression_vit_base_patch16_224_seed42\vit_base_patch16_224_best_regressor.pth"  # UPDATE THIS PATH
$REGRESSION_KEY = "model_state_dict"  # Or whatever key your regression model uses
$RegressionSpecificArgs = @{
    model_name = "vit_base_patch16_224"
    custom_weights_path = $REGRESSION_WEIGHTS
    checkpoint_key = $REGRESSION_KEY
    is_regression_to_classification = $true    # This is the key flag
    freeze_backbone_epochs = 5                 # Freeze backbone for first 5 epochs
    backbone_lr_multiplier = 0.1              # Use 10x lower LR for backbone
    dropout_prob = 0.3
    experiment_tag = "Reg2Class"
    num_epochs = 60                           # Might need fewer epochs since starting from regression
}
$RegressionArgs = $CommonTrainArgs.Clone()
$RegressionArgs.GetEnumerator() | ForEach-Object { $RegressionSpecificArgs[$_.Name] = $_.Value }
$RegressionArgumentList = Build-ArgumentList -ArgsHashTable $RegressionSpecificArgs -DataType $DATA_TYPE_TO_USE -BaseDataRoot $BASE_DATA_ROOT -UseChaksu $ENABLE_CHAKSU -UseAmp $ENABLE_AMP -UseAirogs $ENABLE_AIROGS
Write-Host "Command: $PYTHON_EXE $SCRIPT_PATH $($RegressionArgumentList -join ' ')"
& $PYTHON_EXE $SCRIPT_PATH $RegressionArgumentList
Write-Host "--- Regression-to-Classification Training Finished ---" -ForegroundColor Green


Write-Host "`n======================================================" -ForegroundColor Green
Write-Host "All training runs completed for data_type '$DATA_TYPE_TO_USE'." -ForegroundColor Green
Write-Host "Results saved in subdirectories under: $BASE_OUTPUT_DIR" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Green