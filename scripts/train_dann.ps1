# Set error preference to stop on errors
$ErrorActionPreference = "Stop"

Write-Host "======================================================" -ForegroundColor Green
Write-Host "Starting DANN Multi-Architecture Training Script (PowerShell)" -ForegroundColor Green
Write-Host "======================================================"

# --- Common Configuration ---
$PYTHON_EXE = "python" 
$SCRIPT_PATH = ".\train_classification_dann.py" # MODIFIED: Point to the DANN script
# Ensure 'train_classification_original.py' is in the same directory or accessible
# if train_classification_dann.py imports from it.

$DATA_TYPE_TO_USE = "raw" # Or "processed"
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
$BASE_OUTPUT_DIR = "experiments\multi_arch_dann_comparison_dtype_${DATA_TYPE_TO_USE}_${Timestamp}" # MODIFIED: dann in name

# Training Hyperparameters
$NUM_EPOCHS = 60 
$PATIENCE = 5 # Set to 0 to disable early stopping for DANN, or a higher value
$BATCH_SIZE = 16 # DANN might require smaller batch sizes depending on memory
$GRAD_ACCUM_STEPS = 4 
$EVAL_BATCH_SIZE = 16
$LEARNING_RATE = 1e-05 
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

# --- DANN Specific Configuration ---
$ENABLE_DANN = $true # Set to $true to enable DANN training for the runs below
$DANN_GRL_BASE_LAMBDA = 1.0
$DANN_LAMBDA_SCHEDULE_MODE = "paper" # "paper" or "fixed"
$DANN_GAMMA = 10.0
$DANN_TRADEOFF_LAMBDA = 0.1 # Alpha for domain loss, start small (e.g., 0.1 - 1.0)

New-Item -ItemType Directory -Path $BASE_OUTPUT_DIR -Force | Out-Null
Write-Host "Base output directory: $BASE_OUTPUT_DIR" -ForegroundColor Cyan
Write-Host "Training for $($NUM_EPOCHS) epochs with seed $($SEED)." -ForegroundColor Cyan
if ($ENABLE_SAMPLE_SAVING) { Write-Host "Sample saving: YES, $($NUM_SAMPLES_TO_SAVE_PER_SOURCE) per source." -ForegroundColor Cyan }
if ($ENABLE_AIROGS) { Write-Host "AIROGS data: YES, RG Samples: $($AIROGS_NUM_RG), NRG Samples: $($AIROGS_NUM_NRG)" -ForegroundColor Cyan}

if ($ENABLE_DANN) { 
    Write-Host "DANN Training: YES" -ForegroundColor Cyan
    Write-Host "  GRL Base Lambda: $DANN_GRL_BASE_LAMBDA" -ForegroundColor Cyan
    Write-Host "  GRL Schedule Mode: $DANN_LAMBDA_SCHEDULE_MODE" -ForegroundColor Cyan
    Write-Host "  GRL Gamma: $DANN_GAMMA" -ForegroundColor Cyan
    Write-Host "  Domain Loss Tradeoff (Alpha): $DANN_TRADEOFF_LAMBDA" -ForegroundColor Cyan
} else {
    Write-Host "DANN Training: NO" -ForegroundColor Cyan
}


function Build-ArgumentList {
    param(
        [hashtable]$ArgsHashTable, 
        [string]$DataType,
        [string]$BaseDataRoot,
        [bool]$UseChaksu,
        [bool]$UseAmp,
        [bool]$UseAirogs,
        # --- DANN Args for function ---
        [bool]$UseDann,
        [double]$DannGrlBaseLambda,
        [string]$DannLambdaScheduleMode,
        [double]$DannGamma,
        [double]$DannTradeoffLambda
    )
    $ArgumentList = New-Object System.Collections.Generic.List[string]
    $ArgumentList.Add("--data_type"); $ArgumentList.Add($DataType)
    $ArgumentList.Add("--base_data_root"); $ArgumentList.Add($BaseDataRoot)

    foreach ($key in $ArgsHashTable.Keys) {
        $value = $ArgsHashTable[$key]
        if ($value -is [bool]) {
            # For boolean flags, Python argparse with action='store_true' only needs the flag if true.
            # If false, the flag is omitted, and it defaults to false in Python.
            if ($value -eq $true) { $ArgumentList.Add("--$key") }
        } elseif ($value -ne $null -and $value -ne "") {
            $ArgumentList.Add("--$key"); $ArgumentList.Add($value.ToString()) # Ensure value is string
        }
    }

    # Explicit handling for flags that might need a --no_ prefix if false,
    # or flags that are true by default in Python and need explicit --no_...
    # Python script's argparse should handle store_true/store_false correctly.
    # The flags like use_chaksu, use_airogs, use_amp have --no_... options in Python script.
    if ($UseChaksu) { $ArgumentList.Add("--use_chaksu") } else { $ArgumentList.Add("--no_chaksu")}
    if ($UseAirogs) { $ArgumentList.Add("--use_airogs") } else { $ArgumentList.Add("--no_airogs")}
    if ($UseAmp) { $ArgumentList.Add("--use_amp") } else { $ArgumentList.Add("--no_amp")}
    
    # --- Add DANN Arguments ---
    if ($UseDann) { 
        $ArgumentList.Add("--use_dann")
        $ArgumentList.Add("--dann_grl_base_lambda"); $ArgumentList.Add($DannGrlBaseLambda.ToString())
        $ArgumentList.Add("--dann_lambda_schedule_mode"); $ArgumentList.Add($DannLambdaScheduleMode)
        $ArgumentList.Add("--dann_gamma"); $ArgumentList.Add($DannGamma.ToString())
        $ArgumentList.Add("--dann_tradeoff_lambda"); $ArgumentList.Add($DannTradeoffLambda.ToString())
    }
    # If --use_dann is not passed, the Python script defaults it to False.
    
    return $ArgumentList
}

# --- Common Arguments for most training runs ---
$CommonTrainArgs = @{
    base_output_dir = $BASE_OUTPUT_DIR # This will be overridden by experiment-specific output dir
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
    # Removed GradCAM args as per simplification
}


# --- Example Run: VisionFM (Custom Weights) with DANN ---
Write-Host "`n----------------------------------------" -ForegroundColor Yellow
Write-Host "Training VisionFM (Custom Weights) with DANN settings..." -ForegroundColor Yellow
$VFM_WEIGHTS = "D:\glaucoma\models\VFM_Fundus_weights.pth" # Example path
$VFM_KEY = "teacher" # Example key

$ExperimentSpecificOutputDir = Join-Path -Path $BASE_OUTPUT_DIR -ChildPath "VFM_Custom_DANN"
New-Item -ItemType Directory -Path $ExperimentSpecificOutputDir -Force | Out-Null

$VFMSpecificArgs = @{
    base_output_dir = $ExperimentSpecificOutputDir # Override base_output_dir
    model_name = "vit_base_patch16_224"; custom_weights_path = $VFM_WEIGHTS; checkpoint_key = $VFM_KEY
    dropout_prob = 0.3; experiment_tag = "VFM_custom_DANN" # MODIFIED tag
}
# Merge common args into specific args. Specific args will overwrite if keys overlap.
$VFMArgsToPass = $CommonTrainArgs.Clone()
$VFMSpecificArgs.GetEnumerator() | ForEach-Object { $VFMArgsToPass[$_.Name] = $_.Value }

$VFMArgumentList = Build-ArgumentList -ArgsHashTable $VFMArgsToPass `
                                      -DataType $DATA_TYPE_TO_USE `
                                      -BaseDataRoot $BASE_DATA_ROOT `
                                      -UseChaksu $ENABLE_CHAKSU `
                                      -UseAmp $ENABLE_AMP `
                                      -UseAirogs $ENABLE_AIROGS `
                                      -UseDann $ENABLE_DANN `
                                      -DannGrlBaseLambda $DANN_GRL_BASE_LAMBDA `
                                      -DannLambdaScheduleMode $DANN_LAMBDA_SCHEDULE_MODE `
                                      -DannGamma $DANN_GAMMA `
                                      -DannTradeoffLambda $DANN_TRADEOFF_LAMBDA

Write-Host "Command: $PYTHON_EXE $SCRIPT_PATH $($VFMArgumentList -join ' ')"
& $PYTHON_EXE $SCRIPT_PATH $VFMArgumentList
Write-Host "--- VisionFM with DANN Training Finished ---" -ForegroundColor Green


# --- Example Run: DINOv2 ViT-B/14 (Custom Weights) with DANN ---
# Write-Host "`n----------------------------------------" -ForegroundColor Yellow
# Write-Host "Training DINOv2 ViT-B/14 (Custom Weights) with DANN settings..." -ForegroundColor Yellow
# $DINOV2_WEIGHTS = "D:\glaucoma\models\dinov2_vitb14_reg4_pretrain.pth"
# $DINOV2_KEY = "teacher" 

# $ExperimentSpecificOutputDir_DINO = Join-Path -Path $BASE_OUTPUT_DIR -ChildPath "DINOv2_Custom_DANN"
# New-Item -ItemType Directory -Path $ExperimentSpecificOutputDir_DINO -Force | Out-Null

# $DINOV2SpecificArgs = @{
#     base_output_dir = $ExperimentSpecificOutputDir_DINO
#     model_name = "dinov2_vitb14"; custom_weights_path = $DINOV2_WEIGHTS; checkpoint_key = $DINOV2_KEY 
#     dropout_prob = 0.3; experiment_tag = "DINOv2_DANN"
# }
# $DINOV2ArgsToPass = $CommonTrainArgs.Clone()
# $DINOV2SpecificArgs.GetEnumerator() | ForEach-Object { $DINOV2ArgsToPass[$_.Name] = $_.Value }

# $DINOV2ArgumentList = Build-ArgumentList -ArgsHashTable $DINOV2ArgsToPass `
#                                       -DataType $DATA_TYPE_TO_USE `
#                                       -BaseDataRoot $BASE_DATA_ROOT `
#                                       -UseChaksu $ENABLE_CHAKSU `
#                                       -UseAmp $ENABLE_AMP `
#                                       -UseAirogs $ENABLE_AIROGS `
#                                       -UseDann $ENABLE_DANN `
#                                       -DannGrlBaseLambda $DANN_GRL_BASE_LAMBDA `
#                                       -DannLambdaScheduleMode $DANN_LAMBDA_SCHEDULE_MODE `
#                                       -DannGamma $DANN_GAMMA `
#                                       -DannTradeoffLambda $DANN_TRADEOFF_LAMBDA
# Write-Host "Command: $PYTHON_EXE $SCRIPT_PATH $($DINOV2ArgumentList -join ' ')"
# & $PYTHON_EXE $SCRIPT_PATH $DINOV2ArgumentList
# Write-Host "--- DINOv2 ViT-B/14 with DANN Training Finished ---" -ForegroundColor Green


# You can add more runs here for other architectures or DANN parameter sweeps
# For example, to run WITHOUT DANN for comparison using the same script:
# Set $ENABLE_DANN_FOR_THIS_RUN = $false and pass it to Build-ArgumentList

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host "All DANN training runs completed for data_type '$DATA_TYPE_TO_USE'." -ForegroundColor Green
Write-Host "Results saved in subdirectories under: $BASE_OUTPUT_DIR" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Green