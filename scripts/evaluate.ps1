# evaluate.ps1

param(
    [Parameter(Mandatory=$true, HelpMessage="Path to the PARENT directory of completed training experiments, e.g., experiments\multi_arch_comparison_dtype_raw_20250517_1149. This directory should contain subfolders for each model run.")]
    [string]$ParentOfTrainedExperimentDirs,

    [Parameter(Mandatory=$false, HelpMessage="Data type ('raw' or 'processed') of images to use for THIS evaluation run. This affects how image paths for external test sets are resolved.")]
    [ValidateSet('raw', 'processed')]
    [string]$EvaluationDataType = 'raw' 
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================" -ForegroundColor Green
Write-Host "Starting External Model Evaluation Script (PowerShell)" -ForegroundColor Green
Write-Host "======================================================"
if (-not (Test-Path $ParentOfTrainedExperimentDirs -PathType Container)) { Write-Error "Parent directory of trained experiments does not exist: $ParentOfTrainedExperimentDirs"; exit 1 }
Write-Host "Evaluating models from subdirectories within: $ParentOfTrainedExperimentDirs" -ForegroundColor Cyan
Write-Host "Evaluation will use data_type: $EvaluationDataType for resolving test set image paths." -ForegroundColor Cyan

$PYTHON_EXE = "python" # Ensure python is in your PATH or provide full path
$EVAL_SCRIPT_PATH = ".\evaluate_ood.py" # Assuming it's in the same directory

# --- Paths to RAW Data Manifests and Base Directories ---
# These are passed to the Python script, which then handles adjustments based on its own --data_type arg.
$BASE_DATA_ROOT_FOR_PYTHON = "D:\glaucoma\data"
$SMDG_META_CSV_RAW = "D:\glaucoma\data\raw\SMDG-19\metadata - standardized.csv"
$SMDG_IMG_DIR_RAW = "D:\glaucoma\data\raw\SMDG-19\full-fundus\full-fundus"
$CHAKSU_BASE_DIR_RAW = "D:\glaucoma\data\raw\Chaksu\Train\Train\1.0_Original_Fundus_Images" # Python will adjust this path if its --data_type is 'processed'
$CHAKSU_DECISION_DIR_RAW = "D:\glaucoma\data\raw\Chaksu\Train\Train\6.0_Glaucoma_Decision"
$CHAKSU_META_DIR_RAW = "D:\glaucoma\data\raw\Chaksu\Train\Train\6.0_Glaucoma_Decision\Majority"

# --- Common Evaluation Parameters ---
$EVAL_BATCH_SIZE_COMMON = 64
$N_BOOTSTRAPS_COMMON = 0 # Set to 0 for faster initial testing, increase to 1000+ for CIs
$ALPHA_COMMON = 0.05
$N_CALIBRATION_BINS_COMMON = 15
$MIN_SAMPLES_FOR_CALIBRATION_COMMON = 50
$SEED_FOR_EVAL_COMMON = 42
$NUM_WORKERS_COMMON = 0 # Set to a higher value (e.g., 4) if I/O is a bottleneck and CPU cores are available

# --- Flags for Controlling Evaluation ---
$EVAL_PAPILLA_FLAG_COMMON = $true
$EVAL_OIAODIR_TEST_FLAG_COMMON = $false
$EVAL_CHAKSU_FLAG_COMMON = $false
$RUN_BIAS_ANALYSIS_FLAG_COMMON = $false # Set to $false for faster initial testing if not needed immediately
$PLOT_ROC_PER_SOURCE_EVAL_FLAG_COMMON = $true
$USE_AMP_FOR_BIAS_EXTRACTION_FLAG_COMMON = $true

# --- Model Defaults (passed to Python script) ---
$MODEL_NUM_CLASSES_COMMON = 2
$MODEL_IMAGE_SIZE_COMMON = 224

# --- Build-EvaluationArgumentList function ---
function Build-EvaluationArgumentList {
    param(
        [hashtable]$EvalSpecificArgs,
        [string]$EvalDataTypeToUse,
        [string]$BaseDataRootForPythonScript,
        [string]$SmdgMetaFileRaw, [string]$SmdgImageDirRaw,
        [string]$ChaksuBaseDirRaw, [string]$ChaksuDecisionDirRaw, [string]$ChaksuMetadataDirRaw,
        [int]$EvalBatchSize, [int]$NBootstraps, [double]$Alpha,
        [int]$NCalibrationBins, [int]$MinSamplesCalib, [int]$Seed, [int]$NumWorkers,
        [int]$NumClasses, [int]$ImageSize,
        [bool]$FlagEvalPapilla, [bool]$FlagEvalOiaodirTest, [bool]$FlagEvalChaksu,
        [bool]$FlagRunBiasAnalysis, [bool]$FlagPlotRocPerSource, [bool]$FlagUseAmpForBias
    )
    $ArgumentList = New-Object System.Collections.Generic.List[string]
    # Add specific args first
    foreach ($key in $EvalSpecificArgs.Keys) { $ArgumentList.Add("--$key"); $ArgumentList.Add($EvalSpecificArgs[$key]) }

    # Add common args
    $ArgumentList.Add("--data_type"); $ArgumentList.Add($EvalDataTypeToUse)
    $ArgumentList.Add("--base_data_root"); $ArgumentList.Add($BaseDataRootForPythonScript)
    $ArgumentList.Add("--smdg_metadata_file_raw"); $ArgumentList.Add($SmdgMetaFileRaw)
    $ArgumentList.Add("--smdg_image_dir_raw"); $ArgumentList.Add($SmdgImageDirRaw)
    $ArgumentList.Add("--chaksu_base_dir"); $ArgumentList.Add($ChaksuBaseDirRaw) # Python's adjust_path will handle this based on its --data_type
    $ArgumentList.Add("--chaksu_decision_dir_raw"); $ArgumentList.Add($ChaksuDecisionDirRaw)
    $ArgumentList.Add("--chaksu_metadata_dir_raw"); $ArgumentList.Add($ChaksuMetadataDirRaw)
    $ArgumentList.Add("--eval_batch_size"); $ArgumentList.Add($EvalBatchSize)
    $ArgumentList.Add("--n_bootstraps"); $ArgumentList.Add($NBootstraps); $ArgumentList.Add("--alpha"); $ArgumentList.Add($Alpha)
    $ArgumentList.Add("--n_calibration_bins"); $ArgumentList.Add($NCalibrationBins); $ArgumentList.Add("--min_samples_for_calibration"); $ArgumentList.Add($MinSamplesCalib)
    $ArgumentList.Add("--seed"); $ArgumentList.Add($Seed); $ArgumentList.Add("--num_workers"); $ArgumentList.Add($NumWorkers)
    $ArgumentList.Add("--num_classes"); $ArgumentList.Add($NumClasses); $ArgumentList.Add("--image_size"); $ArgumentList.Add($ImageSize)

    # Boolean flags
    if ($FlagEvalPapilla) { $ArgumentList.Add("--eval_papilla") } else { $ArgumentList.Add("--no_eval_papilla") }
    if ($FlagEvalOiaodirTest) { $ArgumentList.Add("--eval_oiaodir_test") } else { $ArgumentList.Add("--no_eval_oiaodir_test") }
    if ($FlagEvalChaksu) { $ArgumentList.Add("--eval_chaksu") } else { $ArgumentList.Add("--no_eval_chaksu") }
    return $ArgumentList
}

# Function to find the specific experiment subdirectory for a model
function Find-ModelExperimentSubDir {
    param(
        [string]$ParentDir,
        [string]$ModelRunDirPattern
    )
    # Find directories matching the pattern, sort by creation time descending, take the first (most recent)
    $foundDir = Get-ChildItem -Path $ParentDir -Directory -Filter $ModelRunDirPattern |
                Sort-Object CreationTime -Descending |
                Select-Object -First 1
    return $foundDir
}

# --- Define Models to Evaluate ---
# ModelNameInTraining: Exact name passed to --model_name in Python training script.
# TrainingModelShortName: The short name used in the directory structure (e.g., "resnet50", "vit")
#                         Must match Python's model_short_name = args.model_name.replace(...).
# TrainingTag: The --experiment_tag used during training.
# TrainingDropout: The --dropout_prob used during training.
$ModelsToEvaluate = @(
    @{ ModelNameInTraining="dinov2_vitb14"; TrainingModelShortName="d2_vitb14"; TrainingTag="DINOv2"; TrainingDropout=0.3 }
    @{ ModelNameInTraining="vit_base_patch16_224"; TrainingModelShortName="vit"; TrainingTag="VFM_custom"; TrainingDropout=0.3 }
    @{ ModelNameInTraining="convnext_base"; TrainingModelShortName="convnext"; TrainingTag="ConvNeXtBase_timm"; TrainingDropout=0.3 }
    @{ ModelNameInTraining="resnet50"; TrainingModelShortName="resnet50"; TrainingTag="ResNet50_timm"; TrainingDropout=0.3 }
    @{ ModelNameInTraining="vit_base_patch16_224"; TrainingModelShortName="vit"; TrainingTag="ViTBase_timm"; TrainingDropout=0.3 }
)

# Determine the data_type used for the *training runs* based on the $ParentOfTrainedExperimentDirs name
$trainingRunParentDataType = "raw" # Default assumption
if ($ParentOfTrainedExperimentDirs -match "dtype_processed") {
    $trainingRunParentDataType = "processed"
}
Write-Host "Parent experiment directory implies training runs used data_type: $trainingRunParentDataType" -ForegroundColor Magenta


# --- Main Evaluation Loop ---
foreach ($modelInfo in $ModelsToEvaluate) {
    $modelNameForPythonArg = $modelInfo.ModelNameInTraining
    $dropoutForPythonArg = $modelInfo.TrainingDropout
    $trainingModelShortName = $modelInfo.TrainingModelShortName
    $trainingTag = $modelInfo.TrainingTag

    # Construct the search pattern for this specific model's training run directory
    # Format: <model_short_name>_<training_data_type_used_in_training>_<training_tag>_<timestamp_wildcard>
    $modelDirSearchPattern = "$($trainingModelShortName)_$($trainingRunParentDataType)_$($trainingTag)*"

    Write-Host ""; Write-Host "-------------------------------------------------" -ForegroundColor Yellow
    Write-Host "Attempting to Evaluate Model: $($modelNameForPythonArg) (Tag: $($trainingTag))" -ForegroundColor Yellow
    Write-Host "Searching for training run directory with pattern: '$($modelDirSearchPattern)' under '$($ParentOfTrainedExperimentDirs)'" -ForegroundColor Yellow

    $modelActualExperimentDir = Find-ModelExperimentSubDir -ParentDir $ParentOfTrainedExperimentDirs -ModelRunDirPattern $modelDirSearchPattern

    if ($modelActualExperimentDir) {
        Write-Host "Found training run directory: $($modelActualExperimentDir.FullName)" -ForegroundColor Cyan
        $checkpointDir = Join-Path -Path $modelActualExperimentDir.FullName -ChildPath "checkpoints"

        if (-not (Test-Path $checkpointDir -PathType Container)) {
            Write-Warning "Checkpoints directory not found: $checkpointDir for model $($modelNameForPythonArg). Skipping."
            continue # Skip to the next model
        }

        # --- CHECKPOINT SELECTION LOGIC (Prioritizing "best_model") ---
        $checkpointFile = $null

        # 1. Prioritize the "best_model" checkpoint from the highest epoch number
        $bestCheckpointFile = Get-ChildItem -Path $checkpointDir -File -Filter "*_best_model_epoch*.pth" |
            Sort-Object @{Expression = {$_.Name -replace '.*_epoch(\d+)\.pth$', '$1' -as [int]}} -Descending |
            Select-Object -First 1

        if ($bestCheckpointFile) {
            Write-Host "Selected 'best_model' checkpoint: $($bestCheckpointFile.FullName)" -ForegroundColor Magenta
            $checkpointFile = $bestCheckpointFile
        } else {
            Write-Warning "No '*_best_model_epoch*.pth' checkpoint found for model $($modelNameForPythonArg)."
            # 2. Fallback to "final_epoch" if no "best_model" found
            $finalEpochCheckpointFile = Get-ChildItem -Path $checkpointDir -File -Filter "*_final_epoch*.pth" |
                Sort-Object @{Expression = {$_.Name -replace '.*_epoch(\d+)\.pth$', '$1' -as [int]}} -Descending |
                Select-Object -First 1

            if ($finalEpochCheckpointFile) {
                Write-Host "Falling back to 'final_epoch' checkpoint: $($finalEpochCheckpointFile.FullName)" -ForegroundColor Yellow
                $checkpointFile = $finalEpochCheckpointFile
            } else {
                Write-Warning "No '*_final_epoch*.pth' checkpoint found for model $($modelNameForPythonArg)."
                # 3. Further fallback to the absolute latest modified .pth file in the directory
                $latestModifiedCheckpointFile = Get-ChildItem -Path $checkpointDir -File -Filter "*.pth" |
                    Sort-Object LastWriteTime -Descending |
                    Select-Object -First 1
                
                if ($latestModifiedCheckpointFile) {
                    Write-Host "Falling back to latest modified checkpoint (could be 'last_checkpoint.pth', etc.): $($latestModifiedCheckpointFile.FullName)" -ForegroundColor Yellow
                    $checkpointFile = $latestModifiedCheckpointFile
                }
            }
        }
        # --- END OF CHECKPOINT SELECTION LOGIC ---

        if ($checkpointFile) {
            Write-Host "Using checkpoint for evaluation: $($checkpointFile.FullName)" -ForegroundColor Cyan

            $currentModelEvalSpecificArgs = @{
                checkpoint_path = $checkpointFile.FullName
                model_name_from_training = $modelNameForPythonArg
                dropout_prob_from_training = $dropoutForPythonArg
            }

            $evalArgumentList = Build-EvaluationArgumentList `
                -EvalSpecificArgs $currentModelEvalSpecificArgs `
                -EvalDataTypeToUse $EvaluationDataType `
                -BaseDataRootForPythonScript $BASE_DATA_ROOT_FOR_PYTHON `
                -SmdgMetaFileRaw $SMDG_META_CSV_RAW `
                -SmdgImageDirRaw $SMDG_IMG_DIR_RAW `
                -ChaksuBaseDirRaw $CHAKSU_BASE_DIR_RAW `
                -ChaksuDecisionDirRaw $CHAKSU_DECISION_DIR_RAW `
                -ChaksuMetadataDirRaw $CHAKSU_META_DIR_RAW `
                -EvalBatchSize $EVAL_BATCH_SIZE_COMMON `
                -NBootstraps $N_BOOTSTRAPS_COMMON `
                -Alpha $ALPHA_COMMON `
                -NCalibrationBins $N_CALIBRATION_BINS_COMMON `
                -MinSamplesCalib $MIN_SAMPLES_FOR_CALIBRATION_COMMON `
                -Seed $SEED_FOR_EVAL_COMMON `
                -NumWorkers $NUM_WORKERS_COMMON `
                -NumClasses $MODEL_NUM_CLASSES_COMMON `
                -ImageSize $MODEL_IMAGE_SIZE_COMMON `
                -FlagEvalPapilla $EVAL_PAPILLA_FLAG_COMMON `
                -FlagEvalOiaodirTest $EVAL_OIAODIR_TEST_FLAG_COMMON `
                -FlagEvalChaksu $EVAL_CHAKSU_FLAG_COMMON `
                -FlagRunBiasAnalysis $RUN_BIAS_ANALYSIS_FLAG_COMMON `
                -FlagPlotRocPerSource $PLOT_ROC_PER_SOURCE_EVAL_FLAG_COMMON `
                -FlagUseAmpForBias $USE_AMP_FOR_BIAS_EXTRACTION_FLAG_COMMON

            Write-Host "Executing Evaluation for $($modelNameForPythonArg):"
            Write-Host "$PYTHON_EXE $EVAL_SCRIPT_PATH $($evalArgumentList -join ' ')"
            try {
                & $PYTHON_EXE $EVAL_SCRIPT_PATH $evalArgumentList
                Write-Host "--- $($modelNameForPythonArg) (Tag: $($trainingTag)) External Evaluation Finished ---" -ForegroundColor Green
            } catch {
                Write-Error "Error during Python script execution for $($modelNameForPythonArg): $($_.Exception.Message)"
                Write-Warning "Continuing to the next model if any."
            }
        } else { Write-Warning "No suitable checkpoint file ultimately found for $($modelNameForPythonArg) in '$($checkpointDir)'. Skipping." }
    } else { Write-Warning "Experiment subdirectory not found for pattern '$($modelDirSearchPattern)' in '$ParentOfTrainedExperimentDirs'. Skipping model $($modelNameForPythonArg)." }
}

Write-Host ""; Write-Host "======================================================" -ForegroundColor Green
Write-Host "All attempted external evaluations completed." -ForegroundColor Green
Write-Host "Check logs and output directories for each model within their respective training run folders, under 'external_evaluations'." -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Green