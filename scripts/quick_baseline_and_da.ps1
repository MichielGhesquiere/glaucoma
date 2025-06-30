# Quick VFM Model Testing Script
# Tests VFM (Vision Foundation Model) baseline and basic DA techniques with minimal fine-tuning

param(
    [string]$BaseDataRoot = "D:\glaucoma\data",
    [string]$OutputDir = "D:\glaucoma\vfm_quick_experiments",
    [int]$NumEpochs = 2,     # Minimal fine-tuning for pre-trained VFM
    [int]$BatchSize = 32,    # Conservative batch size for RTX A500 Laptop GPU
    [int]$NumWorkers = 4,    # Faster data loading
    [string]$Model = "vit_base_patch16_224",  # VFM model architecture
    [bool]$EnableVFM = $true,   # Enable VFM for medical imaging
    [bool]$ExcludeUnknown = $true,  # Exclude ambiguous SMDG samples
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

# Set environment variables for proper Unicode handling
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Color functions for better output
function Write-Info { param($Text) Write-Host "[INFO] $Text" -ForegroundColor Cyan }
function Write-Success { param($Text) Write-Host "[SUCCESS] $Text" -ForegroundColor Green }
function Write-Warning { param($Text) Write-Host "[WARNING] $Text" -ForegroundColor Yellow }
function Write-Error { param($Text) Write-Host "[ERROR] $Text" -ForegroundColor Red }

Write-Host @"

QUICK VFM MODEL TESTING & BASIC DA TECHNIQUES
================================================================
Model: VFM (Vision Foundation Model) - Pre-trained for medical imaging
Architecture: $Model
Epochs: $NumEpochs (minimal fine-tuning for pre-trained model)
Batch Size: $BatchSize (conservative for RTX A500 Laptop GPU)
Domain Adaptation: Testing core DA techniques only
================================================================

"@ -ForegroundColor Magenta

# Python executable (adjust path if needed)
$PythonExe = "python"
$ScriptPath = "multisource_domain_finetuning.py"

# Check if script exists
if (-not (Test-Path $ScriptPath)) {
    Write-Error "Script not found: $ScriptPath"
    Write-Info "Please run this script from the scripts directory"
    exit 1
}

# Create output directory
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Write-Info "Output directory: $OutputDir"

# Common arguments for all experiments
$CommonArgs = @(
    "--base_data_root", $BaseDataRoot
    "--output_dir", $OutputDir
    "--num_epochs", $NumEpochs
    "--batch_size", $BatchSize
    "--num_workers", $NumWorkers
    "--pin_memory"
    "--persistent_workers"
    "--prefetch_factor", "4"
    "--seed", $Seed
    "--early_stopping_patience", "1"  # Aggressive early stopping for fine-tuning
    "--quiet_mode"  # Reduce output for speed
)

if ($EnableVFM -eq $true) {
    # VFM is enabled by default - it will use the VFM weights automatically
    # No need to specify --default_model when using VFM
    Write-Info "Using VFM (Vision Foundation Model) - optimized for medical fundus images"
} else {
    $CommonArgs += "--disable_vfm"
    $CommonArgs += "--default_model", "resnet18"  # Use ResNet-18 as fallback
}

if ($ExcludeUnknown) {
    $CommonArgs += "--exclude_smdg_unknown"
}

# Function to invoke experiment
function Invoke-Experiment {
    param(
        [string]$Name,
        [string[]]$AdditionalArgs = @(),
        [string]$Description = ""
    )
    
    Write-Host "`n" + "="*80 -ForegroundColor Yellow
    Write-Info "EXPERIMENT: $Name"
    if ($Description) {
        Write-Host "Description: $Description" -ForegroundColor Gray
    }
    Write-Host "="*80 -ForegroundColor Yellow
    
    $AllArgs = $CommonArgs + $AdditionalArgs
    
    Write-Info "Command: $PythonExe $ScriptPath $($AllArgs -join ' ')"
    
    $StartTime = Get-Date
    try {
        & $PythonExe $ScriptPath @AllArgs
        $EndTime = Get-Date
        $Duration = ($EndTime - $StartTime).TotalMinutes
        Write-Success "$Name completed in $([math]::Round($Duration, 2)) minutes"
        return $true
    }
    catch {
        $EndTime = Get-Date
        $Duration = ($EndTime - $StartTime).TotalMinutes
        Write-Error "$Name failed after $([math]::Round($Duration, 2)) minutes"
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Check GPU memory and provide recommendations
Write-Info "Checking GPU availability..."
try {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        Write-Success "NVIDIA GPU detected. Running nvidia-smi for memory info..."
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        Write-Info "Monitor GPU usage: nvidia-smi -l 1"
    } else {
        Write-Warning "nvidia-smi not found. GPU monitoring unavailable."
    }
} catch {
    Write-Warning "Could not check GPU status"
}

Write-Host @"

VFM EXPERIMENT PLAN (Core DA Techniques Only)
==============================================
1. VFM BASELINE: Pure VFM fine-tuning (no DA techniques)
2. VFM + DANN: VFM + Domain Adversarial Neural Networks
3. VFM + MIXSTYLE: VFM + MixStyle domain generalization
4. VFM + AUGMENTATION: VFM + Advanced data augmentation

Estimated: ~15-20 min per experiment × 4 experiments × 4 datasets = 4-5 hours max
With VFM pre-training: Likely 3-4 hours total

"@ -ForegroundColor Cyan

$AllExperiments = @()

# Experiment 1: VFM Baseline (Pure fine-tuning)
$AllExperiments += @{
    Name = "01_VFM_BASELINE"
    Args = @()
    Description = "VFM baseline fine-tuning without domain adaptation techniques"
}

# Experiment 2: VFM + Domain Adversarial Training (most important DA technique)
$AllExperiments += @{
    Name = "02_VFM_DANN"
    Args = @("--use_domain_adversarial")
    Description = "VFM + Domain Adversarial Neural Networks (DANN)"
}

# Experiment 3: VFM + MixStyle (popular and effective DA technique)
$AllExperiments += @{
    Name = "03_VFM_MIXSTYLE"
    Args = @("--use_mixstyle")
    Description = "VFM + MixStyle for domain generalization"
}

# Experiment 4: VFM + Advanced Augmentation (data-level DA)
$AllExperiments += @{
    Name = "04_VFM_AUGMENTATION"
    Args = @("--use_advanced_augmentation")
    Description = "VFM + Advanced data augmentation techniques"
}

# Run all experiments
$SuccessfulExperiments = 0
$TotalExperiments = $AllExperiments.Count
$StartTimeAll = Get-Date

Write-Info "Starting $TotalExperiments VFM experiments with core DA techniques..."

foreach ($Experiment in $AllExperiments) {
    $Success = Invoke-Experiment -Name $Experiment.Name -AdditionalArgs $Experiment.Args -Description $Experiment.Description
    if ($Success) {
        $SuccessfulExperiments++
    }
    
    # Optional: Add a small delay between experiments
    Start-Sleep -Seconds 2
}

$EndTimeAll = Get-Date
$TotalDuration = ($EndTimeAll - $StartTimeAll).TotalMinutes

Write-Host "`n" + "="*80 -ForegroundColor Magenta
Write-Host "VFM EXPERIMENT SUMMARY" -ForegroundColor Magenta
Write-Host "="*80 -ForegroundColor Magenta
Write-Success "Completed: $SuccessfulExperiments / $TotalExperiments experiments"
Write-Info "Total time: $([math]::Round($TotalDuration, 2)) minutes"
Write-Info "Average time per experiment: $([math]::Round($TotalDuration / $TotalExperiments, 2)) minutes"

if ($SuccessfulExperiments -eq $TotalExperiments) {
    Write-Success "All experiments completed successfully!"
} elseif ($SuccessfulExperiments -gt 0) {
    Write-Warning "$($TotalExperiments - $SuccessfulExperiments) experiments failed"
} else {
    Write-Error "All experiments failed"
}

Write-Host @"

RESULTS LOCATION
===================
Check your results in: $OutputDir

NEXT STEPS FOR VFM EXPERIMENTS
===============================
1. Compare VFM baseline vs DA techniques: scripts/compare_experiments.py
2. Review detailed logs in each experiment subdirectory
3. VFM typically converges faster due to pre-training
4. If GPU memory issues, reduce --batch_size to 16
5. For production runs: increase epochs to 5-10 for VFM fine-tuning

VFM PERFORMANCE TIPS
====================
* VFM is pre-trained on medical images - should outperform generic models
* 2 epochs often sufficient for VFM fine-tuning (vs 10+ for from-scratch)
* Monitor GPU usage: nvidia-smi -l 1
* VFM results are typically more stable than training from scratch
* Check if VFM + basic DA beats ResNet baselines significantly

"@ -ForegroundColor Green

Write-Host "VFM baseline and DA experiments complete!" -ForegroundColor Magenta
