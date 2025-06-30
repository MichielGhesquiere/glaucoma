# Ultra-Fast Domain Adaptation Testing (1-2 hour total runtime)
# Minimal epochs and reduced scope for rapid baseline comparison

param(
    [string]$BaseDataRoot = "D:\glaucoma\data",
    [string]$OutputDir = "D:\glaucoma\ultra_fast_experiments",
    [int]$NumEpochs = 1,     # Ultra-minimal: just 1 epoch
    [int]$BatchSize = 128,   # Keep high for GPU efficiency
    [int]$NumWorkers = 4,    # Faster data loading
    [string]$Model = "resnet18",
    [bool]$EnableVFM = $false,
    [bool]$ExcludeUnknown = $true,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

# Color functions
function Write-Info { param($Text) Write-Host "[INFO] $Text" -ForegroundColor Cyan }
function Write-Success { param($Text) Write-Host "[SUCCESS] $Text" -ForegroundColor Green }
function Write-Warning { param($Text) Write-Host "[WARNING] $Text" -ForegroundColor Yellow }
function Write-Error { param($Text) Write-Host "[ERROR] $Text" -ForegroundColor Red }

Write-Host @"

ULTRA-FAST DA BASELINE COMPARISON (1-2 HOUR RUNTIME)
=====================================================
Model: $Model
Epochs: $NumEpochs (ultra-minimal for speed)
Batch Size: $BatchSize 
Estimated Total Time: 1-2 hours
=====================================================

"@ -ForegroundColor Magenta

# Python executable
$PythonExe = "python"
$ScriptPath = "multisource_domain_finetuning.py"

# Check if script exists
if (-not (Test-Path $ScriptPath)) {
    Write-Error "Script not found: $ScriptPath"
    exit 1
}

# Create output directory
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Write-Info "Output directory: $OutputDir"

# Common arguments - ultra-minimal
$CommonArgs = @(
    "--base_data_root", $BaseDataRoot
    "--output_dir", $OutputDir
    "--num_epochs", $NumEpochs
    "--batch_size", $BatchSize
    "--num_workers", $NumWorkers
    "--pin_memory"
    "--prefetch_factor", "4"
    "--seed", $Seed
    "--default_model", $Model
    "--early_stopping_patience", "1"  # Ultra-aggressive early stopping
    "--quiet_mode"
    "--disable_vfm"
    "--exclude_smdg_unknown"
)

# Function to invoke experiment
function Invoke-Experiment {
    param(
        [string]$Name,
        [string[]]$AdditionalArgs = @(),
        [string]$Description = ""
    )
    
    Write-Host "`n" + "="*60 -ForegroundColor Yellow
    Write-Info "EXPERIMENT: $Name"
    if ($Description) {
        Write-Host "Description: $Description" -ForegroundColor Gray
    }
    Write-Host "="*60 -ForegroundColor Yellow
    
    $AllArgs = $CommonArgs + $AdditionalArgs
    
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
        return $false
    }
}

Write-Host @"

ULTRA-FAST EXPERIMENT PLAN (Key DA Techniques Only)
====================================================
1. BASELINE: Pure ResNet-18 
2. DOMAIN ADVERSARIAL: ResNet-18 + DANN (most important DA technique)
3. MIXSTYLE: ResNet-18 + MixStyle (popular DA method)

Estimated: ~20 min per experiment × 3 experiments × 4 datasets = 4 hours max
With 1 epoch: Likely 1-2 hours total

"@ -ForegroundColor Cyan

# Core experiments only - most important DA techniques
$AllExperiments = @()

# Experiment 1: Baseline
$AllExperiments += @{
    Name = "01_BASELINE_ResNet18_1epoch"
    Args = @()
    Description = "Pure ResNet-18 baseline (1 epoch)"
}

# Experiment 2: Domain Adversarial (most important)
$AllExperiments += @{
    Name = "02_DANN_ResNet18_1epoch"
    Args = @("--use_domain_adversarial")
    Description = "ResNet-18 + DANN (1 epoch)"
}

# Experiment 3: MixStyle (popular and effective)
$AllExperiments += @{
    Name = "03_MIXSTYLE_ResNet18_1epoch"
    Args = @("--use_mixstyle")
    Description = "ResNet-18 + MixStyle (1 epoch)"
}

# Run experiments
$SuccessfulExperiments = 0
$TotalExperiments = $AllExperiments.Count
$StartTimeAll = Get-Date

Write-Info "Starting $TotalExperiments ultra-fast experiments..."

foreach ($Experiment in $AllExperiments) {
    $Success = Invoke-Experiment -Name $Experiment.Name -AdditionalArgs $Experiment.Args -Description $Experiment.Description
    if ($Success) {
        $SuccessfulExperiments++
    }
}

$EndTimeAll = Get-Date
$TotalDuration = ($EndTimeAll - $StartTimeAll).TotalMinutes

Write-Host "`n" + "="*60 -ForegroundColor Magenta
Write-Host "ULTRA-FAST EXPERIMENT SUMMARY" -ForegroundColor Magenta
Write-Host "="*60 -ForegroundColor Magenta
Write-Success "Completed: $SuccessfulExperiments / $TotalExperiments experiments"
Write-Info "Total time: $([math]::Round($TotalDuration, 2)) minutes"

Write-Host @"

RESULTS LOCATION
=================
Check: $OutputDir

NEXT STEPS
==========
1. Review ultra-fast baseline comparison
2. Run full experiments if results look promising: .\quick_baseline_and_da.ps1
3. For production runs: increase epochs to 10-20

"@ -ForegroundColor Green

Write-Host "Ultra-fast DA baseline comparison complete!" -ForegroundColor Magenta
