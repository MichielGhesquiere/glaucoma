# Resume VFM Experiments with Unicode Fix
# Continues from where the experiments were interrupted

param(
    [string]$BaseDataRoot = "D:\glaucoma\data",
    [string]$OutputDir = "D:\glaucoma\vfm_quick_experiments",
    [int]$NumEpochs = 2,
    [int]$BatchSize = 32,
    [int]$NumWorkers = 4,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

# Set environment variables for proper Unicode handling
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Color functions
function Write-Info { param($Text) Write-Host "[INFO] $Text" -ForegroundColor Cyan }
function Write-Success { param($Text) Write-Host "[SUCCESS] $Text" -ForegroundColor Green }
function Write-Warning { param($Text) Write-Host "[WARNING] $Text" -ForegroundColor Yellow }
function Write-Error { param($Text) Write-Host "[ERROR] $Text" -ForegroundColor Red }

Write-Host @"

RESUME VFM EXPERIMENTS (Unicode Fixed)
======================================
The previous experiments likely trained successfully but failed 
due to Unicode encoding errors. Resuming from checkpoint...

"@ -ForegroundColor Magenta

# Python executable
$PythonExe = "python"
$ScriptPath = "multisource_domain_finetuning.py"

# Check if script exists
if (-not (Test-Path $ScriptPath)) {
    Write-Error "Script not found: $ScriptPath"
    exit 1
}

Write-Info "Output directory: $OutputDir"
Write-Info "Using VFM (Vision Foundation Model) with Unicode fix applied"

# Common arguments - with resumption enabled
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
    "--early_stopping_patience", "1"
    "--quiet_mode"
    "--exclude_smdg_unknown"
    "--resume_from_checkpoint"  # Try to resume from where we left off
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
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

Write-Host @"

RESUMING VFM EXPERIMENTS
========================
1. VFM BASELINE: Should resume from checkpoint
2. VFM + DANN: Domain Adversarial Networks  
3. VFM + MIXSTYLE: Domain generalization
4. VFM + AUGMENTATION: Advanced augmentation

"@ -ForegroundColor Cyan

# Just run the baseline first to test the fix
$Experiments = @(
    @{
        Name = "01_VFM_BASELINE_RESUME"
        Args = @()
        Description = "VFM baseline - resume with Unicode fix"
    }
)

Write-Info "Testing Unicode fix with VFM baseline experiment..."

foreach ($Experiment in $Experiments) {
    $Success = Invoke-Experiment -Name $Experiment.Name -AdditionalArgs $Experiment.Args -Description $Experiment.Description
    if ($Success) {
        Write-Success "Unicode fix successful! You can now run the full experiment suite."
        Write-Info "To run all experiments: .\quick_baseline_and_da.ps1"
    } else {
        Write-Warning "Still having issues. Check the error messages above."
    }
}

Write-Host @"

NEXT STEPS
==========
If the test succeeded:
1. Run full experiment suite: .\quick_baseline_and_da.ps1
2. The experiments should now complete without Unicode errors
3. Check results in: $OutputDir

If still having issues:
1. Check Python environment: python --version
2. Verify PYTHONIOENCODING is set: echo $env:PYTHONIOENCODING
3. Try reducing batch size to 16 if memory issues

"@ -ForegroundColor Green
