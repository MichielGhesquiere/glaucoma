# Dataset Analysis Script Runner
# This PowerShell script runs the comprehensive dataset analysis

param(
    [string]$OutputDir = ".\dataset_analysis_results",
    [string]$DataType = "raw",
    [string]$BaseDataRoot = "D:\glaucoma\data",
    [string]$DataConfigPath = ".\configs\data_paths.yaml",
    [string]$ModelPath = "",
    [switch]$ProcessedData,
    [switch]$PixelAnalysis,
    [switch]$UmapAnalysis,
    [switch]$UseAirogs,
    [int]$AirogsNumRg = 3000,
    [int]$AirogsNumNrg = 3000,
    [string]$AirogsLabelFile = "D:\glaucoma\data\raw\AIROGS\train_labels.csv",
    [string]$AirogsImageDir = "D:\glaucoma\data\raw\AIROGS\img",
    [switch]$UseAirogsCache,
    [int]$Seed = 42,
    [switch]$SkipPapilla,
    [switch]$SkipChaksu,
    [switch]$SkipAcrima,
    [switch]$SkipHygd,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Dataset Analysis Script Runner

This script runs comprehensive analysis of all glaucoma datasets used in the project.

Parameters:
  -OutputDir      Directory to save analysis results (default: .\dataset_analysis_results)
  -DataType       Type of data to analyze: 'raw' or 'processed' (default: raw)
  -BaseDataRoot   Base directory containing all datasets (default: D:\glaucoma\data)
  -DataConfigPath Path to data configuration YAML file (default: .\configs\data_paths.yaml)
  -ModelPath      Path to pre-trained model for UMAP analysis (default: D:\glaucoma\models\VFM_Fundus_weights.pth)
  -ProcessedData  Use processed data instead of raw (sets DataType to 'processed')
  -PixelAnalysis  Enable pixel distribution analysis for datasets with 200+ samples
  -UmapAnalysis   Enable UMAP embeddings visualization using pre-trained model
  -UseAirogs      Include AIROGS dataset in analysis
  -AirogsNumRg    Number of RG (glaucoma) samples to load from AIROGS (default: 3000)
  -AirogsNumNrg   Number of NRG (normal) samples to load from AIROGS (default: 3000)
  -AirogsLabelFile Path to AIROGS labels CSV file (default: D:\glaucoma\data\raw\AIROGS\train_labels.csv)
  -AirogsImageDir  Path to AIROGS image directory (default: D:\glaucoma\data\raw\AIROGS\img)
  -UseAirogsCache Use cached AIROGS manifest if available
  -Seed           Random seed for reproducible sampling (default: 42)
  -SkipPapilla    Skip PAPILLA/SMDG dataset analysis
  -SkipChaksu     Skip CHAKSU dataset analysis  
  -SkipAcrima     Skip ACRIMA dataset analysis
  -SkipHygd       Skip HYGD dataset analysis
  -Help           Show this help message

Examples:
  .\run_dataset_analysis.ps1
  .\run_dataset_analysis.ps1 -ProcessedData
  .\run_dataset_analysis.ps1 -PixelAnalysis -UmapAnalysis
  .\run_dataset_analysis.ps1 -UseAirogs -AirogsNumRg 2000 -AirogsNumNrg 2000
  .\run_dataset_analysis.ps1 -OutputDir "C:\results" -SkipHygd
  .\run_dataset_analysis.ps1 -DataType "processed" -BaseDataRoot "E:\data"

The script will generate:
- Summary tables (CSV and visual)
- Prevalence comparison plots
- Sample distribution plots
- Metadata availability analysis
- Detailed JSON reports
- Pixel distribution analysis (if enabled)
- UMAP embeddings visualization (if enabled)

"@
    exit 0
}

# Set data type if ProcessedData switch is used
if ($ProcessedData) {
    $DataType = "processed"
}

# Build command arguments
$commandArgs = @(
    "--output_dir", $OutputDir,
    "--data_type", $DataType,
    "--base_data_root", $BaseDataRoot,
    "--data_config_path", $DataConfigPath
)

# Add dataset skip flags
if ($SkipPapilla) { $commandArgs += "--no-eval_papilla" }
if ($SkipChaksu) { $commandArgs += "--no-eval_chaksu" }
if ($SkipAcrima) { $commandArgs += "--no-eval_acrima" }
if ($SkipHygd) { $commandArgs += "--no-eval_hygd" }

# Add advanced analysis flags
if ($PixelAnalysis) { $commandArgs += "--pixel_analysis" }
if ($UmapAnalysis) { $commandArgs += "--umap_analysis" }
if ($ModelPath) { $commandArgs += "--model_path"; $commandArgs += $ModelPath }

# Add AIROGS parameters
if ($UseAirogs) { 
    $commandArgs += "--use_airogs"
    $commandArgs += "--airogs_num_rg"; $commandArgs += $AirogsNumRg
    $commandArgs += "--airogs_num_nrg"; $commandArgs += $AirogsNumNrg
    $commandArgs += "--airogs_label_file"; $commandArgs += $AirogsLabelFile
    $commandArgs += "--airogs_image_dir"; $commandArgs += $AirogsImageDir
    $commandArgs += "--seed"; $commandArgs += $Seed
    if ($UseAirogsCache) { $commandArgs += "--use_airogs_cache" }
}

Write-Host "Running dataset analysis with the following configuration:" -ForegroundColor Green
Write-Host "  Output Directory: $OutputDir" -ForegroundColor Cyan
Write-Host "  Data Type: $DataType" -ForegroundColor Cyan
Write-Host "  Base Data Root: $BaseDataRoot" -ForegroundColor Cyan
Write-Host "  Data Config: $DataConfigPath" -ForegroundColor Cyan
Write-Host "  Pixel Analysis: $PixelAnalysis" -ForegroundColor Cyan
Write-Host "  UMAP Analysis: $UmapAnalysis" -ForegroundColor Cyan
if ($ModelPath) { Write-Host "  Model Path: $ModelPath" -ForegroundColor Cyan }
Write-Host "  Use AIROGS: $UseAirogs" -ForegroundColor Cyan
if ($UseAirogs) {
    Write-Host "  AIROGS RG Samples: $AirogsNumRg" -ForegroundColor Cyan
    Write-Host "  AIROGS NRG Samples: $AirogsNumNrg" -ForegroundColor Cyan
    Write-Host "  AIROGS Label File: $AirogsLabelFile" -ForegroundColor Cyan
    Write-Host "  AIROGS Image Dir: $AirogsImageDir" -ForegroundColor Cyan
    Write-Host "  Use AIROGS Cache: $UseAirogsCache" -ForegroundColor Cyan
    Write-Host "  Random Seed: $Seed" -ForegroundColor Cyan
}
Write-Host "  Skip Papilla: $SkipPapilla" -ForegroundColor Cyan
Write-Host "  Skip Chaksu: $SkipChaksu" -ForegroundColor Cyan
Write-Host "  Skip Acrima: $SkipAcrima" -ForegroundColor Cyan
Write-Host "  Skip HYGD: $SkipHygd" -ForegroundColor Cyan
Write-Host ""

# Check if Python script exists
if (-not (Test-Path ".\scripts\dataset_analysis.py")) {
    Write-Host "Error: dataset_analysis.py not found in .\scripts\" -ForegroundColor Red
    Write-Host "Make sure you're running this from the project root directory" -ForegroundColor Red
    exit 1
}

# Check if data config exists
if (-not (Test-Path $DataConfigPath)) {
    Write-Host "Warning: Data config file not found at $DataConfigPath" -ForegroundColor Yellow
    Write-Host "The script will still run but training data analysis may fail" -ForegroundColor Yellow
}

# Check if base data directory exists
if (-not (Test-Path $BaseDataRoot)) {
    Write-Host "Warning: Base data directory not found at $BaseDataRoot" -ForegroundColor Yellow
    Write-Host "External dataset analysis may fail" -ForegroundColor Yellow
}

try {
    Write-Host "Starting dataset analysis..." -ForegroundColor Green
    python .\scripts\dataset_analysis.py @commandArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Dataset analysis completed successfully!" -ForegroundColor Green
        Write-Host "Results saved to: $OutputDir" -ForegroundColor Cyan
        
        # List generated files if output directory exists
        if (Test-Path $OutputDir) {
            $latestDir = Get-ChildItem $OutputDir -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($latestDir) {
                Write-Host ""
                Write-Host "Generated files:" -ForegroundColor Green
                Get-ChildItem $latestDir.FullName -File | ForEach-Object {
                    Write-Host "  $($_.Name)" -ForegroundColor Cyan
                }
            }
        }
    } else {
        Write-Host "Dataset analysis failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "Error running dataset analysis: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
