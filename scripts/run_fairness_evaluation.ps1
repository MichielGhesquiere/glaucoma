# PowerShell script to run fairness evaluation
# This script evaluates trained models for fairness across demographic subgroups

param(
    [string]$ExperimentDir = "experiments\multisource",
    [string]$OutputDir = "fairness_evaluation_results",
    [string]$DataRoot = "D:\glaucoma\data",
    [switch]$EvalSmdg = $true,
    [switch]$EvalChaksu = $true,
    [switch]$EvalPapilla = $true,
    [switch]$EvalAcrima = $true,
    [switch]$EvalHygd = $true,
    [int]$MinSamplesPerGroup = 10,
    [int]$BatchSize = 32
)

Write-Host "🔍 Starting Fairness Evaluation for Multi-Source Domain Adaptation Models" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

# Construct the command
$Command = "python scripts\evaluate_fairness.py"
$Command += " --experiment_parent_dir `"$ExperimentDir`""
$Command += " --output_dir `"$OutputDir`""
$Command += " --base_data_root `"$DataRoot`""
$Command += " --eval_batch_size $BatchSize"
$Command += " --min_samples_per_group $MinSamplesPerGroup"

# Add dataset flags
if ($EvalSmdg) { $Command += " --eval_smdg" }
if ($EvalChaksu) { $Command += " --eval_chaksu" }
if ($EvalPapilla) { $Command += " --eval_papilla" }
if ($EvalAcrima) { $Command += " --eval_acrima" }
if ($EvalHygd) { $Command += " --eval_hygd" }

Write-Host "📊 Evaluation Configuration:" -ForegroundColor Yellow
Write-Host "  Experiment Directory: $ExperimentDir"
Write-Host "  Output Directory: $OutputDir"
Write-Host "  Data Root: $DataRoot"
Write-Host "  Min Samples per Group: $MinSamplesPerGroup"
Write-Host "  Batch Size: $BatchSize"
Write-Host ""

Write-Host "🔧 Datasets to Evaluate:" -ForegroundColor Yellow
if ($EvalSmdg) { Write-Host "  ✓ SMDG-19 (age, sex, ethnicity)" -ForegroundColor Green }
if ($EvalChaksu) { Write-Host "  ✓ CHAKSU (camera type)" -ForegroundColor Green }
if ($EvalPapilla) { Write-Host "  ✓ PAPILLA" -ForegroundColor Green }
if ($EvalAcrima) { Write-Host "  ✓ ACRIMA" -ForegroundColor Green }
if ($EvalHygd) { Write-Host "  ✓ HYGD" -ForegroundColor Green }
Write-Host ""

Write-Host "🎯 Fairness Metrics to Compute:" -ForegroundColor Yellow
Write-Host "  • Overall AUC and Accuracy"
Write-Host "  • Subgroup Performance (AUC, Accuracy, TPR, FPR)"
Write-Host "  • Demographic Parity (equal positive prediction rates)"
Write-Host "  • Equalized Odds (equal TPR and FPR across groups)"
Write-Host "  • Underdiagnosis Disparities (false negative rate differences)"
Write-Host ""

Write-Host "🚀 Executing command:" -ForegroundColor Magenta
Write-Host "$Command" -ForegroundColor Gray
Write-Host ""

# Execute the command
try {
    Invoke-Expression $Command
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Fairness evaluation completed successfully!" -ForegroundColor Green
        Write-Host "📁 Results saved to: $OutputDir" -ForegroundColor Green
        Write-Host ""
        Write-Host "📊 Generated outputs:" -ForegroundColor Yellow
        Write-Host "  • fairness_evaluation_results.json - Detailed results"
        Write-Host "  • fairness_summary.csv - Summary table"
        Write-Host "  • visualizations/ - Performance plots by demographic groups"
        Write-Host "  • fairness_comparison_*.png - Cross-dataset comparisons"
    } else {
        Write-Host ""
        Write-Host "❌ Fairness evaluation failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host ""
    Write-Host "❌ Error executing fairness evaluation: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "📖 Usage Examples:" -ForegroundColor Cyan
Write-Host "  # Evaluate all datasets with default settings:"
Write-Host "  .\run_fairness_evaluation.ps1"
Write-Host ""
Write-Host "  # Evaluate only CHAKSU and SMDG with custom settings:"
Write-Host "  .\run_fairness_evaluation.ps1 -EvalChaksu -EvalSmdg -MinSamplesPerGroup 20"
Write-Host ""
Write-Host "  # Use custom experiment directory:"
Write-Host "  .\run_fairness_evaluation.ps1 -ExperimentDir `"experiments\my_models`""
