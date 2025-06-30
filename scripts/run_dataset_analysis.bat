@echo off
REM Dataset Analysis Script Runner (Batch Version)
REM This batch script runs the comprehensive dataset analysis

echo Starting Dataset Analysis...
echo.

REM Set default parameters
set OUTPUT_DIR=dataset_analysis_results
set DATA_TYPE=raw
set BASE_DATA_ROOT=D:\glaucoma\data
set DATA_CONFIG=configs\data_paths.yaml

REM Check if Python script exists
if not exist "scripts\dataset_analysis.py" (
    echo Error: dataset_analysis.py not found in scripts\
    echo Make sure you're running this from the project root directory
    pause
    exit /b 1
)

REM Run the analysis
python scripts\dataset_analysis.py ^
    --output_dir %OUTPUT_DIR% ^
    --data_type %DATA_TYPE% ^
    --base_data_root "%BASE_DATA_ROOT%" ^
    --data_config_path %DATA_CONFIG% ^
    --eval_papilla ^
    --eval_chaksu ^
    --eval_acrima ^
    --eval_hygd

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Dataset analysis completed successfully!
    echo Results saved to: %OUTPUT_DIR%
) else (
    echo.
    echo Dataset analysis failed with error code: %ERRORLEVEL%
)

echo.
pause
