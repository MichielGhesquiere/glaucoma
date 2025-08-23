#!/bin/bash
# run_multitask_pipeline.sh
# Automated pipeline for vCDR extraction and multi-task training

echo "Starting vCDR extraction and multi-task training pipeline..."

# Step 1: Extract vCDR labels
echo "Step 1: Extracting vCDR labels..."
python scripts/extract_vcdr_labels.py \
    --output_dir ./vcdr_extraction_results \
    --unet_model_path "D:\glaucoma\models\best_multitask_model_epoch_25.pth"

# Find the most recent vCDR CSV file
VCDR_CSV=$(find ./vcdr_extraction_results -name "vcdr_labels_*.csv" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$VCDR_CSV" ]; then
    echo "Error: No vCDR CSV file found!"
    exit 1
fi

echo "Found vCDR CSV: $VCDR_CSV"

# Step 2: Train multi-task model
echo "Step 2: Training multi-task model..."
python scripts/train_multitask_classification_regression.py \
    --vcdr_csv "$VCDR_CSV" \
    --backbone resnet18 \
    --batch_size 32 \
    --num_epochs 50

echo "Pipeline completed successfully!"
