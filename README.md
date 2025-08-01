# Glaucoma Classification Research

This repository contains code to train and evaluate deep learning classifiers for glaucoma detection from retinal fundus photographs. The goal is to build **generalisable and fair** models that perform well across different cameras and demographic groups. Training uses a mixture of open–source datasets and evaluation is performed on both in–distribution (ID) and out–of–distribution (OOD) test sets.

## Overview
- **train\_classification.py** implements the main training pipeline. It supports loading images and metadata from SMDG-19, CHAKSU and AIROGS. Vision models from the [timm](https://github.com/huggingface/pytorch-image-models) library are used and training options include MixUp, label smoothing and temperature scaling.
- **evaluate\_id.py** and **evaluate\_ood.py** run inference on ID and several external datasets (e.g. PAPILLA, ACRIMA, HYGD, OIA-ODIR). Results include ROC curves and calibration plots.
- Fairness metrics (TPR/FPR parity and underdiagnosis disparity) are computed via functions in `src/evaluation/fairness.py`.

The directory layout roughly matches the structure described in [`project_structure.md`](project_structure.md). Core utilities live under `src/`.

## Installation
1. Create a Python environment (Python 3.9+ recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the datasets (see below) into `data/raw/` following the subdirectory scheme used by the scripts.

## Datasets
Training and evaluation rely on several publicly available datasets. The main ones are:
- **SMDG-19** – images with glaucoma labels and segmentations.
- **CHAKSU** – multi-camera dataset used for training and OOD evaluation.
- **AIROGS** – large dataset of labelled fundus images.
- **PAPILLA**, **ACRIMA**, **HYGD**, **OIA-ODIR** – used only for external testing.

Actual image files are **not** included in the repository. Place them under `data/raw/<dataset>/` and adjust paths when running the scripts.

## Training Example
```bash
python scripts/train_classification.py --config configs/classification_config.yaml
```
Command line arguments still override any values from the YAML file. See `scripts/train_classification.py` for all available options.

## Evaluating
To evaluate the best checkpoint on the held-out ID test set:
```bash
python scripts/evaluate_id.py --experiment_dir /path/to/run
```
For external datasets:
```bash
python scripts/evaluate_ood.py --experiment_parent_dir /path/to/runs
```
Evaluation summaries are saved in each experiment folder.

## Fairness Analysis
Performance by subgroups (age, sex, camera type, etc.) is analysed through `src/evaluation/fairness.py`. The `calculate_underdiagnosis_disparities` function reports false negative and false positive rate gaps between favoured and disfavoured groups.

```python
# src/evaluation/fairness.py
"""
Calculates FNR and FPR disparities between specified subgroups.
"""
```

## Repository Structure
- `scripts/` – entry points for training, evaluation and analysis
- `src/` – modular code for datasets, models and utilities
- `configs/` – configuration files
- `notebooks/` – exploratory notebooks
- `dataset_analysis_results/` – dataset exploration figures


## License
This project is released under the MIT License.

