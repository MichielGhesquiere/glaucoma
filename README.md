# Glaucoma Classification Research

This repository contains code to train and evaluate deep learning classifiers for glaucoma detection from retinal fundus photographs. The goal is to build **generalisable and fair** models that perform well across different cameras and demographic groups. Training uses a mixture of open–source datasets and evaluation is performed on both in–distribution (ID) and out–of–distribution (OOD) test sets.

## Overview
- **train\_classification.py** implements the main training pipeline. It supports loading images and metadata from SMDG-19, CHAKSU and AIROGS. Vision models from the [timm](https://github.com/huggingface/pytorch-image-models) library are used and training options include MixUp, label smoothing and temperature scaling.
- **evaluate\_id.py** and **evaluate\_ood.py** run inference on ID and several external datasets (e.g. PAPILLA, ACRIMA, HYGD, OIA-ODIR). Results include ROC curves and calibration plots.
- Fairness metrics (TPR/FPR parity and underdiagnosis disparity) are computed via functions in `src/evaluation/fairness.py`.

## Datasets
Training and evaluation rely on several publicly available datasets. The main ones are:
- **SMDG-19** – images with glaucoma labels and segmentations.
- **CHAKSU** – multi-camera dataset used for training and OOD evaluation.
- **AIROGS** – large dataset of labelled fundus images.
- **PAPILLA**, **ACRIMA**, **HYGD**, **OIA-ODIR** – used only for external testing.

Actual image files are **not** included in the repository. Place them under `data/raw/<dataset>/` and adjust paths when running the scripts.

## Fairness Analysis
Performance by subgroups (age, sex, camera type, etc.) is analysed through `src/evaluation/fairness.py`. The `calculate_underdiagnosis_disparities` function reports false negative and false positive rate gaps between favoured and disfavoured groups.


```

## Repository Structure
- `scripts/` – entry points for training, evaluation and analysis
- `src/` – modular code for datasets, models and utilities
- `configs/` – configuration files
- `notebooks/` – exploratory notebooks
- `dataset_analysis_results/` – dataset exploration figures

