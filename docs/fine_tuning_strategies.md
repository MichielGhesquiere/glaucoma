# Fine-tuning Strategy Guide

This document explains how to use the three fine-tuning strategies implemented in the training script to mitigate catastrophic forgetting when fine-tuning vision foundation models.

## Overview

The training script now supports three fine-tuning regimes:

| Strategy | What's Trainable? | Why Use It? | When to Stop |
|----------|------------------|-------------|--------------|
| **Linear Probing** | Only classification head | Sanity-check transferability; no feature forgetting | 5-15 epochs or when val-loss plateaus |
| **Gradual Unfreezing** | Head → head+top layers → all layers | Lets higher layers adapt while protecting low-level filters | Unfreeze stages based on validation plateau |
| **Full + LLRD** | All layers with Layer-wise LR Decay | Best accuracy with enough data; tiny LR prevents forgetting | Standard early stopping |

## Command Line Usage

### 1. Linear Probing
```bash
python train_classification.py \
    --ft_strategy linear \
    --linear_probe_epochs 15 \
    --model_name vit_base_patch16_224 \
    --custom_weights_path models/VFM_Fundus_weights.pth \
    --experiment_tag linear_probe
```

### 2. Gradual Unfreezing (ULMFit-style)
```bash
python train_classification.py \
    --ft_strategy gradual \
    --gradual_patience 5 \
    --num_epochs 45 \
    --model_name vit_base_patch16_224 \
    --custom_weights_path models/VFM_Fundus_weights.pth \
    --experiment_tag gradual_unfreeze
```

### 3. Full Fine-tuning with LLRD
```bash
python train_classification.py \
    --ft_strategy full \
    --llrd_decay 0.85 \
    --num_epochs 30 \
    --model_name vit_base_patch16_224 \
    --custom_weights_path models/VFM_Fundus_weights.pth \
    --experiment_tag full_llrd
```

## Parameters Explained

### Fine-tuning Strategy Parameters

- `--ft_strategy`: Choose from `linear`, `gradual`, or `full`
- `--llrd_decay`: Layer-wise learning rate decay factor (default: 0.9)
  - 0.9 means each deeper layer gets 90% of the LR of the layer above
  - Lower values (0.8-0.85) give more aggressive LR differences
- `--gradual_patience`: Epochs to wait before unfreezing next layer group (default: 3)
- `--linear_probe_epochs`: Maximum epochs for linear probing (default: 10)

### Architecture-Specific Patterns

The gradual unfreezing automatically detects your model architecture and uses appropriate layer patterns:

**Vision Transformers (ViT, DeiT, DINOv2):**
- Phase 0: `head|fc|classifier` (head only)
- Phase 1: `head|fc|classifier|norm|blocks\.1[01]` (head + last 2 blocks + norm)
- Phase 2: `.*` (everything)

**ResNet:**
- Phase 0: `head|fc|classifier` (head only)  
- Phase 1: `head|fc|classifier|layer4` (head + last residual block)
- Phase 2: `.*` (everything)

## Expected Results

### Linear Probing
- **Purpose**: Baseline to test if pre-trained features are useful
- **Expected**: Lower accuracy but good sanity check
- **Training time**: Fastest (~15 epochs)
- **Good if**: Accuracy is reasonable (>70-80% for glaucoma), indicating good features

### Gradual Unfreezing  
- **Purpose**: Balance between feature preservation and adaptation
- **Expected**: Good accuracy with retained low-level features
- **Training time**: Medium (~45 epochs across 3 phases)
- **Good if**: Maintains linear probe performance while improving significantly

### Full LLRD
- **Purpose**: Maximum performance while minimizing forgetting
- **Expected**: Best overall accuracy and calibration
- **Training time**: Medium (~30 epochs)
- **Good if**: You have sufficient data and compute budget

## Running Comparisons

Use the provided comparison script:

```bash
python scripts/run_fine_tuning_comparison.py
```

This will run all three strategies with sensible defaults and generate comparison results.

## Monitoring Training

### Key Metrics to Watch

1. **Validation Accuracy**: Should improve across strategies: linear < gradual < full
2. **Training Speed**: Linear fastest, gradual slowest due to phase transitions
3. **Parameter Utilization**: 
   - Linear: ~1-5% of parameters trainable
   - Gradual: Gradually increases to 100%
   - Full: 100% trainable but with varying learning rates

### Debugging Tips

1. **Linear probe fails**: Pre-trained features may not be suitable
2. **Gradual unfreezing worse than linear**: Try increasing patience or reducing LR
3. **Full LLRD unstable**: Reduce LLRD decay (try 0.8) or base learning rate

## Advanced Usage

### Custom Unfreezing Patterns

You can modify the patterns in `src/utils/fine_tune_tools.py` to customize which layers unfreeze at each phase.

### Integration with Other Features

All fine-tuning strategies work with:
- Mixup/CutMix augmentation
- Label smoothing  
- Temperature scaling calibration
- Automatic mixed precision (AMP)

### Example: Medical Domain Fine-tuning

For medical imaging, we typically recommend:

```bash
# Start with linear probing to verify feature quality
python train_classification.py --ft_strategy linear --linear_probe_epochs 10

# If linear probe works well (>75% accuracy), use gradual unfreezing
python train_classification.py --ft_strategy gradual --gradual_patience 4 --num_epochs 40

# For final model, use full LLRD with conservative decay
python train_classification.py --ft_strategy full --llrd_decay 0.8 --learning_rate 5e-5
```

This approach maximizes the retention of important medical imaging features learned during pre-training while allowing adaptation to your specific task.
