#!/usr/bin/env python3
"""
Enhanced fine-tuning strategy comparison script with comprehensive data configuration.

This script runs all three fine-tuning strategies with the same comprehensive
data setup as the regularization study:
1. Linear probing (head-only)
2. Gradual unfreezing (ULMFit-style) 
3. Full fine-tuning with Layer-wise Learning Rate Decay (LLRD)

Includes SMDG-19, CHAKSU, and AIROGS data sources with proper sampling.
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"✓ {description} completed successfully")
        print(f"Duration: {duration}")
        return True, duration
    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"✗ {description} failed with return code {e.returncode}")
        print(f"Failed after: {duration}")
        return False, duration
    except Exception as e:
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"✗ {description} failed with error: {e}")
        print(f"Failed after: {duration}")
        return False, duration

def main():
    # Base directory setup
    script_dir = Path(__file__).parent
    train_script = script_dir / "train_classification.py"
    
    if not train_script.exists():
        print(f"Error: Training script not found at {train_script}")
        sys.exit(1)
    
    # Data configuration - matching regularization study
    data_type = "raw"  # or "processed"
    base_data_root = "D:/glaucoma/data"
    
    # Data paths
    smdg_metadata = "D:/glaucoma/data/raw/SMDG-19/metadata - standardized.csv"
    smdg_image_dir = "D:/glaucoma/data/raw/SMDG-19/full-fundus/full-fundus"
    chaksu_base = "D:/glaucoma/data/raw/Chaksu/Train/Train/1.0_Original_Fundus_Images"
    chaksu_decision = "D:/glaucoma/data/raw/Chaksu/Train/Train/6.0_Glaucoma_Decision"
    chaksu_metadata = "D:/glaucoma/data/raw/Chaksu/Train/Train/6.0_Glaucoma_Decision/Majority"
    airogs_labels = "D:/glaucoma/data/raw/AIROGS/train_labels.csv"
    airogs_images = "D:/glaucoma/data/raw/AIROGS/img"
    
    # VFM configuration
    vfm_weights = "D:/glaucoma/models/VFM_Fundus_weights.pth"
    vfm_key = "teacher"
    
    # Output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"experiments/fine_tuning_comparison_study_{data_type}_{timestamp}"
    
    # Common arguments for all experiments - comprehensive setup like regularization study
    base_args = [
        "python", str(train_script),
        # Model configuration
        "--model_name", "vit_base_patch16_224",
        "--custom_weights_path", vfm_weights,
        "--checkpoint_key", vfm_key,
        # Data configuration
        "--data_type", data_type,
        "--base_data_root", base_data_root,
        "--smdg_metadata_file", smdg_metadata,
        "--smdg_image_dir", smdg_image_dir,
        "--chaksu_base_dir", chaksu_base,
        "--chaksu_decision_dir", chaksu_decision,
        "--chaksu_metadata_dir", chaksu_metadata,
        "--airogs_label_file", airogs_labels,
        "--airogs_image_dir", airogs_images,
        "--use_chaksu",
        "--use_airogs",
        "--airogs_num_rg_samples", "2000",  # Balanced sampling
        "--airogs_num_nrg_samples", "2000",
        "--use_airogs_cache",
        # Training configuration
        "--batch_size", "16",
        "--eval_batch_size", "16",
        "--weight_decay", "0.05",
        "--early_stopping_patience", "8",
        "--seed", "42",
        "--num_workers", "4",
        "--grad_accum_steps", "2",
        # Output configuration
        "--base_output_dir", output_dir,
        "--save_data_samples",
        "--num_data_samples_per_source", "3",
        # Advanced features
        "--use_data_augmentation",
        "--use_mixup",
        "--mixup_alpha", "0.8",
        "--cutmix_alpha", "1.0",
        "--use_temperature_scaling",
        "--calibration_split_ratio", "0.3",
        "--use_amp",
        "--experiment_tag", "ft_strategy_comparison"
    ]
    
    # Fine-tuning strategy experiments with optimized hyperparameters
    experiments = [
        {
            "name": "Linear Probing",
            "args": [
                "--ft_strategy", "linear",
                "--linear_probe_epochs", "15",
                "--learning_rate", "1e-4",  # Higher LR for head-only training
                "--num_epochs", "15"
            ],
            "description": "Head-only training to test feature transferability without forgetting",
            "expected": "Fast convergence, moderate accuracy, perfect feature preservation"
        },
        {
            "name": "Gradual Unfreezing",
            "args": [
                "--ft_strategy", "gradual", 
                "--gradual_patience", "6",
                "--learning_rate", "3e-5",  # Medium LR for gradual approach
                "--num_epochs", "50"  # Longer training for gradual phases
            ],
            "description": "ULMFit-style progressive layer unfreezing for balanced adaptation",
            "expected": "Good accuracy with feature preservation, smooth learning curves"
        },
        {
            "name": "Full LLRD",
            "args": [
                "--ft_strategy", "full",
                "--llrd_decay", "0.85",  # Conservative decay for medical domain
                "--learning_rate", "1e-5",  # Conservative base LR
                "--num_epochs", "35"
            ],
            "description": "Full model training with layer-wise learning rate decay",
            "expected": "Best overall performance with minimal feature forgetting"
        }
    ]
    
    print("Enhanced Fine-tuning Strategy Comparison Study")
    print("=" * 60)
    print("Model: VisionFM (ViT-Base) with custom pre-trained weights")
    print("Focus: Preventing catastrophic forgetting of foundation model features")
    print()
    print("Data Sources:")
    print(f"  • SMDG-19: Full dataset ({data_type} images)")
    print(f"  • CHAKSU: All camera types enabled")
    print(f"  • AIROGS: 2000 RG + 2000 NRG samples")
    print()
    print("Strategies to test:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}: {exp['description']}")
        print(f"     Expected: {exp['expected']}")
    print()
    print(f"Results will be saved to: {output_dir}")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed with all three strategies? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Experiments cancelled.")
        return
    
    results = []
    total_duration = datetime.now() - datetime.now()  # Initialize to zero duration
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n🚀 Starting Strategy {i}/3: {exp['name']}")
        print(f"Description: {exp['description']}")
        print(f"Expected Outcome: {exp['expected']}")
        
        cmd = base_args + exp['args']
        success, duration = run_command(cmd, f"Strategy {i}: {exp['name']}")
        
        total_duration += duration
        
        results.append({
            'name': exp['name'],
            'success': success,
            'description': exp['description'],
            'expected': exp['expected'],
            'duration': duration
        })
        
        if not success:
            print(f"⚠️  Strategy {i} failed, but continuing with remaining strategies...")
        
        # Brief pause between experiments for system cleanup
        if i < len(experiments):
            print("Pausing for system cleanup...")
            import time
            time.sleep(10)
    
    # Print comprehensive summary
    print(f"\n{'=' * 60}")
    print("FINE-TUNING STRATEGY COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    
    successful_count = 0
    for i, result in enumerate(results, 1):
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        if result['success']:
            successful_count += 1
        
        print(f"\n{i}. {result['name']}: {status}")
        print(f"   Description: {result['description']}")
        print(f"   Expected: {result['expected']}")
        print(f"   Duration: {result['duration']}")
    
    print(f"\nOverall Results:")
    print(f"  • Successful strategies: {successful_count}/{len(results)}")
    print(f"  • Total execution time: {total_duration}")
    print(f"  • Results directory: {output_dir}")
    
    if successful_count > 0:
        print(f"\n📊 Next Steps for Analysis:")
        print("1. Compare validation accuracy curves across strategies")
        print("2. Analyze feature retention (compare to linear probe baseline)")
        print("3. Evaluate calibration performance (temperature scaling results)")
        print("4. Check for catastrophic forgetting indicators")
        print("5. Test on held-out datasets (PAPILLA, CHAKSU)")
        print()
        print("Key Questions to Answer:")
        print("• Does linear probe achieve reasonable accuracy (>75%)?")
        print("• Does gradual unfreezing improve without losing features?") 
        print("• Does Full LLRD achieve best performance with good calibration?")
        print("• Which strategy has the smoothest learning curves?")
    
    print(f"\n{'=' * 60}")
    print("STUDY COMPLETE - Ready for comparative analysis!")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
