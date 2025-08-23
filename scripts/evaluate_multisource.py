"""
Evaluation script for Multi-Source Domain Fine-Tuning Models.

This script loads pre-trained models and evaluates them on their corresponding test datasets,
calculating AUC with 95% CI, sensitivity/specificity at Youden's threshold, and ECE.
"""

import argparse
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from scipy import stats

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Subgroup analysis plots will not be generated.")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modular components
from src.data.external_loader import load_external_test_data
from src.data.loaders import load_airogs_data, assign_dataset_source
from src.data.utils import safe_collate
from src.models.classification.build_model import build_classifier_model
from src.evaluation.metrics import calculate_ece
from src.data.transforms import get_transforms
from src.data.datasets import GlaucomaSubgroupDataset
from src.utils.helpers import set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.*compile.*')


def calculate_youden_threshold(fpr, tpr, thresholds):
    """Calculate Youden's J statistic and optimal threshold."""
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_j = j_scores[optimal_idx]
    return optimal_threshold, optimal_j, optimal_idx


def bootstrap_auc(y_true, y_scores, n_bootstraps=1000, random_state=42):
    """Calculate AUC with 95% confidence interval using bootstrapping."""
    np.random.seed(random_state)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        # Bootstrap by sampling with replacement on the prediction indices
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            # Need at least one positive and one negative sample
            continue
        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Calculate 95% confidence interval
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    return np.mean(sorted_scores), confidence_lower, confidence_upper


class MultiSourceEvaluator:
    """Main class for evaluating pre-trained multi-source models."""
    
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(__name__)
        self.results = []
        
        # Dynamically set output directory based on TTA usage
        if args.use_tta and "multisource" in args.output_dir and "TTA" not in args.output_dir:
            # Change from evaluation_results/multisource to evaluation_results/multisource-TTA
            if args.output_dir.endswith("multisource"):
                args.output_dir = args.output_dir.replace("multisource", "multisource-TTA")
            elif "multisource" in args.output_dir:
                args.output_dir = args.output_dir.replace("multisource", "multisource-TTA")
            else:
                args.output_dir = os.path.join(args.output_dir, "multisource-TTA")
            
            self.logger.info(f"TTA enabled: Results will be saved to {args.output_dir}")
        elif args.use_tta:
            self.logger.info("TTA enabled for evaluation")
        
    def load_datasets(self):
        """Load all datasets for evaluation using the same logic as training script."""
        self.logger.info("Loading datasets...")
        
        # Use the proven external_loader approach with correct argument names
        external_datasets = load_external_test_data(
            smdg_metadata_file_raw=self.args.smdg_metadata_file_raw,
            smdg_image_dir_raw=self.args.smdg_image_dir_raw,
            chaksu_base_dir_eval=self.args.chaksu_base_dir,
            chaksu_decision_dir_raw=self.args.chaksu_decision_dir_raw,
            chaksu_metadata_dir_raw=self.args.chaksu_metadata_dir_raw,
            data_type=self.args.data_type,
            base_data_root=self.args.base_data_root,
            raw_dir_name='raw',
            processed_dir_name='processed',
            eval_papilla=self.args.use_papila,
            eval_oiaodir_test=False,
            eval_chaksu=self.args.use_chaksu,
            eval_acrima=self.args.use_acrima,
            eval_hygd=self.args.use_hygd,
            acrima_image_dir_raw=self.args.acrima_image_dir_raw,
            hygd_image_dir_raw=self.args.hygd_image_dir_raw,
            hygd_labels_file_raw=self.args.hygd_labels_file_raw
        )
        
        datasets = {}
        
        # Load SMDG-19 if requested and split into subdatasets
        if getattr(self.args, 'use_smdg', True):
            smdg_metadata_file = os.path.join(self.args.base_data_root, self.args.smdg_metadata_file_raw)
            smdg_image_dir = os.path.join(self.args.base_data_root, self.args.smdg_image_dir_raw)
            
            if os.path.exists(smdg_metadata_file):
                try:
                    df_smdg_all = pd.read_csv(smdg_metadata_file)
                    
                    # Filter out PAPILA to avoid overlap with external datasets
                    smdg_main = df_smdg_all[
                        ~df_smdg_all['names'].str.startswith('PAPILA', na=False)
                    ].copy()
                    
                    if not smdg_main.empty:
                        # Add image paths and filter existing files
                        smdg_main["image_path"] = smdg_main["names"].apply(
                            lambda name: os.path.join(smdg_image_dir, f"{name}.png")
                        )
                        smdg_main["file_exists"] = smdg_main["image_path"].apply(os.path.exists)
                        smdg_main = smdg_main[smdg_main["file_exists"]]
                        
                        # Clean data
                        smdg_main = smdg_main.dropna(subset=["types"])
                        smdg_main = smdg_main[smdg_main["types"].isin([0, 1])]
                        smdg_main["types"] = smdg_main["types"].astype(int)
                        smdg_main["label"] = smdg_main["types"]
                        
                        # Assign dataset sources
                        smdg_main["dataset_source"] = assign_dataset_source(smdg_main["names"])
                        
                        # Split into subdatasets and filter by sample count
                        smdg_subdatasets = smdg_main.groupby('dataset_source')
                        min_samples_per_subdataset = 100
                        
                        # Process subdatasets
                        valid_subdatasets = []
                        oia_odir_combined = []
                        
                        for subdataset_name, subdataset_df in smdg_subdatasets:
                            subdataset_size = len(subdataset_df)
                            
                            if subdataset_name == "SMDG_Unknown":
                                continue
                            
                            # Handle OIA-ODIR train and test - combine them
                            if subdataset_name in ["OIA-ODIR-train", "OIA-ODIR-test"]:
                                oia_odir_combined.append(subdataset_df.copy())
                                continue
                                
                            if subdataset_size >= min_samples_per_subdataset:
                                datasets[f'SMDG-{subdataset_name}'] = subdataset_df.copy()
                                valid_subdatasets.append(subdataset_name)
                        
                        # Combine OIA-ODIR train and test if we have them
                        if oia_odir_combined:
                            combined_oia_odir = pd.concat(oia_odir_combined, ignore_index=True)
                            if len(combined_oia_odir) >= min_samples_per_subdataset:
                                datasets['SMDG-OIA-ODIR'] = combined_oia_odir
                                valid_subdatasets.append('OIA-ODIR')
                        
                        self.logger.info(f"Loaded SMDG-19 subdatasets: {valid_subdatasets}")
                        
                except Exception as e:
                    self.logger.error(f"Error loading SMDG-19: {e}")
                    traceback.print_exc()
        
        # Add AIROGS using proven loader approach
        if self.args.use_airogs:
            try:
                class AirogsConfig:
                    def __init__(self, args):
                        self.airogs_label_file = args.airogs_label_file
                        self.airogs_image_dir = args.airogs_image_dir
                        self.airogs_num_rg_samples = args.airogs_num_rg_samples
                        self.airogs_num_nrg_samples = args.airogs_num_nrg_samples
                        self.use_airogs_cache = args.use_airogs_cache
                        self.seed = args.seed
                        self.data_type = args.data_type
                
                airogs_config = AirogsConfig(self.args)
                df_airogs = load_airogs_data(airogs_config)
                
                if not df_airogs.empty:
                    df_airogs["label"] = df_airogs["types"]
                    datasets['AIROGS'] = df_airogs
                    self.logger.info(f"Loaded AIROGS: {len(df_airogs)} samples")
            except Exception as e:
                self.logger.error(f"Error loading AIROGS: {e}")
        
        # Add external datasets from external_loader
        for dataset_name, dataset_df in external_datasets.items():
            if not dataset_df.empty and dataset_name not in datasets:
                # Ensure label column exists for compatibility
                if 'label' not in dataset_df.columns and 'types' in dataset_df.columns:
                    dataset_df['label'] = dataset_df['types']
                elif 'types' not in dataset_df.columns and 'label' in dataset_df.columns:
                    dataset_df['types'] = dataset_df['label']
                
                # Filter out invalid labels
                if 'label' in dataset_df.columns:
                    original_count = len(dataset_df)
                    dataset_df = dataset_df[dataset_df['label'].isin([0, 1])].copy()
                    filtered_count = len(dataset_df)
                    if filtered_count < original_count:
                        self.logger.info(f"Filtered {dataset_name}: removed {original_count - filtered_count} samples with invalid labels")
                
                if not dataset_df.empty:
                    datasets[dataset_name] = dataset_df
                    self.logger.info(f"Added external dataset {dataset_name}: {len(dataset_df)} samples")
        
        # Validate we have at least some datasets
        if not datasets:
            raise ValueError("No datasets loaded successfully")
            
        # Log dataset statistics
        for name, df in datasets.items():
            pos_count = len(df[df['label'] == 1])
            neg_count = len(df[df['label'] == 0])
            self.logger.info(f"Dataset {name}: {len(df)} samples (Pos: {pos_count}, Neg: {neg_count})")
        
        return datasets
        
    def get_model_configurations(self):
        """Get list of model configurations to test."""
        models = [
            {
                'name': 'VFM',
                'model_name': 'vit_base_patch16_224',
                'custom_weights_path': self.args.vfm_weights_path,
                'description': 'Vision Foundation Model (VFM)'
            }
        ]
        
        # Add additional models if specified
        if self.args.additional_models:
            for model_spec in self.args.additional_models:
                if ':' in model_spec:
                    name, model_name = model_spec.split(':', 1)
                    models.append({
                        'name': name,
                        'model_name': model_name,
                        'custom_weights_path': None,
                        'description': f'Additional model: {name}'
                    })
                    
        return models
        
    def evaluate_single_model(self, model_config, test_dataset_name, test_dataset):
        """Evaluate a single pre-trained model on its test dataset."""
        self.logger.info(f"Evaluating {model_config['name']} on {test_dataset_name}")
        
        # Check if fine-tuned model exists
        model_save_path = os.path.join(
            self.args.model_dir,
            f"{model_config['name']}_{test_dataset_name}_best.pth"
        )
        
        if not os.path.exists(model_save_path):
            self.logger.warning(f"No pre-trained model found at {model_save_path}. Skipping.")
            return None
            
        # Load metadata for subgroup analysis
        metadata = self.load_dataset_with_metadata(test_dataset_name, test_dataset)
            
        try:
            # Build model architecture
            model = build_classifier_model(
                model_name=model_config['model_name'],
                num_classes=self.args.num_classes,
                custom_weights_path=None
            )
            
            # Load pre-trained weights
            self.logger.info(f"Loading pre-trained weights from {model_save_path}")
            checkpoint = torch.load(model_save_path, map_location='cpu', weights_only=True)
            state_dict = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            
            load_result = model.load_state_dict(state_dict, strict=False)
            
            if load_result.missing_keys:
                self.logger.warning(f"Missing keys when loading weights: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                self.logger.info(f"Unexpected keys in weights: {load_result.unexpected_keys}")
            
            # Setup device and model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()
            
            # Get transforms
            try:
                _, eval_transforms = get_transforms(
                    self.args.image_size, 
                    model_config['model_name'], 
                    False  # use_data_augmentation=False for evaluation
                )
            except TypeError as e:
                self.logger.warning(f"Error with get_transforms parameters: {e}. Trying alternative approach...")
                # Fallback to basic transforms if function signature is different
                from torchvision import transforms
                eval_transforms = transforms.Compose([
                    transforms.Resize((self.args.image_size, self.args.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            # Create test dataset and loader
            test_dataset_obj = GlaucomaSubgroupDataset(test_dataset, transform=eval_transforms)
            test_loader = DataLoader(
                test_dataset_obj, 
                batch_size=self.args.batch_size, 
                shuffle=False, 
                num_workers=self.args.num_workers,
                collate_fn=safe_collate, 
                pin_memory=True
            )
            
            # Evaluate model
            all_predictions = []
            all_probabilities = []
            all_targets = []
            
            self.logger.info(f"Running inference on {len(test_dataset)} samples{'with TTA' if self.args.use_tta else ''}...")
            
            with torch.no_grad():
                batch_count = 0
                total_batches = len(test_loader)
                
                for batch_data in test_loader:
                    if len(batch_data) == 3:
                        images, targets, _ = batch_data
                    else:
                        images, targets = batch_data
                        
                    images, targets = images.to(device), targets.to(device).long()
                    
                    # Apply Test-Time Augmentation if enabled
                    if self.args.use_tta:
                        if batch_count == 0:  # Only log on first batch to avoid spam
                            self.logger.info(f"🔄 Applying TTA to {images.shape[0]} images in batch...")
                        outputs = self._apply_tta(model, images)
                    else:
                        outputs = model(images)
                    
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
                    batch_count += 1
                    if self.args.use_tta and batch_count % 10 == 0:  # Progress update every 10 batches
                        self.logger.info(f"🔄 TTA Progress: {batch_count}/{total_batches} batches processed")
            
            # Convert to numpy arrays
            y_true = np.array(all_targets)
            y_scores = np.array(all_probabilities)
            y_pred = np.array(all_predictions)
            
            # Calculate ROC curve and Youden's threshold
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            optimal_threshold, youden_j, optimal_idx = calculate_youden_threshold(fpr, tpr, thresholds)
            
            # Calculate metrics
            auc = roc_auc_score(y_true, y_scores)
            auc_mean, auc_ci_lower, auc_ci_upper = bootstrap_auc(y_true, y_scores, 
                                                                n_bootstraps=self.args.n_bootstraps, 
                                                                random_state=self.args.seed)
            
            # Calculate sensitivity and specificity at Youden's threshold
            y_pred_youden = (y_scores >= optimal_threshold).astype(int)
            
            # Calculate confusion matrix components
            tn = np.sum((y_true == 0) & (y_pred_youden == 0))
            fp = np.sum((y_true == 0) & (y_pred_youden == 1))
            fn = np.sum((y_true == 1) & (y_pred_youden == 0))
            tp = np.sum((y_true == 1) & (y_pred_youden == 1))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Calculate ECE
            ece, _, _, _, _ = calculate_ece(y_true, y_scores)
            
            # Calculate basic accuracy for reference
            accuracy = accuracy_score(y_true, y_pred_youden)
            
            # Log results
            self.logger.info(f"Results for {model_config['name']} on {test_dataset_name}:")
            self.logger.info(f"  AUC: {auc:.4f} (95% CI: {auc_ci_lower:.4f}-{auc_ci_upper:.4f})")
            self.logger.info(f"  Youden's threshold: {optimal_threshold:.4f}")
            self.logger.info(f"  Sensitivity: {sensitivity:.4f}")
            self.logger.info(f"  Specificity: {specificity:.4f}")
            self.logger.info(f"  Accuracy: {accuracy:.4f}")
            self.logger.info(f"  ECE: {ece:.4f}")
            
            # Perform subgroup analysis if metadata is available and enabled
            subgroup_results_age = None
            subgroup_results_sex = None
            
            if metadata and self.args.enable_subgroup_analysis:
                self.logger.info(f"Performing subgroup analysis for {test_dataset_name}")
                
                # Age-based subgroup analysis
                if 'age' in metadata:
                    age_groups = self.create_age_groups(metadata['age'], threshold=self.args.age_threshold)
                    valid_age_mask = ~age_groups.isin(['Unknown'])
                    
                    if np.sum(valid_age_mask) > self.args.min_subgroup_size:  # Need sufficient samples
                        subgroup_results_age = self.calculate_subgroup_metrics(
                            y_true[valid_age_mask], y_scores[valid_age_mask], 
                            age_groups[valid_age_mask], 'age'
                        )
                        self.logger.info(f"Age subgroup analysis completed for {len(subgroup_results_age)} groups")
                
                # Sex-based subgroup analysis
                if 'sex' in metadata:
                    sex_groups = metadata['sex']
                    valid_sex_mask = ~sex_groups.isin(['Unknown', None]) & ~sex_groups.isna()
                    
                    if np.sum(valid_sex_mask) > self.args.min_subgroup_size:  # Need sufficient samples
                        subgroup_results_sex = self.calculate_subgroup_metrics(
                            y_true[valid_sex_mask], y_scores[valid_sex_mask], 
                            sex_groups[valid_sex_mask], 'sex'
                        )
                        self.logger.info(f"Sex subgroup analysis completed for {len(subgroup_results_sex)} groups")
                
                # Generate subgroup analysis plot
                if subgroup_results_age is not None or subgroup_results_sex is not None:
                    from pathlib import Path
                    output_dir = Path(self.args.output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    self.plot_subgroup_analysis(
                        test_dataset_name, model_config['name'], 
                        subgroup_results_age, subgroup_results_sex, output_dir
                    )
            
            return {
                'model_name': model_config['name'],
                'test_dataset': test_dataset_name,
                'n_samples': len(y_true),
                'n_positive': int(np.sum(y_true)),
                'n_negative': int(len(y_true) - np.sum(y_true)),
                'auc': auc,
                'auc_ci_lower': auc_ci_lower,
                'auc_ci_upper': auc_ci_upper,
                'youden_threshold': optimal_threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'accuracy': accuracy,
                'ece': ece,
                'youden_j': youden_j,
                'has_age_metadata': 'age' in metadata if metadata else False,
                'has_sex_metadata': 'sex' in metadata if metadata else False,
                'subgroup_results_age': subgroup_results_age.to_dict('records') if subgroup_results_age is not None else None,
                'subgroup_results_sex': subgroup_results_sex.to_dict('records') if subgroup_results_sex is not None else None,
                'tta_enabled': self.args.use_tta
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate {model_config['name']} on {test_dataset_name}: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def run_evaluation(self):
        """Run evaluation on all available pre-trained models."""
        # Load datasets
        datasets = self.load_datasets()
        
        # Get model configurations
        models = self.get_model_configurations()
        
        # Run evaluations
        total_evaluations = len(datasets) * len(models)
        completed = 0
        
        for dataset_name, dataset_df in datasets.items():
            for model_config in models:
                result = self.evaluate_single_model(model_config, dataset_name, dataset_df)
                if result is not None:
                    self.results.append(result)
                
                completed += 1
                progress_pct = (completed / total_evaluations) * 100
                self.logger.info(f"Progress: {completed}/{total_evaluations} ({progress_pct:.1f}%)")
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate and save summary table."""
        if not self.results:
            self.logger.warning("No results to summarize.")
            return
        
        # Create DataFrame
        df_results = pd.DataFrame(self.results)
        
        # Sort by test dataset for better readability
        df_results = df_results.sort_values('test_dataset')
        
        # Create formatted summary table
        summary_data = []
        for _, row in df_results.iterrows():
            model_display = row['model_name']
            if row.get('tta_enabled', False):
                model_display += " (TTA)"
                
            summary_data.append({
                'Test Dataset': row['test_dataset'],
                'Model': model_display,
                'N': f"{row['n_samples']} ({row['n_positive']}/{row['n_negative']})",
                'AUC (95% CI)': f"{row['auc']:.3f} ({row['auc_ci_lower']:.3f}-{row['auc_ci_upper']:.3f})",
                'Sensitivity': f"{row['sensitivity']:.3f}",
                'Specificity': f"{row['specificity']:.3f}",
                'ECE': f"{row['ece']:.3f}",
                'Youden Threshold': f"{row['youden_threshold']:.3f}",
                'Accuracy': f"{row['accuracy']:.3f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Save detailed results
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add TTA info to filename if enabled
        results_filename = 'detailed_evaluation_results'
        summary_filename = 'evaluation_summary_table'
        if self.args.use_tta:
            results_filename += '_TTA'
            summary_filename += '_TTA'
        
        detailed_results_path = output_dir / f'{results_filename}.csv'
        df_results.to_csv(detailed_results_path, index=False)
        self.logger.info(f"Detailed results saved to {detailed_results_path}")
        
        # Save summary table
        summary_path = output_dir / f'{summary_filename}.csv'
        df_summary.to_csv(summary_path, index=False)
        self.logger.info(f"Summary table saved to {summary_path}")
        
        if self.args.use_tta:
            self.logger.info("🔄 All results include Test-Time Augmentation (TTA)")
        
        # Print summary to console
        print("\n" + "="*120)
        if self.args.use_tta:
            print("MULTI-SOURCE DOMAIN ADAPTATION EVALUATION SUMMARY - 🔄 WITH TEST-TIME AUGMENTATION")
        else:
            print("MULTI-SOURCE DOMAIN ADAPTATION EVALUATION SUMMARY")
        print("="*120)
        print(df_summary.to_string(index=False))
        print("="*120)
        
        # Print additional statistics
        print(f"\nEvaluation completed for {len(df_results)} model-dataset combinations")
        if self.args.use_tta:
            print("🔄 Results obtained with Test-Time Augmentation (TTA)")
            print("   - Multiple augmented versions of each image were averaged")
            print("   - Performance may be improved compared to single-image inference")
        if len(df_results) > 0:
            print(f"Mean AUC: {df_results['auc'].mean():.3f} ± {df_results['auc'].std():.3f}")
            print(f"Mean Sensitivity: {df_results['sensitivity'].mean():.3f} ± {df_results['sensitivity'].std():.3f}")
            print(f"Mean Specificity: {df_results['specificity'].mean():.3f} ± {df_results['specificity'].std():.3f}")
            print(f"Mean ECE: {df_results['ece'].mean():.3f} ± {df_results['ece'].std():.3f}")
            
            # Print subgroup analysis summary
            datasets_with_metadata = df_results[
                (df_results['has_age_metadata'] == True) | 
                (df_results['has_sex_metadata'] == True)
            ]
            if len(datasets_with_metadata) > 0:
                print(f"\nSubgroup analysis available for {len(datasets_with_metadata)} datasets:")
                for _, row in datasets_with_metadata.iterrows():
                    metadata_types = []
                    if row['has_age_metadata']:
                        metadata_types.append('age')
                    if row['has_sex_metadata']:
                        metadata_types.append('sex')
                    print(f"  - {row['test_dataset']}: {', '.join(metadata_types)} metadata")
                print("Subgroup analysis plots saved in the output directory.")
    
    def load_dataset_with_metadata(self, dataset_name, dataset_df):
        """Load dataset and extract additional metadata if available."""
        metadata = {}
        
        # For PAPILLA dataset, try to extract age and sex information
        if 'PAPILLA' in dataset_name.upper():
            metadata = self.load_papilla_metadata(dataset_df)
        
        # Try to infer metadata from image paths or filenames for other datasets
        if not metadata and 'image_path' in dataset_df.columns:
            # Extract any metadata encoded in filenames
            filenames = dataset_df['image_path'].apply(lambda x: str(x).split('/')[-1] if isinstance(x, str) else '')
            
            # Look for age patterns in filenames (common formats: age_XX, _XX_years, etc.)
            age_pattern = r'(?:age|Age|AGE)[_\-]?(\d{1,3})'
            ages = filenames.str.extract(age_pattern, expand=False)
            if not ages.isna().all():
                metadata['age'] = pd.to_numeric(ages, errors='coerce')
            
            # Look for sex patterns in filenames (M/F, Male/Female, etc.)
            sex_pattern = r'(?:sex|Sex|SEX|gender|Gender)[_\-]?([MFmf]|Male|Female|male|female)'
            sexes = filenames.str.extract(sex_pattern, expand=False)
            if not sexes.isna().all():
                # Standardize sex values
                sex_mapping = {'M': 'Male', 'm': 'Male', 'Male': 'Male', 'male': 'Male',
                              'F': 'Female', 'f': 'Female', 'Female': 'Female', 'female': 'Female'}
                metadata['sex'] = sexes.map(sex_mapping)
        
        return metadata
    
    def load_papilla_metadata(self, dataset_df):
        """Specialized function to load PAPILLA metadata from various possible sources."""
        metadata = {}
        
        # Check for direct metadata columns
        age_columns = ['age', 'Age', 'AGE', 'patient_age', 'Age_at_diagnosis']
        sex_columns = ['sex', 'Sex', 'SEX', 'gender', 'Gender', 'patient_sex', 'patient_gender']
        
        # Try to find age column
        for col in age_columns:
            if col in dataset_df.columns:
                ages = pd.to_numeric(dataset_df[col], errors='coerce')
                if not ages.isna().all():
                    metadata['age'] = ages
                    self.logger.info(f"Found age data in column '{col}': {len(ages.dropna())} valid ages")
                    break
        
        # Try to find sex column
        for col in sex_columns:
            if col in dataset_df.columns:
                sexes = dataset_df[col].astype(str).str.strip()
                # Standardize sex values
                sex_mapping = {
                    'M': 'Male', 'm': 'Male', 'Male': 'Male', 'male': 'Male', '1': 'Male',
                    'F': 'Female', 'f': 'Female', 'Female': 'Female', 'female': 'Female', '0': 'Female'
                }
                sexes_mapped = sexes.map(sex_mapping)
                if not sexes_mapped.isna().all():
                    metadata['sex'] = sexes_mapped
                    self.logger.info(f"Found sex data in column '{col}': {sexes_mapped.value_counts().to_dict()}")
                    break
        
        # If no direct columns found, try to extract from filenames
        if not metadata and 'image_path' in dataset_df.columns:
            self.logger.info("No direct metadata columns found, trying to extract from filenames...")
            
            # Extract filenames
            filenames = dataset_df['image_path'].apply(
                lambda x: str(x).split('/')[-1].split('\\')[-1] if isinstance(x, str) else ''
            )
            
            # Try various patterns for age extraction
            age_patterns = [
                r'(?:age|Age|AGE)[_\-]?(\d{1,3})',
                r'(\d{2,3})(?:y|Y|years|Years)',
                r'_(\d{2,3})_',
                r'-(\d{2,3})-'
            ]
            
            for pattern in age_patterns:
                ages = filenames.str.extract(pattern, expand=False)
                ages_numeric = pd.to_numeric(ages, errors='coerce')
                # Filter reasonable age range
                ages_filtered = ages_numeric[(ages_numeric >= 18) & (ages_numeric <= 120)]
                if len(ages_filtered.dropna()) > 10:  # Need reasonable number of valid ages
                    metadata['age'] = ages_numeric
                    self.logger.info(f"Extracted age from filenames using pattern '{pattern}': {len(ages_filtered)} valid ages")
                    break
            
            # Try various patterns for sex extraction
            sex_patterns = [
                r'(?:sex|Sex|SEX|gender|Gender)[_\-]?([MFmf]|Male|Female|male|female)',
                r'([MF])_',
                r'_([MF])_',
                r'-([MF])-'
            ]
            
            for pattern in sex_patterns:
                sexes = filenames.str.extract(pattern, expand=False)
                if not sexes.isna().all():
                    sex_mapping = {
                        'M': 'Male', 'm': 'Male', 'Male': 'Male', 'male': 'Male',
                        'F': 'Female', 'f': 'Female', 'Female': 'Female', 'female': 'Female'
                    }
                    sexes_mapped = sexes.map(sex_mapping)
                    if len(sexes_mapped.dropna()) > 10:  # Need reasonable number of valid sexes
                        metadata['sex'] = sexes_mapped
                        self.logger.info(f"Extracted sex from filenames using pattern '{pattern}': {sexes_mapped.value_counts().to_dict()}")
                        break
        
        return metadata
    
    def create_age_groups(self, ages, threshold=65):
        """Create age groups based on threshold."""
        age_groups = pd.Series(['Unknown'] * len(ages), index=ages.index)
        valid_ages = pd.to_numeric(ages, errors='coerce')
        
        age_groups[valid_ages < threshold] = f'Age < {threshold}'
        age_groups[valid_ages >= threshold] = f'Age ≥ {threshold}'
        
        return age_groups
    
    def calculate_subgroup_metrics(self, y_true, y_scores, subgroup_labels, subgroup_name):
        """Calculate metrics for each subgroup."""
        results = []
        
        # Overall metrics first
        overall_auc = roc_auc_score(y_true, y_scores)
        overall_fpr, overall_tpr, overall_thresholds = roc_curve(y_true, y_scores)
        overall_threshold, _, _ = calculate_youden_threshold(overall_fpr, overall_tpr, overall_thresholds)
        overall_pred = (y_scores >= overall_threshold).astype(int)
        
        # Calculate confusion matrix for overall
        overall_tn = np.sum((y_true == 0) & (overall_pred == 0))
        overall_fp = np.sum((y_true == 0) & (overall_pred == 1))
        overall_fn = np.sum((y_true == 1) & (overall_pred == 0))
        overall_tp = np.sum((y_true == 1) & (overall_pred == 1))
        
        overall_sens = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_spec = overall_tn / (overall_tn + overall_fp) if (overall_tn + overall_fp) > 0 else 0
        
        # Bootstrap confidence intervals for overall AUC
        overall_auc_mean, overall_ci_lower, overall_ci_upper = bootstrap_auc(
            y_true, y_scores, n_bootstraps=500, random_state=self.args.seed
        )
        
        results.append({
            'subgroup': 'Overall',
            'n_total': len(y_true),
            'n_positive': int(np.sum(y_true)),
            'n_negative': int(len(y_true) - np.sum(y_true)),
            'auc': overall_auc,
            'auc_ci_lower': overall_ci_lower,
            'auc_ci_upper': overall_ci_upper,
            'sensitivity': overall_sens,
            'specificity': overall_spec,
            'threshold': overall_threshold
        })
        
        # Subgroup metrics
        unique_groups = subgroup_labels.unique()
        for group in unique_groups:
            if pd.isna(group) or group == 'Unknown':
                continue
                
            mask = subgroup_labels == group
            if np.sum(mask) < 10:  # Skip groups with too few samples
                continue
                
            group_y_true = y_true[mask]
            group_y_scores = y_scores[mask]
            
            # Check if we have both classes in the subgroup
            if len(np.unique(group_y_true)) < 2:
                continue
                
            try:
                group_auc = roc_auc_score(group_y_true, group_y_scores)
                group_fpr, group_tpr, group_thresholds = roc_curve(group_y_true, group_y_scores)
                group_threshold, _, _ = calculate_youden_threshold(group_fpr, group_tpr, group_thresholds)
                group_pred = (group_y_scores >= group_threshold).astype(int)
                
                # Calculate confusion matrix for subgroup
                group_tn = np.sum((group_y_true == 0) & (group_pred == 0))
                group_fp = np.sum((group_y_true == 0) & (group_pred == 1))
                group_fn = np.sum((group_y_true == 1) & (group_pred == 0))
                group_tp = np.sum((group_y_true == 1) & (group_pred == 1))
                
                group_sens = group_tp / (group_tp + group_fn) if (group_tp + group_fn) > 0 else 0
                group_spec = group_tn / (group_tn + group_fp) if (group_tn + group_fp) > 0 else 0
                
                # Bootstrap confidence intervals for subgroup AUC
                group_auc_mean, group_ci_lower, group_ci_upper = bootstrap_auc(
                    group_y_true, group_y_scores, n_bootstraps=500, random_state=self.args.seed
                )
                
                results.append({
                    'subgroup': group,
                    'n_total': len(group_y_true),
                    'n_positive': int(np.sum(group_y_true)),
                    'n_negative': int(len(group_y_true) - np.sum(group_y_true)),
                    'auc': group_auc,
                    'auc_ci_lower': group_ci_lower,
                    'auc_ci_upper': group_ci_upper,
                    'sensitivity': group_sens,
                    'specificity': group_spec,
                    'threshold': group_threshold
                })
                
            except Exception as e:
                self.logger.warning(f"Could not calculate metrics for subgroup {group}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def plot_subgroup_analysis(self, dataset_name, model_name, subgroup_results_age, subgroup_results_sex, output_dir):
        """Create subgroup analysis plot similar to the provided example."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("matplotlib not available. Skipping subgroup analysis plots.")
            return
            
        try:
            fig, axes = plt.subplots(1, 4, figsize=(16, 6))
            
            # Define colors for different metrics
            colors = {
                'Overall': 'black',
                'Male': '#D946EF',  # Purple/magenta
                'Female': '#D946EF',  # Purple/magenta  
                'Age < 65': '#10B981',  # Green
                'Age ≥ 65': '#10B981'   # Green
            }
            
            # Define markers
            markers = {
                'Overall': 'o',
                'Male': 's',
                'Female': 's',
                'Age < 65': '^',
                'Age ≥ 65': '^'
            }
            
            # Plot 1: AUC
            ax1 = axes[0]
            y_pos = 0
            
            # Combine all results for plotting
            all_results = []
            if subgroup_results_age is not None:
                age_results = subgroup_results_age.copy()
                age_results['category'] = 'Age'
                all_results.append(age_results)
            if subgroup_results_sex is not None:
                sex_results = subgroup_results_sex.copy()
                sex_results['category'] = 'Sex'
                all_results.append(sex_results)
            
            if not all_results:
                return
                
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Group by category and plot
            for category in ['Overall', 'Sex', 'Age']:
                if category == 'Overall':
                    # Plot overall result once
                    overall_row = combined_results[combined_results['subgroup'] == 'Overall'].iloc[0]
                    ax1.scatter([overall_row['auc']], [y_pos], 
                              color=colors['Overall'], marker=markers['Overall'], s=100, zorder=3)
                    ax1.plot([overall_row['auc_ci_lower'], overall_row['auc_ci_upper']], 
                            [y_pos, y_pos], color=colors['Overall'], linewidth=2, zorder=2)
                    ax1.text(-0.05, y_pos, f"Overall (n = {overall_row['n_total']})", 
                            ha='right', va='center', transform=ax1.get_yaxis_transform())
                    y_pos += 1
                else:
                    category_data = combined_results[
                        (combined_results['category'] == category) & 
                        (combined_results['subgroup'] != 'Overall')
                    ]
                    
                    if len(category_data) == 0:
                        continue
                        
                    # Add category header
                    ax1.text(-0.05, y_pos, category, ha='right', va='center', 
                            transform=ax1.get_yaxis_transform(), weight='bold')
                    y_pos += 0.5
                    
                    for _, row in category_data.iterrows():
                        color = colors.get(row['subgroup'], 'gray')
                        marker = markers.get(row['subgroup'], 'o')
                        
                        ax1.scatter([row['auc']], [y_pos], 
                                  color=color, marker=marker, s=100, zorder=3)
                        ax1.plot([row['auc_ci_lower'], row['auc_ci_upper']], 
                                [y_pos, y_pos], color=color, linewidth=2, zorder=2)
                        ax1.text(-0.05, y_pos, f"  {row['subgroup']} (n = {row['n_total']})", 
                                ha='right', va='center', transform=ax1.get_yaxis_transform())
                        y_pos += 1
                    
                    y_pos += 0.5  # Extra space between categories
            
            ax1.set_xlim(0.5, 1.0)
            ax1.set_ylim(-0.5, y_pos)
            ax1.set_xlabel('Subgroup\nAUROC')
            ax1.set_title('AUROC by Subgroup')
            ax1.grid(True, alpha=0.3)
            ax1.invert_yaxis()
            
            # Plot 2: Sensitivity
            ax2 = axes[1]
            y_pos = 0
            
            for category in ['Overall', 'Sex', 'Age']:
                if category == 'Overall':
                    overall_row = combined_results[combined_results['subgroup'] == 'Overall'].iloc[0]
                    ax2.scatter([overall_row['sensitivity']], [y_pos], 
                              color=colors['Overall'], marker=markers['Overall'], s=100, zorder=3)
                    y_pos += 1
                else:
                    category_data = combined_results[
                        (combined_results['category'] == category) & 
                        (combined_results['subgroup'] != 'Overall')
                    ]
                    
                    if len(category_data) == 0:
                        continue
                        
                    y_pos += 0.5
                    
                    for _, row in category_data.iterrows():
                        color = colors.get(row['subgroup'], 'gray')
                        marker = markers.get(row['subgroup'], 'o')
                        
                        ax2.scatter([row['sensitivity']], [y_pos], 
                                  color=color, marker=marker, s=100, zorder=3)
                        y_pos += 1
                    
                    y_pos += 0.5
            
            ax2.set_xlim(0.5, 1.0)
            ax2.set_ylim(-0.5, y_pos)
            ax2.set_xlabel('Subgroup\nsensitivity')
            ax2.set_title('Sensitivity by Subgroup')
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()
            
            # Plot 3: Specificity
            ax3 = axes[2]
            y_pos = 0
            
            for category in ['Overall', 'Sex', 'Age']:
                if category == 'Overall':
                    overall_row = combined_results[combined_results['subgroup'] == 'Overall'].iloc[0]
                    ax3.scatter([overall_row['specificity']], [y_pos], 
                              color=colors['Overall'], marker=markers['Overall'], s=100, zorder=3)
                    y_pos += 1
                else:
                    category_data = combined_results[
                        (combined_results['category'] == category) & 
                        (combined_results['subgroup'] != 'Overall')
                    ]
                    
                    if len(category_data) == 0:
                        continue
                        
                    y_pos += 0.5
                    
                    for _, row in category_data.iterrows():
                        color = colors.get(row['subgroup'], 'gray')
                        marker = markers.get(row['subgroup'], 'o')
                        
                        ax3.scatter([row['specificity']], [y_pos], 
                                  color=color, marker=marker, s=100, zorder=3)
                        y_pos += 1
                    
                    y_pos += 0.5
            
            ax3.set_xlim(0.0, 1.0)
            ax3.set_ylim(-0.5, y_pos)
            ax3.set_xlabel('Subgroup\nspecificity')
            ax3.set_title('Specificity by Subgroup')
            ax3.grid(True, alpha=0.3)
            ax3.invert_yaxis()
            
            # Plot 4: Sample counts
            ax4 = axes[3]
            y_pos = 0
            
            for category in ['Overall', 'Sex', 'Age']:
                if category == 'Overall':
                    overall_row = combined_results[combined_results['subgroup'] == 'Overall'].iloc[0]
                    # Plot stacked bar: green for non-glaucoma, purple for glaucoma
                    ax4.barh(y_pos, overall_row['n_negative'], color='#10B981', height=0.6)
                    ax4.barh(y_pos, overall_row['n_positive'], left=overall_row['n_negative'], 
                            color='#D946EF', height=0.6)
                    y_pos += 1
                else:
                    category_data = combined_results[
                        (combined_results['category'] == category) & 
                        (combined_results['subgroup'] != 'Overall')
                    ]
                    
                    if len(category_data) == 0:
                        continue
                        
                    y_pos += 0.5
                    
                    for _, row in category_data.iterrows():
                        ax4.barh(y_pos, row['n_negative'], color='#10B981', height=0.6)
                        ax4.barh(y_pos, row['n_positive'], left=row['n_negative'], 
                               color='#D946EF', height=0.6)
                        y_pos += 1
                    
                    y_pos += 0.5
            
            ax4.set_ylim(-0.5, y_pos)
            ax4.set_xlabel('Number of SLE (green) and non-\nSLE (purple) in each subgroup')
            ax4.set_title('Sample Distribution')
            ax4.invert_yaxis()
            
            # Add overall title
            plt.suptitle(f'Subgroup Analysis: {model_name} on {dataset_name}', fontsize=16, y=1.02)
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / f'subgroup_analysis_{model_name}_{dataset_name}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            self.logger.info(f"Subgroup analysis plot saved to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create subgroup analysis plot: {e}")
            import traceback
            traceback.print_exc()

    def _apply_tta(self, model, images):
        """Apply Test-Time Augmentation for more robust predictions."""
        
        batch_size = images.size(0)
        tta_predictions = []
        
        # Original image
        outputs = model(images)
        tta_predictions.append(torch.softmax(outputs, dim=1))
        
        # Horizontal flip
        flipped_images = TF.hflip(images)
        flipped_outputs = model(flipped_images)
        tta_predictions.append(torch.softmax(flipped_outputs, dim=1))
        
        # Rotation augmentations (small rotations that preserve medical relevance)
        for angle in [-5, 5]:  # Small rotations to avoid losing medical information
            rotated_images = TF.rotate(images, angle)
            rotated_outputs = model(rotated_images)
            tta_predictions.append(torch.softmax(rotated_outputs, dim=1))
        
        # Brightness adjustments (common in medical imaging)
        for brightness_factor in [0.9, 1.1]:
            bright_images = TF.adjust_brightness(images, brightness_factor)
            bright_outputs = model(bright_images)
            tta_predictions.append(torch.softmax(bright_outputs, dim=1))
        
        # Contrast adjustments
        for contrast_factor in [0.9, 1.1]:
            contrast_images = TF.adjust_contrast(images, contrast_factor)
            contrast_outputs = model(contrast_images)
            tta_predictions.append(torch.softmax(contrast_outputs, dim=1))
        
        # Average all predictions
        avg_probabilities = torch.stack(tta_predictions).mean(dim=0)
        
        # Convert back to logits for consistency with rest of code
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        avg_probabilities = torch.clamp(avg_probabilities, epsilon, 1 - epsilon)
        avg_logits = torch.log(avg_probabilities / (1 - avg_probabilities + epsilon))
        
        return avg_logits


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluation script for Multi-Source Domain Fine-Tuning Models"
    )
    
    # Data arguments (same as training script)
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data_type', type=str, default='raw', 
                           choices=['raw', 'processed'],
                           help="Target type of image data ('raw' or 'processed')")
    data_group.add_argument('--base_data_root', type=str, default=r'D:\glaucoma\data',
                           help="Absolute base root directory for all datasets")
    
    # SMDG/PAPILLA dataset
    data_group.add_argument('--smdg_metadata_file_raw', type=str, 
                           default=os.path.join('raw','SMDG-19','metadata - standardized.csv'))
    data_group.add_argument('--smdg_image_dir_raw', type=str, 
                           default=os.path.join('raw','SMDG-19','full-fundus','full-fundus'))
    
    # CHAKSU dataset
    data_group.add_argument('--chaksu_base_dir', type=str, 
                           default=os.path.join('raw','Chaksu','Train','Train','1.0_Original_Fundus_Images'))
    data_group.add_argument('--chaksu_decision_dir_raw', type=str, 
                           default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision'))
    data_group.add_argument('--chaksu_metadata_dir_raw', type=str, 
                           default=os.path.join('raw','Chaksu','Train','Train','6.0_Glaucoma_Decision','Majority'))
    
    # AIROGS dataset configuration
    data_group.add_argument('--airogs_label_file', type=str, 
                           default=r'D:\glaucoma\data\raw\AIROGS\train_labels.csv',
                           help="Path to AIROGS training labels CSV file")
    data_group.add_argument('--airogs_image_dir', type=str, 
                           default=r'D:\glaucoma\data\raw\AIROGS\img',
                           help="Path to AIROGS image directory")
    data_group.add_argument('--airogs_num_rg_samples', type=int, default=3000,
                           help="Number of RG (glaucoma) samples to load from AIROGS")
    data_group.add_argument('--airogs_num_nrg_samples', type=int, default=3000,
                           help="Number of NRG (normal) samples to load from AIROGS")
    data_group.add_argument('--use_airogs_cache', action='store_true', default=True,
                           help="Use cached AIROGS manifest if available")
    
    # Additional external datasets
    data_group.add_argument('--acrima_image_dir_raw', type=str, 
                           default=os.path.join('raw','ACRIMA','Database','Images'))
    data_group.add_argument('--hygd_image_dir_raw', type=str, 
                           default=os.path.join('raw','HYGD','HYGD','Images'))
    data_group.add_argument('--hygd_labels_file_raw', type=str, 
                           default=os.path.join('raw','HYGD','HYGD','Labels.csv'))
    
    # Dataset selection flags
    data_group.add_argument('--use_smdg', action='store_true', default=True)
    data_group.add_argument('--use_chaksu', action='store_true', default=True)
    data_group.add_argument('--use_airogs', action='store_true', default=True)
    data_group.add_argument('--use_acrima', action='store_true', default=True)
    data_group.add_argument('--use_hygd', action='store_true', default=True)
    data_group.add_argument('--use_papila', action='store_true', default=True)
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--vfm_weights_path', type=str,
                            default=os.path.join(os.path.dirname(__file__), '..', 'models', 'VFM_Fundus_weights.pth'))
    model_group.add_argument('--additional_models', nargs='*', default=[],
                            help='Additional models in format "name:model_name"')
    model_group.add_argument('--num_classes', type=int, default=2)
    model_group.add_argument('--image_size', type=int, default=224)
    
    # Evaluation arguments
    eval_group = parser.add_argument_group('Evaluation Configuration')
    eval_group.add_argument('--model_dir', type=str, default='experiments/multisource',
                           help='Directory containing pre-trained model weights')
    eval_group.add_argument('--output_dir', type=str, default='evaluation_results',
                           help='Directory to save evaluation results')
    eval_group.add_argument('--batch_size', type=int, default=32,
                           help='Batch size for evaluation')
    eval_group.add_argument('--n_bootstraps', type=int, default=1000,
                           help='Number of bootstrap iterations for AUC confidence intervals')
    eval_group.add_argument('--age_threshold', type=int, default=65,
                           help='Age threshold for creating age groups (default: 65)')
    eval_group.add_argument('--enable_subgroup_analysis', action='store_true', default=True,
                           help='Enable subgroup analysis for datasets with metadata')
    eval_group.add_argument('--min_subgroup_size', type=int, default=20,
                           help='Minimum sample size required for subgroup analysis')
    
    # Domain adaptation for evaluation
    eval_group.add_argument('--use_tta', action='store_true', default=False,
                           help='Apply Test-Time Augmentation during evaluation for more robust predictions')
    
    # System arguments
    sys_group = parser.add_argument_group('System Configuration')
    sys_group.add_argument('--seed', type=int, default=42)
    sys_group.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Automatically adjust output directory for TTA
    if args.use_tta:
        if args.output_dir == 'evaluation_results':
            args.output_dir = os.path.join('evaluation_results', 'multisource-TTA')
        elif 'multisource' in args.output_dir and 'TTA' not in args.output_dir:
            args.output_dir = args.output_dir.replace('multisource', 'multisource-TTA')
    else:
        if args.output_dir == 'evaluation_results':
            args.output_dir = os.path.join('evaluation_results', 'multisource')
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Print prominent TTA message
    if args.use_tta:
        print("\n" + "="*80)
        print("🔄 TEST-TIME AUGMENTATION (TTA) ENABLED")
        print("="*80)
        print("📋 TTA Configuration:")
        print("   • Horizontal flips: YES")
        print("   • Small rotations (-5° to +5°): YES")
        print("   • Brightness adjustments (±10%): YES") 
        print("   • Contrast adjustments (±10%): YES")
        print("   • Prediction averaging: YES")
        print("⚠️  Note: TTA will significantly increase evaluation time")
        print("📁 Results will be saved to TTA-specific directory")
        print("="*80 + "\n")
    
    try:
        # Create and run evaluator
        evaluator = MultiSourceEvaluator(args)
        evaluator.run_evaluation()
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("Evaluation interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
