"""
Experiment management utilities for multi-source domain fine-tuning.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentCheckpoint:
    """
    Handles experiment checkpointing and resumption for robust experiments.
    """
    
    def __init__(self, checkpoint_dir: str, experiment_name: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.checkpoint_file = self.checkpoint_dir / f"{experiment_name}_checkpoint.pkl"
        self.state = {
            'completed_experiments': set(),
            'results': [],
            'start_time': datetime.now().isoformat(),
            'last_update': None
        }
        
    def load_checkpoint(self) -> bool:
        """Load existing checkpoint if available."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    self.state = pickle.load(f)
                logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
                logger.info(f"Previously completed: {len(self.state['completed_experiments'])} experiments")
                return True
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                return False
        return False
        
    def save_checkpoint(self):
        """Save current experiment state."""
        self.state['last_update'] = datetime.now().isoformat()
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(self.state, f)
            logger.debug(f"Checkpoint saved to {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    def is_completed(self, experiment_key: str) -> bool:
        """Check if an experiment has been completed."""
        return experiment_key in self.state['completed_experiments']
        
    def mark_completed(self, experiment_key: str, result: Dict[str, Any]):
        """Mark an experiment as completed and save result."""
        self.state['completed_experiments'].add(experiment_key)
        self.state['results'].append(result)
        self.save_checkpoint()
        
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all completed experiment results."""
        return self.state['results']
        
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get experiment progress summary."""
        return {
            'completed_count': len(self.state['completed_experiments']),
            'completed_experiments': list(self.state['completed_experiments']),
            'start_time': self.state['start_time'],
            'last_update': self.state['last_update']
        }


class ExperimentLogger:
    """Enhanced logging utilities for experiments."""
    
    def __init__(self, log_dir: str, experiment_name: str, detailed_logging: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.detailed_logging = detailed_logging
        
        # Setup file logging
        log_file = self.log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Add file handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.file_handler)
        
        logger.info(f"Experiment logging initialized: {log_file}")
        
    def log_experiment_start(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        logger.info("=" * 80)
        logger.info(f"STARTING EXPERIMENT: {self.experiment_name}")
        logger.info("=" * 80)
        logger.info("Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
        
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information."""
        logger.info("Dataset Information:")
        for dataset_name, info in dataset_info.items():
            logger.info(f"  {dataset_name}: {info['total_samples']} samples")
            if 'class_distribution' in info:
                for class_name, count in info['class_distribution'].items():
                    logger.info(f"    {class_name}: {count}")
                    
    def log_training_progress(self, epoch: int, total_epochs: int, metrics: Dict[str, float]):
        """Log training progress."""
        if self.detailed_logging:
            progress = f"Epoch {epoch}/{total_epochs}"
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"{progress} - {metric_str}")
            
    def log_experiment_result(self, experiment_key: str, result: Dict[str, Any]):
        """Log experiment completion."""
        logger.info(f"COMPLETED: {experiment_key}")
        logger.info(f"  AUC: {result.get('auc', 'N/A'):.4f}")
        logger.info(f"  Accuracy: {result.get('accuracy', 'N/A'):.4f}")
        logger.info(f"  Sensitivity@95%Spec: {result.get('sensitivity_at_95_specificity', 'N/A'):.4f}")
        
    def cleanup(self):
        """Clean up logging handlers."""
        root_logger = logging.getLogger()
        root_logger.removeHandler(self.file_handler)
        self.file_handler.close()


class ResultsManager:
    """Manages experiment results and generates summary reports."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_results(self, results: List[Dict[str, Any]], filename: str = "experiment_results.json"):
        """Save results to JSON file."""
        results_file = self.output_dir / filename
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            
    def create_summary_table(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a summary table from results."""
        if not results:
            return pd.DataFrame()
            
        # Extract key metrics
        summary_data = []
        for result in results:
            summary_data.append({
                'Model': result.get('model_name', 'Unknown'),
                'Test_Dataset': result.get('test_dataset', 'Unknown'),
                'Train_Datasets': ', '.join(result.get('train_datasets', [])),
                'AUC': result.get('auc', 0.0),
                'Accuracy': result.get('accuracy', 0.0),
                'Sensitivity@95%Spec': result.get('sensitivity_at_95_specificity', 0.0),
                'ECE': result.get('ece', 0.0),
                'Training_Time': result.get('training_time', 0.0)
            })
            
        df = pd.DataFrame(summary_data)
        
        # Save summary table
        summary_file = self.output_dir / "experiment_summary.csv"
        df.to_csv(summary_file, index=False)
        logger.info(f"Summary table saved to {summary_file}")
        
        return df
        
    def generate_model_comparison(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate model comparison across datasets."""
        if not results:
            return pd.DataFrame()
            
        # Create pivot table for model comparison
        df = pd.DataFrame(results)
        if df.empty:
            return pd.DataFrame()
            
        pivot_data = []
        for model in df['model_name'].unique():
            model_results = df[df['model_name'] == model]
            row = {'Model': model}
            
            for dataset in model_results['test_dataset'].unique():
                dataset_result = model_results[model_results['test_dataset'] == dataset]
                if not dataset_result.empty:
                    row[f'{dataset}_AUC'] = dataset_result['auc'].iloc[0]
                    row[f'{dataset}_Acc'] = dataset_result['accuracy'].iloc[0]
                    
            pivot_data.append(row)
            
        comparison_df = pd.DataFrame(pivot_data)
        
        # Save comparison table
        comparison_file = self.output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Model comparison saved to {comparison_file}")
        
        return comparison_df
        
    def print_final_summary(self, results: List[Dict[str, Any]]):
        """Print final experiment summary."""
        if not results:
            logger.info("No results to summarize")
            return
            
        logger.info("=" * 80)
        logger.info("FINAL EXPERIMENT SUMMARY")
        logger.info("=" * 80)
        
        df = pd.DataFrame(results)
        
        # Overall statistics
        logger.info(f"Total experiments completed: {len(results)}")
        logger.info(f"Models tested: {df['model_name'].nunique()}")
        logger.info(f"Datasets tested: {df['test_dataset'].nunique()}")
        
        # Performance statistics
        logger.info(f"Average AUC: {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
        logger.info(f"Average Accuracy: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
        
        # Best performing experiments
        best_auc = df.loc[df['auc'].idxmax()]
        logger.info(f"Best AUC: {best_auc['auc']:.4f} ({best_auc['model_name']} on {best_auc['test_dataset']})")
        
        best_acc = df.loc[df['accuracy'].idxmax()]
        logger.info(f"Best Accuracy: {best_acc['accuracy']:.4f} ({best_acc['model_name']} on {best_acc['test_dataset']})")
        
        logger.info("=" * 80)
