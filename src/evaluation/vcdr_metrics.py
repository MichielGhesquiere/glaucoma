"""
Evaluation utilities for VCDR regression models.
"""

import logging
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def calculate_regression_metrics(
    y_true: List[float], 
    y_pred: List[float]
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing various regression metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mean_error = np.mean(y_pred - y_true)  # Bias
    std_error = np.std(y_pred - y_true)    # Standard deviation of errors
    
    # Percentage metrics (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    
    # Correlation coefficient
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mean_error': mean_error,
        'std_error': std_error,
        'mape': mape,
        'correlation': correlation,
        'n_samples': len(y_true)
    }
    
    return metrics


def print_evaluation_results(metrics: Dict[str, float], title: str = "Evaluation Results") -> None:
    """
    Print formatted evaluation results.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the results section
    """
    logger.info("=" * 60)
    logger.info(f"{title.upper()}")
    logger.info("=" * 60)
    
    logger.info(f"Number of samples: {metrics.get('n_samples', 'N/A')}")
    logger.info(f"Mean Absolute Error (MAE): {metrics.get('mae', 0):.6f}")
    logger.info(f"Mean Squared Error (MSE): {metrics.get('mse', 0):.6f}")
    logger.info(f"Root Mean Squared Error (RMSE): {metrics.get('rmse', 0):.6f}")
    logger.info(f"R² Score: {metrics.get('r2', 0):.6f}")
    logger.info(f"Mean Error (Bias): {metrics.get('mean_error', 0):.6f}")
    logger.info(f"Standard Deviation of Errors: {metrics.get('std_error', 0):.6f}")
    logger.info(f"Mean Absolute Percentage Error (MAPE): {metrics.get('mape', 0):.2f}%")
    logger.info(f"Correlation Coefficient: {metrics.get('correlation', 0):.6f}")
    
    logger.info("=" * 60)


def evaluate_model_performance(
    predictions: List[float],
    targets: List[float],
    title: str = "Model Performance"
) -> Dict[str, float]:
    """
    Evaluate model performance and print results.
    
    Args:
        predictions: Model predictions
        targets: True target values
        title: Title for the evaluation
        
    Returns:
        Dictionary of calculated metrics
    """
    logger.info(f"Evaluating {title.lower()}...")
    
    metrics = calculate_regression_metrics(targets, predictions)
    print_evaluation_results(metrics, title)
    
    return metrics
