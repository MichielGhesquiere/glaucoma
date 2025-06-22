import argparse
import os
import pandas as pd
import joblib # For saving sklearn models
import json
import logging

# Make sure Python can find the src modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logging
from src.utils.plotting import plot_roc_curve, plot_feature_importance # Import plotting functions
# Import model initializer and trainer
from src.models.progression.basic_rf import get_rf_progression_model
from src.training.trainers import train_sklearn_progression_model

def main(args):
    """Main function to train progression model on extracted features."""
    # Load configuration
    config = load_config(args.config)
    paths_config = config.get('paths', {})
    train_config = config.get('training', {}).get('progression', {})
    model_config = config.get('model', {}).get('progression', {})

    # Setup logging
    log_dir = paths_config.get('log_dir', 'logs')
    model_type = model_config.get('type', 'RandomForest') # Example model type
    log_file = os.path.join(log_dir, f'train_progression_{model_type}.log')
    setup_logging(log_level=config.get('log_level', 'INFO'), log_dir=log_dir, log_file=log_file)
    logging.info("Starting progression model training script...")
    logging.info(f"Loaded configuration from: {args.config}")

    # --- Setup ---
    features_dir = paths_config.get('features_dir', os.path.join(paths_config.get('processed_data_dir', 'data/processed'), 'features'))
    model_save_dir = paths_config.get('model_save_dir', 'models') # Dir to save sklearn models
    results_dir = paths_config.get('results_dir', 'results')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # --- Load Features ---
    feature_file = train_config.get('feature_file', 'progression_features.csv')
    feature_path = os.path.join(features_dir, feature_file)
    logging.info(f"Loading features from: {feature_path}")
    try:
        features_df = pd.read_csv(feature_path)
        if features_df.empty:
            raise ValueError("Features file is empty.")
        logging.info(f"Loaded {len(features_df)} sequences with features.")
    except FileNotFoundError:
        logging.error(f"Features file not found: {feature_path}")
        return
    except Exception as e:
        logging.error(f"Error loading features file {feature_path}: {e}", exc_info=True)
        return

    # --- Initialize Model ---
    # Example for RandomForest, extend for other sklearn models if needed
    if model_type.lower() == 'randomforest':
        rf_params = model_config.get('params', {})
        model_instance = get_rf_progression_model(
            n_estimators=rf_params.get('n_estimators', 100),
            random_state=train_config.get('random_seed', 42),
            max_depth=rf_params.get('max_depth', None), # Add other RF params
            min_samples_leaf=rf_params.get('min_samples_leaf', 1),
            class_weight=rf_params.get('class_weight', None) # e.g., 'balanced'
        )
        logging.info(f"Initialized RandomForestClassifier with params: {rf_params}")
    else:
        logging.error(f"Unsupported progression model type: {model_type}")
        return

    # --- Train and Evaluate Model ---
    target_col = train_config.get('target_col', 'progression_label') # Make sure this matches feature file
    results = train_sklearn_progression_model(
        features_df=features_df,
        model_instance=model_instance,
        feature_cols=train_config.get('feature_cols', None), # Specify in config or infer
        target_col=target_col,
        test_size=train_config.get('test_split', 0.25),
        random_state=train_config.get('random_seed', 42),
        scale_features=train_config.get('scale_features', True)
    )

    # --- Save Results ---
    if 'error' in results:
        logging.error(f"Training failed: {results['error']}")
        return

    # Save trained model and scaler
    model_filename = f"{model_type}_progression_model.joblib"
    scaler_filename = f"{model_type}_progression_scaler.joblib"
    model_save_path = os.path.join(model_save_dir, model_filename)
    scaler_save_path = os.path.join(model_save_dir, scaler_filename)

    try:
        joblib.dump(results['model'], model_save_path)
        logging.info(f"Trained progression model saved to: {model_save_path}")
        if 'scaler' in results and results['scaler'] is not None:
            joblib.dump(results['scaler'], scaler_save_path)
            logging.info(f"Scaler saved to: {scaler_save_path}")
    except Exception as e:
        logging.error(f"Error saving model or scaler: {e}")

    # Save evaluation metrics
    results_filename = f"{model_type}_progression_evaluation.json"
    results_save_path = os.path.join(results_dir, results_filename)
    try:
        # Prepare results for JSON (convert numpy arrays/objects if necessary)
        saveable_results = {k: v for k, v in results.items() if k not in ['model', 'scaler']}
        # feature_importance is already list of dicts
        # classification_report is already dict
        # confusion_matrix is list of lists
        with open(results_save_path, 'w') as f:
            json.dump(saveable_results, f, indent=4)
        logging.info(f"Evaluation results saved to: {results_save_path}")
    except Exception as e:
        logging.error(f"Error saving evaluation results: {e}")

    # --- Plot Results ---
    # Need y_true, y_prob from the test split - currently not returned by trainer
    # Modification needed in train_sklearn_progression_model to return these if plotting is desired here.
    # Or, plotting must be done in a separate evaluation script that reloads model/data.
    logging.warning("Plotting (ROC, Feature Importance) currently not implemented directly after training in this script.")
    # Example if y_test and y_prob were returned:
    # if 'roc_auc' in results:
    #     plot_roc_curve(y_test, y_prob, title=f'{model_type} Progression ROC',
    #                    save_path=os.path.join(results_dir, f'{model_type}_roc_curve.png'))
    # if 'feature_importance' in results:
    #     fi_df = pd.DataFrame(results['feature_importance'])
    #     plot_feature_importance(fi_df, top_n=20, title=f'{model_type} Feature Importance',
    #                            save_path=os.path.join(results_dir, f'{model_type}_feature_importance.png'))

    logging.info("Progression model training script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Progression Model on Extracted Features")
    parser.add_argument('--config', type=str, required=True, help='Path to the main configuration YAML file.')
    # Add other args if needed, e.g., override feature file path
    args = parser.parse_args()
    main(args)