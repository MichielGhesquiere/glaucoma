import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn as nn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier # Keep import if default RF is used

from tqdm import tqdm
import os
import time
from datetime import datetime
import logging
import copy

# Import loss function (assuming it's in the same directory or package)
from .losses import multi_task_segmentation_loss

# Placeholder imports for plotting (should be moved to utils/plotting.py)
# We will remove the direct calls to plotting from here later.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- PyTorch Model Trainers ---

def train_classification_model(model: nn.Module,
                               train_loader: DataLoader,
                               val_loader: DataLoader,
                               criterion: nn.Module,
                               optimizer: optim.Optimizer,
                               scheduler=None, # Optional scheduler
                               num_epochs: int = 25,
                               device: str = 'cuda',
                               checkpoint_dir: str = 'checkpoints',
                               model_name: str = 'classification_model') -> tuple[nn.Module, dict]:
    """
    Trains a PyTorch classification model.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        criterion: The loss function (e.g., nn.CrossEntropyLoss).
        optimizer: The optimizer (e.g., optim.Adam).
        scheduler: Optional learning rate scheduler.
        num_epochs (int): Number of epochs to train.
        device (str): Device to use ('cuda' or 'cpu').
        checkpoint_dir (str): Directory to save model checkpoints.
        model_name (str): Base name for saving the best model checkpoint.

    Returns:
        tuple[nn.Module, dict]:
            - The best performing model based on validation loss.
            - A history dictionary containing training/validation losses and accuracies.
    """
    start_time = time.time()
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f'best_{model_name}.pth')

    model.to(device)
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict()) # Store best weights

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    logging.info(f"Starting classification training for {num_epochs} epochs on {device}.")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logging.info(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        # === Training Phase ===
        model.train()
        running_loss = 0.0
        train_corrects = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
        for inputs, labels in train_pbar:
            # Handle potential None values from Dataset __getitem__ if error occurred
            # This requires a custom collate_fn in DataLoader or filtering Nones here.
            # Simple approach: skip batch if None encountered (might skew results)
            # if inputs is None or labels is None:
            #      logging.warning("Skipping a batch due to None input/label.")
            #      continue
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            train_total += labels.size(0)

            # Update progress bar postfix
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_corrects.double() / train_total:.2f}%'
            })

        epoch_train_loss = running_loss / train_total
        epoch_train_acc = train_corrects.double() / train_total * 100
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.item()) # Store as float

        logging.info(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}%")

        # === Validation Phase ===
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]', leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                # if inputs is None or labels is None: continue # Handle None if necessary
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_corrects.double() / val_total * 100
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item()) # Store as float

        logging.info(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")

        # --- Checkpoint Saving ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            # Save the best model state
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                # Include scheduler state if exists
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'history': history # Optionally save history in checkpoint
            }, best_model_path)
            logging.info(f"*** New best model saved to {best_model_path} (Val Loss: {best_val_loss:.4f}) ***")

        # --- Learning Rate Scheduling ---
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step() # For schedulers like StepLR, MultiStepLR

        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch completed in {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s")
        if scheduler:
             logging.info(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")


    # --- Training Complete ---
    total_time = time.time() - start_time
    logging.info(f"\nTraining completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Remove plotting call from here - should be done after training finishes
    # plot_training_history_classification(history) # Moved to utils/plotting.py

    return model, history


def train_segmentation_model(model: nn.Module,
                             train_loader: DataLoader,
                             val_loader: DataLoader,
                             optimizer: optim.Optimizer,
                             scheduler=None,
                             loss_fn=multi_task_segmentation_loss, # Can pass custom loss function
                             loss_alpha: float = 0.5, # Weight for OC loss
                             loss_type: str = 'dice', # 'dice' or 'bce'
                             num_epochs: int = 50,
                             device: str = 'cuda',
                             checkpoint_dir: str = 'checkpoints',
                             model_name: str = 'segmentation_model',
                             log_file: str = None) -> tuple[nn.Module, dict]:
    """
    Trains a multi-task segmentation model (e.g., MultiTaskUNet).

    Args:
        model: The PyTorch segmentation model.
        train_loader: DataLoader for training data (expects image, oc_mask, od_mask).
        val_loader: DataLoader for validation data.
        optimizer: The optimizer.
        scheduler: Optional learning rate scheduler.
        loss_fn: The loss function to use. Should accept (pred_oc, target_oc, pred_od, target_od, ...)
                 and return (combined_loss, loss_oc_item, loss_od_item).
        loss_alpha (float): Weight for the first task (OC) loss.
        loss_type (str): Argument passed to loss_fn if it's the default multi_task_segmentation_loss.
        num_epochs (int): Number of epochs.
        device (str): Device ('cuda' or 'cpu').
        checkpoint_dir (str): Directory for saving checkpoints.
        model_name (str): Base name for saving the best model.
        log_file (str, optional): Path to write training logs. If None, logs only to console.

    Returns:
        tuple[nn.Module, dict]:
            - The best performing model based on validation loss.
            - A history dictionary containing detailed losses.
    """
    start_time = time.time()
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f'best_{model_name}.pth')

    if log_file is None:
         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
         log_file = os.path.join(checkpoint_dir, f'training_log_{model_name}_{timestamp}.txt')
         logging.info(f"Log file not specified, using default: {log_file}")

    def write_log(message):
        logging.info(message)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')

    model.to(device)
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {
        'train_loss': [], 'train_oc_loss': [], 'train_od_loss': [],
        'val_loss': [], 'val_oc_loss': [], 'val_od_loss': []
    }

    log_header = f"Starting {model_name} training for {num_epochs} epochs on {device}."
    write_log(log_header)
    write_log("=" * 80)
    log_fmt = f"{'Epoch':^6} | {'Phase':^7} | {'Loss':^12} | {'OC Loss':^12} | {'OD Loss':^12} | {'Time':^10}"
    write_log(log_fmt)
    write_log("-" * 80)


    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        write_log(f"--- Epoch {epoch+1}/{num_epochs} ---")

        # === Training Phase ===
        model.train()
        running_loss = 0.0
        running_oc_loss = 0.0
        running_od_loss = 0.0
        num_batches = len(train_loader)

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
        for batch_idx, batch_data in enumerate(train_pbar):
            # Adapt based on Dataset output, expecting (image, oc_mask, od_mask)
            # Handle potential Nones if necessary
            try:
                 images, oc_masks, od_masks = batch_data
                 if images is None: raise ValueError("Received None image tensor") # Handle None properly
                 images = images.to(device)
                 oc_masks = oc_masks.to(device)
                 od_masks = od_masks.to(device)
            except Exception as e:
                 logging.error(f"Error unpacking or moving batch {batch_idx} to device: {e}. Skipping batch.")
                 continue # Skip this batch

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Assuming model outputs (oc_pred, od_pred)
                pred_oc, pred_od = model(images)

                # Calculate loss using the provided function
                loss, oc_loss_item, od_loss_item = loss_fn(
                    pred_oc, oc_masks, pred_od, od_masks,
                    loss_type=loss_type, alpha=loss_alpha # Pass necessary args
                )

                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_oc_loss += oc_loss_item
            running_od_loss += od_loss_item

            # Update progress bar postfix
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'oc': f'{oc_loss_item:.4f}',
                'od': f'{od_loss_item:.4f}'
            })

        epoch_train_loss = running_loss / num_batches
        epoch_train_oc_loss = running_oc_loss / num_batches
        epoch_train_od_loss = running_od_loss / num_batches
        history['train_loss'].append(epoch_train_loss)
        history['train_oc_loss'].append(epoch_train_oc_loss)
        history['train_od_loss'].append(epoch_train_od_loss)

        epoch_time_so_far = time.time() - epoch_start_time
        log_train = f"{epoch+1:^6d} | {'Train':^7} | {epoch_train_loss:^12.4f} | {epoch_train_oc_loss:^12.4f} | {epoch_train_od_loss:^12.4f} | {f'{epoch_time_so_far:.1f}s':^10}"
        write_log(log_train)


        # === Validation Phase ===
        model.eval()
        val_loss = 0.0
        val_oc_loss = 0.0
        val_od_loss = 0.0
        num_val_batches = len(val_loader)

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]', leave=False)
        with torch.no_grad():
            for batch_data in val_pbar:
                 try:
                     images, oc_masks, od_masks = batch_data
                     if images is None: raise ValueError("Received None image tensor")
                     images = images.to(device)
                     oc_masks = oc_masks.to(device)
                     od_masks = od_masks.to(device)
                 except Exception as e:
                     logging.error(f"Error unpacking or moving validation batch to device: {e}. Skipping batch.")
                     continue

                 pred_oc, pred_od = model(images)
                 loss, oc_loss_item, od_loss_item = loss_fn(
                    pred_oc, oc_masks, pred_od, od_masks,
                    loss_type=loss_type, alpha=loss_alpha
                 )

                 val_loss += loss.item()
                 val_oc_loss += oc_loss_item
                 val_od_loss += od_loss_item
                 val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_val_loss = val_loss / num_val_batches
        epoch_val_oc_loss = val_oc_loss / num_val_batches
        epoch_val_od_loss = val_od_loss / num_val_batches
        history['val_loss'].append(epoch_val_loss)
        history['val_oc_loss'].append(epoch_val_oc_loss)
        history['val_od_loss'].append(epoch_val_od_loss)

        epoch_time_total = time.time() - epoch_start_time
        log_val = f"{epoch+1:^6d} | {'Val':^7} | {epoch_val_loss:^12.4f} | {epoch_val_oc_loss:^12.4f} | {epoch_val_od_loss:^12.4f} | {f'{epoch_time_total:.1f}s':^10}"
        write_log(log_val)

        # --- Checkpoint Saving ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'history': history # Optionally save history
            }, best_model_path)
            write_log(f"*** New best model saved to {best_model_path} (Val Loss: {best_val_loss:.4f}) ***")

        # --- Learning Rate Scheduling ---
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()
            write_log(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        write_log("-" * 80) # End of epoch marker

    # --- Training Complete ---
    total_time = time.time() - start_time
    write_log(f"\nTraining completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    write_log(f"Best validation loss: {best_val_loss:.4f}")
    write_log("=" * 80)

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Remove plotting - should be done externally
    # plot_training_history_segmentation(history)

    return model, history


# --- Sklearn Model Trainer ---

def train_sklearn_progression_model(features_df: pd.DataFrame,
                                    model_instance: object, # Pre-initialized sklearn model (e.g., RandomForestClassifier)
                                    feature_cols: list = None, # Optional: specify feature columns explicitly
                                    target_col: str = 'progression_label', # Assumes 0/1 label
                                    test_size: float = 0.25,
                                    random_state: int = 42,
                                    scale_features: bool = True) -> dict:
    """
    Trains a scikit-learn model on pre-extracted progression features.

    Args:
        features_df (pd.DataFrame): DataFrame containing features and the target variable.
                                    Must include columns like 'subject_id', 'laterality' (or other identifiers)
                                    and the target_col.
        model_instance: An initialized scikit-learn compatible classifier (e.g., RandomForestClassifier()).
        feature_cols (list, optional): Explicit list of column names to use as features.
                                        If None, uses all columns except identifiers and target. Defaults to None.
        target_col (str): Name of the target variable column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for splitting and model training (if applicable).
        scale_features (bool): Whether to apply StandardScaler to the features.

    Returns:
        dict: A dictionary containing the trained model, scaler (if used), classification report,
              confusion matrix, ROC AUC score, and feature importances (if applicable).
    """
    logging.info(f"Starting sklearn model training for progression prediction.")
    results = {}

    if features_df is None or features_df.empty:
        logging.error("Features DataFrame is empty.")
        return {"error": "Input DataFrame is empty."}
    if target_col not in features_df.columns:
         logging.error(f"Target column '{target_col}' not found in DataFrame.")
         return {"error": f"Target column '{target_col}' not found."}

    # Identify identifier columns (modify if your identifiers are different)
    id_cols = ['subject_id', 'laterality']
    cols_to_drop = [col for col in id_cols if col in features_df.columns] + [target_col]

    if feature_cols is None:
        # Infer feature columns if not provided
        feature_cols = [col for col in features_df.columns if col not in cols_to_drop]
        logging.info(f"Inferred {len(feature_cols)} feature columns.")
    else:
        # Validate provided feature columns
        missing_features = [col for col in feature_cols if col not in features_df.columns]
        if missing_features:
            logging.error(f"Provided feature columns not found in DataFrame: {missing_features}")
            return {"error": f"Missing feature columns: {missing_features}"}
        logging.info(f"Using specified {len(feature_cols)} feature columns.")


    X = features_df[feature_cols]
    y = features_df[target_col]

    # Handle potential NaN/Inf values in features
    if X.isnull().sum().sum() > 0:
        logging.warning(f"Found {X.isnull().sum().sum()} NaN values in features. Imputing with column mean.")
        # Simple imputation - consider more sophisticated strategies if needed
        X = X.fillna(X.mean())
    if np.isinf(X.values).sum() > 0:
         logging.warning(f"Found {np.isinf(X.values).sum()} infinite values in features. Replacing with large finite numbers.")
         X = X.replace([np.inf, -np.inf], np.nan)
         X = X.fillna(X.mean()) # Re-impute NaNs created by replacement


    # Split data - stratify by target variable for balanced splits
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
    except ValueError as e:
        logging.error(f"Error during train/test split (potentially too few samples per class for stratify): {e}")
        return {"error": f"Train/test split failed: {e}"}


    # Scale features
    scaler = None
    if scale_features:
        logging.info("Applying StandardScaler to features.")
        scaler = StandardScaler()
        try:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            results['scaler'] = scaler
        except ValueError as e:
             logging.error(f"Error during feature scaling (check for columns with zero variance): {e}")
             # Fallback: train without scaling? Or return error.
             logging.warning("Proceeding without feature scaling due to error.")
             X_train_scaled = X_train.values # Use numpy arrays
             X_test_scaled = X_test.values
             scale_features = False # Mark that scaling wasn't actually done
    else:
        X_train_scaled = X_train.values # Use numpy arrays if not scaling
        X_test_scaled = X_test.values

    # Train model
    model = model_instance # Use the pre-initialized model
    logging.info(f"Training model: {type(model).__name__}")
    try:
        model.fit(X_train_scaled, y_train)
        results['model'] = model
    except Exception as e:
         logging.error(f"Error during model training: {e}")
         return {"error": f"Model training failed: {e}"}

    # Evaluate
    logging.info("Evaluating model on the test set.")
    try:
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] # Probability of positive class

        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        results['classification_report'] = report_dict
        results['confusion_matrix'] = conf_matrix.tolist() # Convert to list for easier saving (e.g., JSON)
        results['roc_auc'] = roc_auc

        logging.info("Classification Report (Test Set):\n" + classification_report(y_test, y_pred, zero_division=0))
        logging.info(f"Confusion Matrix (Test Set):\n{conf_matrix}")
        logging.info(f"ROC AUC Score (Test Set): {roc_auc:.4f}")

        # Feature Importance (if applicable)
        if hasattr(model, 'feature_importances_'):
            feature_importance_df = pd.DataFrame({
                'Feature': feature_cols, # Use original feature names
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
            results['feature_importance'] = feature_importance_df.to_dict(orient='records') # Save as list of dicts
            logging.info("Top 10 Feature Importances:\n" + feature_importance_df.head(10).to_string())

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        # Return partial results if training succeeded but evaluation failed?
        results['evaluation_error'] = str(e)


    # Remove plotting - should be done externally using the results dict
    # plot_progression_evaluation(y_test, y_prob, results.get('feature_importance', None))

    logging.info("Sklearn model training and evaluation finished.")
    return results


# --- Placeholder Plotting Functions (to be moved) ---
def plot_training_history_classification(history):
    # Placeholder - Actual implementation in utils/plotting.py
    logging.warning("Plotting function 'plot_training_history_classification' called from trainer (should be in utils).")
    pass

def plot_training_history_segmentation(history):
     # Placeholder - Actual implementation in utils/plotting.py
    logging.warning("Plotting function 'plot_training_history_segmentation' called from trainer (should be in utils).")
    pass

def plot_progression_evaluation(y_test, y_prob, feature_importance_list):
     # Placeholder - Actual implementation in utils/plotting.py
    logging.warning("Plotting function 'plot_progression_evaluation' called from trainer (should be in utils).")
     # Example: ROC Curve
    # fpr, tpr, _ = roc_curve(y_test, y_prob)
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.title('ROC Curve')
    # plt.show()
    pass