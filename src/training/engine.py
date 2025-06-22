import torch
import torch.nn as nn
import os
import time
import logging
from tqdm import tqdm
import numpy as np
import sys

# Import for saving images (for Mixup samples)
from torchvision.utils import save_image
# Import for type hinting Mixup object
from timm.data import Mixup # Assuming Mixup class is from timm.data.mixup

try:
    # Ensure this path is correct relative to your project structure
    # If engine.py is in src/training/ and plotting.py is in src/utils/
    from ..utils.plotting import plot_training_history
except ImportError:
    # Fallback or log a warning if plotting is optional/not found
    def plot_training_history(*args, **kwargs):
        # Conditional logging: only log if logger is already configured
        if logger.hasHandlers():
            logger.warning("plot_training_history function not found or import failed. Skipping plot generation.")
        else:
            print("Warning: plot_training_history function not found or import failed. Skipping plot generation.")

logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, checkpoint_dir, experiment_name="train",
                early_stopping_patience=None, use_amp=False,
                gradient_accumulation_steps=1,
                start_epoch=0,
                resume_history=None,
                mixup_fn: Mixup = None,  # <<< MODIFIED: Added mixup_fn argument
                mixup_sample_dir: str = None, # <<< NEW: Directory to save mixup samples
                num_mixup_samples_to_save: int = 0): # <<< NEW: Number of samples to save
    """
    Trains a PyTorch model with gradient accumulation, resuming, AMP, Mixup/CutMix, and logging.

    Args:
        model (nn.Module): PyTorch model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        criterion (callable): Loss function (should handle soft targets if mixup_fn is used, e.g., SoftTargetCrossEntropy)
        optimizer (torch.optim.Optimizer): PyTorch optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        num_epochs (int): TOTAL number of epochs the model should be trained for.
        device (torch.device): Device to train on (cpu or cuda)
        checkpoint_dir (str): Directory to save checkpoints
        experiment_name (str): Name for this training run (used for checkpoint filenames)
        early_stopping_patience (int, optional): Epochs to wait before early stopping. Defaults to None (disabled).
        use_amp (bool): Whether to use automatic mixed precision. Defaults to False.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients over. Defaults to 1.
        start_epoch (int): The epoch number to start training from (0-indexed).
        resume_history (dict, optional): Existing history dict to append to if resuming.
        mixup_fn (Mixup, optional): timm Mixup object. If None, Mixup/CutMix is not applied.
        mixup_sample_dir (str, optional): Directory to save Mixup/CutMix sample images.
        num_mixup_samples_to_save (int): Number of Mixup/CutMix samples to save from the first batch of each epoch.

    Returns:
        tuple: (trained_model, history_dict for the epochs trained IN THIS SESSION)
    """

    if train_loader is None:
        logger.error("Error: train_loader is None. Cannot start training.")
        return model, {}

    # --- Setup ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    history_this_session = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_epoch_this_session = -1

    # === AMP Setup ===
    device_type = device.type
    amp_enabled = use_amp and (device_type == 'cuda')
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    # Ensure amp_dtype is torch.bfloat16 if supported and desired for specific hardware (e.g., Ampere+ GPUs)
    # For general compatibility, float16 is common.
    amp_dtype = torch.float16 if device_type == 'cuda' else None # Can be torch.bfloat16 for A100+
    logger.info(f"AMP enabled: {amp_enabled}{f' with dtype {amp_dtype}' if amp_enabled and amp_dtype else ''}")


    # --- Checkpoint Paths ---
    best_model_path_template = os.path.join(checkpoint_dir, f'{experiment_name}_best_model_epoch{{}}.pth')
    last_checkpoint_path = os.path.join(checkpoint_dir, f'{experiment_name}_last_checkpoint.pth')

    # --- Gradient Accumulation Check ---
    if gradient_accumulation_steps < 1:
        logger.warning("gradient_accumulation_steps must be >= 1. Setting to 1.")
        gradient_accumulation_steps = 1
    if gradient_accumulation_steps > 1:
        logger.info(f"Using gradient accumulation with {gradient_accumulation_steps} steps.")

    # --- Resume Logic ---
    if start_epoch > 0:
        logger.info(f"--- Resuming training from epoch {start_epoch + 1} ---")
        if resume_history and resume_history.get('val_loss'):
            valid_past_losses = [l for l in resume_history['val_loss'] if l is not None and np.isfinite(l)]
            if valid_past_losses:
                best_val_loss = min(valid_past_losses)
                logger.info(f"Initialized best_val_loss from resumed history: {best_val_loss:.4f}")
        epochs_no_improve = 0 # Typically reset for resumed phase, or load from checkpoint if saved

    def write_log(message, use_tqdm_write=False):
        if use_tqdm_write: tqdm.write(message, file=sys.stderr)
        else: logger.info(message)

    if start_epoch == 0:
        patience_str = f"Patience: {early_stopping_patience}" if early_stopping_patience else "Patience: Disabled"
        amp_status_str = f"AMP: {'Enabled' if amp_enabled else 'Disabled'}"
        accum_str = f"Grad Accum: {gradient_accumulation_steps if gradient_accumulation_steps > 1 else 'Disabled'}"
        mixup_str = f"Mixup/CutMix: {'Enabled' if mixup_fn else 'Disabled'}"
        header_parts = [f"{'Epoch':^7}", f"{'Batch':^12}", f"{'Loss':^10}", f"{'Accuracy':^10}",
                        f"{'Val Loss':^10}", f"{'Val Acc':^10}", f"{'LR':^8}", f"{'Best':^6}"]
        header = " | ".join(header_parts)
        separator = "-" * len(header)
        write_log("\n" + "=" * len(header))
        write_log(f"Training Started: {experiment_name} ({patience_str}, {amp_status_str}, {accum_str}, {mixup_str})")
        write_log(separator); write_log(header); write_log(separator)

    overall_start_time = time.time()
    early_stop_triggered = False
    
    for epoch in range(start_epoch, num_epochs):
        current_epoch_display = epoch + 1
        epoch_start_time = time.time()
        write_log(f"--- Starting Epoch {current_epoch_display}/{num_epochs} ---")
        
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        loader_len = len(train_loader) if hasattr(train_loader, '__len__') else None
        saved_mixup_this_epoch_count = 0

        optimizer.zero_grad(set_to_none=True)

        batch_loop = tqdm(train_loader, desc=f"Epoch {current_epoch_display}/{num_epochs} Train", leave=False, unit="batch", disable=(loader_len is None))
        for batch_idx, batch_data in enumerate(batch_loop):
            if batch_data is None: continue

            try:
                if len(batch_data) == 3: inputs, labels_orig, _ = batch_data
                elif len(batch_data) == 2: inputs, labels_orig = batch_data
                else: raise ValueError(f"Expected 2 or 3 items from train_loader, got {len(batch_data)}")
                
                inputs_device = inputs.to(device, non_blocking=True)
                labels_for_loss = labels_orig.to(device, non_blocking=True).long() # This will be modified by mixup if active
            except ValueError as e:
                 logger.error(f"Error unpacking train batch {batch_idx+1} in epoch {current_epoch_display}: {e}. Skipping.")
                 continue
            batch_size = inputs_device.size(0)

            inputs_for_model = inputs_device
            if mixup_fn is not None:
                mixed_inputs, labels_for_loss = mixup_fn(inputs_device, labels_for_loss.long())
                inputs_for_model = mixed_inputs

                if mixup_sample_dir and saved_mixup_this_epoch_count < num_mixup_samples_to_save and batch_idx == 0:
                    num_to_save_now = min(mixed_inputs.size(0), num_mixup_samples_to_save - saved_mixup_this_epoch_count)
                    for i in range(num_to_save_now):
                        sample_path = os.path.join(mixup_sample_dir, f"epoch{current_epoch_display}_batch{batch_idx}_sample{i}.png")
                        try:
                            save_image(mixed_inputs[i].cpu(), sample_path) # Saves normalized tensor
                            saved_mixup_this_epoch_count += 1
                        except Exception as e_save:
                            logger.warning(f"Could not save mixup sample {sample_path}: {e_save}")
                    if saved_mixup_this_epoch_count >= num_mixup_samples_to_save:
                         logger.info(f"Saved {saved_mixup_this_epoch_count} mixup samples for epoch {current_epoch_display}.")
            
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                 outputs = model(inputs_for_model)
                 loss = criterion(outputs, labels_for_loss.long()) # criterion should handle soft targets if mixup_fn
                 if gradient_accumulation_steps > 1:
                     loss = loss / gradient_accumulation_steps

            if not torch.isfinite(loss):
                 logger.warning(f"Non-finite loss ({loss.item()}) encountered in epoch {current_epoch_display}, batch {batch_idx+1}. Skipping update for this batch.")
                 optimizer.zero_grad(set_to_none=True) 
                 continue 
            
            scaler.scale(loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (loader_len and (batch_idx + 1) == loader_len):
                # Optional: Gradient clipping before optimizer step
                # if args.clip_grad_norm > 0:
                #     scaler.unscale_(optimizer) # Unscale before clipping
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += (loss.item() * gradient_accumulation_steps) * batch_size 
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels_orig.to(device)).sum().item() # Acc against original hard labels
            train_total += batch_size
            
            current_acc_tqdm = 100. * train_correct / train_total if train_total > 0 else 0
            batch_loop.set_postfix(loss=(loss.item() * gradient_accumulation_steps), acc=f"{current_acc_tqdm:.1f}%")

            log_freq = max(1, loader_len // 10 if loader_len else 50)
            if (batch_idx + 1) % log_freq == 0 or (loader_len and (batch_idx + 1) == loader_len):
                 current_loss_log = running_loss / train_total if train_total > 0 else 0
                 current_acc_log = 100. * train_correct / train_total if train_total > 0 else 0
                 lr_str = f"{optimizer.param_groups[0]['lr']:.2e}"
                 batch_str_log = f"{batch_idx+1}{'/' + str(loader_len) if loader_len else ''}"
                 write_log(f"{current_epoch_display:^7d} | {batch_str_log:^12} | {current_loss_log:^10.4f} | {current_acc_log:^10.1f} | {'---':^10} | {'---':^10} | {lr_str:^8s} | {'':^6}", use_tqdm_write=True)
        
        batch_loop.close()
        epoch_train_time = time.time() - epoch_start_time
        train_loss_epoch = running_loss / train_total if train_total > 0 else float('nan')
        train_acc_epoch = 100. * train_correct / train_total if train_total > 0 else float('nan')
        history_this_session['train_loss'].append(train_loss_epoch if np.isfinite(train_loss_epoch) else None)
        history_this_session['train_acc'].append(train_acc_epoch if np.isfinite(train_acc_epoch) else None)
        history_this_session['lr'].append(optimizer.param_groups[0]['lr'])

        # --- Validation Phase ---
        epoch_val_loss, epoch_val_acc = None, None
        is_best_this_epoch = False
        epoch_val_time = 0.0

        if val_loader:
            val_start_time = time.time()
            model.eval()
            val_loss_running = 0.0; val_correct_running = 0; val_total_running = 0
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f"Epoch {current_epoch_display}/{num_epochs} Val  ", leave=False, unit="batch")
                for batch_data_val in val_loop:
                    if batch_data_val is None: continue
                    try:
                        if len(batch_data_val) == 3: inputs_val, labels_val, _ = batch_data_val
                        elif len(batch_data_val) == 2: inputs_val, labels_val = batch_data_val
                        else: raise ValueError(f"Expected 2 or 3 items from val_loader, got {len(batch_data_val)}")
                        inputs_val, labels_val = inputs_val.to(device, non_blocking=True), labels_val.to(device, non_blocking=True)
                    except (ValueError, TypeError) as e: 
                        logger.error(f"Error unpacking val batch epoch {current_epoch_display}: {e}. Skipping.")
                        continue
                    
                    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                        outputs_val = model(inputs_val)
                        # For validation, always use hard labels with the criterion.
                        # SoftTargetCrossEntropy handles hard labels correctly (by one-hot encoding them).
                        # Other criteria like nn.CrossEntropyLoss expect long tensor for hard labels.
                        loss_val = criterion(outputs_val, labels_val.long()) 
                    
                    if torch.isfinite(loss_val): val_loss_running += loss_val.item() * inputs_val.size(0)
                    _, predicted_val = torch.max(outputs_val.data, 1)
                    val_correct_running += (predicted_val == labels_val).sum().item()
                    val_total_running += labels_val.size(0)
            val_loop.close()
            
            epoch_val_loss = val_loss_running / val_total_running if val_total_running > 0 else float('inf')
            epoch_val_acc = 100. * val_correct_running / val_total_running if val_total_running > 0 else 0.0
            history_this_session['val_loss'].append(epoch_val_loss if np.isfinite(epoch_val_loss) else None)
            history_this_session['val_acc'].append(epoch_val_acc if np.isfinite(epoch_val_acc) else None)
            epoch_val_time = time.time() - val_start_time

            if val_total_running > 0 and epoch_val_loss is not None and np.isfinite(epoch_val_loss):
                 val_loss_log_str_val = f"{epoch_val_loss:.4f}" # Renamed to avoid conflict
                 write_log(f"Validation Summary: Loss={val_loss_log_str_val}, Acc={epoch_val_acc:.2f}%")
                 best_loss_str_val = f"{best_val_loss:.4f}" if np.isfinite(best_val_loss) else "inf" # Renamed
                 if epoch_val_loss < best_val_loss:
                     write_log(f"*** Val loss improved from {best_loss_str_val} to {val_loss_log_str_val}. Saving best model for epoch {current_epoch_display}...")
                     best_val_loss = epoch_val_loss
                     is_best_this_epoch = True
                     best_epoch_this_session = current_epoch_display
                     epochs_no_improve = 0
                     try:
                         best_model_save_path = best_model_path_template.format(current_epoch_display)
                         state_to_save_best = {
                             'epoch': epoch, 
                             'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                             'amp_state_dict': scaler.state_dict() if amp_enabled else None,
                             'best_val_loss': best_val_loss,
                             'epochs_no_improve': epochs_no_improve,
                             'history': {** (resume_history or {}), **history_this_session} # Save combined history
                         }
                         torch.save(state_to_save_best, best_model_save_path)
                         logger.info(f"Best model checkpoint saved to {best_model_save_path}")
                     except Exception as e_save_best: write_log(f"ERROR saving best model checkpoint: {e_save_best}")
                 else:
                     epochs_no_improve += 1
                     if early_stopping_patience: write_log(f"Val loss did not improve from {best_loss_str_val}. Patience: {epochs_no_improve}/{early_stopping_patience}")
                 
                 if scheduler:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(epoch_val_loss)
                    # For schedulers not dependent on metrics, step them after optimizer.step()
                    # However, it's common to step them per epoch like this for many types.
                    # If scheduler is None, this block is skipped.
                    elif not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                         scheduler.step()


                 if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
                     early_stop_triggered = True
                     write_log(f"\n! EARLY STOPPING triggered after epoch {current_epoch_display}.")
            else:
                 write_log("--- Validation metrics could not be computed or were invalid. ---")
                 if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): 
                     logger.warning("Cannot step ReduceLROnPlateau scheduler without valid loss.")
                 elif scheduler : scheduler.step() # Step other types of schedulers
        else: 
            write_log("--- No validation loader provided, skipping validation phase. ---")
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): 
                scheduler.step()

        if epoch == start_epoch and start_epoch == 0:
            header_parts_log = [f"{'Epoch':^7}", f"{'Batch':^12}", f"{'Loss':^10}", f"{'Accuracy':^10}",
                                f"{'Val Loss':^10}", f"{'Val Acc':^10}", f"{'LR':^8}", f"{'Best':^6}"]
            header_log = " | ".join(header_parts_log) # Renamed
            separator_log = "-" * len(header_log) # Renamed
            write_log(separator_log); write_log(header_log); write_log(separator_log)
            
        lr_str_log = f"{optimizer.param_groups[0]['lr']:.2e}" # Renamed
        train_loss_str_log = f"{train_loss_epoch:.4f}" if train_loss_epoch is not None and np.isfinite(train_loss_epoch) else "---" # Added isfinite check
        train_acc_str_log = f"{train_acc_epoch:.1f}" if train_acc_epoch is not None and np.isfinite(train_acc_epoch) else "---" # Added isfinite check
        val_loss_str_log = f"{epoch_val_loss:.4f}" if epoch_val_loss is not None and np.isfinite(epoch_val_loss) else "---" # Added isfinite check
        val_acc_str_log = f"{epoch_val_acc:.1f}" if epoch_val_acc is not None and np.isfinite(epoch_val_acc) else "---" # Added isfinite check
        best_marker_log = '*' if is_best_this_epoch else '' # Renamed
        
        epoch_log_msg = f"{current_epoch_display:^7d} | {'Complete':^12} | {train_loss_str_log:^10} | {train_acc_str_log:^10} | {val_loss_str_log:^10} | {val_acc_str_log:^10} | {lr_str_log:^8s} | {best_marker_log:^6}"
        if epoch == start_epoch and start_epoch > 0:
             header_parts_resume = [f"{'Epoch':^7}", f"{'Batch':^12}", f"{'Loss':^10}", f"{'Accuracy':^10}",
                                    f"{'Val Loss':^10}", f"{'Val Acc':^10}", f"{'LR':^8}", f"{'Best':^6}"]
             header_resume = " | ".join(header_parts_resume) # Renamed
             separator_resume = "-" * len(header_resume) # Renamed
             write_log(separator_resume); write_log(header_resume); write_log(separator_resume)
        write_log(epoch_log_msg)
        epoch_total_time = time.time() - epoch_start_time
        write_log(f"Epoch {current_epoch_display} Timings: Train={epoch_train_time:.2f}s, Val={epoch_val_time:.2f}s, Total={epoch_total_time:.2f}s\n")

        try:
            state_to_save_last = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'amp_state_dict': scaler.state_dict() if amp_enabled else None,
                'best_val_loss': best_val_loss, 
                'epochs_no_improve': epochs_no_improve,
                'history': {** (resume_history or {}), **history_this_session} # Save combined history
            }
            torch.save(state_to_save_last, last_checkpoint_path)
        except Exception as e_save_last:
            write_log(f"ERROR saving last checkpoint {last_checkpoint_path}: {e_save_last}")

        if early_stop_triggered: break

    total_duration = time.time() - overall_start_time
    # Ensure 'epoch' is defined if loop didn't run (e.g., num_epochs <= start_epoch)
    # This typically shouldn't happen if num_epochs is the *total* desired epochs.
    if 'epoch' not in locals(): epoch = start_epoch -1 # If loop never ran, epoch completed is start_epoch-1

    epochs_completed_this_run = epoch - start_epoch + 1
    final_msg = f"\nTraining finished for {experiment_name} after {epochs_completed_this_run} epochs (total epochs completed: {epoch + 1})."
    if early_stop_triggered: final_msg += " (Early stopping triggered)"
    final_msg += f"\nTotal duration for this run: {total_duration//60:.0f}m {total_duration%60:.0f}s"
    best_loss_str_final = f"{best_val_loss:.4f}" if np.isfinite(best_val_loss) else "N/A" # Renamed
    
    final_best_epoch_overall = -1
    final_best_val_loss_overall = float('inf')

    # Combine history for determining overall best
    combined_history_for_plot = {**(resume_history or {}), **history_this_session}
    if resume_history: # Merge lists properly
        for key in resume_history:
            if key in history_this_session:
                combined_history_for_plot[key] = resume_history[key] + history_this_session[key]


    if combined_history_for_plot.get('val_loss'):
        valid_combined_losses = [l for l in combined_history_for_plot['val_loss'] if l is not None and np.isfinite(l)]
        if valid_combined_losses:
             final_best_val_loss_overall = min(valid_combined_losses)
             # Find index in the *original combined list* to account for Nones before valid ones
             final_best_epoch_overall = combined_history_for_plot['val_loss'].index(final_best_val_loss_overall) + 1
             final_msg += f"\nOverall Best validation loss: {final_best_val_loss_overall:.4f} at overall epoch {final_best_epoch_overall}"
    elif best_epoch_this_session != -1 :
        final_msg += f"\nBest validation loss for this run: {best_loss_str_final} at epoch {best_epoch_this_session}"
    else:
        final_msg += f"\nBest validation loss for this run: {best_loss_str_final}"

    separator_final = "-" * 80 # Renamed
    write_log("=" * len(separator_final))
    write_log(final_msg)
    write_log("=" * len(separator_final))

    history_plot_path = os.path.join(checkpoint_dir, f'{experiment_name}_training_history_epochs_{start_epoch + 1}_to_{epoch + 1}.png')
    try:
        if any(combined_history_for_plot.get(k) for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc']):
            plot_training_history(combined_history_for_plot, save_path=history_plot_path)
        else:
            logger.warning("No valid data in combined history to plot.")
    except Exception as plot_e:
        logger.error(f"Error generating training plot: {plot_e}")

    final_model_to_return = model
    
    # Attempt to load the weights of the overall best model identified
    if final_best_epoch_overall != -1: # If an overall best epoch was found
        path_of_overall_best_model_chkpt = best_model_path_template.format(final_best_epoch_overall)
        if os.path.exists(path_of_overall_best_model_chkpt):
            logger.info(f"Loading weights from overall best model (epoch {final_best_epoch_overall}) for final return from {path_of_overall_best_model_chkpt}.")
            try:
                checkpoint = torch.load(path_of_overall_best_model_chkpt, map_location=device)
                # Handle cases where model is nested or directly in checkpoint
                if 'model_state_dict' in checkpoint:
                    state_dict_to_load = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint: # Some checkpoints use 'state_dict'
                    state_dict_to_load = checkpoint['state_dict']
                else: # Assume checkpoint IS the state_dict
                    state_dict_to_load = checkpoint
                
                model.load_state_dict(state_dict_to_load)
                final_model_to_return = model
            except Exception as e_load_best:
                logger.error(f"ERROR loading overall best model state_dict from {path_of_overall_best_model_chkpt}: {e_load_best}. Returning model from last epoch of current run.")
        else:
            logger.warning(f"Overall best model checkpoint ({path_of_overall_best_model_chkpt}) not found. Returning model from last epoch of current run.")
    elif best_epoch_this_session != -1 : # Fallback to best of this session if no overall history or current best is overall best
        path_of_session_best_model_chkpt = best_model_path_template.format(best_epoch_this_session)
        if os.path.exists(path_of_session_best_model_chkpt):
            logger.info(f"Loading weights from best model of this session (epoch {best_epoch_this_session}) for final return from {path_of_session_best_model_chkpt}.")
            try:
                checkpoint = torch.load(path_of_session_best_model_chkpt, map_location=device)
                if 'model_state_dict' in checkpoint: state_dict_to_load = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint: state_dict_to_load = checkpoint['state_dict']
                else: state_dict_to_load = checkpoint
                model.load_state_dict(state_dict_to_load)
                final_model_to_return = model
            except Exception as e_load_session_best:
                logger.error(f"ERROR loading session best model state_dict from {path_of_session_best_model_chkpt}: {e_load_session_best}. Returning model from last epoch.")
        else:
             logger.warning(f"Session best model checkpoint ({path_of_session_best_model_chkpt}) not found. Returning model from last epoch.")
    else:
         logger.info("No best model identified or saved. Returning model from last epoch of current run.")

    return final_model_to_return, history_this_session