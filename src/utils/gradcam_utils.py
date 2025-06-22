import logging
import os
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
# Import various CAM methods
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM, XGradCAM, LayerCAM,
    EigenGradCAM, FullGrad # Add more if you want to experiment
)
# ViT specific (if you want to add them later as options)
# from pytorch_grad_cam import ViTAttentionRollout
# from pytorch_grad_cam. τότε_attention_cam import AttentionGradCAM

from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import timm
from timm.data import resolve_data_config

logger = logging.getLogger(__name__)

# Helper for un-normalization (remains the same)
class UnNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    def __call__(self, tensor):
        return tensor * self.std + self.mean

def get_default_target_layers(model: torch.nn.Module, model_name_str: str) -> list[torch.nn.Module]:
    """
    Attempts to get suitable target layers for CAM based on model architecture.
    Returns a list of nn.Module objects.
    """
    logger.info(f"Attempting to find target layers for CAM for model: {model_name_str}")
    model_name_lower = model_name_str.lower()
    target_layers = []

    # ResNet Family
    if "resnet" in model_name_lower or "resnext" in model_name_lower:
        if hasattr(model, 'layer4') and isinstance(model.layer4, torch.nn.Module) and len(list(model.layer4.children())) > 0 :
            target = model.layer4[-1]
            target_layers.append(target)
            logger.info(f"Identified ResNet-like. Target: model.layer4[-1] ({type(target).__name__})")
        else:
            logger.warning("ResNet-like model name, but model.layer4 not found or structured as expected.")

    # ViT Family (vit_base_patch16_224, dinov2_vitb14, VFM if it's vit_base)
    elif "vit" in model_name_lower or "dinov2" in model_name_lower:
        # Common targets for GradCAM on ViTs:
        # 1. The norm layer before the MLP in the last block
        # 2. The MLP itself in the last block
        # 3. The norm layer after the MLP in the last block
        # 4. The final model normalization layer (model.norm)
        if hasattr(model, 'blocks') and isinstance(model.blocks, (torch.nn.ModuleList, torch.nn.Sequential)) and len(model.blocks) > 0:
            last_block = model.blocks[-1]
            if hasattr(last_block, 'norm2') and isinstance(last_block.norm2, torch.nn.Module): # After MLP
                target_layers.append(last_block.norm2)
                logger.info(f"Identified ViT-like. Target: model.blocks[-1].norm2 ({type(last_block.norm2).__name__})")
            elif hasattr(last_block, 'mlp') and isinstance(last_block.mlp, torch.nn.Module):
                target_layers.append(last_block.mlp)
                logger.info(f"Identified ViT-like. Target: model.blocks[-1].mlp ({type(last_block.mlp).__name__})")
            elif hasattr(last_block, 'norm1') and isinstance(last_block.norm1, torch.nn.Module): # Before Attention
                target_layers.append(last_block.norm1)
                logger.info(f"Identified ViT-like. Target: model.blocks[-1].norm1 ({type(last_block.norm1).__name__})")
        # If no blocks, or couldn't find in last block, try final model.norm
        if not target_layers and hasattr(model, 'norm') and isinstance(model.norm, torch.nn.Module):
            target_layers.append(model.norm)
            logger.info(f"Identified ViT-like (fallback). Target: model.norm ({type(model.norm).__name__})")
        
        if not target_layers:
             logger.warning(f"ViT-like model {model_name_str}, but standard target layers (blocks[-1].normX, model.norm) not found.")


    # ConvNeXt Family
    elif "convnext" in model_name_lower:
        # Timm ConvNeXt typically has model.stages and model.norm
        # The last layer of the last stage, or the final norm layer.
        if hasattr(model, 'stages') and isinstance(model.stages, (torch.nn.ModuleList, torch.nn.Sequential)) and len(model.stages) > 0:
            last_stage = model.stages[-1]
            if hasattr(last_stage, 'blocks') and isinstance(last_stage.blocks, (torch.nn.ModuleList, torch.nn.Sequential)) and len(last_stage.blocks) > 0:
                target = last_stage.blocks[-1] # The last block itself
                target_layers.append(target)
                logger.info(f"Identified ConvNeXt-like. Target: model.stages[-1].blocks[-1] ({type(target).__name__})")
        # Fallback to final norm if stages not found as expected or if above failed
        if not target_layers and hasattr(model, 'norm') and isinstance(model.norm, torch.nn.Module):
            target_layers.append(model.norm)
            logger.info(f"Identified ConvNeXt-like (fallback to model.norm). Target: model.norm ({type(model.norm).__name__})")
        
        if not target_layers:
            logger.warning(f"ConvNeXt-like model {model_name_str}, but model.stages or model.norm not found as expected.")
    
    # Fallback for other CNNs (e.g., DenseNet, EfficientNet, MobileNet from timm)
    # These often have a 'model.features' attribute
    if not target_layers and hasattr(model, 'features') and isinstance(model.features, torch.nn.Sequential) and len(model.features) > 0:
        # Try to find the last nn.Conv2d in model.features by iterating backwards
        # This can be complex due to nested blocks in models like EfficientNet
        found_conv_fallback = False
        for i in range(len(model.features) -1, -1, -1):
            module_level1 = model.features[i]
            if isinstance(module_level1, torch.nn.Conv2d):
                target_layers.append(module_level1)
                logger.info(f"Generic CNN fallback. Target: model.features[{i}] (Conv2d) ({type(module_level1).__name__})")
                found_conv_fallback = True
                break
            # Check if it's a block that might contain a conv layer (common in EfficientNet/MobileNet)
            # This part can be made more sophisticated by introspecting common block structures
            if hasattr(module_level1, '_project_conv') and isinstance(module_level1._project_conv, torch.nn.Conv2d): # E.g. MBConv block in EfficientNet
                target_layers.append(module_level1._project_conv)
                logger.info(f"Generic CNN fallback. Target: model.features[{i}]._project_conv ({type(module_level1._project_conv).__name__})")
                found_conv_fallback = True
                break
            if hasattr(module_level1, 'conv') and isinstance(module_level1.conv, torch.nn.Conv2d): # More generic block
                 target_layers.append(module_level1.conv)
                 logger.info(f"Generic CNN fallback. Target: model.features[{i}].conv ({type(module_level1.conv).__name__})")
                 found_conv_fallback = True
                 break
        if not found_conv_fallback:
            # If no conv found, take the last module in features as a last resort
            last_feature_module = model.features[-1]
            target_layers.append(last_feature_module)
            logger.warning(f"Generic CNN fallback: No obvious Conv2d in model.features for {model_name_str}. Using model.features[-1] ({type(last_feature_module).__name__}). CAM might be suboptimal.")

    if not target_layers:
        logger.error(f"CRITICAL: Failed to find any suitable target layer for {model_name_str} ({type(model).__name__}). Grad-CAM will likely fail or be incorrect.")
    
    return target_layers


def reshape_transform_vit(tensor: torch.Tensor, config_args):
    """
    Reshapes ViT output for CAM.
    Input tensor: (batch, num_tokens, embed_dim) -> (batch, embed_dim, H, W)
    """
    patch_size = 0
    model_name_l = config_args.model_name.lower()
    if "patch16" in model_name_l: patch_size = 16
    elif "patch32" in model_name_l: patch_size = 32
    elif "patch14" in model_name_l: patch_size = 14
    elif "patch8" in model_name_l: patch_size = 8
    else:
        logger.warning(f"Could not determine patch size from model_name {config_args.model_name} for ViT reshape. Assuming 16.")
        patch_size = 16

    if patch_size == 0:
        logger.error("Patch size is 0, cannot reshape ViT tensor.")
        return None
        
    # Calculate grid dimensions
    # Ensure image_size is an int; it should be from argparse
    image_size = int(config_args.image_size)
    if image_size % patch_size != 0:
        logger.warning(f"Image_size ({image_size}) not perfectly divisible by patch_size ({patch_size}). Approximating grid size for ViT reshape.")
        # Approximate grid size (this might lead to slight inaccuracies if not perfectly divisible)
        num_patches_approx = (image_size // patch_size) * (image_size // patch_size)
        # If tensor has CLS token, num_tokens = num_patches_approx + 1
        # If tensor has only patch tokens, num_tokens = num_patches_approx
        if tensor.shape[1] == num_patches_approx + 1 or tensor.shape[1] == num_patches_approx:
             height = width = image_size // patch_size
        else: # Fallback if approximation doesn't match token count
            num_tokens_for_grid = tensor.shape[1]
            if tensor.shape[1] > 1 and (tensor.shape[1]-1)**0.5 == int((tensor.shape[1]-1)**0.5) : # Check if num_tokens-1 is a perfect square (CLS token present)
                height = width = int(math.sqrt(tensor.shape[1] - 1))
            elif tensor.shape[1] > 0 and (tensor.shape[1])**0.5 == int((tensor.shape[1])**0.5): # Check if num_tokens is a perfect square (no CLS token)
                height = width = int(math.sqrt(tensor.shape[1]))
            else:
                logger.error(f"Cannot determine grid size for ViT reshape. Tensor tokens: {tensor.shape[1]}, approx patches: {num_patches_approx}. CAM will likely fail.")
                return None
    else:
        height = width = image_size // patch_size

    # Handle CLS token
    # Expected num_tokens = height * width (+1 if CLS token is present)
    if tensor.ndim == 3 and tensor.shape[1] == (height * width + 1):
        result = tensor[:, 1:, :]  # Skip CLS token
    elif tensor.ndim == 3 and tensor.shape[1] == (height * width):
        result = tensor # No CLS token, or already handled
    else:
        logger.warning(f"Unexpected ViT tensor shape {tensor.shape} for reshape. Expected num_tokens: {height*width} or {height*width+1}. CAM might be incorrect.")
        # Attempt to make it work if tokens are slightly off but > H*W
        if tensor.shape[1] > height * width:
            # If more tokens than H*W, assume CLS might be present or extra tokens, try taking last H*W
            start_index = tensor.shape[1] - (height * width)
            result = tensor[:, start_index:, :]
            logger.warning(f"  Adjusted: taking last {result.shape[1]} tokens for reshape from index {start_index}.")
        elif tensor.shape[1] < height * width :
            logger.error(f"ViT tensor has {tensor.shape[1]} tokens, less than expected {height*width}. Cannot reshape properly for CAM.")
            return None # Indicate failure
        else: # Equal, but previous checks failed (e.g. tensor.ndim !=3)
            logger.error(f"ViT tensor has an unexpected dimension or structure for reshape: {tensor.shape}")
            return None


    if result is None or result.shape[1] != height * width:
        logger.error(f"After token handling, ViT result has {result.shape[1] if result is not None else 'None'} tokens, but expected {height*width} for reshape. CAM failed for this sample.")
        return None
        
    embedding_dim = result.shape[2]
    final_result = result.reshape(tensor.size(0), height, width, embedding_dim)
    final_result = final_result.permute(0, 3, 1, 2)  # B, C (embed_dim), H, W
    return final_result


def visualize_gradcam_misclassifications(
    model,
    dataloader,
    device,
    output_dir,
    config_args, 
    num_samples_per_category=5,
    class_names=None,
    use_rgb_images=True,
    # Removed cam_method_class and target_layers_fn from params
    # as they will be determined internally
):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(config_args.num_classes)]
    if len(class_names) < 2:
        logger.error("Grad-CAM needs at least 2 class_names for binary TP/TN/FP/FN context.")
        return

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    try:
        dummy_model_for_norm = timm.create_model(config_args.model_name, pretrained=False)
        data_config = resolve_data_config({}, model=dummy_model_for_norm)
        mean, std = data_config.get('mean', (0.485, 0.456, 0.406)), data_config.get('std', (0.229, 0.224, 0.225))
    except Exception:
        logger.warning("Could not get model-specific mean/std for Grad-CAM unnormalization. Using ImageNet defaults.")
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    un_normalizer = UnNormalize(mean, std)

    all_img_paths, all_inputs, all_true_labels, all_pred_probs, all_pred_labels = [], [], [], [], []
    logger.info("Collecting predictions for Grad-CAM sample selection...")
    # ... (Data collection remains the same as your provided code) ...
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Grad-CAM Preds"):
            if batch is None: continue
            inputs, labels, metadata = batch[0], batch[1], batch[2] if len(batch) > 2 else {}
            inputs_device = inputs.to(device)
            outputs = model(inputs_device)
            probs = F.softmax(outputs, dim=1)
            all_inputs.extend(inputs.cpu())
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_probs.extend(probs.cpu().numpy())
            all_pred_labels.extend(torch.argmax(probs, dim=1).cpu().numpy())
            img_paths_batch = metadata.get('image_path', [f"unknown_img_{i}" for i in range(len(labels))])
            all_img_paths.extend(img_paths_batch)

    if not all_inputs:
        logger.warning("No inputs collected. Skipping Grad-CAM visualization.")
        return
    # ... (DataFrame creation and sample category identification remains the same) ...
    df_preds = pd.DataFrame({
        'image_path': all_img_paths, 'true_label': all_true_labels, 'pred_label': all_pred_labels,
        'prob_class0': [p[0] for p in all_pred_probs], 'prob_class1': [p[1] for p in all_pred_probs]
    })
    df_preds['input_tensor'] = all_inputs
    df_preds['is_correct'] = (df_preds['true_label'] == df_preds['pred_label'])
    df_preds['TP'] = (df_preds['true_label'] == 1) & (df_preds['pred_label'] == 1)
    df_preds['TN'] = (df_preds['true_label'] == 0) & (df_preds['pred_label'] == 0)
    df_preds['FP'] = (df_preds['true_label'] == 0) & (df_preds['pred_label'] == 1)
    df_preds['FN'] = (df_preds['true_label'] == 1) & (df_preds['pred_label'] == 0)
    df_fp_extreme = df_preds[df_preds['FP']].sort_values(by='prob_class1', ascending=False)
    df_fn_extreme = df_preds[df_preds['FN']].sort_values(by='prob_class1', ascending=True)
    samples_to_visualize = {
        'TP': df_preds[df_preds['TP']].sample(min(num_samples_per_category, len(df_preds[df_preds['TP']]))),
        'TN': df_preds[df_preds['TN']].sample(min(num_samples_per_category, len(df_preds[df_preds['TN']]))),
        'FP': df_preds[df_preds['FP']].sample(min(num_samples_per_category, len(df_preds[df_preds['FP']]))),
        'FN': df_preds[df_preds['FN']].sample(min(num_samples_per_category, len(df_preds[df_preds['FN']]))),
        'RandomCorrect': df_preds[df_preds['is_correct']].sample(min(num_samples_per_category, len(df_preds[df_preds['is_correct']]))),
        'RandomIncorrect': df_preds[~df_preds['is_correct']].sample(min(num_samples_per_category, len(df_preds[~df_preds['is_correct']]))),
        'FP_Extreme': df_fp_extreme.head(num_samples_per_category),
        'FN_Extreme': df_fn_extreme.head(num_samples_per_category)
    }

    # --- 4. Setup CAM Method and Target Layers based on Architecture ---
    model_name_lower = config_args.model_name.lower()
    target_layers = get_default_target_layers(model, config_args.model_name)
    
    cam_algorithm = None
    cam_method_name = "UnknownCAM"
    current_reshape_transform = None

    if not target_layers:
        logger.error(f"No target layers identified for model {config_args.model_name}. Cannot proceed with Grad-CAM.")
        return

    if "vit" in model_name_lower or "dinov2" in model_name_lower:
        logger.info(f"Configuring GradCAM for ViT model: {config_args.model_name}")
        cam_method_name = "GradCAM_for_ViT"
        # Bind config_args to reshape_transform_vit for the lambda
        current_reshape_transform = lambda x: reshape_transform_vit(x, config_args)
        cam_algorithm = GradCAM(model=model, target_layers=target_layers, reshape_transform=current_reshape_transform)
    elif "convnext" in model_name_lower or "resnet" in model_name_lower or "resnext" in model_name_lower:
        logger.info(f"Configuring GradCAMPlusPlus for CNN model: {config_args.model_name}")
        cam_method_name = "GradCAMPlusPlus"
        cam_algorithm = GradCAMPlusPlus(model=model, target_layers=target_layers)
    else: # Generic fallback for other CNNs
        logger.info(f"Configuring GradCAM for generic CNN model: {config_args.model_name}")
        cam_method_name = "GradCAM"
        cam_algorithm = GradCAM(model=model, target_layers=target_layers)

    if cam_algorithm is None:
        logger.error(f"Failed to initialize CAM algorithm for model {config_args.model_name}.")
        return

    logger.info(f"Using CAM method: {cam_method_name} with target layers: {[type(layer).__name__ for layer in target_layers]}")
    if current_reshape_transform:
        logger.info("Reshape transform is active for ViT.")

    # --- 5. Generate and Save Visualizations ---
    # ... (Visualization loop remains largely the same, using `cam_algorithm` and `cam_method_name`) ...
    for category_name, df_category_samples in samples_to_visualize.items():
        if df_category_samples.empty:
            logger.info(f"No samples for Grad-CAM category: {category_name}")
            continue
        
        category_output_dir = os.path.join(output_dir, category_name)
        os.makedirs(category_output_dir, exist_ok=True)
        logger.info(f"Generating CAM ({cam_method_name}) for {category_name} ({len(df_category_samples)} samples)...")

        for idx, row in tqdm(df_category_samples.iterrows(), total=len(df_category_samples), desc=f"CAM {category_name}"):
            input_tensor_cpu = row['input_tensor']
            input_tensor_device = input_tensor_cpu.unsqueeze(0).to(device)
            true_label = int(row['true_label'])
            pred_label = int(row['pred_label'])
            cam_target_class_idx = pred_label 
            targets_for_cam = [ClassifierOutputTarget(cam_target_class_idx)]
            
            try:
                # For ViTs, the reshape_transform is part of the cam_algorithm constructor.
                # It will be applied internally if it was set.
                grayscale_cam = cam_algorithm(input_tensor=input_tensor_device, targets=targets_for_cam, eigen_smooth=True)
                if grayscale_cam is None : # reshape_transform_vit might return None on failure
                    logger.warning(f"Grayscale CAM is None for {os.path.basename(row['image_path'])} in {category_name}. Skipping this sample.")
                    continue
                grayscale_cam = grayscale_cam[0, :] 
            except Exception as e_cam:
                logger.error(f"Error generating CAM for {os.path.basename(row['image_path'])} in {category_name}: {e_cam}", exc_info=True)
                continue


            img_pil_unnorm = to_pil_image(un_normalizer(input_tensor_cpu.cpu()).squeeze())
            if img_pil_unnorm.mode == 'L' and use_rgb_images:
                img_pil_unnorm = img_pil_unnorm.convert("RGB")
            
            rgb_img_for_overlay = np.array(img_pil_unnorm) / 255.0 
            if rgb_img_for_overlay.ndim == 2:
                 rgb_img_for_overlay = np.stack([rgb_img_for_overlay]*3, axis=-1)

            cam_image = show_cam_on_image(rgb_img_for_overlay, grayscale_cam, use_rgb=True, image_weight=0.5)

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(img_pil_unnorm)
            axs[0].set_title(f"Original Image\nFile: {os.path.basename(row['image_path'])}")
            axs[0].axis('off')
            axs[1].imshow(cam_image)
            axs[1].set_title(f"{cam_method_name} on Pred: {class_names[pred_label]}")
            axs[1].axis('off')
            prob_pred_class = row[f'prob_class{pred_label}']
            fig.suptitle(f"True: {class_names[true_label]}, Pred: {class_names[pred_label]} (Prob: {prob_pred_class:.3f})\n"
                         f"Category: {category_name}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            base_img_name = os.path.splitext(os.path.basename(row['image_path']))[0]
            safe_img_name = "".join(c if c.isalnum() else "_" for c in base_img_name)
            save_path = os.path.join(category_output_dir, f"{safe_img_name}_{cam_method_name}.png")
            try:
                plt.savefig(save_path)
            except Exception as e_save:
                logger.error(f"Failed to save CAM image {save_path}: {e_save}")
            plt.close(fig)