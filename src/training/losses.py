import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """
    Calculates the Dice Loss between predicted and target masks.

    Args:
        pred (torch.Tensor): Predicted mask tensor (usually after sigmoid), shape (N, 1, H, W).
        target (torch.Tensor): Ground truth mask tensor (binary 0 or 1), shape (N, 1, H, W).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: The calculated Dice loss (scalar).
    """
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)

    dice_coefficient = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    loss = 1 - dice_coefficient

    return loss.mean() # Return average loss over the batch

def multi_task_segmentation_loss(pred_oc: torch.Tensor, target_oc: torch.Tensor,
                                 pred_od: torch.Tensor, target_od: torch.Tensor,
                                 loss_type: str = 'dice', alpha: float = 0.5, smooth: float = 1e-5) -> tuple[torch.Tensor, float, float]:
    """
    Combined loss function for multi-task segmentation (OC and OD).
    Can use Dice loss or Binary Cross-Entropy (BCE).

    Args:
        pred_oc (torch.Tensor): Predicted optic cup mask (after sigmoid).
        target_oc (torch.Tensor): Ground truth optic cup mask.
        pred_od (torch.Tensor): Predicted optic disc mask (after sigmoid).
        target_od (torch.Tensor): Ground truth optic disc mask.
        loss_type (str): Type of loss to use ('dice' or 'bce'). Defaults to 'dice'.
        alpha (float): Weighting factor for the optic cup loss (0 <= alpha <= 1).
                       Weight for optic disc loss will be (1 - alpha). Defaults to 0.5.
        smooth (float): Smoothing factor for Dice loss. Defaults to 1e-5.

    Returns:
        tuple[torch.Tensor, float, float]:
            - Combined weighted loss (tensor).
            - Optic cup loss value (float).
            - Optic disc loss value (float).
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    if loss_type.lower() == 'dice':
        loss_oc = dice_loss(pred_oc, target_oc, smooth)
        loss_od = dice_loss(pred_od, target_od, smooth)
    elif loss_type.lower() == 'bce':
        # Ensure target tensors are float for BCE
        loss_oc = F.binary_cross_entropy(pred_oc, target_oc.float())
        loss_od = F.binary_cross_entropy(pred_od, target_od.float())
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'dice' or 'bce'.")

    # Weighted sum of both losses
    combined_loss = alpha * loss_oc + (1 - alpha) * loss_od

    return combined_loss, loss_oc.item(), loss_od.item()


# Example usage (optional, for testing)
if __name__ == '__main__':
    # Dummy data
    N, C, H, W = 2, 1, 64, 64 # Batch size 2, 1 channel, 64x64
    pred_oc_t = torch.sigmoid(torch.randn(N, C, H, W))
    target_oc_t = (torch.rand(N, C, H, W) > 0.5).float() # Binary target
    pred_od_t = torch.sigmoid(torch.randn(N, C, H, W))
    target_od_t = (torch.rand(N, C, H, W) > 0.5).float()

    print("--- Testing Dice Loss ---")
    loss_d, oc_d, od_d = multi_task_segmentation_loss(pred_oc_t, target_oc_t, pred_od_t, target_od_t, loss_type='dice', alpha=0.6)
    print(f"Combined Dice Loss (alpha=0.6): {loss_d.item():.4f}")
    print(f"OC Dice Loss: {oc_d:.4f}")
    print(f"OD Dice Loss: {od_d:.4f}")

    print("\n--- Testing BCE Loss ---")
    loss_b, oc_b, od_b = multi_task_segmentation_loss(pred_oc_t, target_oc_t, pred_od_t, target_od_t, loss_type='bce', alpha=0.5)
    print(f"Combined BCE Loss (alpha=0.5): {loss_b.item():.4f}")
    print(f"OC BCE Loss: {oc_b:.4f}")
    print(f"OD BCE Loss: {od_b:.4f}")