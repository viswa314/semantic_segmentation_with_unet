"""
Utility functions for semantic segmentation

This module provides utility functions for metrics computation, visualization,
and model analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os


def calculate_iou(pred, target, num_classes):
    """
    Calculate Intersection over Union (IoU) for semantic segmentation.
    
    Args:
        pred (torch.Tensor): Predicted segmentation (B, H, W) or (B, C, H, W)
        target (torch.Tensor): Ground truth segmentation (B, H, W)
        num_classes (int): Number of classes
        
    Returns:
        dict: IoU scores per class and mean IoU
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = float('nan')
        else:
            iou = (intersection / union).item()
        
        ious.append(iou)
    
    # Calculate mean IoU (ignoring NaN values)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    return {
        'per_class_iou': ious,
        'mean_iou': mean_iou,
        'num_valid_classes': len(valid_ious)
    }


def calculate_dice_score(pred, target, num_classes):
    """
    Calculate Dice Score (F1 Score) for semantic segmentation.
    
    Args:
        pred (torch.Tensor): Predicted segmentation (B, H, W) or (B, C, H, W)
        target (torch.Tensor): Ground truth segmentation (B, H, W)
        num_classes (int): Number of classes
        
    Returns:
        dict: Dice scores per class and mean Dice
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    dice_scores = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        pred_sum = pred_cls.sum().float()
        target_sum = target_cls.sum().float()
        
        if (pred_sum + target_sum) == 0:
            dice = float('nan')
        else:
            dice = (2 * intersection / (pred_sum + target_sum)).item()
        
        dice_scores.append(dice)
    
    # Calculate mean Dice (ignoring NaN values)
    valid_dice = [d for d in dice_scores if not np.isnan(d)]
    mean_dice = np.mean(valid_dice) if valid_dice else 0.0
    
    return {
        'per_class_dice': dice_scores,
        'mean_dice': mean_dice,
        'num_valid_classes': len(valid_dice)
    }


def calculate_pixel_accuracy(pred, target):
    """
    Calculate pixel-wise accuracy.
    
    Args:
        pred (torch.Tensor): Predicted segmentation (B, H, W) or (B, C, H, W)
        target (torch.Tensor): Ground truth segmentation (B, H, W)
        
    Returns:
        float: Pixel accuracy
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    correct = (pred == target).sum().item()
    total = target.numel()
    
    return correct / total


class DiceLoss(torch.nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted logits (B, C, H, W)
            target (torch.Tensor): Ground truth (B, H, W)
            
        Returns:
            torch.Tensor: Dice loss
        """
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten
        pred = pred.contiguous().view(-1)
        target_one_hot = target_one_hot.contiguous().view(-1)
        
        intersection = (pred * target_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target_one_hot.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(torch.nn.Module):
    """
    Combined Cross Entropy and Dice Loss.
    
    Args:
        ce_weight (float): Weight for cross entropy loss
        dice_weight (float): Weight for dice loss
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


def visualize_prediction(image, target, prediction, num_classes=19, save_path=None):
    """
    Visualize RGB-D image, ground truth, and prediction side by side.
    
    Args:
        image (torch.Tensor): Input image (C, H, W) where C is 3 or 4
        target (torch.Tensor): Ground truth mask (H, W)
        prediction (torch.Tensor): Predicted mask (H, W) or (C, H, W)
        num_classes (int): Number of classes for colormap
        save_path (str, optional): Path to save the figure
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        if prediction.dim() == 3:
            prediction = torch.argmax(prediction, dim=0)
        prediction = prediction.cpu().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # RGB image (first 3 channels)
    rgb = image[:3].transpose(1, 2, 0)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # Normalize for display
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Depth channel (if available)
    if image.shape[0] == 4:
        depth = image[3]
        axes[1].imshow(depth, cmap='gray')
        axes[1].set_title('Depth Channel')
    else:
        axes[1].text(0.5, 0.5, 'No Depth', ha='center', va='center')
        axes[1].set_title('Depth Channel')
    axes[1].axis('off')
    
    # Ground truth
    axes[2].imshow(target, cmap='tab20', vmin=0, vmax=num_classes-1)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    # Prediction
    axes[3].imshow(prediction, cmap='tab20', vmin=0, vmax=num_classes-1)
    axes[3].set_title('Prediction')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_model_summary(model, input_size=(4, 256, 256)):
    """
    Print a summary of the model architecture.
    
    Args:
        model (torch.nn.Module): The model
        input_size (tuple): Input size (C, H, W)
    """
    from model import count_parameters
    
    print("=" * 80)
    print("Model Architecture Summary")
    print("=" * 80)
    
    print(f"\nModel: {model.__class__.__name__}")
    
    # Count parameters
    params = count_parameters(model)
    print(f"\nTotal Parameters: {params['total_readable']}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, *input_size)
    
    print(f"\nInput Shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output Shape: {output.shape}")
    
    print("\n" + "=" * 80)


def save_checkpoint(model, optimizer, epoch, loss, save_dir='checkpoints', 
                   filename=None, best=False):
    """
    Save model checkpoint.
    
    Args:
        model (torch.nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer state
        epoch (int): Current epoch
        loss (float): Current loss
        save_dir (str): Directory to save checkpoint
        filename (str, optional): Checkpoint filename
        best (bool): Whether this is the best model so far
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    
    checkpoint_path = os.path.join(save_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    if best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        model (torch.nn.Module): Model to load weights into
        checkpoint_path (str): Path to checkpoint file
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
        
    Returns:
        int: Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {epoch}, Loss: {loss:.4f}")
    
    return epoch


if __name__ == "__main__":
    print("Testing Utility Functions...")
    
    # Test metrics
    pred = torch.randint(0, 5, (2, 128, 128))
    target = torch.randint(0, 5, (2, 128, 128))
    
    iou_result = calculate_iou(pred, target, num_classes=5)
    print(f"\nIoU Test: Mean IoU = {iou_result['mean_iou']:.4f}")
    
    dice_result = calculate_dice_score(pred, target, num_classes=5)
    print(f"Dice Test: Mean Dice = {dice_result['mean_dice']:.4f}")
    
    pixel_acc = calculate_pixel_accuracy(pred, target)
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    
    print("\n✓ Utility functions test passed!")
