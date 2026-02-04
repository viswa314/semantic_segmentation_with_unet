"""
Training script for U-Net RGB-D Semantic Segmentation

This script provides a complete training pipeline for the U-Net model,
including training loop, validation, metrics, and checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import UNetRGBD
from dataset import RGBDDataset, CityscapesDataset, get_training_augmentation, get_validation_augmentation
from utils import (
    calculate_iou, calculate_dice_score, calculate_pixel_accuracy,
    CombinedLoss, visualize_prediction, save_checkpoint, load_checkpoint
)


class Trainer:
    """
    Trainer class for U-Net semantic segmentation.
    
    Args:
        model (nn.Module): U-Net model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        num_classes (int): Number of segmentation classes
        checkpoint_dir (str): Directory to save checkpoints
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 device, num_classes, checkpoint_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_ious = []
        self.val_ious = []
        
        self.best_val_iou = 0.0
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        iou_result = calculate_iou(all_preds, all_targets, self.num_classes)
        mean_iou = iou_result['mean_iou']
        
        self.train_losses.append(avg_loss)
        self.train_ious.append(mean_iou)
        
        print(f'Epoch {epoch} Training - Loss: {avg_loss:.4f}, mIoU: {mean_iou:.4f}')
        
        return avg_loss, mean_iou
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                
                running_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(self.val_loader)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        iou_result = calculate_iou(all_preds, all_targets, self.num_classes)
        mean_iou = iou_result['mean_iou']
        
        dice_result = calculate_dice_score(all_preds, all_targets, self.num_classes)
        mean_dice = dice_result['mean_dice']
        
        pixel_acc = calculate_pixel_accuracy(all_preds, all_targets)
        
        self.val_losses.append(avg_loss)
        self.val_ious.append(mean_iou)
        
        print(f'Epoch {epoch} Validation - Loss: {avg_loss:.4f}, mIoU: {mean_iou:.4f}, '
              f'Dice: {mean_dice:.4f}, Pixel Acc: {pixel_acc:.4f}')
        
        return avg_loss, mean_iou, mean_dice, pixel_acc
    
    def train(self, num_epochs, save_interval=5):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs (int): Number of epochs to train
            save_interval (int): Save checkpoint every N epochs
        """
        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 80)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_iou = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_iou, val_dice, val_pixel_acc = self.validate(epoch)
            
            # Save checkpoint
            if epoch % save_interval == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    save_dir=self.checkpoint_dir,
                    filename=f'checkpoint_epoch_{epoch}.pth'
                )
            
            # Save best model
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    save_dir=self.checkpoint_dir,
                    filename='best_model.pth',
                    best=True
                )
            
            print("-" * 80)
        
        print("\nTraining completed!")
        print(f"Best validation mIoU: {self.best_val_iou:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        epochs = range(1, len(self.train_losses) + 1)
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # IoU plot
        axes[1].plot(epochs, self.train_ious, 'b-', label='Train mIoU')
        axes[1].plot(epochs, self.val_ious, 'r-', label='Val mIoU')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mIoU')
        axes[1].set_title('Training and Validation mIoU')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.checkpoint_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
        plt.close()


def main():
    """Main training function."""
    
    # Configuration
    config = {
        'in_channels': 3,  # 3 for RGB, 4 for RGB-D
        'num_classes': 19,  # Cityscapes has 19 classes
        'base_channels': 64,
        'batch_size': 4,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'img_size': 256,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_dir': 'checkpoints',
        'dataset_root': '/home/viswa/Downloads/Cityscapes',
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create model
    model = UNetRGBD(
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        base_channels=config['base_channels']
    )
    model = model.to(config['device'])
    
    print(f"Model created: {model.get_num_params():,} parameters")
    
    # Create datasets and dataloaders
    # Note: This assumes Cityscapes dataset structure
    # Adjust paths as needed for your dataset
    try:
        train_dataset = CityscapesDataset(
            root_dir=config['dataset_root'],
            split='train',
            transform=get_training_augmentation(config['img_size'], config['img_size']),
            num_classes=config['num_classes'],
            use_depth=False  # Set to True if you have depth data
        )
        
        val_dataset = CityscapesDataset(
            root_dir=config['dataset_root'],
            split='val',
            transform=get_validation_augmentation(config['img_size'], config['img_size']),
            num_classes=config['num_classes'],
            use_depth=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset path is correct and the dataset is properly formatted.")
        return
    
    # Loss function and optimizer
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config['device'],
        num_classes=config['num_classes'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'], save_interval=5)


if __name__ == "__main__":
    main()
