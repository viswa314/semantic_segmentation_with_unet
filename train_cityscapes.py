"""
Training script configured for Cityscapes dataset structure
Located at: /home/viswa/Downloads/Cityscapes/
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UNetRGBD
from dataset import CityscapesDataset, get_training_augmentation, get_validation_augmentation
from utils import CombinedLoss, save_checkpoint
from train import Trainer


def main():
    """Main training function with custom Cityscapes paths."""
    
    # Configuration - Optimized for CPU training
    config = {
        'in_channels': 3,  # Start with RGB only (3 channels)
        'num_classes': 34,  # Cityscapes has 34 label classes total
        'base_channels': 32,  # Reduced from 64 for less memory
        'batch_size': 1,  # Reduced to 1 for CPU stability
        'num_epochs': 20,  # Reduced epochs
        'learning_rate': 1e-4,
        'img_size': 128,  # Reduced from 256 for faster training
        'num_workers': 0,  # No multiprocessing on CPU
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_dir': 'checkpoints',
        
        # Custom paths for the dataset structure
        'images_base': '/home/viswa/Downloads/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit',
        'labels_base': '/home/viswa/Downloads/Cityscapes/gtFine_trainvaltest/gtFine',
        'depth_base': '/home/viswa/Downloads/Cityscapes/disparity_trainvaltest/disparity',
    }
    
    print("=" * 80)
    print("U-Net Training on Cityscapes Dataset")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in config.items():
        if not key.endswith('_base'):
            print(f"  {key}: {value}")
    print()
    
    print("Dataset paths:")
    print(f"  Images: {config['images_base']}")
    print(f"  Labels: {config['labels_base']}")
    print(f"  Depth: {config['depth_base']}")
    print()
    
    # Create model
    print("Creating model...")
    model = UNetRGBD(
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        base_channels=config['base_channels']
    )
    model = model.to(config['device'])
    print(f"Model created: {model.get_num_params():,} parameters")
    print(f"Device: {config['device']}")
    
    # Create custom dataset class for this structure
    from torch.utils.data import Dataset
    import cv2
    import numpy as np
    
    class CustomCityscapesDataset(Dataset):
        """Custom dataset for the specific Cityscapes structure."""
        
        def __init__(self, images_base, labels_base, split='train', transform=None, num_classes=34):
            self.images_base = images_base
            self.labels_base = labels_base
            self.split = split
            self.transform = transform
            self.num_classes = num_classes
            
            # Find all images
            self.samples = []
            split_dir = os.path.join(images_base, split)
            
            if os.path.exists(split_dir):
                for city in os.listdir(split_dir):
                    city_dir = os.path.join(split_dir, city)
                    if os.path.isdir(city_dir):
                        for img_file in os.listdir(city_dir):
                            if img_file.endswith('_leftImg8bit.png'):
                                self.samples.append((city, img_file))
            
            print(f"  {split}: Found {len(self.samples)} images")
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            city, img_file = self.samples[idx]
            
            # Load RGB
            img_path = os.path.join(self.images_base, self.split, city, img_file)
            rgb = cv2.imread(img_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            # Load label
            label_file = img_file.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            label_path = os.path.join(self.labels_base, self.split, city, label_file)
            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            # Resize
            rgb = cv2.resize(rgb, (config['img_size'], config['img_size']))
            mask = cv2.resize(mask, (config['img_size'], config['img_size']), 
                            interpolation=cv2.INTER_NEAREST)
            
            # Normalize RGB
            rgb = rgb.astype(np.float32) / 255.0
            
            # Convert to tensors
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()
            
            # Clamp mask values to valid range
            mask = torch.clamp(mask, 0, self.num_classes - 1)
            
            return rgb, mask
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CustomCityscapesDataset(
        images_base=config['images_base'],
        labels_base=config['labels_base'],
        split='train',
        num_classes=config['num_classes']
    )
    
    val_dataset = CustomCityscapesDataset(
        images_base=config['images_base'],
        labels_base=config['labels_base'],
        split='val',
        num_classes=config['num_classes']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Training batches per epoch: {len(train_loader)}")
    print()
    
    # Loss and optimizer
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
    print("=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer.train(num_epochs=config['num_epochs'], save_interval=5)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation mIoU: {trainer.best_val_iou:.4f}")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
