"""
RGB-D Training Script - Train U-Net with RGB + Depth channels
Uses Cityscapes RGB images + Disparity maps
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2
import numpy as np

from model import UNetRGBD
from utils import CombinedLoss, save_checkpoint
from train import Trainer


class CityscapesRGBDDataset(Dataset):
    """Cityscapes dataset with RGB + Disparity (depth)."""
    
    def __init__(self, images_base, labels_base, depth_base, split='train', 
                 transform=None, num_classes=34, img_size=128):
        self.images_base = images_base
        self.labels_base = labels_base
        self.depth_base = depth_base
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Find all samples
        self.samples = []
        split_dir = os.path.join(images_base, split)
        
        if os.path.exists(split_dir):
            for city in os.listdir(split_dir):
                city_dir = os.path.join(split_dir, city)
                if os.path.isdir(city_dir):
                    for img_file in os.listdir(city_dir):
                        if img_file.endswith('_leftImg8bit.png'):
                            # Check if matching depth exists
                            depth_file = img_file.replace('_leftImg8bit.png', '_disparity.png')
                            depth_path = os.path.join(depth_base, split, city, depth_file)
                            
                            if os.path.exists(depth_path):
                                self.samples.append((city, img_file))
        
        print(f"  {split}: Found {len(self.samples)} RGB-D images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        city, img_file = self.samples[idx]
        
        # Load RGB
        img_path = os.path.join(self.images_base, self.split, city, img_file)
        rgb = cv2.imread(img_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        
        # Load Depth (disparity)
        depth_file = img_file.replace('_leftImg8bit.png', '_disparity.png')
        depth_path = os.path.join(self.depth_base, self.split, city, depth_file)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, (self.img_size, self.img_size))
        
        # Load Label
        label_file = img_file.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        label_path = os.path.join(self.labels_base, self.split, city, label_file)
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size), 
                         interpolation=cv2.INTER_NEAREST)
        
        # Normalize
        rgb = rgb.astype(np.float32) / 255.0
        depth = depth.astype(np.float32) / 255.0
        
        # Create RGBD (4 channels)
        rgbd = np.concatenate([rgb, depth[:, :, np.newaxis]], axis=-1)
        
        # Convert to tensors
        rgbd = torch.from_numpy(rgbd).permute(2, 0, 1).float()  # (4, H, W)
        mask = torch.from_numpy(mask).long()
        
        # Clamp mask
        mask = torch.clamp(mask, 0, self.num_classes - 1)
        
        return rgbd, mask


def main():
    """Train U-Net with RGB-D input."""
    
    # Configuration for RGB-D
    config = {
        'in_channels': 4,  # RGB-D (4 channels)
        'num_classes': 34,
        'base_channels': 32,
        'batch_size': 1,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'img_size': 128,
        'num_workers': 0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_dir': 'checkpoints_rgbd',
        
        # Dataset paths
        'images_base': '/home/viswa/Downloads/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit',
        'labels_base': '/home/viswa/Downloads/Cityscapes/gtFine_trainvaltest/gtFine',
        'depth_base': '/home/viswa/Downloads/Cityscapes/disparity_trainvaltest/disparity',
    }
    
    print("=" * 80)
    print("U-Net RGB-D Training on Cityscapes Dataset")
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
    print("Creating RGB-D model...")
    model = UNetRGBD(
        in_channels=config['in_channels'],  # 4 channels!
        num_classes=config['num_classes'],
        base_channels=config['base_channels']
    )
    model = model.to(config['device'])
    print(f"Model created: {model.get_num_params():,} parameters")
    print(f"Device: {config['device']}")
    print(f"Input channels: {config['in_channels']} (RGB + Depth)")
    print()
    
    # Create datasets
    print("Loading RGB-D datasets...")
    train_dataset = CityscapesRGBDDataset(
        images_base=config['images_base'],
        labels_base=config['labels_base'],
        depth_base=config['depth_base'],
        split='train',
        num_classes=config['num_classes'],
        img_size=config['img_size']
    )
    
    val_dataset = CityscapesRGBDDataset(
        images_base=config['images_base'],
        labels_base=config['labels_base'],
        depth_base=config['depth_base'],
        split='val',
        num_classes=config['num_classes'],
        img_size=config['img_size']
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
    print("Starting RGB-D training...")
    print("=" * 80)
    trainer.train(num_epochs=config['num_epochs'], save_interval=5)
    
    print("\n" + "=" * 80)
    print("RGB-D Training completed!")
    print(f"Best validation mIoU: {trainer.best_val_iou:.4f}")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
