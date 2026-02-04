"""
Dataset module for RGB-D Semantic Segmentation

This module provides dataset classes and utilities for loading RGB-D images
and their corresponding segmentation masks. Supports the Cityscapes dataset
and custom RGB-D datasets.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class RGBDDataset(Dataset):
    """
    Dataset class for RGB-D semantic segmentation.
    
    This dataset can handle:
    1. RGB + Depth image pairs
    2. RGB-only images (depth channel filled with zeros)
    3. Cityscapes dataset format
    
    Args:
        rgb_dir (str): Directory containing RGB images
        mask_dir (str): Directory containing segmentation masks
        depth_dir (str, optional): Directory containing depth maps. If None, 
                                   depth channel is filled with zeros.
        transform (albumentations.Compose, optional): Data augmentation pipeline
        num_classes (int): Number of segmentation classes
        rgb_only (bool): If True, use 3-channel RGB instead of 4-channel RGB-D
    """
    def __init__(self, rgb_dir, mask_dir, depth_dir=None, transform=None, 
                 num_classes=19, rgb_only=False):
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.num_classes = num_classes
        self.rgb_only = rgb_only
        
        # Get list of image files
        self.images = sorted([f for f in os.listdir(rgb_dir) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(self.images)} images in {rgb_dir}")
        if depth_dir:
            print(f"Using depth maps from {depth_dir}")
        else:
            print("No depth maps provided, using RGB-only mode")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load RGB image
        img_name = self.images[idx]
        rgb_path = os.path.join(self.rgb_dir, img_name)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = img_name  # Assuming same name, adjust if needed
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Load depth if available
        if self.depth_dir is not None and not self.rgb_only:
            depth_path = os.path.join(self.depth_dir, img_name)
            if os.path.exists(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            else:
                # If depth file doesn't exist, create zero depth
                depth = np.zeros_like(mask)
        else:
            depth = None
        
        # Apply transformations
        if self.transform:
            if depth is not None:
                transformed = self.transform(image=rgb, mask=mask, depth=depth)
                rgb = transformed['image']
                mask = transformed['mask']
                depth = transformed['depth']
            else:
                transformed = self.transform(image=rgb, mask=mask)
                rgb = transformed['image']
                mask = transformed['mask']
        
        # Convert to tensors if not already
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        
        # Concatenate RGB and Depth
        if depth is not None and not self.rgb_only:
            if not isinstance(depth, torch.Tensor):
                depth = torch.from_numpy(depth).unsqueeze(0).float() / 255.0
            rgbd = torch.cat([rgb, depth], dim=0)  # (4, H, W)
        else:
            if self.rgb_only:
                rgbd = rgb  # (3, H, W)
            else:
                # Add zero depth channel
                depth_zeros = torch.zeros(1, rgb.shape[1], rgb.shape[2])
                rgbd = torch.cat([rgb, depth_zeros], dim=0)  # (4, H, W)
        
        return rgbd, mask


class CityscapesDataset(Dataset):
    """
    Specialized dataset for Cityscapes format.
    
    Cityscapes uses specific naming conventions and directory structures.
    This class handles the Cityscapes dataset format with optional depth.
    
    Args:
        root_dir (str): Root directory of Cityscapes dataset
        split (str): 'train', 'val', or 'test'
        transform (albumentations.Compose, optional): Data augmentation
        num_classes (int): Number of classes (default: 19 for Cityscapes)
        use_depth (bool): Whether to use depth information
    """
    def __init__(self, root_dir, split='train', transform=None, 
                 num_classes=19, use_depth=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        self.use_depth = use_depth
        
        # Cityscapes directory structure
        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.labels_dir = os.path.join(root_dir, 'gtFine', split)
        
        if use_depth:
            self.depth_dir = os.path.join(root_dir, 'disparity', split)
        
        # Collect all images
        self.images = []
        if os.path.exists(self.images_dir):
            for city in os.listdir(self.images_dir):
                city_img_dir = os.path.join(self.images_dir, city)
                if os.path.isdir(city_img_dir):
                    for img_file in os.listdir(city_img_dir):
                        if img_file.endswith('_leftImg8bit.png'):
                            self.images.append((city, img_file))
        
        print(f"Cityscapes {split}: Found {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        city, img_file = self.images[idx]
        
        # Load RGB image
        img_path = os.path.join(self.images_dir, city, img_file)
        rgb = cv2.imread(img_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load label (use labelIds version)
        label_file = img_file.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        label_path = os.path.join(self.labels_dir, city, label_file)
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # Load depth/disparity if requested
        depth = None
        if self.use_depth:
            depth_file = img_file.replace('_leftImg8bit.png', '_disparity.png')
            depth_path = os.path.join(self.depth_dir, city, depth_file)
            if os.path.exists(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations
        if self.transform:
            if depth is not None:
                transformed = self.transform(image=rgb, mask=mask, depth=depth)
                rgb = transformed['image']
                mask = transformed['mask']
                depth = transformed['depth']
            else:
                transformed = self.transform(image=rgb, mask=mask)
                rgb = transformed['image']
                mask = transformed['mask']
        
        # Convert to tensors
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        
        # Create RGBD tensor
        if depth is not None and self.use_depth:
            if not isinstance(depth, torch.Tensor):
                depth = torch.from_numpy(depth).unsqueeze(0).float() / 255.0
            rgbd = torch.cat([rgb, depth], dim=0)
        else:
            if self.use_depth:
                # Add zero depth channel
                depth_zeros = torch.zeros(1, rgb.shape[1], rgb.shape[2])
                rgbd = torch.cat([rgb, depth_zeros], dim=0)
            else:
                rgbd = rgb  # RGB only
        
        return rgbd, mask


def get_training_augmentation(height=256, width=256):
    """
    Get training augmentation pipeline.
    
    Args:
        height (int): Target height
        width (int): Target width
        
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    train_transform = A.Compose([
        A.Resize(height, width),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_transform


def get_validation_augmentation(height=256, width=256):
    """
    Get validation augmentation pipeline (minimal augmentation).
    
    Args:
        height (int): Target height
        width (int): Target width
        
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    val_transform = A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return val_transform


def create_dataloaders(rgb_dir, mask_dir, depth_dir=None, batch_size=8, 
                       num_workers=4, img_size=256, num_classes=19, 
                       rgb_only=False, val_split=0.2):
    """
    Create training and validation dataloaders.
    
    Args:
        rgb_dir (str): Directory with RGB images
        mask_dir (str): Directory with segmentation masks
        depth_dir (str, optional): Directory with depth maps
        batch_size (int): Batch size
        num_workers (int): Number of data loading workers
        img_size (int): Image size (assumes square images)
        num_classes (int): Number of segmentation classes
        rgb_only (bool): Use RGB only (3 channels)
        val_split (float): Fraction of data to use for validation
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create dataset
    dataset = RGBDDataset(
        rgb_dir=rgb_dir,
        mask_dir=mask_dir,
        depth_dir=depth_dir,
        transform=get_training_augmentation(img_size, img_size),
        num_classes=num_classes,
        rgb_only=rgb_only
    )
    
    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing Dataset Module...")
    
    # This is a basic test - you would need actual data to run it
    print("Dataset module loaded successfully!")
    print("Classes defined:")
    print("  - RGBDDataset: General RGB-D dataset")
    print("  - CityscapesDataset: Cityscapes-specific dataset")
    print("  - get_training_augmentation()")
    print("  - get_validation_augmentation()")
    print("  - create_dataloaders()")
