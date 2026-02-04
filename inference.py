"""
Example Inference Script for U-Net Semantic Segmentation

This script demonstrates how to use the trained U-Net model for inference
on new RGB-D images.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from model import UNetRGBD
from utils import visualize_prediction


def load_model(checkpoint_path, in_channels=4, num_classes=19, device='cuda'):
    """
    Load a trained U-Net model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        in_channels (int): Number of input channels (3 for RGB, 4 for RGB-D)
        num_classes (int): Number of segmentation classes
        device (str): Device to load model on
        
    Returns:
        model: Loaded U-Net model in eval mode
    """
    # Create model
    model = UNetRGBD(in_channels=in_channels, num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Final loss: {checkpoint['loss']:.4f}")
    
    return model


def preprocess_image(rgb_path, depth_path=None, target_size=(256, 256)):
    """
    Preprocess RGB-D image for inference.
    
    Args:
        rgb_path (str): Path to RGB image
        depth_path (str, optional): Path to depth map
        target_size (tuple): Target (height, width)
        
    Returns:
        torch.Tensor: Preprocessed image (1, C, H, W)
    """
    # Load RGB
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (target_size[1], target_size[0]))
    
    # Normalize RGB (ImageNet stats)
    rgb = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb = (rgb - mean) / std
    
    # Load depth if available
    if depth_path is not None:
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, (target_size[1], target_size[0]))
        depth = depth.astype(np.float32) / 255.0
        depth = np.expand_dims(depth, axis=-1)
        
        # Concatenate RGB and depth
        rgbd = np.concatenate([rgb, depth], axis=-1)
    else:
        # Add zero depth channel
        depth = np.zeros((target_size[0], target_size[1], 1), dtype=np.float32)
        rgbd = np.concatenate([rgb, depth], axis=-1)
    
    # Convert to tensor (H, W, C) -> (C, H, W)
    rgbd = torch.from_numpy(rgbd).permute(2, 0, 1).unsqueeze(0)
    
    return rgbd


def inference(model, image, device='cuda'):
    """
    Run inference on a single image.
    
    Args:
        model: U-Net model
        image (torch.Tensor): Preprocessed image (1, C, H, W)
        device (str): Device for inference
        
    Returns:
        prediction (torch.Tensor): Predicted segmentation map (H, W)
        probabilities (torch.Tensor): Class probabilities (num_classes, H, W)
    """
    image = image.to(device)
    
    with torch.no_grad():
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)[0]  # (num_classes, H, W)
        prediction = torch.argmax(logits, dim=1)[0]  # (H, W)
    
    return prediction.cpu(), probabilities.cpu()


def save_prediction(prediction, save_path, colormap='tab20'):
    """
    Save segmentation prediction as colored image.
    
    Args:
        prediction (torch.Tensor or np.ndarray): Segmentation map (H, W)
        save_path (str): Path to save image
        colormap (str): Matplotlib colormap name
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(prediction, cmap=colormap)
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Prediction saved to {save_path}")


def main():
    """Example usage."""
    
    # Configuration
    config = {
        'checkpoint_path': 'checkpoints/best_model.pth',
        'rgb_path': 'example_rgb.png',
        'depth_path': 'example_depth.png',  # Set to None if no depth
        'output_path': 'prediction.png',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'in_channels': 4,  # 3 for RGB, 4 for RGB-D
        'num_classes': 19,
        'img_size': (256, 256)
    }
    
    print("=" * 80)
    print("U-Net Inference")
    print("=" * 80)
    print(f"Device: {config['device']}")
    print(f"RGB Image: {config['rgb_path']}")
    print(f"Depth Image: {config['depth_path']}")
    print("=" * 80)
    
    # Load model
    model = load_model(
        config['checkpoint_path'],
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        device=config['device']
    )
    
    # Preprocess image
    print("\nPreprocessing image...")
    image = preprocess_image(
        config['rgb_path'],
        config['depth_path'],
        target_size=config['img_size']
    )
    print(f"Input shape: {image.shape}")
    
    # Run inference
    print("\nRunning inference...")
    prediction, probabilities = inference(model, image, config['device'])
    print(f"Prediction shape: {prediction.shape}")
    print(f"Unique classes: {torch.unique(prediction).tolist()}")
    
    # Save prediction
    print("\nSaving results...")
    save_prediction(prediction, config['output_path'])
    
    # Optional: Compute confidence
    confidence = probabilities.max(dim=0)[0]
    mean_confidence = confidence.mean().item()
    print(f"Mean prediction confidence: {mean_confidence:.4f}")
    
    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Example of batch inference
    print("""
    Usage Examples:
    
    1. Single Image Inference:
       python inference.py
    
    2. Custom paths in code:
       config = {
           'checkpoint_path': 'checkpoints/best_model.pth',
           'rgb_path': 'path/to/your/image.png',
           'depth_path': 'path/to/your/depth.png',  # or None
           'output_path': 'output.png'
       }
    
    3. Batch inference - modify main() to loop over directory:
       import os
       for img_file in os.listdir('input_dir'):
           rgb_path = os.path.join('input_dir', img_file)
           # ... run inference ...
    
    Note: Before running, ensure you have a trained model checkpoint at
    'checkpoints/best_model.pth' from training.
    """)
    
    # Uncomment to run example
    # main()
