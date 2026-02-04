"""
RGB-D Inference Demo - Visualize predictions on RGB-D images
Shows RGB, Depth, Ground Truth, and Prediction side by side
"""

import os
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import UNetRGBD


def load_rgbd_model(checkpoint_path, num_classes=34, device='cpu'):
    """Load trained RGB-D model."""
    print(f"Loading RGB-D model from {checkpoint_path}...")
    model = UNetRGBD(in_channels=4, num_classes=num_classes, base_channels=64)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✓ RGB-D Model loaded (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
    return model


def run_rgbd_inference(model, rgb_path, depth_path, label_path, img_size=128, device='cpu'):
    """Run inference on RGB-D image."""
    # Load RGB
    rgb = cv2.imread(rgb_path)
    original_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # Load Depth
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    original_depth = depth.copy()
    depth = cv2.resize(depth, (img_size, img_size))
    
    # Load ground truth label
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    
    # Normalize
    rgb_normalized = rgb.astype(np.float32) / 255.0
    depth_normalized = depth.astype(np.float32) / 255.0
    
    # Create RGBD tensor (4 channels)
    rgbd = np.concatenate([rgb_normalized, depth_normalized[:, :, np.newaxis]], axis=-1)
    rgbd_tensor = torch.from_numpy(rgbd).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(rgbd_tensor)
        prediction = torch.argmax(output, dim=1)[0].cpu().numpy()
    
    return original_rgb, rgb, depth, label, prediction


def visualize_rgbd_results(original_rgb, rgb, depth, label, prediction, save_path):
    """Create comprehensive RGB-D visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Original, RGB Input, Depth Input
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original RGB Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rgb)
    axes[0, 1].set_title('RGB Input (128×128)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(depth, cmap='plasma')
    axes[0, 2].set_title('Depth Map (Disparity)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: RGBD Overlay, Ground Truth, Prediction
    # Create RGB-D overlay visualization
    depth_colored = cv2.applyColorMap((depth).astype(np.uint8), cv2.COLORMAP_PLASMA)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    rgbd_overlay = cv2.addWeighted(rgb, 0.6, depth_colored, 0.4, 0)
    
    axes[1, 0].imshow(rgbd_overlay)
    axes[1, 0].set_title('RGB-D Overlay', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(label, cmap='tab20', vmin=0, vmax=33)
    axes[1, 1].set_title('Ground Truth Segmentation', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(prediction, cmap='tab20', vmin=0, vmax=33)
    axes[1, 2].set_title('RGB-D U-Net Prediction', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ RGB-D Visualization saved to {save_path}")
    plt.close()


def main():
    """Run RGB-D inference demo."""
    
    config = {
        'checkpoint_path': 'checkpoints_rgbd/best_rgbd_model.pth',
        'num_classes': 34,
        'img_size': 256,
        'device': 'cpu',
        'images_base': '/home/viswa/Downloads/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val',
        'labels_base': '/home/viswa/Downloads/Cityscapes/gtFine_trainvaltest/gtFine/val',
        'depth_base': '/home/viswa/Downloads/Cityscapes/disparity_trainvaltest/disparity/val',
        'num_samples': 5,
        'output_dir': 'inference_results_rgbd'
    }
    
    print("=" * 80)
    print("U-Net RGB-D Inference Demo")
    print("=" * 80)
    print(f"Device: {config['device']}")
    print(f"Input: RGB + Depth (4 channels)")
    print(f"Image size: {config['img_size']}×{config['img_size']}")
    print(f"Testing on {config['num_samples']} validation images")
    print()
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load model
    model = load_rgbd_model(config['checkpoint_path'], config['num_classes'], config['device'])
    print()
    
    # Find validation RGB-D samples
    print("Finding RGB-D validation images...")
    samples = []
    for city in os.listdir(config['images_base']):
        city_img_dir = os.path.join(config['images_base'], city)
        if os.path.isdir(city_img_dir):
            for img_file in os.listdir(city_img_dir):
                if img_file.endswith('_leftImg8bit.png'):
                    # Check for depth
                    depth_file = img_file.replace('_leftImg8bit.png', '_disparity.png')
                    depth_path = os.path.join(config['depth_base'], city, depth_file)
                    
                    # Check for label
                    label_file = img_file.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    label_path = os.path.join(config['labels_base'], city, label_file)
                    
                    if os.path.exists(depth_path) and os.path.exists(label_path):
                        samples.append((
                            os.path.join(city_img_dir, img_file),
                            depth_path,
                            label_path,
                            f"{city}_{img_file}"
                        ))
                    if len(samples) >= config['num_samples']:
                        break
        if len(samples) >= config['num_samples']:
            break
    
    print(f"✓ Found {len(samples)} RGB-D images")
    print()
    
    # Run inference on each sample
    print("Running RGB-D inference...")
    print("-" * 80)
    
    for idx, (rgb_path, depth_path, label_path, name) in enumerate(samples, 1):
        print(f"\nSample {idx}/{len(samples)}: {name}")
        
        # Run inference
        original_rgb, rgb, depth, label, prediction = run_rgbd_inference(
            model, rgb_path, depth_path, label_path, config['img_size'], config['device']
        )
        
        # Calculate metrics
        correct = (prediction == label).sum()
        total = label.size
        accuracy = correct / total
        
        print(f"  Pixel Accuracy: {accuracy:.2%}")
        print(f"  Unique classes in GT: {len(np.unique(label))}")
        print(f"  Unique classes predicted: {len(np.unique(prediction))}")
        
        # Visualize
        output_path = os.path.join(config['output_dir'], f'rgbd_result_{idx}.png')
        visualize_rgbd_results(original_rgb, rgb, depth, label, prediction, output_path)
    
    print()
    print("=" * 80)
    print("RGB-D Inference completed!")
    print(f"Results saved to: {config['output_dir']}/")
    print("=" * 80)
    print("\nVisualization includes:")
    print("  - Original RGB image")
    print("  - RGB input (resized)")
    print("  - Depth map (disparity)")
    print("  - RGB-D overlay")
    print("  - Ground truth segmentation")
    print("  - RGB-D U-Net prediction")


if __name__ == "__main__":
    main()
