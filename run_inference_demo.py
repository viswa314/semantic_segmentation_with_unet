"""
Demo Inference Script - Run U-Net on Cityscapes validation images
"""

import os
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from model import UNetRGBD


def load_model(checkpoint_path, num_classes=34, device='cpu'):
    """Load trained model."""
    print(f"Loading model from {checkpoint_path}...")
    model = UNetRGBD(in_channels=3, num_classes=num_classes, base_channels=32)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
    return model


def run_inference_on_sample(model, rgb_path, label_path, img_size=128, device='cpu'):
    """Run inference on a single image."""
    # Load RGB
    rgb = cv2.imread(rgb_path)
    original_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # Load ground truth label
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    original_label = label.copy()
    label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    
    # Preprocess
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(rgb_tensor)
        prediction = torch.argmax(output, dim=1)[0].cpu().numpy()
    
    return original_rgb, rgb, label, prediction


def visualize_results(original_rgb, rgb, label, prediction, save_path):
    """Create visualization of results."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Resized input
    axes[1].imshow(rgb)
    axes[1].set_title('Input (128×128)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Ground truth
    im2 = axes[2].imshow(label, cmap='tab20', vmin=0, vmax=33)
    axes[2].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Prediction
    im3 = axes[3].imshow(prediction, cmap='tab20', vmin=0, vmax=33)
    axes[3].set_title('Prediction (U-Net)', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.close()


def main():
    """Run inference on multiple validation samples."""
    
    config = {
        'checkpoint_path': 'checkpoints/best_model.pth',
        'num_classes': 34,
        'img_size': 128,
        'device': 'cpu',
        'images_base': '/home/viswa/Downloads/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val',
        'labels_base': '/home/viswa/Downloads/Cityscapes/gtFine_trainvaltest/gtFine/val',
        'num_samples': 5,  # Number of samples to test
        'output_dir': 'inference_results'
    }
    
    print("=" * 80)
    print("U-Net Inference Demo")
    print("=" * 80)
    print(f"Device: {config['device']}")
    print(f"Image size: {config['img_size']}×{config['img_size']}")
    print(f"Testing on {config['num_samples']} validation images")
    print()
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load model
    model = load_model(config['checkpoint_path'], config['num_classes'], config['device'])
    print()
    
    # Find validation images
    print("Finding validation images...")
    samples = []
    for city in os.listdir(config['images_base']):
        city_img_dir = os.path.join(config['images_base'], city)
        if os.path.isdir(city_img_dir):
            for img_file in os.listdir(city_img_dir):
                if img_file.endswith('_leftImg8bit.png'):
                    label_file = img_file.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    label_path = os.path.join(config['labels_base'], city, label_file)
                    if os.path.exists(label_path):
                        samples.append((
                            os.path.join(city_img_dir, img_file),
                            label_path,
                            f"{city}_{img_file}"
                        ))
                    if len(samples) >= config['num_samples']:
                        break
        if len(samples) >= config['num_samples']:
            break
    
    print(f"✓ Found {len(samples)} images")
    print()
    
    # Run inference on each sample
    print("Running inference...")
    print("-" * 80)
    
    for idx, (rgb_path, label_path, name) in enumerate(samples, 1):
        print(f"\nSample {idx}/{len(samples)}: {name}")
        
        # Run inference
        original_rgb, rgb, label, prediction = run_inference_on_sample(
            model, rgb_path, label_path, config['img_size'], config['device']
        )
        
        # Calculate metrics
        correct = (prediction == label).sum()
        total = label.size
        accuracy = correct / total
        
        print(f"  Pixel Accuracy: {accuracy:.2%}")
        print(f"  Unique classes in GT: {len(np.unique(label))}")
        print(f"  Unique classes predicted: {len(np.unique(prediction))}")
        
        # Visualize
        output_path = os.path.join(config['output_dir'], f'result_{idx}.png')
        visualize_results(original_rgb, rgb, label, prediction, output_path)
    
    print()
    print("=" * 80)
    print("Inference completed!")
    print(f"Results saved to: {config['output_dir']}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
