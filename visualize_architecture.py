"""
Visualization script for U-Net Architecture

This script generates visualizations of the U-Net model architecture including:
1. Model architecture diagram
2. Feature map size progression
3. Parameter count breakdown
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from model import UNetRGBD, count_parameters


def create_architecture_diagram(save_path='architecture_diagram.png'):
    """
    Create a visual diagram of the U-Net architecture.
    
    Args:
        save_path (str): Path to save the diagram
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    color_input = '#E8F4F8'
    color_conv = '#B3D9FF'
    color_down = '#66B2FF'
    color_bottleneck = '#FF6B6B'
    color_up = '#95E1D3'
    color_skip = '#FFA07A'
    color_output = '#F9E79F'
    
    # Title
    ax.text(8, 11.5, 'U-Net Architecture for RGB-D Semantic Segmentation', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Layer dimensions
    layers = [
        # Encoder
        {'name': 'Input\nRGB-D', 'channels': '4', 'size': 'H×W', 'x': 1, 'y': 10, 'color': color_input},
        {'name': 'Conv64', 'channels': '64', 'size': 'H×W', 'x': 3, 'y': 10, 'color': color_conv},
        {'name': 'Down128', 'channels': '128', 'size': 'H/2×W/2', 'x': 3, 'y': 8, 'color': color_down},
        {'name': 'Down256', 'channels': '256', 'size': 'H/4×W/4', 'x': 3, 'y': 6, 'color': color_down},
        {'name': 'Down512', 'channels': '512', 'size': 'H/8×W/8', 'x': 3, 'y': 4, 'color': color_down},
        {'name': 'Bottleneck\n1024', 'channels': '1024', 'size': 'H/16×W/16', 'x': 3, 'y': 2, 'color': color_bottleneck},
        
        # Decoder
        {'name': 'Up512', 'channels': '512', 'size': 'H/8×W/8', 'x': 13, 'y': 4, 'color': color_up},
        {'name': 'Up256', 'channels': '256', 'size': 'H/4×W/4', 'x': 13, 'y': 6, 'color': color_up},
        {'name': 'Up128', 'channels': '128', 'size': 'H/2×W/2', 'x': 13, 'y': 8, 'color': color_up},
        {'name': 'Up64', 'channels': '64', 'size': 'H×W', 'x': 13, 'y': 10, 'color': color_up},
        {'name': 'Output\nClasses', 'channels': '19', 'size': 'H×W', 'x': 15, 'y': 10, 'color': color_output},
    ]
    
    # Draw boxes
    boxes = {}
    for i, layer in enumerate(layers):
        box = FancyBboxPatch(
            (layer['x'] - 0.6, layer['y'] - 0.4),
            1.2, 0.8,
            boxstyle="round,pad=0.05",
            edgecolor='black',
            facecolor=layer['color'],
            linewidth=2
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(layer['x'], layer['y'] + 0.15, layer['name'], 
                fontsize=9, fontweight='bold', ha='center', va='center')
        ax.text(layer['x'], layer['y'] - 0.15, f"{layer['channels']}ch\n{layer['size']}", 
                fontsize=7, ha='center', va='center')
        
        boxes[layer['name']] = (layer['x'], layer['y'])
    
    # Draw arrows - Encoder path
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    ax.annotate('', xy=(2.4, 10), xytext=(1.6, 10), arrowprops=arrow_props)
    ax.annotate('', xy=(3, 9.5), xytext=(3, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(3, 7.5), xytext=(3, 6.5), arrowprops=arrow_props)
    ax.annotate('', xy=(3, 5.5), xytext=(3, 4.5), arrowprops=arrow_props)
    ax.annotate('', xy=(3, 3.5), xytext=(3, 2.5), arrowprops=arrow_props)
    
    # Draw arrows - Decoder path
    ax.annotate('', xy=(13, 4.5), xytext=(13, 3.5), arrowprops=arrow_props)
    ax.annotate('', xy=(13, 6.5), xytext=(13, 5.5), arrowprops=arrow_props)
    ax.annotate('', xy=(13, 8.5), xytext=(13, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(13, 10.5), xytext=(13, 9.5), arrowprops=arrow_props)
    ax.annotate('', xy=(14.4, 10), xytext=(13.6, 10), arrowprops=arrow_props)
    
    # Draw arrows - Bottleneck to decoder
    arrow_curve = FancyArrowPatch((3.6, 2), (12.4, 4), 
                                  connectionstyle="arc3,rad=.3", 
                                  arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow_curve)
    
    # Draw skip connections
    skip_props = dict(arrowstyle='->', lw=2.5, color=color_skip, linestyle='--')
    ax.annotate('', xy=(12.4, 10), xytext=(3.6, 10), arrowprops=skip_props)
    ax.annotate('', xy=(12.4, 8), xytext=(3.6, 8), arrowprops=skip_props)
    ax.annotate('', xy=(12.4, 6), xytext=(3.6, 6), arrowprops=skip_props)
    ax.annotate('', xy=(12.4, 4), xytext=(3.6, 4), arrowprops=skip_props)
    
    # Add labels
    ax.text(8, 10.8, 'Skip Connection', fontsize=9, color=color_skip, 
            style='italic', ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.text(1.5, 0.5, 'ENCODER\n(Contracting Path)', fontsize=11, fontweight='bold', 
            ha='center', bbox=dict(boxstyle='round', facecolor=color_down, alpha=0.3))
    ax.text(8, 0.5, 'BOTTLENECK', fontsize=11, fontweight='bold', 
            ha='center', bbox=dict(boxstyle='round', facecolor=color_bottleneck, alpha=0.3))
    ax.text(14.5, 0.5, 'DECODER\n(Expansive Path)', fontsize=11, fontweight='bold', 
            ha='center', bbox=dict(boxstyle='round', facecolor=color_up, alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Architecture diagram saved to {save_path}")
    plt.close()


def print_architecture_table():
    """Print a detailed table of the architecture."""
    print("\n" + "=" * 100)
    print("U-Net Architecture - Layer-by-Layer Breakdown")
    print("=" * 100)
    
    print(f"\n{'Stage':<15} {'Layer':<20} {'Channels':<15} {'Spatial Size':<20} {'Parameters':<15}")
    print("-" * 100)
    
    # Create model to get actual parameter counts
    model = UNetRGBD(in_channels=4, num_classes=19, base_channels=64)
    params = count_parameters(model)
    
    # Manually define architecture table (approximate)
    arch_table = [
        ("INPUT", "Input Image", "4", "H × W", "-"),
        ("ENCODER", "DoubleConv", "4 → 64", "H × W", "~37K"),
        ("ENCODER", "MaxPool + DoubleConv", "64 → 128", "H/2 × W/2", "~148K"),
        ("ENCODER", "MaxPool + DoubleConv", "128 → 256", "H/4 × W/4", "~590K"),
        ("ENCODER", "MaxPool + DoubleConv", "256 → 512", "H/8 × W/8", "~2.4M"),
        ("BOTTLENECK", "MaxPool + DoubleConv", "512 → 1024", "H/16 × W/16", "~9.4M"),
        ("DECODER", "UpConv + Concat + Conv", "1024 → 512", "H/8 × W/8", "~9.4M"),
        ("DECODER", "UpConv + Concat + Conv", "512 → 256", "H/4 × W/4", "~2.4M"),
        ("DECODER", "UpConv + Concat + Conv", "256 → 128", "H/2 × W/2", "~590K"),
        ("DECODER", "UpConv + Concat + Conv", "128 → 64", "H × W", "~148K"),
        ("OUTPUT", "Conv 1×1", "64 → 19", "H × W", "~1.2K"),
    ]
    
    for stage, layer, channels, spatial, param_count in arch_table:
        print(f"{stage:<15} {layer:<20} {channels:<15} {spatial:<20} {param_count:<15}")
    
    print("-" * 100)
    print(f"\nTotal Parameters: {params['total_readable']}")
    print("=" * 100)


def visualize_feature_maps_progression():
    """Visualize how feature map sizes change through the network."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define layers and their properties
    layers = [
        ('Input', 4, 256, 'Encoder'),
        ('Conv64', 64, 256, 'Encoder'),
        ('Down128', 128, 128, 'Encoder'),
        ('Down256', 256, 64, 'Encoder'),
        ('Down512', 512, 32, 'Encoder'),
        ('Bottleneck', 1024, 16, 'Bottleneck'),
        ('Up512', 512, 32, 'Decoder'),
        ('Up256', 256, 64, 'Decoder'),
        ('Up128', 128, 128, 'Decoder'),
        ('Up64', 64, 256, 'Decoder'),
        ('Output', 19, 256, 'Output'),
    ]
    
    x_positions = np.arange(len(layers))
    channels = [l[1] for l in layers]
    spatial_sizes = [l[2] for l in layers]
    names = [l[0] for l in layers]
    stages = [l[3] for l in layers]
    
    # Create color map
    color_map = {
        'Encoder': '#66B2FF',
        'Bottleneck': '#FF6B6B',
        'Decoder': '#95E1D3',
        'Output': '#F9E79F'
    }
    colors = [color_map[s] for s in stages]
    
    # Plot channels
    ax1 = ax
    bars1 = ax1.bar(x_positions, channels, color=colors, alpha=0.7, label='Channels')
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Channels', fontsize=12, fontweight='bold', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot spatial size on secondary axis
    ax2 = ax1.twinx()
    line = ax2.plot(x_positions, spatial_sizes, 'ro-', linewidth=2, markersize=8, label='Spatial Size')
    ax2.set_ylabel('Spatial Size (H or W)', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add value labels on bars
    for i, (bar, ch, ss) in enumerate(zip(bars1, channels, spatial_sizes)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{ch}ch',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax2.text(i, ss + 10, f'{ss}×{ss}',
                ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')
    
    # Title
    ax1.set_title('U-Net Feature Maps Progression\n(Channels and Spatial Dimensions)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['Encoder'], alpha=0.7, label='Encoder'),
        Patch(facecolor=color_map['Bottleneck'], alpha=0.7, label='Bottleneck'),
        Patch(facecolor=color_map['Decoder'], alpha=0.7, label='Decoder'),
        Patch(facecolor=color_map['Output'], alpha=0.7, label='Output'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    save_path = 'feature_progression.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature map progression saved to {save_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 80)
    print("Generating U-Net Architecture Visualizations")
    print("=" * 80)
    
    # Create architecture diagram
    print("\n1. Creating architecture diagram...")
    create_architecture_diagram('architecture_diagram.png')
    
    # Print architecture table
    print("\n2. Generating architecture table...")
    print_architecture_table()
    
    # Create feature map progression
    print("\n3. Creating feature map progression chart...")
    visualize_feature_maps_progression()
    
    print("\n" + "=" * 80)
    print("All visualizations generated successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - architecture_diagram.png")
    print("  - feature_progression.png")


if __name__ == "__main__":
    main()
