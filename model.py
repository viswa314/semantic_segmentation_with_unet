"""
U-Net Model for RGB-D Semantic Segmentation

This module implements a U-Net architecture designed for semantic segmentation
of RGB-D images. The model uses an encoder-decoder structure with skip connections
to preserve spatial information while capturing semantic context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution block: (Conv2D -> BatchNorm -> ReLU) x 2
    
    This is the basic building block of the U-Net architecture.
    Each block performs two 3x3 convolutions, each followed by 
    batch normalization and ReLU activation.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        mid_channels (int, optional): Number of channels in intermediate layer.
                                      If None, uses out_channels.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Downsampling block in the encoder path.
    
    Applies MaxPooling followed by a DoubleConv block to reduce spatial dimensions
    while increasing feature channels.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """
    Upsampling block in the decoder path.
    
    Upsamples the input using transposed convolution, concatenates with the 
    corresponding skip connection from the encoder, then applies DoubleConv.
    
    Args:
        in_channels (int): Number of input channels (from previous layer)
        out_channels (int): Number of output channels
        bilinear (bool): If True, use bilinear upsampling instead of transposed conv
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UpBlock, self).__init__()
        
        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Input from previous decoder layer (lower resolution)
            x2: Skip connection from encoder (higher resolution)
        """
        x1 = self.up(x1)
        
        # Handle size mismatches between x1 and x2 due to pooling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetRGBD(nn.Module):
    """
    U-Net architecture for RGB-D semantic segmentation.
    
    This implementation supports both RGB (3 channels) and RGB-D (4 channels) inputs.
    The architecture consists of:
    - Encoder: 4 downsampling blocks that extract hierarchical features
    - Bottleneck: Deepest layer with highest number of channels
    - Decoder: 4 upsampling blocks that reconstruct spatial resolution
    - Skip connections: Connect encoder to decoder at each level
    
    Args:
        in_channels (int): Number of input channels (3 for RGB, 4 for RGB-D)
        num_classes (int): Number of segmentation classes
        base_channels (int): Number of channels in first layer (doubles at each level)
        bilinear (bool): Use bilinear upsampling instead of transposed convolutions
        dropout_p (float): Dropout probability in bottleneck layer
    
    Input:
        x: Tensor of shape (batch_size, in_channels, height, width)
    
    Output:
        Tensor of shape (batch_size, num_classes, height, width)
    """
    def __init__(self, in_channels=4, num_classes=19, base_channels=64, 
                 bilinear=False, dropout_p=0.5):
        super(UNetRGBD, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # Initial convolution (no downsampling)
        self.inc = DoubleConv(in_channels, base_channels)
        
        # Encoder (contracting path)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck with dropout for regularization
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(base_channels * 8, base_channels * 16 // factor)
        self.dropout = nn.Dropout2d(p=dropout_p)
        
        # Decoder (expansive path)
        self.up1 = UpBlock(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = UpBlock(base_channels * 2, base_channels, bilinear)
        
        # Output convolution (1x1 conv to map to class scores)
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor (B, C, H, W) where C is in_channels
            
        Returns:
            Output tensor (B, num_classes, H, W) with class logits
        """
        # Encoder with skip connections stored
        x1 = self.inc(x)      # Level 1: (B, 64, H, W)
        x2 = self.down1(x1)   # Level 2: (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # Level 3: (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # Level 4: (B, 512, H/8, W/8)
        x5 = self.down4(x4)   # Bottleneck: (B, 1024, H/16, W/16)
        
        # Apply dropout at bottleneck
        x5 = self.dropout(x5)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # (B, 512, H/8, W/8)
        x = self.up2(x, x3)   # (B, 256, H/4, W/4)
        x = self.up3(x, x2)   # (B, 128, H/2, W/2)
        x = self.up4(x, x1)   # (B, 64, H, W)
        
        # Final classification layer
        logits = self.outc(x)  # (B, num_classes, H, W)
        
        return logits
    
    def get_num_params(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Dictionary with total parameters and parameters per layer
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    layer_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_params[name] = param.numel()
    
    return {
        'total': total,
        'total_readable': f"{total:,}",
        'layers': layer_params
    }


if __name__ == "__main__":
    # Test the model
    print("=" * 80)
    print("U-Net RGB-D Model Test")
    print("=" * 80)
    
    # Create model instance
    model = UNetRGBD(in_channels=4, num_classes=19, base_channels=64)
    
    # Print model summary
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Input channels: {model.in_channels}")
    print(f"Number of classes: {model.num_classes}")
    
    # Count parameters
    params = count_parameters(model)
    print(f"\nTotal trainable parameters: {params['total_readable']}")
    
    # Test forward pass with dummy data
    batch_size = 2
    height, width = 256, 256
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: ({batch_size}, {model.in_channels}, {height}, {width})")
    
    x = torch.randn(batch_size, model.in_channels, height, width)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {model.num_classes}, {height}, {width})")
    
    assert output.shape == (batch_size, model.num_classes, height, width), \
        "Output shape mismatch!"
    
    print("\n✓ Model test passed successfully!")
    print("=" * 80)
