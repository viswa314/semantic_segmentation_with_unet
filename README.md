# U-Net Semantic Segmentation for RGB-D Images

A PyTorch implementation of U-Net architecture for semantic segmentation of RGB-D images, designed for datasets like Cityscapes with optional depth information.

## 📺 Demonstration
[![Click to Watch](https://img.shields.io/badge/▶_Watch_Demo-blue?style=for-the-badge&logo=youtube)](media/Task%207.webm)
*U-Net performing semantic segmentation on RGB-D input. (Click badge to view)*

## 📋 Project Overview

This project implements a U-Net-based semantic segmentation model that can process both RGB (3-channel) and RGB-D (4-channel with depth) images. The architecture features an encoder-decoder structure with skip connections, optimized for pixel-wise classification tasks.

## 🏗️ Architecture

The U-Net architecture consists of:
- **Encoder (Contracting Path)**: 4 downsampling levels that capture semantic context
- **Bottleneck**: Deepest layer with largest receptive field (1024 channels)
- **Decoder (Expansive Path)**: 4 upsampling levels that reconstruct spatial resolution
- **Skip Connections**: Link encoder to decoder at each level, preserving spatial details

For detailed architecture explanation, see [architecture_explanation.md](architecture_explanation.md).

## 📁 Project Structure

```
unet/
├── model.py                      # U-Net model implementation
├── dataset.py                    # Dataset classes and data loaders
├── train.py                      # Training pipeline
├── utils.py                      # Utility functions (metrics, visualization)
├── visualize_architecture.py     # Architecture visualization script
├── architecture_explanation.md   # Detailed architecture documentation
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### Installation

```bash
cd /home/viswa/unet
pip install -r requirements.txt
```

### Test Model

```bash
# Test the model architecture
python model.py

# Generate architecture visualizations
python visualize_architecture.py

# Test utility functions
python utils.py
```

### Training

```bash
# Edit train.py to configure dataset path and hyperparameters
# Then run:
python train.py
```

## 🔧 Model Configuration

Key parameters in the U-Net model:

```python
model = UNetRGBD(
    in_channels=4,        # 3 for RGB, 4 for RGB-D
    num_classes=19,       # Number of segmentation classes
    base_channels=64,     # Starting number of channels
    bilinear=False,       # Use bilinear upsampling (vs transposed conv)
    dropout_p=0.5         # Dropout probability at bottleneck
)
```

## 📊 Features

### Model Features
- ✅ Supports both RGB and RGB-D inputs
- ✅ Configurable architecture depth and channels
- ✅ Batch normalization for training stability
- ✅ Dropout regularization at bottleneck
- ✅ Skip connections for precise localization
- ✅ ~34.6M parameters (base configuration)

### Training Features
- ✅ Combined Cross-Entropy + Dice Loss
- ✅ IoU, Dice Score, and Pixel Accuracy metrics
- ✅ Automatic checkpointing
- ✅ Training curve visualization
- ✅ Data augmentation pipeline

### Dataset Support
- ✅ Cityscapes dataset format
- ✅ Custom RGB-D datasets
- ✅ Handles missing depth gracefully

## 📈 Performance Metrics

The model is evaluated using:

1. **Mean Intersection over Union (mIoU)**: Primary metric for segmentation quality
2. **Dice Score**: Harmonic mean of precision and recall
3. **Pixel Accuracy**: Overall pixel-wise correctness

## 🎨 Visualization

Generate architecture diagrams and feature progression charts:

```bash
python visualize_architecture.py
```

This creates:
- `architecture_diagram.png`: Visual representation of the U-Net structure
- `feature_progression.png`: Chart showing channel and spatial dimension changes

## 📚 Dataset Format

### Cityscapes Dataset
Expected directory structure:
```
Cityscapes/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

### Custom Dataset
For custom datasets, organize as:
```
dataset/
├── images/        # RGB images
├── masks/         # Segmentation masks
└── depth/         # Depth maps (optional)
```

## 🔬 Technical Details

- **Framework**: PyTorch
- **Input Size**: Flexible (default: 256×256)
- **Batch Size**: 4-8 (depends on GPU memory)
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: 0.5×CrossEntropy + 0.5×Dice
- **Training Time**: ~30 min/epoch on RTX 3080 (Cityscapes train set)

## 📄 Documentation

- [architecture_explanation.md](architecture_explanation.md): Comprehensive architecture documentation
  - Theoretical background
  - Layer-by-layer breakdown
  - Mathematical formulations
  - Design decisions

## 🎯 Use Cases

This model is suitable for:
- Urban scene understanding (autonomous driving)
- Indoor scene segmentation
- Robot navigation
- Augmented reality applications
- Any pixel-wise classification task with optional depth

## 🔑 Key Design Decisions

1. **Early Fusion**: RGB and Depth concatenated at input for joint learning
2. **Batch Normalization**: After every convolution for training stability
3. **Transposed Convolution**: Learnable upsampling (better than bilinear)
4. **Combined Loss**: CrossEntropy for classification + Dice for boundary refinement
5. **Dropout at Bottleneck**: Regularization where overfitting is most likely

## 📝 Citation

Original U-Net paper:
```
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  year={2015}
}
```

## 📧 Notes

- The model automatically detects CUDA availability
- Checkpoints are saved to `checkpoints/` directory
- Training curves are automatically plotted after training
- The model can be easily adapted for different numbers of classes

---

**Dataset Path (Cityscapes)**: `/home/viswa/Downloads/Cityscapes`

For questions or issues, please refer to the architecture documentation or code comments.
