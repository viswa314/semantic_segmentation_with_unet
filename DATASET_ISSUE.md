# Dataset Issue and Solutions

## Problem

The Cityscapes dataset at `/home/viswa/Downloads/Cityscapes/` is **incomplete** for semantic segmentation training.

### What You Have ✅
- **RGB Images**: 2,975 training images in `leftImg8bit_trainvaltest/`
- **Disparity Maps**: Depth data in `disparity_trainvaltest/`  
- **3D Bounding Boxes**: In `gtBbox3d_trainvaltest/`

### What's Missing ❌
- **Segmentation Labels** (`gtFine`): Ground truth pixel-wise annotations required for training

---

## Solutions

### Option 1: Download Missing Labels (Recommended)

1. **Visit Cityscapes Download Page**:
   - URL: https://www.cityscapes-dataset.com/downloads/
   - You may need to register/login

2. **Download Required File**:
   - File: `gtFine_trainvaltest.zip` (~241 MB)
   - Contains pixel-wise semantic segmentation annotations

3. **Extract to Dataset Directory**:
   ```bash
   cd /home/viswa/Downloads/Cityscapes/
   unzip /path/to/gtFine_trainvaltest.zip
   ```

4. **Start Training**:
   ```bash
   cd /home/viswa/unet
   python3 train.py
   ```

### Option 2: Demo with Synthetic Data

I can create a small synthetic dataset (fake images and labels) to demonstrate that the training pipeline works correctly. This won't produce a useful model but will verify all code is working.

### Option 3: Use Different Dataset

Alternative datasets with RGB-D and labels:
- **NYU Depth V2**: Indoor scenes, smaller dataset
- **SUN RGB-D**: Indoor scenes with depth
- **KITTI**: Autonomous driving (similar to Cityscapes)

---

## Current Dataset Structure

```
/home/viswa/Downloads/Cityscapes/
├── leftImg8bit_trainvaltest/
│   └── leftImg8bit/
│       ├── train/  (2975 images)
│       └── val/
├── disparity_trainvaltest/
│   └── disparity/
│       ├── train/
│       └── val/
└── gtBbox3d_trainvaltest/
    └── gtBbox3d/
        ├── train/
        └── val/
```

### Required Structure (After Download)

```
/home/viswa/Downloads/Cityscapes/
├── leftImg8bit_trainvaltest/
├── disparity_trainvaltest/
├── gtBbox3d_trainvaltest/
└── gtFine_trainvaltest/  ← MISSING
    └── gtFine/
        ├── train/
        │   └── {city}/
        │       └── *_labelIds.png
        └── val/
```

---

## What to Do Next

**If you want to proceed with training**, please either:

1. Download `gtFine_trainvaltest.zip` from Cityscapes website
2. Let me know if you'd like me to create a demo with synthetic data
3. Choose a different dataset

Once you have the labels, training will work automatically.
