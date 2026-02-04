# Google Colab Training Instructions

## Quick Start Guide for Training RGB-D U-Net on Google Colab

### Prerequisites
1. Google account
2. Cityscapes dataset downloaded locally
3. ~5GB free space in Google Drive

---

## Step-by-Step Instructions

### Step 1: Upload Dataset to Google Drive

**Option A: Manual Upload (Recommended for first time)**
1. Go to [Google Drive](https://drive.google.com)
2. Create a folder named `Cityscapes`
3. Upload these 3 folders to the `Cityscapes` folder:
   - `leftImg8bit_trainvaltest/`
   - `gtFine_trainvaltest/`
   - `disparity_trainvaltest/`

**Option B: Use Google Drive Desktop App**
1. Install Google Drive for Desktop
2. Copy your Cityscapes folder to your Google Drive folder
3. Wait for sync to complete

**Expected structure in Google Drive:**
```
MyDrive/
└── Cityscapes/
    ├── leftImg8bit_trainvaltest/
    ├── gtFine_trainvaltest/
    └── disparity_trainvaltest/
```

---

### Step 2: Open Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `UNet_RGBD_Training_Colab.ipynb` from `/home/viswa/unet/`

**OR**

Upload the notebook to Google Drive and open it from there:
1. Upload `UNet_RGBD_Training_Colab.ipynb` to Google Drive
2. Right-click → Open with → Google Colaboratory

---

### Step 3: Enable GPU

**CRITICAL STEP** - This will make training 10-15x faster!

1. In Colab: **Runtime → Change runtime type**
2. Hardware accelerator: Select **GPU** (ideally **T4 GPU**)
3. Click **Save**

---

### Step 4: Run the Notebook

1. **Run all cells**: Runtime → Run all
2. **Mount Google Drive**: 
   - A popup will ask for permission
   - Click "Connect to Google Drive"
   - Choose your Google account
   - Allow access

3. **Verify GPU**: First cell should show:
   ```
   CUDA available: True
   GPU: Tesla T4 (or similar)
   ```

4. **Update dataset path** (if needed):
   - In Cell 2, check the `DATASET_PATH` variable
   - Default: `/content/drive/MyDrive/Cityscapes`
   - Update if your folder is elsewhere

5. **Training starts automatically**:
   - Progress bars will show training progress
   - Each epoch takes ~2-3 minutes on GPU
   - Total time: **15-20 minutes** for 20 epochs

---

### Step 5: Monitor Training

You'll see output like:
```
Epoch 1/20 [Train]: 100%|██████| 744/744 [02:15<00:00]
Epoch 1: Train Loss: 0.452, Val Loss: 0.389, Val Acc: 0.754
  ✓ Best model saved!
```

**What to watch for:**
- ✅ Loss should decrease
- ✅ Accuracy should increase
- ✅ "Best model saved!" appears when improving

---

### Step 6: Download Trained Model

After training completes, the last cell will automatically download:
1. `best_rgbd_model.pth` - Your trained model (~360MB)
2. `training_curves.png` - Training visualization

These files will be in your Downloads folder.

---

## Configuration Details

### Default Settings (Optimized for GPU)
```python
- Input channels: 4 (RGB-D)
- Image size: 256×256 (larger than CPU version)
- Batch size: 4 (larger than CPU version)
- Base channels: 64 (more parameters)
- Epochs: 20
- Learning rate: 1e-4
- Device: GPU (CUDA)
```

### Expected Results
- **Training time**: 15-20 minutes (vs 2.5 hours on CPU)
- **Validation accuracy**: 75-85% pixel accuracy
- **Model size**: ~360MB
- **Parameters**: ~31 million

---

## Troubleshooting

### "No GPU detected"
- **Fix**: Runtime → Change runtime type → GPU → Save
- **Note**: Free Colab has limited GPU hours (~12 hours/day)

### "Dataset not found"
- **Fix**: Check Google Drive folder structure
- **Update**: `DATASET_PATH` variable in Cell 2
- **Verify**: Run `!ls /content/drive/MyDrive/Cityscapes`

### "Out of memory"
- **Fix 1**: Reduce batch size (change `batch_size: 4` to `2`)
- **Fix 2**: Reduce image size (change `img_size: 256` to `128`)
- **Fix 3**: Reduce base channels (change `base_channels: 64` to `32`)

### Training is slow
- **Check**: Verify GPU is enabled (first cell output)
- **Check**: GPU usage (Runtime → View resources)
- **Note**: First epoch is always slower (data loading)

---

## After Training

### Use the Model Locally

1. **Move downloaded model** to your project:
   ```bash
   mv ~/Downloads/best_rgbd_model.pth /home/viswa/unet/checkpoints_rgbd/
   ```

2. **Run inference**:
   ```bash
   cd /home/viswa/unet
   python3 run_inference_rgbd.py
   ```

3. **Model will load** from `checkpoints_rgbd/best_model.pth`

---

## Tips for Best Results

### GPU Usage
- Free tier: ~12 hours/day GPU time
- Pro tier: More GPU hours + better GPUs (A100)
- Check usage: Settings (⚙️) → Usage limits

### Training Time
- 20 epochs: ~15-20 minutes
- 50 epochs: ~40-50 minutes (better accuracy)
- Adjust `num_epochs` in config cell

### Save Intermediate Results
To save checkpoints during training:
```python
# Add this in the training loop (after epoch completes)
if (epoch + 1) % 5 == 0:
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')
```

### Monitor from Phone
- Colab notebooks work on mobile
- Check progress from anywhere
- Auto-saves to Google Drive

---

## Comparison: CPU vs GPU

| Metric | CPU (Local) | GPU (Colab) |
|--------|-------------|-------------|
| Training time (20 epochs) | 2.5 hours | 15-20 minutes |
| Batch size | 1 | 4 |
| Image size | 128×128 | 256×256 |
| Model size | 7.7M params | 31M params |
| Expected accuracy | 75-80% | 80-85% |

**Conclusion**: GPU training is **10-15x faster** and achieves **better accuracy**!

---

## Files Provided

1. `UNet_RGBD_Training_Colab.ipynb` - The Colab notebook
2. `COLAB_INSTRUCTIONS.md` - This guide

**Location**: `/home/viswa/unet/`

---

## Need Help?

Common issues and solutions:
- GPU not available → Enable in Runtime settings
- Dataset not found → Check Google Drive path
- Out of memory → Reduce batch size or image size
- Slow training → Verify GPU is being used

For more help, check the Colab FAQ or Google Colab documentation.

---

**Ready to start?** Upload the notebook to Colab and follow Step 3!
