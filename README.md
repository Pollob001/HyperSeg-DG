<h1 id="HyperSeg-DG">HyperSeg-DG: Multi-Scale Hyper Feature Context for Domain Generalized Medical image Segmentation</h1>

<p align="center">
  <!-- Adjust width as needed (e.g., 800–1100) -->
  <img src="images/HyperSeg-DG.jpg" alt="Model Architecture" width="900">
</p>
<p align="center"><em>Figure 1. The proposed HyperSeg-DG framework for segmentation and domain generalization.</em></p>




# HyperSeg-DG
## 🏗️ Model Architecture

### Backbone Options
- **WMamba-T** (Tiny): Lightweight backbone for resource-constrained scenarios
- **WMamba-S** (Small): Balanced performance-efficiency trade-off
- **WMamba-B** (Base): Maximum accuracy for high-end systems

### Training Stages
1. **Stage 1**: Pre-trains the WMamba backbone with auxiliary heads
2. **Stage 2**: End-to-end training with HyperSeg-DG component

## 🚀 Getting Started

### Prerequisites

# Create conda environment
conda create -n hyperseg python=3.8
conda activate hyperseg

# Install dependencies
pip install torch>=1.9.0 torchvision
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.9.0/index.html
pip install timm einops yacs



## Dataset Preparation

### ImageNet-1K Dataset (Backbone Pre-training)

```bash
data/
├── imagenet/
│   ├── train/
│   │   ├── n01440764/
│   │   │   ├── n01440764_10026.JPEG
│   │   │   └── ...
│   │   ├── n01443537/
│   │   └── ... (1000 classes)
│   └── val/
│       ├── n01440764/
│       │   ├── ILSVRC2012_val_00000293.JPEG
│       │   └── ...
│       └── ... (1000 classes)



### Segmentation Dataset (Stage 1 & Stage 2)

data/
├── dataset_name/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/


# WMamba-Tiny backbone
python train_stage1.py --backbone wmamba_t --batch_size 16 --epochs 100

# WMamba-Small backbone
python train_stage1.py --backbone wmamba_s --batch_size 12 --epochs 120

# WMamba-Base backbone
python train_stage1.py --backbone wmamba_b --batch_size 8 --epochs 150


# Resume from Stage 1 backbone checkpoint
python train_stage2.py --backbone wmamba_t --stage1_ckpt checkpoints/stage1/wmamba_t/best_model.pth

# Train from scratch (not recommended)
python train_stage2.py --backbone wmamba_s --batch_size 8 --epochs 80

# Resume from Stage 1 backbone checkpoint
python train_stage2.py --backbone wmamba_t --stage1_ckpt checkpoints/stage1/wmamba_t/best_model.pth

# Train from scratch (not recommended)
python train_stage2.py --backbone wmamba_s --batch_size 8 --epochs 80



### Acknowledgements

This project builds upon the following open-source works:

- **ConDSeg** - https://github.com/Mengqi-Lei/ConDSeg
- **Dofe** - https://github.com/emma-sjwang/Dofe  
- **RAM-DSIR** - https://github.com/zzzqzhou/RAM-DSIR

We thank the authors for their valuable contributions.
