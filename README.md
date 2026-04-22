<h1 id="HyperSeg-DG">HyperSeg-DG: Multi-Scale Hyper Feature Context for Domain Generalized Medical image Segmentation</h1>

<p align="center">
  <!-- Adjust width as needed (e.g., 800–1100) -->
  <img src="images/HyperSeg-DG.jpg" alt="Model Architecture" width="900">
</p>
<p align="center"><em>Figure 1. The proposed HyperSeg-DG framework for segmentation and domain generalization.</em></p>

<h2 id="requirements">Requirements</h2>
<ul>
  <li>Python 3.9.21</li>
  <li>numpy==2.0.2</li>
  <li>pandas==2.2.3</li>
  <li>torch==2.6.0</li>
</ul>

<h2 id="clone-repository">Clone Repository</h2>
<pre><code>git clone https://github.com/khanonuvov/KANPM-DTA.git
cd KANPM-DTA
</code></pre>

<h2 id="Generate Pretrained Models">Generate Pretrained Models</h2>
<p>For each dataset (<code>davis</code>, <code>kiba</code>, <code>metz</code>, <code>bindingdb</code>, <code>Lung Cancer Test</code>) run these three commands or download the pretrained models from <a href="https://www.kaggle.com/datasets/khanonuvov/kanpm-dta-pretrained-models"><code>here</code></a>.</p>

<pre><code>python pretrained/chemberta_pretraiend.py</code></pre>
<pre><code>python pretrained/esmC_pretraiend.py</code></pre>
<pre><code>python pretrained/esm2_map.py</code></pre>

<h2 id="Generate Warm & Cold-start Datasets">Generate Cold-start Datasets</h2>
<pre><code>python code/cold_split.py</code></pre>

<h2 id="Train the Model">Train the Model</h2>
<pre><code>python code/train.py</code></pre>

<h2 id="Prediction">Prediction</h2>
<pre><code>python code/pred.py</code></pre>

<h2 id="contact">Contact</h2>
<p>
  For inquiries, please contact
  <strong>Md Aynul Islam</strong> (Email: 
  <a href="mailto:aynulislam1997@mail.ustc.edu.cn">aynulislam1997@mail.ustc.edu.cn</a>).
</p>






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
