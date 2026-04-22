<h2 id="domain-generalization">Domain Generalization Datasets</h2>

<h3>Fundus Dataset (Provided by DoFE)</h3>
<p>Download the Fundus dataset and organize as follows:</p>

<pre>
data/
├── fundus/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
</pre>

<p><strong>Download Link</strong>: <a href="https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view?pli=1">Fundus Dataset</a></p>

<h3>Prostate Dataset (Originally Provided by SAML)</h3>
<p>Download our pre-processed Prostate dataset:</p>

<pre>
data/
├── prostate/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
</pre>

<p><strong>Download Link</strong>: <a href="https://drive.google.com/file/d/1sx2FpNySQNjU6_zBa4DPnb9RAmesN0P6/view">Prostate Dataset</a></p>

<h2 id="in-distribution">In-Distribution Segmentation Datasets</h2>

<p>Organize each dataset in the following structure:</p>

<pre>
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
</pre>

<h3>CVC-ClinicDB</h3>
<ul>
<li><strong>Description</strong>: Endoscopic images for polyp segmentation</li>
<li><strong>Download</strong>: <a href="https://www.kaggle.com/datasets/balraj98/cvcclinicdb">CVC-ClinicDB on Kaggle</a></li>
</ul>

<h3>ISIC 2016, 2017, 2018</h3>
<ul>
<li><strong>Description</strong>: Skin lesion images for melanoma segmentation</li>
<li><strong>Download</strong>: <a href="https://challenge.isic-archive.com/data/">ISIC Archive</a></li>
</ul>

<h3>GlaS (Gland Segmentation)</h3>
<ul>
<li><strong>Description</strong>: Histology images for gland segmentation</li>
<li><strong>Download</strong>: <a href="https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation">GlaS on Kaggle</a></li>
</ul>

<h3>Breast Ultrasound Images Dataset</h3>
<ul>
<li><strong>Description</strong>: Breast ultrasound images for tumor segmentation</li>
<li><strong>Download</strong>: <a href="https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset">Breast Ultrasound on Kaggle</a></li>
</ul>

<h3>Kvasir-SEG</h3>
<ul>
<li><strong>Description</strong>: Gastrointestinal polyp segmentation</li>
<li><strong>Download</strong>: <a href="https://datasets.simula.no/kvasir-seg/">Kvasir-SEG Dataset</a></li>
</ul>

<h3>Kvasir-Sessile</h3>
<ul>
<li><strong>Description</strong>: Sessile polyp segmentation</li>
<li><strong>Download</strong>: <a href="https://datasets.simula.no/kvasir-seg/">Kvasir-Sessile Dataset</a></li>
</ul>
