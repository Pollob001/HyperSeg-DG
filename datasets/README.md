<h2 id="Datasets_">Datasets</h2>
<p>To ensure a comprehensive evaluation, we utilize a suite of ten datasets spanning diverse imaging modalities. Our experimental protocol includes eight in-distribution benchmarks for standard segmentation performance such as <a href="https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset">BUSI</a>, <a href="https://datasets.simula.no/kvasir-seg/">Kvasir-SEG</a>, <a href="https://datasets.simula.no/kvasir-seg/">Kvasir-Sessile</a>, <a href="https://www.kaggle.com/datasets/balraj98/cvcclinicdb">CVC-ClinicDB</a>, <a href="https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation">GlaS</a>, <a href="https://challenge.isic-archive.com/data/">ISIC 2016</a>, <a href="https://challenge.isic-archive.com/data/">ISIC 2017</a>, and <a href="https://challenge.isic-archive.com/data/">ISIC 2018</a>. Furthermore, to rigorously assess the model's generalizability to unseen domains without fine-tuning, we employ two domain generalization benchmarks <a href="https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view">DGFundus</a> for retinal fundus images and <a href="https://drive.google.com/file/d/1sx2FpNySQNjU6_zBa4DPnb9RAmesN0P6/view">DGProstate</a> for prostate MRI images.</p>

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; margin: 20px 0;">
  <caption style="text-align: center; margin-bottom: 10px; font-weight: bold;">Table: Segmentation and generalization benchmark datasets used in our study.</caption>
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Benchmark/Dataset</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Modality</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Organ</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Target</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Images/Cases</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #e6f3ff;">
      <td colspan="5" style="border: 1px solid #ddd; padding: 8px; text-align: center; font-weight: bold;">Segmentation</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;"><a href="https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset">BUSI</a></td>
      <td style="border: 1px solid #ddd; padding: 8px;">Ultrasound</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Breast</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Lesion</td>
      <td style="border: 1px solid #ddd; padding: 8px;">647</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;"><a href="https://datasets.simula.no/kvasir-seg/">Kvasir-SEG</a></td>
      <td style="border: 1px solid #ddd; padding: 8px;">Endoscope</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Colon</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Polyp</td>
      <td style="border: 1px solid #ddd; padding: 8px;">1000</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;"><a href="https://datasets.simula.no/kvasir-seg/">Kvasir-Sessile</a></td>
      <td style="border: 1px solid #ddd; padding: 8px;">Endoscope</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Colon</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Polyp</td>
      <td style="border: 1px solid #ddd; padding: 8px;">196</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;"><a href="https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation">GlaS</a></td>
      <td style="border: 1px solid #ddd; padding: 8px;">WSI</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Colorectum</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Gland</td>
      <td style="border: 1px solid #ddd; padding: 8px;">165</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;"><a href="https://challenge.isic-archive.com/data/">ISIC-2016</a></td>
      <td style="border: 1px solid #ddd; padding: 8px;">Dermoscope</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Skin</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Skin Lesion</td>
      <td style="border: 1px solid #ddd; padding: 8px;">1279</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;"><a href="https://challenge.isic-archive.com/data/">ISIC-2017</a></td>
      <td style="border: 1px solid #ddd; padding: 8px;">Dermoscope</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Skin</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Skin Lesion</td>
      <td style="border: 1px solid #ddd; padding: 8px;">2750</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;"><a href="https://challenge.isic-archive.com/data/">ISIC-2018</a></td>
      <td style="border: 1px solid #ddd; padding: 8px;">Dermoscope</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Skin</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Skin Lesion</td>
      <td style="border: 1px solid #ddd; padding: 8px;">2594</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;"><a href="https://www.kaggle.com/datasets/balraj98/cvcclinicdb">CVC</a></td>
      <td style="border: 1px solid #ddd; padding: 8px;">Images</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Colon</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Polyp</td>
      <td style="border: 1px solid #ddd; padding: 8px;">612</td>
    </tr>
    <tr style="background-color: #e6f3ff;">
      <td colspan="5" style="border: 1px solid #ddd; padding: 8px; text-align: center; font-weight: bold;">Generalization</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;" rowspan="4"><a href="https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view">DGFundus</a></td>
      <td style="border: 1px solid #ddd; padding: 8px;">Fundus</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Eye</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Optic Cup/Disc</td>
      <td style="border: 1px solid #ddd; padding: 8px;">800</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">Domain-1 (Fundus)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Eye</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Optic Cup/Disc</td>
      <td style="border: 1px solid #ddd; padding: 8px;">-</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">Domain-2 (Fundus)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Eye</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Optic Cup/Disc</td>
      <td style="border: 1px solid #ddd; padding: 8px;">-</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">Domain-3 (Fundus)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Eye</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Optic Cup/Disc</td>
      <td style="border: 1px solid #ddd; padding: 8px;">-</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">Domain-4 (Fundus)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Eye</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Optic Cup/Disc</td>
      <td style="border: 1px solid #ddd; padding: 8px;">-</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;" rowspan="6"><a href="https://drive.google.com/file/d/1sx2FpNySQNjU6_zBa4DPnb9RAmesN0P6/view">DGProstate</a></td>
      <td style="border: 1px solid #ddd; padding: 8px;">T2-weighted MRI</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Prostate</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Cancer</td>
      <td style="border: 1px solid #ddd; padding: 8px;">116</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">Domain-1 (T2-weighted MRI)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Prostate</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Cancer</td>
      <td style="border: 1px solid #ddd; padding: 8px;">-</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">Domain-2 (T2-weighted MRI)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Prostate</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Cancer</td>
      <td style="border: 1px solid #ddd; padding: 8px;">-</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">Domain-3 (T2-weighted MRI)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Prostate</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Cancer</td>
      <td style="border: 1px solid #ddd; padding: 8px;">-</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">Domain-4 (T2-weighted MRI)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Prostate</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Cancer</td>
      <td style="border: 1px solid #ddd; padding: 8px;">-</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">Domain-5 (T2-weighted MRI)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Prostate</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Cancer</td>
      <td style="border: 1px solid #ddd; padding: 8px;">-</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">Domain-6 (T2-weighted MRI)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Prostate</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Cancer</td>
      <td style="border: 1px solid #ddd; padding: 8px;">-</td>
    </tr>
  </tbody>
</table>
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
