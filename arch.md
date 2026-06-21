# Fake Dataset Factory — Architecture

## one line
Generate synthetic labeled chest X-ray images using 6 generative AI architectures (2014–2022), evaluate each individually using domain-specific medical metrics, export a usable labeled dataset.

---

## the problem
Medical chest X-ray datasets are expensive, private, and class-imbalanced.
Synthetic data generation is an active research solution.
arxiv 2503.00266 (March 2025) validates flow matching for medical image synthesis.

---

## dataset
- **Source:** paultimothymooney/chest-xray-pneumonia (Kaggle, free)
- **Size:** 5,863 JPEG images
- **Classes:** Normal, Pneumonia
- **Preprocessing:** resize all to 64x64 grayscale
- **Usage:** train each model separately on normal/ and pneumonia/ folder

---

## pipeline (same for all 6 models)

```
real xray images (64x64 grayscale)
          ↓
    generation engine
          ↓
  100 synthetic images
          ↓
  FID score (domain-adapted)     ← always first
          ↓
  torchxrayvision label + score
          ↓
  proxy TSTR accuracy
          ↓
  export: images/ + labels.csv + metrics.json → ZIP
          ↓
  Gradio UI on HuggingFace Spaces
```

---

## 6 generation engines

### 01 — Stable Diffusion (img2img mode)
- **Year:** 2022 (but used as bonus/exploratory engine here)
- **How:** feed real xray as input image → SD generates synthetic variation
- **Model:** runwayml/stable-diffusion-v1-5
- **Why img2img:** keeps outputs domain-relevant, not random artistic xrays
- **Note:** FID comparison is apple vs orange here — document as exploratory

### 02 — DCGAN
- **Year:** 2014
- **How:** Generator makes fake xrays from noise. Discriminator catches fakes. They compete.
- **Loss:** Binary Cross-Entropy (BCE)
- **Known issue:** mode collapse, training instability — documented findings, not bugs
- **Starting point:** Lab 2 code, swap MNIST for chest xrays
- **Reference:** kaledhoshme123/Using-GAN-to-Generate-Chest-X-Ray-Images

### 03 — WGAN-GP
- **Year:** 2017
- **How:** Same architecture as DCGAN. Different loss = Wasserstein distance + gradient penalty
- **Loss:** W-distance + GP (lambda=10)
- **Key fix:** no mode collapse, stable training
- **Reference:** MDPI 2023 paper — exact same setup at 64x64 chest xrays

### 04 — VQ-VAE
- **Year:** 2017
- **How:** Encoder → Vector Quantizer (codebook lookup) → Decoder
- **Loss:** reconstruction + commitment + codebook loss
- **Note:** without PixelCNN prior, generation is limited — documented as "VQ-VAE baseline without learned prior"
- **Reference:** kamenbliznashki/generative_models (CheXpert dataset)
- **Historical note:** foundation of DALL-E 1

### 05 — DDPM
- **Year:** 2020
- **How:** learns to reverse a noising process. Pure noise → realistic xray in N steps
- **Training:** from scratch on chest xray data using HF diffusers UNet2D
- **⚠️ No pretrained chest xray DDPM exists** — train overnight on Kaggle T4 (4-6 hrs)

### 06 — Flow Matching
- **Year:** 2022
- **How:** learns a vector field that transports Gaussian noise → data in straight paths
- **Loss:** MSE between predicted and target velocity — simpler than DDPM
- **Library:** torchcfm (pip installable)
- **Reference:** atong01/conditional-flow-matching mnist_example.ipynb
- **Research backing:** arxiv 2503.00266 (March 2025)
- **This is the flex model** — barely implemented at student level

---

## evaluation

### domain-adapted FID
- standard FID uses Inception V3 (trained on ImageNet) — wrong for medical images
- we use DenseNet121 features from torchxrayvision (trained on real chest xrays)
- call it: "domain-adapted FID using DenseNet121 features"

### proxy TSTR
- train classifier on synthetic images, test on real chest xrays
- use torchxrayvision as the test classifier (already pretrained on real xrays)
- call it: "proxy TSTR evaluation using domain-pretrained medical classifier"

### torchxrayvision
```python
pip install torchxrayvision
import torchxrayvision as xrv
model = xrv.models.DenseNet(weights='densenet121-res224-all')
# returns pneumonia probability scores — domain-specific, medically meaningful
```

---

## expected results (ballpark)

| Model         | Year | FID↓   | TSTR↑  | Speed   |
|---------------|------|--------|--------|---------|
| SD (img2img)  | 2022 | N/A*   | N/A*   | medium  |
| DCGAN         | 2014 | 40-80  | 60-70% | fast    |
| WGAN-GP       | 2017 | 25-50  | 70-80% | fast    |
| VQ-VAE        | 2017 | 30-60  | 65-75% | medium  |
| DDPM          | 2020 | 10-25  | 80-88% | slow    |
| Flow Matching | 2022 | 15-35  | 75-85% | medium  |
| Real (ceiling)| —    | —      | ~90%   | —       |

*SD uses img2img — different input distribution, not directly comparable via FID

---

## code structure

```
fake-dataset-factory/
│
├── data/
│   └── prepare.ipynb           # download + resize + visualize dataset
│
├── notebooks/
│   ├── 01_stable_diffusion.ipynb
│   ├── 02_dcgan.ipynb
│   ├── 03_wgan_gp.ipynb
│   ├── 04_vqvae.ipynb
│   ├── 05_ddpm.ipynb
│   ├── 06_flow_matching.ipynb
│   └── 07_comparison.ipynb     # all 6 results side by side
│
├── src/
│   ├── models/
│   │   ├── dcgan.py
│   │   ├── wgan_gp.py
│   │   ├── vqvae.py
│   │   ├── ddpm.py
│   │   └── flow_match.py
│   ├── evaluate.py             # FID + TSTR functions (shared)
│   ├── label.py                # torchxrayvision labeling (shared)
│   └── export.py               # save images + labels.csv + metrics.json + zip
│
├── app.py                      # Gradio UI — 6 tabs, pick class, generate, download
├── arch.md                     # this file
├── rules.md                    # coding + evaluation rules
└── install.md                  # setup instructions
```

---

## infra
- **Training:** Kaggle T4 GPU (free, 30hr/week)
- **Hosting:** HuggingFace Spaces (free CPU tier)
- **Total cost:** Rs. 0
