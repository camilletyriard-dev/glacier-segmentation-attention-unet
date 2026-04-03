# Attention U-Net for Glacier Extent Segmentation in the Swiss Alps

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![TensorFlow 2.19](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Weights-yellow)](https://huggingface.co/camilletyriard/glacier-segmentation-attention-unet)

Semantic segmentation of glacier extent from Sentinel-2 multispectral imagery
using an attention-gated U-Net. This repository first replicates the deforestation
detection methodology of John & Zhang (2022) across all reported dataset configurations,
then transfers the architecture to binary glacier/non-glacier segmentation over the
Jungfrau–Aletsch region of the Swiss Alps — achieving a test IoU of 0.9839.

---

## Overview

Rapid glacier retreat in the Swiss Alps threatens freshwater supply for an estimated
170 million people across the Rhine, Rhône, Po and Danube river basins, disrupts
hydropower generation, and increases the frequency of high-mountain hazards including
glacial lake outburst floods and slope destabilisation. Monitoring at the required
spatial and temporal resolution using traditional field surveys or manual
photointerpretation is neither scalable nor cost-effective.

This work investigates whether the attention-gated U-Net architecture proposed for
tropical deforestation detection by John & Zhang (2022) can be successfully transferred
to high-alpine glacier segmentation — a domain with fundamentally different spectral
signatures, terrain characteristics, and label distributions.

**Two research questions are addressed:**

1. Can the attention U-Net of John & Zhang (2022) be reliably replicated under updated
   software dependencies (Keras 3.x / TensorFlow 2.19), and how do results compare
   across all published dataset configurations?
2. Does the architecture transfer effectively to glacier extent segmentation when adapted
   to six-band Sentinel-2 input and trained on a curated dataset of summer Alpine imagery?

---

## Scientific Contributions

- **Full replication study** across all configurations reported by John & Zhang (2022):
  Amazon (in-domain and cross-biome), Atlantic Forest (in-domain and cross-biome), and
  RGB-only Amazon. Results are compared quantitatively against published metrics with
  explicit analysis of discrepancies introduced by framework migration from
  Keras 2.x / TensorFlow 2.15 to Keras 3.x / TensorFlow 2.19.

- **Domain transfer to cryosphere monitoring**: Adaptation of the attention U-Net to
  glacier extent segmentation using Sentinel-2 Level-2A summer imagery over the
  Jungfrau–Aletsch region (2020–2025). The adapted model achieves test IoU of 0.9839,
  statistically significantly outperforming the original Amazon baseline across all
  metrics (one-sample t-test, p < 0.05).

- **Spectral extension to six bands**: Systematic expansion from four-band (RGB + NIR)
  to six-band input by incorporating SWIR bands B11 and B12, which improve the
  separability of glacier ice from water, shadow, and debris-covered surfaces.

- **NDSI-based automated labelling pipeline**: Binary glacier masks generated using
  an NDSI threshold of 0.25 with NDWI-based water filtering, enabling scalable
  label production without manual annotation at the cost of known spectral biases
  at debris-covered glacier margins.

- **Systematic hyperparameter search**: Six training configurations evaluated across
  60 epochs each, exploring batch size, learning rate, and LR reduction schedule
  interactions, with full fine-tuning of all 2.01M parameters motivated by the
  substantial spectral domain shift between tropical forest and Alpine glacier imagery.

---

## Results

### Part 1 — Replication: Deforestation Detection (John & Zhang 2022)

| Configuration | Split | IoU | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 4-band Amazon — **Ours** | Test | 0.7898 | 0.9164 | 0.8727 | 0.8940 |
| 4-band Amazon — **Paper** | Test | 0.9516 | 0.9758 | 0.9748 | 0.9753 |
| 4-band Amazon — **Ours** | Validation | 0.8488 | 0.9358 | 0.9123 | 0.9239 |
| 4-band Amazon — **Paper** | Validation | 0.9581 | 0.9790 | 0.9779 | 0.9785 |
| 4-band Amazon → Atlantic — **Ours** | Test | 0.8206 | 0.9220 | 0.8884 | 0.9049 |
| 4-band Amazon → Atlantic — **Paper** | Test | 0.8143 | 0.9222 | 0.8829 | 0.9021 |
| 4-band Atlantic Forest — **Ours** | Test | 0.8287 | 0.9325 | 0.8929 | 0.9123 |
| 4-band Atlantic Forest — **Paper** | Test | 0.9199 | 0.9591 | 0.9571 | 0.9581 |
| 4-band Atlantic Forest — **Ours** | Validation | 0.8178 | 0.9303 | 0.8833 | 0.9062 |
| 4-band Atlantic Forest — **Paper** | Validation | 0.9120 | 0.9563 | 0.9520 | 0.9541 |
| 4-band Atlantic → Amazon — **Ours** | Test | 0.8620 | 0.9396 | 0.9214 | 0.9304 |
| 4-band Atlantic → Amazon — **Paper** | Test | 0.8722 | 0.9445 | 0.9266 | 0.9355 |
| RGB Amazon (3-band) — **Ours** | Validation | 0.9072 | 0.9563 | 0.9490 | 0.9526 |
| RGB Amazon (3-band) — **Paper** | Validation | 0.9028 | 0.9574 | 0.9526 | 0.9550 |

**On the Amazon in-distribution test gap.** The 16-point IoU gap on the Amazon test set
does not appear in cross-biome transfer results, where scores match the paper within 1 IoU
point. This is consistent with two factors: (1) the original training seed and
data-loading order are unpublished, making in-distribution results more sensitive than
transfer results; and (2) migration to Keras 3.x / TensorFlow 2.19 required replacing
deprecated API calls (e.g. `K.int_shape()` → `x.shape`, `adam_v2.Adam` → `Adam`) which
alter gradient dynamics and convergence behaviour. Full analysis in
`notebooks/01_replication.ipynb`.

### Part 2 — Glacier Segmentation (Swiss Alps, Jungfrau–Aletsch)

| Configuration | Split | IoU | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 6-band Jungfrau–Aletsch — **Ours** | Test | **0.9839** | **0.9919** | **0.9915** | **0.9917** |

The glacier model significantly outperforms the original Amazon baseline across all
metrics (one-sample t-test, p < 0.05). However, this should be interpreted cautiously:
ice exhibits a more spectrally distinct signature than vegetation, the SWIR bands provide
additional discriminative power, and the test set covers only 101 images from a single
region and season. See `notebooks/05_evaluation.ipynb` for a full critical discussion,
confidence intervals, and confusion matrix analysis.

---

## Repository Structure

```text
├── notebooks/
│   ├── 01_replication.ipynb          # Full replication of John & Zhang (2022)
│   ├── 02_problem_formulation.ipynb  # Glacier monitoring context, impact, scalability
│   ├── 03_dataset_curation.ipynb     # Sentinel-2 acquisition, NDSI labelling, tiling
│   ├── 04_model_adaptation.ipynb     # Architecture changes, hyperparameter search, training
│   └── 05_evaluation.ipynb           # Metrics, confidence intervals, qualitative analysis
│
├── preprocessing_switzerland/        # Python package: glacier preprocessing pipeline
│   ├── preprocessing.py              # preprocess_gletscher_data, PreprocessingConfig
│   └── train_val_test_split.py       # setup_and_split_data_nested
│
├── preprocessing_paper/              # Python package: deforestation preprocessing utilities
│
├── scripts/
│   └── predict.py                    # Inference on new Sentinel-2 imagery
│
├── results/
│   ├── replication/                  # Saved metrics from Part 1
│   └── glacier/                      # Saved metrics from Part 2
│
├── checkpoints/                      # Empty — model weights hosted externally
├── docs/
│   ├── poster.pdf                    # Technical poster
│   └── figures/                      # Images referenced in notebooks and README
│
├── data/
│   └── README.md                     # Data acquisition and preprocessing instructions
│
├── environment.yml                   # Full Conda environment specification
├── requirements.txt                  # Minimal pip-compatible dependencies
└── CITATION.cff
```

---

## Setup

### 1. Clone
```bash
git clone --recurse-submodules https://github.com/camilletyriard-dev/glacier-segmentation-attention-unet.git
cd glacier-segmentation-attention-unet
```

`--recurse-submodules` is required to clone the `attention_unet` submodule alongside
the main repository.

### 2. Environment

**Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate glacier-seg
```

**pip:**
```bash
pip install -r requirements.txt
```

> Training was conducted on an AWS GPU instance (NVIDIA T4, 16 GB VRAM).
> The full glacier model hyperparameter search required approximately 12–24 GPU hours
> across six configurations of 60 epochs each.

### 3. Data

See [`data/README.md`](data/README.md) for full acquisition and preprocessing
instructions. Raw imagery and processed tiles are not committed to this repository.

- **Deforestation data**: [Zenodo record 3233081](https://zenodo.org/record/3233081)
  (Amazon) and the [original repository](https://github.com/davej23/attention-mechanism-unet)
- **Glacier data**: Sentinel-2 Level-2A via the
  [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
  (free registration required)

### 4. Model checkpoints

Pre-trained weights are hosted on Hugging Face. Download and place files in `checkpoints/`.

🤗 **[huggingface.co/camilletyriard/glacier-segmentation-attention-unet](https://huggingface.co/camilletyriard/glacier-segmentation-attention-unet)**

| File | Description |
|---|---|
| `glacier_segmentation_6band.keras` | Glacier segmentation — main model |
| `deforestation_amazon_4band.keras` | Amazon replication |
| `deforestation_atlantic_4band.keras` | Atlantic Forest replication |
| `deforestation_amazon_rgb_3band.keras` | Amazon RGB replication |

### 5. Run notebooks
```bash
jupyter lab notebooks/
```

Execute in numerical order. Each notebook is self-contained.

### 6. Inference on new imagery
```bash
python scripts/predict.py \
  --image path/to/sentinel2_tile.tif \
  --model checkpoints/attention_unet_glacier_6band.hdf5 \
  --output predictions/mask.png \
  --threshold 0.5
```

---

## Model Architecture

Attention-gated U-Net (Oktay et al., 2018) as implemented by John & Zhang (2022),
with the following modifications for glacier segmentation:

| Component | John & Zhang (2022) | This work |
|---|---|---|
| Input channels | 3 (RGB) or 4 (RGB + NIR) | **6** (B02, B03, B04, B08, B11, B12) |
| Bottleneck dropout | None | **p = 0.5** |
| Loss function | Binary cross-entropy | **Dice loss** |
| Augmentation | Horizontal flip (3-band only) | **Flip + 90° rotation + ±brightness** |
| Batch size | 1 | **4** |
| LR scheduling | None | **ReduceLROnPlateau (patience=5)** |
| Early stopping | None | **patience = 10 epochs** |
| Fine-tuning strategy | Full training | **Full training (all 2.01M parameters)** |
| Total parameters | 2.01M | 2.01M |

**Rationale for full fine-tuning:** Initial experiments with partial fine-tuning
(frozen encoder, trainable decoder only) produced unstable learning curves and poor
convergence. This is consistent with the substantial spectral domain shift between
tropical vegetation and Alpine glacier ice, which requires significant weight
adaptation throughout the network rather than only in the decoder.

**Rationale for Dice loss:** Ice covers only approximately 12% of pixels across the
dataset. Dice loss directly optimises the overlap between prediction and ground truth,
making it more robust than binary cross-entropy under this class imbalance.

**Rationale for SWIR bands:** Bands B11 (1610 nm) and B12 (2190 nm) exhibit strong
absorption by liquid water and high reflectance for snow and ice, providing spectral
discrimination that RGB and NIR alone cannot achieve — particularly for separating
clean glacier ice from supraglacial lakes and debris-covered tongues.

---

## Dataset

| Property | Deforestation (John & Zhang) | Glacier (This work) |
|---|---|---|
| Source | Zenodo / original repo | Copernicus Data Space |
| Region | Amazon, Atlantic Forest (Brazil) | Jungfrau–Aletsch, Swiss Alps |
| Sensor | Sentinel-2 | Sentinel-2 Level-2A |
| Bands | 3 (RGB) or 4 (RGB + NIR) | 6 (RGB + NIR + SWIR1 + SWIR2) |
| Acquisition period | Various | July–September, 2020–2025 |
| Max cloud cover | — | 4% |
| Approx. dataset size | ~250 (Amazon) | ~1,000 image–mask pairs |
| Tile size | 512 × 512 px | 512 × 512 px |
| Label method | Manual annotation | NDSI ≥ 0.25 + NDWI filter |
| Test set size | — | 101 images |

---

## Limitations

- **Geographic scope**: The glacier model was trained and evaluated exclusively on
  the Jungfrau–Aletsch region. Generalisation to other Alpine systems, Himalayan
  glaciers, or surge-type glaciers is not validated.

- **Seasonal constraint**: Only July–September imagery was used. The model is not
  expected to generalise to winter or spring conditions where seasonal snowpack
  cannot be distinguished from glacier ice.

- **Label noise at debris-covered margins**: NDSI-based labelling systematically
  under-detects debris-covered glacier tongues, which reflect light similarly to
  surrounding rock. This introduces boundary noise in both training and evaluation.

- **Test set scope**: With 101 test images from a single region and season, reported
  metrics should be interpreted as indicative of within-distribution performance, not
  broad generalisation.

- **Computational cost**: The hyperparameter search required an estimated 12–24 GPU
  hours. Operational deployment would incur ongoing infrastructure costs that should
  be weighed against the environmental benefits of replacing helicopter-based surveys.

---

## Acknowledgements

This work was conducted as part of the *Artificial Intelligence for Sustainable
Development* module at University College London. The attention U-Net
implementation is based on the open-source code of John & Zhang (2022).
Sentinel-2 imagery is provided by the European Space Agency under the
Copernicus Open Data Licence.

---

## Citation
```bibtex
@misc{tyriard2025glacier,
  author = {Tyriard, Camille},
  title  = {Attention {U-Net} for Glacier Extent Segmentation in the {Swiss Alps}},
  year   = {2025},
  url    = {https://github.com/camilletyriard-dev/glacier-segmentation-attention-unet}
}
```

This work builds on:
```bibtex
@article{john2022attention,
  title   = {An attention-based {U-Net} for detecting deforestation within satellite sensor imagery},
  author  = {John, David A. and Zhang, Chuanxin},
  journal = {International Journal of Applied Earth Observation and Geoinformation},
  volume  = {107},
  year    = {2022},
  doi     = {10.1016/j.jag.2022.102685}
}
```

---

## License

[MIT License](LICENSE). The `attention_unet` submodule retains the license of its original authors.
