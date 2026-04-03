# Data

Raw imagery and processed tiles are not committed to this repository.
Instructions for obtaining both datasets are provided below.

---

## 1. Deforestation Dataset (John & Zhang 2022 — Replication)

The Amazon and Atlantic Forest Sentinel-2 datasets are provided by the original authors.

- **Source**: https://github.com/davej23/attention-mechanism-unet
- **Amazon dataset**: https://zenodo.org/record/3233081
- Follow the download instructions in the original repository README.
- Place data in: `data/deforestation/amazon/` and `data/deforestation/atlantic/`

The download and extraction pipeline is implemented in `notebooks/01_replication.ipynb`.

---

## 2. Glacier Dataset — Swiss Alps (This Work)

Sentinel-2 Level-2A imagery was acquired from the Copernicus Data Space Ecosystem.
Free registration is required at: https://dataspace.copernicus.eu/

### Acquisition parameters

| Parameter | Value |
|---|---|
| Region | Jungfrau–Aletsch, Switzerland |
| Sensor | Sentinel-2A/B MSI |
| Processing level | Level-2A (surface reflectance) |
| Acquisition period | July–September, 2020–2025 |
| Maximum cloud cover | 4% |
| Bands used | B02, B03, B04, B08, B11, B12 |
| Approximate dataset size | ~1,000 image–mask pairs |
| Tile size | 512 × 512 px |

### Label generation

Binary glacier masks were generated algorithmically using an NDSI threshold of 0.25
combined with an NDWI-based water filter to suppress misclassification of lakes and
rivers. Labels were not manually annotated. A random sample of 50 images was
visually inspected to verify mask quality. Known limitations include occasional
misclassification of clean snow as glacier ice and under-detection of debris-covered
glacier tongues. See `notebooks/03_dataset_curation.ipynb` for the full pipeline.

### Preprocessing

After downloading scenes from the Copernicus portal, run
`notebooks/03_dataset_curation.ipynb`. The preprocessing pipeline is implemented in
`preprocessing_switzerland/preprocessing.py` and
`preprocessing_switzerland/train_val_test_split.py`.

Processed tiles are written to `data_switzerland/preprocessed_data/` with the
following structure:

data_switzerland/preprocessed_data/
├── train/
│   ├── X/      ← 6-band input tiles (.npy)
│   └── Y/      ← binary glacier masks (.npy)
├── val/
│   ├── X/
│   └── Y/
└── test/
├── X/
└── Y/

### Model checkpoints

Pre-trained weights are hosted externally due to file size constraints.

| Model | Description | Link |
|---|---|---|
| `attention_unet_amazon_4band.hdf5` | Amazon replication | [add link] |
| `attention_unet_atlantic_4band.hdf5` | Atlantic Forest replication | [add link] |
| `attention_unet_glacier_6band.hdf5` | Glacier segmentation (best model) | [add link] |

Place downloaded `.hdf5` files in `checkpoints/`.