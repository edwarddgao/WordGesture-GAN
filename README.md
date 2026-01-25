# WordGesture-GAN Reproduction

This project reproduces **WordGesture-GAN** (Chu et al., 2023) and the **Minimum Jerk** gesture-production model (Quinn & Zhai, 2018) on the *How We Swipe* dataset.

## Project Structure

```
WordGesture-GAN/
├── shared/                 # Common utilities
│   ├── keyboard.py         # QWERTY layout and key positions
│   ├── data.py             # Dataset loading and preprocessing
│   ├── prototypes.py       # Word prototype generation
│   └── parse_swipelogs.py  # Raw log parsing
│
├── wg_gan/                 # WordGesture-GAN package
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation script
│   ├── models.py           # Generator, Discriminator, Encoder
│   ├── reproduce_figures.py # GIF animation generation
│   ├── modal_app.py        # Modal cloud training
│   ├── config.yaml
│   ├── checkpoints/
│   ├── results/
│   ├── metrics/            # Evaluation metrics
│   │   ├── distance.py     # L2, DTW, Wasserstein
│   │   ├── fid.py          # Fréchet Inception Distance
│   │   ├── precision_recall.py
│   │   ├── duration.py     # CLC baseline
│   │   └── dynamics.py     # Velocity, acceleration, jerk
│   └── minjerk/            # Minimum jerk baseline
│       ├── fit.py          # Distribution fitting
│       ├── sample.py       # Trajectory sampling
│       └── via_points.py   # Via-point extraction
│
├── contrastive/            # Contrastive learning package
│   ├── train.py
│   ├── eval.py
│   ├── models.py           # Two-tower encoder
│   ├── losses.py           # InfoNCE loss
│   ├── data.py             # Augmentation pipeline
│   ├── config.yaml
│   ├── checkpoints/
│   └── results/
│
└── data/                   # Dataset storage
    └── processed/          # Preprocessed .npz files
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preprocessing

Place raw swipe logs in `data/swipelogs/` (one `.log` per user/session).

```bash
# Parse raw logs to JSONL
python -m shared.parse_swipelogs \
  --log_dir data/swipelogs \
  --out_dir data/processed/raw

# Preprocess to fixed-length tensors
python -m shared.data \
  --raw_dir data/processed/raw \
  --out_dir data/processed \
  --n_points 128 \
  --max_per_word 5 \
  --train_split 0.8 \
  --seed 1337
```

Creates `data/processed/train.npz`, `data/processed/test.npz`, and `split.json`.

## WordGesture-GAN

### Train

```bash
# Local
python -m wg_gan.train --config wg_gan/config.yaml

# Modal (cloud GPU)
modal volume create wordgesture-gan-data
modal volume put wordgesture-gan-data data/processed/train.npz processed/train.npz
modal volume put wordgesture-gan-data data/processed/test.npz processed/test.npz
modal run wg_gan/modal_app.py --epochs 50
modal volume get wordgesture-gan-data checkpoints/wg_gan_latest.pt ./wg_gan/checkpoints/
```

### Evaluate

```bash
python -m wg_gan.eval \
  --config wg_gan/config.yaml \
  --minjerk_config wg_gan/minjerk/config.yaml \
  --checkpoint wg_gan/checkpoints/wg_gan_latest.pt \
  --data_dir data/processed \
  --out_dir wg_gan/results
```

### Animate

```bash
# Multi-word overlay (Figure 7/8 style)
python -m wg_gan.reproduce_figures \
  --n_words 6 \
  --models user,wg_gan,minjerk \
  --n_samples 5 \
  --collapse_samples true \
  --out wg_gan/results/gifs/overlay.gif

# Per-model grid (Figure 6 style)
python -m wg_gan.reproduce_figures \
  --n_words 6 \
  --models user,wg_gan,minjerk \
  --n_samples 1 \
  --collapse_samples false \
  --out wg_gan/results/gifs/grid.gif
```

## Contrastive Learning

Two-tower encoder for gesture-to-word matching.

### Train

```bash
python -m contrastive.train --config contrastive/config.yaml
```

### Evaluate

```bash
python -m contrastive.eval \
  --checkpoint contrastive/checkpoints/contrastive_latest.pt \
  --data_dir data/processed
```

## Notes

- Coordinates normalized to `[-1, 1]`, timestamps as `dt` (seconds)
- Minimum jerk baseline uses aggregate via-point distributions
- CLC duration: `T(P) = Σ m * ||AB||^n` over prototype segments
- Shape-only metrics use uniform `dt=1` for comparable dynamics
- WG-GAN reports both shape-only and spatiotemporal `(x, y, dt)` metrics
