# WordGesture-GAN Reproduction

This project reproduces **WordGesture-GAN** (Chu et al., 2023) and the **Minimum Jerk** gesture-production model (Quinn & Zhai, 2018) on the *How We Swipe* dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data preprocessing

Place the raw swipe logs in `data/swipelogs/` (one `.log` per user/session).

```bash
python -m wordgesture_gan.data.parse_swipelogs \
  --log_dir data/swipelogs \
  --out_dir data/processed/raw

python -m wordgesture_gan.data.preprocess \
  --raw_dir data/processed/raw \
  --out_dir data/processed \
  --n_points 128 \
  --max_per_word 5 \
  --train_split 0.8 \
  --seed 1337
```

This creates `data/processed/train.npz`, `data/processed/test.npz`, and split metadata.

## Train WordGesture-GAN

**Local:**
```bash
python train_wg_gan.py --config configs/wg_gan.yaml
```

**Modal (cloud GPU):**
```bash
# One-time setup: upload data to Modal volume
modal volume create wordgesture-gan-data
modal volume put wordgesture-gan-data data/processed/train.npz processed/train.npz
modal volume put wordgesture-gan-data data/processed/test.npz processed/test.npz

# Train on L40S GPU
modal run modal_app.py --epochs 50

# Download checkpoint
modal volume get wordgesture-gan-data checkpoints/wg_gan_latest.pt ./checkpoints/
```

Checkpoints are written under `checkpoints/`.

## Evaluate (WordGesture-GAN + Minimum Jerk)

```bash
python eval.py \
  --config configs/wg_gan.yaml \
  --minjerk_config configs/minjerk.yaml \
  --checkpoint checkpoints/wg_gan_latest.pt \
  --data_dir data/processed \
  --out_dir results
```

Outputs JSON summaries and prints paper-style tables to stdout.

## Animate gesture comparisons

GIF export uses Pillow (included in requirements).

```bash
# Multi-word overlays (Figure 7/8 style)
python reproduce_figures.py \
  --n_words 6 \
  --models user,wg_gan,minjerk \
  --n_samples 5 \
  --collapse_samples true \
  --out results/gifs/fig8_overlay.gif

# Per-model x per-word grid (Figure 6 style)
python reproduce_figures.py \
  --n_words 6 \
  --models user,wg_gan,minjerk \
  --n_samples 1 \
  --collapse_samples false \
  --out results/gifs/fig6_grid.gif
```

## Notes

- Coordinates are normalized to `[-1, 1]` with timestamps stored as `dt` (seconds).
- The minimum-jerk baseline uses aggregate via-point distributions and a minimum-jerk trajectory solver; evaluation treats it as spatial-only.
- The CLC duration baseline uses the paper equation `T(P) = Σ m * ||AB||^n` over prototype segments.
- Shape-only metrics use a flattened L2 (Frobenius) distance and DTW with sqrt of accumulated squared cost, matching the paper scale.
- Shape-only dynamics (velocity/accel/jerk) use uniform `dt=1` (index-based) for comparable magnitudes.
- WG-GAN evaluation reports both shape-only metrics and spatiotemporal Wasserstein on `(x, y, dt)`.
