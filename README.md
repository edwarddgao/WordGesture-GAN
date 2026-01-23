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

```bash
python train_wg_gan.py --config configs/wg_gan.yaml
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

## Notes

- Coordinates are normalized to `[-1, 1]` with timestamps stored as `dt` (seconds).
- The minimum-jerk baseline is implemented as described in Quinn & Zhai (2018) with aggregate via-point distributions.
