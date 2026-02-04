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
├── features/               # Gesture feature extraction
│   └── gesture.py          # GestureFeatureExtractor (31-dim features)
│
├── wg_gan/                 # WordGesture-GAN (gesture synthesis)
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
├── contrastive/            # Contrastive learning (embedding space)
│   ├── train.py            # Training script
│   ├── models.py           # Two-tower encoder
│   ├── losses.py           # InfoNCE loss
│   ├── data.py             # Dataset with augmentation
│   ├── config.yaml
│   └── checkpoints/
│
├── ctc/                    # CTC decoder (character-level)
│   ├── train.py            # Training script
│   ├── models.py           # BLSTM-CTC model
│   ├── decode.py           # CTCDecoder for inference
│   ├── data.py             # Dataset loader
│   ├── modal_app.py        # Modal cloud training
│   └── config.yaml
│
├── recognition/            # Recognition pipeline
│   ├── eval.py             # Full pipeline evaluation
│   ├── reranker.py         # LLM reranking (Gemini)
│   └── sentence_data.py    # Sentence dataset loader
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

Two-tower encoder for gesture-to-word embedding matching.

### Train

```bash
python -m contrastive.train --config contrastive/config.yaml
```

## CTC Decoder

Character-level CTC decoder for OOV word recovery.

### Train

```bash
# Local
python -m ctc.train --config ctc/config.yaml

# Modal (cloud GPU)
modal run ctc/modal_app.py --epochs 100
modal volume get wordgesture-gan-data checkpoints/ctc/ctc_best.pt ./ctc/checkpoints/
```

## Recognition Pipeline

Evaluates the full pipeline: contrastive retrieval → CTC decoder → LLM reranking.

### Setup (one-time for Gemini)

```bash
pip install google-genai>=1.51.0 python-dotenv
gcloud auth application-default login

# Create .env with your GCP project
echo "GOOGLE_CLOUD_PROJECT=your-project-id" >> .env
echo "GOOGLE_CLOUD_LOCATION=global" >> .env
```

### Evaluate

```bash
# Top-1 retrieval (no reranking)
python -m recognition.eval \
  --checkpoint contrastive/checkpoints/contrastive_latest.pt \
  --ctc-checkpoint ctc/checkpoints/ctc_best.pt \
  --max_sentences 200

# With Gemini reranking
python -m recognition.eval \
  --checkpoint contrastive/checkpoints/contrastive_latest.pt \
  --ctc-checkpoint ctc/checkpoints/ctc_best.pt \
  --reranker gemini \
  --max_sentences 200
```

### Logging

Use `--rerank-log` to write JSONL logs for analysis:

```bash
python -m recognition.eval \
  --checkpoint contrastive/checkpoints/contrastive_latest.pt \
  --ctc-checkpoint ctc/checkpoints/ctc_best.pt \
  --reranker gemini \
  --rerank-log logs/eval.jsonl

# View failed sentences
jq 'select(.is_correct == false)' logs/eval.jsonl

# View summary metrics (last line)
tail -1 logs/eval.jsonl | jq
```

Log format:
- Per-sentence entries: `ground_truth`, `predictions`, `candidates`, `ctc_words`, `raw_response`, `parse_details`
- Summary entry (last line): `word_accuracy`, `sentence_accuracy`, `wer`, `errors`, `fallback_reasons`

Use `--seed` to fix the random seed for reproducibility.

## Notes

- Coordinates normalized to `[-1, 1]`, timestamps as `dt` (seconds)
- Minimum jerk baseline uses aggregate via-point distributions
- CLC duration: `T(P) = Σ m * ||AB||^n` over prototype segments
- Shape-only metrics use uniform `dt=1` for comparable dynamics
- WG-GAN reports both shape-only and spatiotemporal `(x, y, dt)` metrics
