# WordGesture-GAN Reproduction

This project reproduces **WordGesture-GAN** (Chu et al., 2023) and the **Minimum Jerk** gesture-production model (Quinn & Zhai, 2018) on the *How We Swipe* dataset.

## Project Structure

```
WordGesture-GAN/
├── shared/                 # Common utilities
│   ├── keyboard.py         # QWERTY layout and key positions
│   ├── data.py             # Dataset loading and preprocessing
│   ├── prototypes.py       # Word prototype generation
│   ├── features.py         # GestureFeatureExtractor (31-dim features)
│   ├── config.py           # YAML config loading
│   ├── utils.py            # Device detection utilities
│   └── parse_swipelogs.py  # Raw log parsing
│
├── wg_gan/                 # WordGesture-GAN (gesture synthesis)
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation script
│   ├── models.py           # Generator, Discriminator, Encoder
│   ├── reproduce_figures.py # GIF animation generation
│   ├── modal_app.py        # Modal cloud training
│   ├── config.yaml
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
├── ctc/                    # CTC decoder and recognition pipeline
│   ├── train.py            # Training script
│   ├── models.py           # BLSTM-CTC model
│   ├── decode.py           # CTCDecoder for inference
│   ├── trie.py             # Vocabulary trie for beam search
│   ├── beam_search.py      # Trie-constrained beam search
│   ├── data.py             # Dataset loader
│   ├── eval.py             # Full pipeline evaluation
│   ├── reranker.py         # LLM reranking (OpenRouter)
│   ├── sentence_data.py    # Sentence dataset loader
│   ├── modal_app.py        # Modal cloud training
│   └── config.yaml
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

## CTC Decoder

Character-level CTC decoder with trie-constrained beam search for gesture recognition.

### Train

```bash
# Local
python -m ctc.train --config ctc/config.yaml

# Modal (cloud GPU)
modal run ctc/modal_app.py --epochs 100
modal volume get wordgesture-gan-data checkpoints/ctc/ctc_best.pt ./ctc/checkpoints/
```

### Inference

```python
from ctc import CTCDecoder
from ctc.eval import load_wordfreq_vocabulary

# Load vocabulary and decoder
vocab = load_wordfreq_vocabulary(10000)
decoder = CTCDecoder.from_checkpoint("ctc/checkpoints/ctc_best.pt", vocabulary=vocab)

# Decode single gesture
candidates = decoder.decode_top_k(gesture, k=10, beam_width=100)
# Returns: [("hello", -5.2), ("help", -6.1), ...]
```

## Recognition Pipeline

Evaluates the full pipeline: CTC beam search → LLM reranking.

### Setup (one-time for OpenRouter)

```bash
pip install openai python-dotenv

# Create .env with your OpenRouter API key
# Get your API key at https://openrouter.ai/keys
echo "OPENROUTER_API_KEY=sk-or-..." >> .env
```

### Evaluate

```bash
# Top-1 (no reranking)
python -m ctc.eval

# With LLM reranking
python -m ctc.eval --reranker

# Larger vocabulary
python -m ctc.eval --vocab-size 50000
```

Options:
- `--reranker`: Enable LLM reranking (default: top-1 only)
- `--checkpoint`: CTC model (default: `ctc/checkpoints/ctc_best.pt`)
- `--k`: Number of candidates (default: 10)
- `--beam-width`: Beam search width (default: 100)
- `--vocab-size`: Vocabulary size (default: 20000)
- `--max_sentences`: Sentences to evaluate (default: 50)
- `--seed`: Random seed for reproducibility
- `--model`: OpenRouter model (default: `google/gemini-3-flash-preview`)

### Logging

```bash
python -m ctc.eval --reranker --rerank-log logs/eval.jsonl

# View failed sentences
jq 'select(.is_correct == false)' logs/eval.jsonl

# View summary (last line)
tail -1 logs/eval.jsonl | jq
```

## Notes

- Coordinates normalized to `[-1, 1]`, timestamps as `dt` (seconds)
- Minimum jerk baseline uses aggregate via-point distributions
- CLC duration: `T(P) = Σ m * ||AB||^n` over prototype segments
- Shape-only metrics use uniform `dt=1` for comparable dynamics
- WG-GAN reports both shape-only and spatiotemporal `(x, y, dt)` metrics
