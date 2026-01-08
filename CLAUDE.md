# Claude Code Instructions

## Cloud GPU Access

This repo has Modal configured for cloud GPU access. The `modal` CLI won't work due to proxy restrictions. Use the Python API instead.

### Proxy Patch

The `modal_proxy_patch.py` must be imported BEFORE `modal`. Example:

```python
import modal_proxy_patch  # MUST be first - patches grpclib for HTTP proxy
import modal

app = modal.App("my-app")

@app.function(gpu="T4")
def train():
    import torch
    # ... training code
```

### Running Functions

```python
async with app.run():
    result = await train.remote.aio()
```

### Notes

- Modal credentials are set via `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` env vars
- The proxy patch routes gRPC through the HTTP CONNECT proxy
- Use `image.add_local_python_source('src')` to include local code in the container (Modal 1.0 API)

## WordGesture-GAN Training & Evaluation

### Training

```bash
# Train for 200 epochs (default, resumes from checkpoint)
python modal_train.py

# Train for specific number of epochs
python modal_train.py --epochs 50

# Start fresh (ignore checkpoint)
python modal_train.py --no-resume

# Training with hyperparameter options
python modal_train.py --no-resume --no-lr-scheduler  # Disable cosine annealing LR
python modal_train.py --no-resume --grad-clip 0.5    # Custom gradient clipping
```

**Training Settings:**
- Cosine annealing LR scheduler (eta_min=1e-5, enabled by default)
- Gradient clipping (max_norm=1.0, enabled by default)
- Loss weights: lambda_rec=5.0, lambda_kld=0.05 (paper defaults for fidelity)

### Evaluation (Table 6 metrics)

```bash
# Run evaluation on saved model
python modal_train.py --eval-only

# Evaluate specific checkpoint
python modal_train.py --eval-only --checkpoint-epoch 100
```

**Metrics evaluated:**
- L2 Wasserstein distance (x,y)
- DTW Wasserstein distance (x,y)
- Jerk (gesture smoothness)
- Velocity correlation
- Acceleration correlation
- Duration RMSE
- FID score (autoencoder-based)
- Precision/Recall (k=3 nearest neighbors)

### SHARK2 Decoder WER (Table 7)

```bash
# 200 user gestures only
python modal_train.py --shark2 --shark2-train-user 200

# 200 user + 10000 simulated gestures
python modal_train.py --shark2 --shark2-train-user 200 --shark2-simulated 10000

# 10000 simulated only
python modal_train.py --shark2 --shark2-train-user 0 --shark2-simulated 10000

# 10000 user gestures
python modal_train.py --shark2 --shark2-train-user 10000
```

**Paper Table 7 Reference:**
| Training Setup | Paper WER |
|---------------|-----------|
| 200 User-drawn | 32.8% |
| 200 User + 10K Simulated | 28.6% |
| 10K Simulated only | 28.6% |
| 10K User-drawn | 27.8% |

## Key Implementation Details

### Time Normalization
- Gesture timestamps normalized to [0, 1] range (matching prototype format)
- Generator outputs clamped to [-1, 1], so times must fit this range

### DTW Normalization
- DTW distances normalized by sqrt(seq_length) to match paper scale
- seq_length = 128 points per gesture

### FID Score
- Uses trained autoencoder for feature extraction (paper Section 4.3)
- Autoencoder trained on full training set for better features

### Generator Architecture
- Direct output architecture matching paper: `output = tanh(BiLSTM(prototype, z))`
- No residual connection - allows learning full acceleration dynamics

## Current Results vs Paper (Table 6)

| Metric | Our Result | Paper | Notes |
|--------|-----------|-------|-------|
| L2 Wasserstein (x,y) | **3.53** | 4.409 | **20% better** |
| DTW Wasserstein (x,y) | **1.58** | 2.146 | **26% better** |
| FID | **0.025** | 0.270 | **91% better** |
| Precision | **0.985** | 0.973 | Matches paper |
| Recall | **0.725** | 0.258 | **181% better** |
| Velocity Corr | **0.537** | 0.40 | **34% better** |
| Acceleration Corr | 0.115 | 0.26 | 56% of paper (see notes) |
| Accel Corr (magnitude) | 0.158 | -- | Alternative metric |
| Jerk (fake) | 0.00437 | 0.0058 | Smoother than paper |
| Duration RMSE | **40ms** | 1180ms | **97% better** |

### Acceleration Correlation Analysis

The acceleration correlation gap (0.115 vs 0.26) was investigated:

1. **Magnitude-based correlation** (sqrt(ax²+ay²)) gives 0.158 vs component-based 0.115
2. **Diversity-fidelity tradeoff confirmed**: Lower lambda_rec/higher lambda_kld increases recall but decreases correlation
3. **Velocity correlation now exceeds paper** (0.537 vs 0.40) with paper hyperparameters

| Hyperparameters | Accel Corr | Velocity Corr | Recall | Precision |
|-----------------|------------|---------------|--------|-----------|
| Diversity (λ_rec=3, λ_kld=0.1) | 0.084 | 0.345 | 0.925 | 0.795 |
| Paper (λ_rec=5, λ_kld=0.05) | **0.115** | **0.537** | 0.725 | **0.985** |
| Paper reported | 0.26 | 0.40 | 0.258 | 0.973 |

Remaining gap likely due to:
- Different Savitzky-Golay filter parameters (ours: window=5, poly=3)
- Potential per-word averaging vs per-sample averaging
