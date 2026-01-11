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

```

**Training Settings:**
- Cosine annealing LR scheduler (eta_min=1e-5, enabled by default)
- Gradient clipping (max_norm=1.0, enabled by default)
- Loss weights: lambda_rec=4.0, lambda_kld=0.02
- Generator: gen_hidden_dim=48 (increased from 32 for better capacity)

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
- Velocity correlation (time-aware, d/dt)
- Acceleration correlation (time-aware, d²/dt²)
- Speed profile correlation
- Time delta correlation
- FID score (autoencoder-based)
- Precision/Recall (k=3 nearest neighbors)

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

**Configuration: λ_rec=4.0, λ_kld=0.02, gen_hidden_dim=48**

| Metric | GAN | Min Jerk | Paper GAN | Notes |
|--------|-----|----------|-----------|-------|
| L2 Wasserstein (x,y) | **2.81** | 2.76 | 4.409 | **36% better** |
| DTW Wasserstein (x,y) | **1.46** | 1.36 | 2.146 | **32% better** |
| FID | **0.024** | 0.026 | 0.270 | **91% better** |
| Precision (k=3) | **1.000** | 1.000 | 0.973 | Matches paper |
| Recall (k=3) | **0.525** | 0.525 | 0.258 | **104% better** |
| Velocity Corr | 0.177 | **0.716** | 0.40 | Min jerk better |
| Acceleration Corr | 0.003 | **0.162** | 0.26 | Min jerk better |
| Speed Profile Corr | 0.094 | **0.313** | -- | Min jerk better |
| Time Delta Corr | 0.190 | **0.250** | -- | Min jerk better |

**Spatial metrics (L2, DTW, FID, Precision/Recall) are excellent.** However, the GAN underperforms on temporal dynamics compared to minimum jerk baseline. The model learns spatial shape well but not when to speed up/slow down.

### Time-Aware Dynamics Metrics

Velocity and acceleration correlations now use proper temporal derivatives:
- `velocity = d(position) / d(time)` instead of `d(position) / d(index)`
- This correctly measures whether the model learns human-like speed patterns

The minimum jerk baseline performs better on dynamics because it explicitly models the physics of human motor control (minimizing jerk produces natural velocity profiles).

### Precision/Recall k Parameter

The paper uses k=3 for k-NN manifold estimation. Our implementation defaults to k=3 to match.

### Minimum Jerk Model (Quinn & Zhai 2018)

The minimum jerk baseline follows the paper's approach of learning distributions from data:

1. **Key Center Offsets**: Distribution of how far users deviate from key centers (x, y independently)
2. **Midpoint Angles**: Distribution of perpendicular deviation for midpoints between consecutive keys

The `MinimumJerkModel` class in `src/keyboard.py`:
- `fit(gestures_by_word)`: Learns distributions from training data
- `generate_trajectory(word)`: Generates trajectories using learned distributions

```python
from src.keyboard import QWERTYKeyboard, MinimumJerkModel

keyboard = QWERTYKeyboard()
model = MinimumJerkModel(keyboard)
model.fit(training_gestures_by_word)  # Learn from data
trajectory = model.generate_trajectory("hello", num_points=128)
```

Run evaluation with: `python eval_min_jerk.py`
