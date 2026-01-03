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
- Residual architecture: `output = prototype + delta`
- Delta scale tuning affects acceleration dynamics
