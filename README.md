# WordGesture-GAN

Implementation of **"WordGesture-GAN: Modeling Word-Gesture Movement with Generative Adversarial Network"** from CHI '23.

This model generates realistic word-gesture (swipe typing) movements for arbitrary words on a virtual keyboard, producing both spatial (x, y) and temporal (t) sequences.

## Paper Reference

```
Jeremy Chu, Dongsheng An, Yan Ma, Wenzhe Cui, Shumin Zhai, Xianfeng Gu, and Xiaojun Bi. 2023.
WordGesture-GAN: Modeling Word-Gesture Movement with Generative Adversarial Network.
In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems (CHI '23).
https://doi.org/10.1145/3544548.3581279
```

## Architecture Overview

WordGesture-GAN consists of three main components:

1. **Generator (BiLSTM)**: Takes a word prototype and sampled latent code to generate gesture sequences
2. **Variational Encoder (MLP)**: Encodes user-drawn gestures into a Gaussian latent space
3. **Discriminator (MLP + Spectral Norm)**: Distinguishes real from generated gestures

### Key Features

- Generates both spatial (x, y) and temporal (t) information
- Uses Wasserstein GAN loss for stable training
- Two-cycle training procedure (similar to BicycleGAN)
- No user reference gestures needed at inference time

## Project Structure

```
WordGesture-GAN/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration dataclasses
│   ├── keyboard.py        # QWERTY keyboard layout and word prototypes
│   ├── data.py            # Dataset loading and preprocessing
│   ├── models.py          # Generator, Discriminator, Encoder architectures
│   ├── losses.py          # Loss functions (Wasserstein, feature matching, etc.)
│   ├── trainer.py         # Training loop with two-cycle procedure
│   ├── evaluation.py      # Evaluation metrics (Wasserstein dist, FID, etc.)
│   └── visualization.py   # Plotting utilities
├── dataset/
│   └── swipelogs.zip      # Mobile word-gesture dataset
├── train.py               # Main training script
├── evaluate.py            # Evaluation script
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

This implementation uses the **"How We Swipe"** dataset (Leiva et al., MobileHCI 2021) containing ~38k word-gesture samples from 1,338 users.

The dataset should be placed at `dataset/swipelogs.zip`.

## Training

Basic training:
```bash
python train.py --data_path dataset/swipelogs.zip --epochs 200
```

With custom parameters:
```bash
python train.py \
    --data_path dataset/swipelogs.zip \
    --epochs 200 \
    --batch_size 512 \
    --lr 0.0002 \
    --latent_dim 32 \
    --lambda_feat 1.0 \
    --lambda_rec 5.0 \
    --lambda_lat 0.5 \
    --lambda_kld 0.05 \
    --checkpoint_dir checkpoints
```

Resume training from checkpoint:
```bash
python train.py --resume checkpoints/checkpoint_epoch_100.pt
```

## Cloud GPU Training (Modal)

For environments without local GPU access (e.g., Claude Code web containers), training can be run on Modal cloud GPUs.

### Setup

1. Set Modal credentials as environment variables:
   ```bash
   export MODAL_TOKEN_ID="ak-..."
   export MODAL_TOKEN_SECRET="as-..."
   ```

2. Run training:
   ```bash
   python run_training.py test                # Test GPU access
   python run_training.py train --epochs 200  # Run training
   python run_training.py list                # List checkpoints
   ```

### How It Works

The `modal_proxy_patch.py` module patches grpclib to route gRPC connections through HTTP proxies, enabling Modal access from restricted network environments. Training functions are defined in `modal_train.py` and executed on Tesla T4 GPUs via Modal's infrastructure.

### Training Parameters (from paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch size | 512 | Training batch size |
| Learning rate | 0.0002 | Adam optimizer LR |
| n_critic | 5 | Discriminator updates per generator update |
| λ_feat | 1.0 | Feature matching loss weight |
| λ_rec | 5.0 | Reconstruction loss weight |
| λ_lat | 0.5 | Latent encoding loss weight |
| λ_KLD | 0.05 | KL divergence loss weight |
| Latent dim | 32 | Gaussian latent code dimension |
| Sequence length | 128 | Number of points per gesture |

## Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --data_path dataset/swipelogs.zip \
    --output_dir results
```

### Evaluation Metrics

The model is evaluated using:

1. **L2 Wasserstein Distance**: Measures similarity between generated and real gesture distributions
2. **DTW Wasserstein Distance**: Uses Dynamic Time Warping for temporal alignment
3. **Fréchet Inception Distance (FID)**: Measures realism and diversity
4. **Precision & Recall**: Assesses coverage and realism using k-NN manifolds
5. **Velocity/Acceleration Correlation**: Compares movement dynamics
6. **Jerk Analysis**: Compares smoothness of generated gestures

## Model Architecture Details

### Generator (BiLSTM)
- Input: Word prototype (128×3) + Repeated latent code (128×32)
- 4 BiLSTM layers with hidden dimension 32
- Linear output layer with Tanh activation
- Output: Generated gesture (128×3)

### Variational Encoder
- Input: Flattened gesture (384-dimensional)
- MLP: 384 → 192 → 96 → 48 → 32 (Leaky ReLU)
- Two heads: μ and log(σ²) for reparameterization trick

### Discriminator
- Input: Flattened gesture (384-dimensional)
- MLP: 384 → 192 → 96 → 48 → 24 → 1
- Spectral normalization on all layers
- Leaky ReLU activation

## Loss Functions

The generator loss combines five components:

```
L_gen = -L_disc + λ_feat·L_feat + λ_rec·L_rec + λ_lat·L_lat + λ_KLD·L_KLD
```

- **L_disc**: Wasserstein discriminator loss
- **L_feat**: Feature matching loss (L1 between discriminator features)
- **L_rec**: Reconstruction loss (L1 between real and generated gestures)
- **L_lat**: Latent encoding loss (L1 between original and recovered z)
- **L_KLD**: KL divergence to keep encoder close to N(0,1)

## Two-Cycle Training

Following BicycleGAN, training uses two cycles:

1. **Cycle 1 (z → X' → z')**: Sample random z, generate gesture X', recover z' from X', minimize |z - z'|
2. **Cycle 2 (X → z → X')**: Encode real gesture X to z, generate X', minimize |X - X'|

Each cycle has its own discriminator.

## Generating Gestures

```python
from src.models import Generator
from src.keyboard import QWERTYKeyboard
from src.config import ModelConfig
import torch

# Initialize
config = ModelConfig()
generator = Generator(config)
keyboard = QWERTYKeyboard()

# Load trained weights
checkpoint = torch.load('checkpoints/checkpoint_final.pt')
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

# Generate gesture for a word
word = "hello"
prototype = keyboard.get_word_prototype(word, num_points=128)
prototype_tensor = torch.FloatTensor(prototype).unsqueeze(0)

# Sample latent code and generate
z = torch.randn(1, config.latent_dim)
with torch.no_grad():
    gesture = generator(prototype_tensor, z)

# gesture shape: (1, 128, 3) - (batch, seq_length, [x, y, t])
```

## Results (from paper)

| Metric | WordGesture-GAN | Minimum Jerk | Style-Transfer GAN |
|--------|-----------------|--------------|-------------------|
| L2 Wasserstein (x,y) | **4.409** | 5.004 | 10.48 |
| DTW Wasserstein (x,y) | **2.146** | 2.752 | 8.11 |
| FID Score | **0.270** | 0.354 | 2.733 |
| Precision | **0.973** | 0.785 | 0.229 |
| Recall | 0.258 | **0.575** | 0.569 |

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{chu2023wordgesture,
  title={WordGesture-GAN: Modeling Word-Gesture Movement with Generative Adversarial Network},
  author={Chu, Jeremy and An, Dongsheng and Ma, Yan and Cui, Wenzhe and Zhai, Shumin and Gu, Xianfeng and Bi, Xiaojun},
  booktitle={Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems},
  year={2023},
  publisher={ACM}
}
```

## License

This implementation is for research purposes. Please refer to the original paper for usage terms.
