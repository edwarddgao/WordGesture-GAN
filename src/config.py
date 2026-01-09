"""
Configuration for WordGesture-GAN training and evaluation.
Based on the paper: "WordGesture-GAN: Modeling Word-Gesture Movement with
Generative Adversarial Network" (CHI '23)
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Sequence parameters
    seq_length: int = 128  # Number of points in gesture sequence
    input_dim: int = 3  # (x, y, t) coordinates

    # Latent space
    latent_dim: int = 32  # Dimension of Gaussian latent code

    # Generator (BiLSTM)
    gen_hidden_dim: int = 48  # Hidden dimension for BiLSTM layers (increased for precision)
    gen_num_layers: int = 4  # Number of BiLSTM layers

    # Discriminator (MLP)
    disc_hidden_dims: Tuple[int, ...] = (192, 96, 48, 24)

    # Encoder (MLP)
    enc_hidden_dims: Tuple[int, ...] = (192, 96, 48, 32)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic training params
    batch_size: int = 512
    learning_rate: float = 0.0002
    num_epochs: int = 200

    # WGAN training: update discriminator n_critic times per generator update
    n_critic: int = 5

    # Loss weights (from paper Section 4.2)
    lambda_feat: float = 1.0  # Feature matching loss weight
    lambda_rec: float = 4.0  # Reconstruction loss weight (tuned between 3.0-5.0)
    lambda_lat: float = 0.5  # Latent encoding loss weight
    lambda_kld: float = 0.02  # KL divergence loss weight (tuned between 0.01-0.05)

    # Dataset
    max_samples_per_word: int = 5  # Cap samples per word to balance dataset
    train_ratio: float = 0.8  # Train/test split ratio

    # Checkpointing
    save_every: int = 10  # Save checkpoint every N epochs
    log_every: int = 100  # Log every N batches


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    # Sample size
    n_samples: int = 2000  # Number of samples for evaluation (increased for stable FID)

    # Generation
    truncation: float = 1.0  # No truncation for evaluation (full diversity)

    # FID score (paper Section 4.3)
    fid_autoencoder_epochs: int = 100  # Epochs to train autoencoder for FID
    fid_hidden_dim: int = 32  # Paper: "32 dimensional space"

    # Precision/Recall
    precision_recall_k: int = 3  # k for k-NN manifold estimation (paper uses k=3)

    # Signal processing
    savgol_window: int = 21  # Savitzky-Golay filter window (tuned to match paper metrics)
    savgol_poly_order: int = 3  # Savitzky-Golay polynomial order


@dataclass
class ModalConfig:
    """Configuration for Modal remote execution."""
    checkpoint_dir: str = '/data/checkpoints'
    data_path: str = '/data/swipelogs.zip'
    wandb_project: str = 'wordgesture-gan'
    random_seed: int = 42


@dataclass
class KeyboardConfig:
    """Virtual keyboard layout configuration."""
    # Standard QWERTY layout dimensions (normalized to [0, 1])
    width: float = 1.0
    height: float = 1.0

    # Number of rows and their key layouts
    rows: Tuple[str, ...] = ('qwertyuiop', 'asdfghjkl', 'zxcvbnm')
    row_offsets: Tuple[float, ...] = (0.0, 0.05, 0.15)  # Horizontal offset for each row

    # Key dimensions (relative to keyboard width)
    key_width: float = 0.1
    key_height: float = 0.333


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_EVALUATION_CONFIG = EvaluationConfig()
DEFAULT_MODAL_CONFIG = ModalConfig()
DEFAULT_KEYBOARD_CONFIG = KeyboardConfig()
