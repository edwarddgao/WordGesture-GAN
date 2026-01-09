# WordGesture-GAN: Modeling Word-Gesture Movement with Generative Adversarial Network
# Implementation based on CHI'23 paper by Chu et al.

__version__ = "1.0.0"

# Configuration
from .config import (
    ModelConfig, TrainingConfig, EvaluationConfig, ModalConfig, KeyboardConfig,
    DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG, DEFAULT_EVALUATION_CONFIG,
    DEFAULT_MODAL_CONFIG, DEFAULT_KEYBOARD_CONFIG
)

# Models
from .models import Generator, Discriminator, VariationalEncoder, AutoEncoder

# Utils
from .utils import seed_everything, log, train_epoch_with_grad_clip

# SHARK2 decoder
from .shark2 import (
    SHARK2Decoder, SHARK2Config, DEFAULT_SHARK2_CONFIG,
    evaluate_decoder, load_word_frequencies
)

# Evaluation
from .evaluation import evaluate_all_metrics, compute_duration_rmse

# Trainer
from .trainer import WordGestureGANTrainer

# Keyboard
from .keyboard import QWERTYKeyboard
