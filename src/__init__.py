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

# Evaluation
from .evaluation import evaluate_all_metrics

# Trainer
from .trainer import WordGestureGANTrainer

# Keyboard
from .keyboard import QWERTYKeyboard

# Visualization
from .visualization import (
    plot_gestures_on_keyboard,
    create_comparison_figure,
    create_overlay_figure,
)

# Contrastive Learning
from .contrastive_model import (
    ContrastiveConfig, ContrastiveEncoder, SupervisedContrastiveLoss,
    DEFAULT_CONTRASTIVE_CONFIG, word_labels_to_tensor
)
from .contrastive_dataset import (
    ContrastiveGestureDataset, ContrastiveBatchSampler,
    create_contrastive_datasets, create_contrastive_data_loader
)
from .contrastive_trainer import ContrastiveTrainer
