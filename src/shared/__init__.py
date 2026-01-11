"""
Shared utilities for WordGesture-GAN and contrastive learning.
"""

from .config import (
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    ModalConfig,
    KeyboardConfig,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_MODAL_CONFIG,
    DEFAULT_KEYBOARD_CONFIG,
)
from .keyboard import (
    QWERTYKeyboard,
    MinimumJerkModel,
    MinimumJerkDistributions,
)
from .data import (
    load_dataset_from_zip,
    GestureDataset,
    infer_key_positions,
)
from .utils import (
    seed_everything,
    log,
    train_epoch_with_grad_clip,
)

__all__ = [
    # Config
    'ModelConfig',
    'TrainingConfig',
    'EvaluationConfig',
    'ModalConfig',
    'KeyboardConfig',
    'DEFAULT_MODEL_CONFIG',
    'DEFAULT_TRAINING_CONFIG',
    'DEFAULT_EVALUATION_CONFIG',
    'DEFAULT_MODAL_CONFIG',
    'DEFAULT_KEYBOARD_CONFIG',
    # Keyboard
    'QWERTYKeyboard',
    'MinimumJerkModel',
    'MinimumJerkDistributions',
    # Data
    'load_dataset_from_zip',
    'GestureDataset',
    'infer_key_positions',
    # Utils
    'seed_everything',
    'log',
    'train_epoch_with_grad_clip',
]
