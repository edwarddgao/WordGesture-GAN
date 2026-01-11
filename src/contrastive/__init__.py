"""
Contrastive learning for gesture-to-word recognition.
"""

from .model import (
    ContrastiveEncoder,
    ContrastiveConfig,
    SupervisedContrastiveLoss,
    DEFAULT_CONTRASTIVE_CONFIG
)
from .trainer import ContrastiveTrainer
from .dataset import (
    ContrastiveGestureDataset,
    ContrastiveBatchSampler,
    create_contrastive_datasets,
    create_contrastive_data_loader,
    augment_with_minimum_jerk
)

__all__ = [
    # Model
    'ContrastiveEncoder',
    'ContrastiveConfig',
    'SupervisedContrastiveLoss',
    'DEFAULT_CONTRASTIVE_CONFIG',
    # Training
    'ContrastiveTrainer',
    # Dataset
    'ContrastiveGestureDataset',
    'ContrastiveBatchSampler',
    'create_contrastive_datasets',
    'create_contrastive_data_loader',
    'augment_with_minimum_jerk',
]
