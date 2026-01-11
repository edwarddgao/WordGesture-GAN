"""
WordGesture-GAN: Generative adversarial network for gesture synthesis.
"""

from .models import (
    Generator,
    Discriminator,
    VariationalEncoder,
    TemporalDiscriminator,
    AutoEncoder
)
from .trainer import WordGestureGANTrainer
from .losses import (
    WassersteinLoss,
    ReconstructionLoss,
    LatentEncodingLoss,
    KLDivergenceLoss,
    FeatureMatchingLoss
)
from .evaluation import evaluate_all_metrics
from .visualization import (
    plot_gestures_on_keyboard,
    create_comparison_figure,
    create_overlay_figure
)

__all__ = [
    # Models
    'Generator',
    'Discriminator',
    'VariationalEncoder',
    'TemporalDiscriminator',
    'AutoEncoder',
    # Training
    'WordGestureGANTrainer',
    # Losses
    'WassersteinLoss',
    'ReconstructionLoss',
    'LatentEncodingLoss',
    'KLDivergenceLoss',
    'FeatureMatchingLoss',
    # Evaluation
    'evaluate_all_metrics',
    # Visualization
    'plot_gestures_on_keyboard',
    'create_comparison_figure',
    'create_overlay_figure',
]
