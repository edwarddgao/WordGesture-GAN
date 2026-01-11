"""
Contrastive Gesture Encoder for similarity-based gesture retrieval.

Trains embeddings where:
- Gestures for the same word -> close embeddings
- Gestures for different words -> far embeddings

Architecture: BiLSTM encoder with mean pooling and L2 normalization.
Loss: Supervised Contrastive Loss (InfoNCE-style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive gesture encoder."""
    # Architecture
    embedding_dim: int = 64  # Output embedding dimension
    lstm_hidden_dim: int = 64  # BiLSTM hidden size (output = 2x due to bidirectional)
    num_lstm_layers: int = 2  # Number of BiLSTM layers

    # Contrastive Loss
    temperature: float = 0.07  # Temperature for similarity scaling

    # Training
    learning_rate: float = 1e-3
    batch_words: int = 32  # Number of unique words per batch
    gestures_per_word: int = 2  # Gestures sampled per word
    num_epochs: int = 100

    # LR Schedule
    use_cosine_annealing: bool = True
    eta_min: float = 1e-5

    # Sequence parameters (inherited from ModelConfig)
    seq_length: int = 128
    input_dim: int = 3  # (x, y, t)


DEFAULT_CONTRASTIVE_CONFIG = ContrastiveConfig()


class ContrastiveEncoder(nn.Module):
    """
    1D CNN encoder for contrastive gesture embedding.

    Faster than BiLSTM due to parallelization. Architecture:
        Input: (batch, 128, 3) → transpose → (batch, 3, 128)
            ↓
        Conv1d(3→32, k=7, s=2) + BN + ReLU → (batch, 32, 64)
            ↓
        Conv1d(32→64, k=5, s=2) + BN + ReLU → (batch, 64, 32)
            ↓
        Conv1d(64→128, k=3, s=2) + BN + ReLU → (batch, 128, 16)
            ↓
        AdaptiveAvgPool1d(1) → (batch, 128)
            ↓
        Linear(128→64) + ReLU + Linear(64→64) → (batch, 64)
            ↓
        L2 Normalize → unit sphere
    """

    def __init__(self, config: ContrastiveConfig = DEFAULT_CONTRASTIVE_CONFIG):
        super().__init__()
        self.config = config

        self.conv_layers = nn.Sequential(
            # (batch, 3, 128) → (batch, 32, 64)
            nn.Conv1d(config.input_dim, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # (batch, 32, 64) → (batch, 64, 32)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # (batch, 64, 32) → (batch, 128, 16)
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.projection = nn.Sequential(
            nn.Linear(128, config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Encode gesture to embedding space.

        Args:
            x: Gesture tensor of shape (batch, seq_length, 3)
            normalize: Whether to L2 normalize output (default: True)

        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        # (batch, seq_len, 3) → (batch, 3, seq_len) for Conv1d
        x = x.transpose(1, 2)

        x = self.conv_layers(x)  # (batch, 128, 16)
        x = self.pool(x).squeeze(-1)  # (batch, 128)
        x = self.projection(x)  # (batch, embedding_dim)

        if normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.config.embedding_dim


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).

    For each anchor, treats samples with the same label as positives
    and all other samples as negatives.

    Loss = -sum over positives[ log(exp(sim(z_i, z_p)/tau) / sum_all(exp(sim(z_i, z_a)/tau))) ]

    Reference: "Supervised Contrastive Learning" (Khosla et al., 2020)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            embeddings: L2-normalized embeddings of shape (batch, embedding_dim)
            labels: Integer labels of shape (batch,)

        Returns:
            Scalar loss value
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Compute pairwise similarity: (batch, batch)
        similarity = embeddings @ embeddings.T / self.temperature

        # Create positive mask: 1 if same label, 0 otherwise
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Remove self-comparisons from mask
        mask = mask - torch.eye(batch_size, device=device)

        # For numerical stability, subtract max from similarity
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Compute log_prob: log(exp(sim_ij) / sum_k(exp(sim_ik)))
        # Exclude self from denominator
        exp_logits = torch.exp(logits)
        # Mask out self in denominator
        self_mask = 1.0 - torch.eye(batch_size, device=device)
        log_prob = logits - torch.log((exp_logits * self_mask).sum(dim=1, keepdim=True) + 1e-8)

        # Average over positive pairs only
        # For each row, compute mean log_prob over positive columns
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1.0)  # Avoid division by zero

        mean_log_prob = (mask * log_prob).sum(dim=1) / mask_sum

        # Loss is negative mean log probability
        loss = -mean_log_prob.mean()

        return loss


def word_labels_to_tensor(word_labels: List[str], device: torch.device) -> torch.Tensor:
    """
    Convert list of word strings to integer label tensor.

    Args:
        word_labels: List of word strings
        device: Target device for tensor

    Returns:
        Integer tensor of shape (len(word_labels),)
    """
    # Create mapping from unique words to integers
    unique_words = list(set(word_labels))
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}

    # Convert to integer labels
    labels = torch.tensor([word_to_idx[w] for w in word_labels], device=device)
    return labels
