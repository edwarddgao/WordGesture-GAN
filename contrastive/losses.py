"""Contrastive learning losses."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE loss for contrastive learning."""

    def forward(
        self,
        gesture_emb: torch.Tensor,
        prototype_emb: torch.Tensor,
        temperature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute symmetric InfoNCE loss.

        Args:
            gesture_emb: Gesture embeddings, L2 normalized (B, D)
            prototype_emb: Prototype embeddings, L2 normalized (B, D)
            temperature: Temperature for softmax scaling

        Returns:
            loss: Scalar loss value
            metrics: Dict with loss components and accuracy
        """
        batch_size = gesture_emb.size(0)
        device = gesture_emb.device

        # Compute similarity matrix
        logits = (gesture_emb @ prototype_emb.T) / temperature  # (B, B)

        # Labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=device)

        # Symmetric loss
        loss_g2p = F.cross_entropy(logits, labels)
        loss_p2g = F.cross_entropy(logits.T, labels)
        loss = (loss_g2p + loss_p2g) / 2

        # Compute accuracy for monitoring
        with torch.no_grad():
            pred_g2p = logits.argmax(dim=1)
            pred_p2g = logits.argmax(dim=0)
            acc_g2p = (pred_g2p == labels).float().mean()
            acc_p2g = (pred_p2g == labels).float().mean()

        metrics = {
            "loss_g2p": loss_g2p.item(),
            "loss_p2g": loss_p2g.item(),
            "acc_g2p": acc_g2p.item(),
            "acc_p2g": acc_p2g.item(),
            "temperature": temperature.item(),
        }

        return loss, metrics
