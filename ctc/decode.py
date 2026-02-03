"""CTC decoding utilities for inference."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from features import GestureFeatureExtractor
from .models import BLSTMCTCModel


class CTCDecoder:
    """CTC decoder for gesture-to-word inference.

    Wraps a trained BLSTMCTCModel for easy inference on raw gestures.
    """

    def __init__(
        self,
        model: BLSTMCTCModel,
        feature_extractor: GestureFeatureExtractor,
        device: torch.device | None = None,
    ):
        """Initialize the CTC decoder.

        Args:
            model: Trained BLSTM CTC model.
            feature_extractor: Feature extractor matching training config.
            device: Device to run inference on. Defaults to model's device.
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device or next(model.parameters()).device
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device | None = None,
    ) -> "CTCDecoder":
        """Load a CTCDecoder from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint .pt file.
            device: Device to load model on.

        Returns:
            Initialized CTCDecoder.
        """
        from shared import KeyboardLayout

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = checkpoint["config"]

        # Create feature extractor
        feature_extractor = GestureFeatureExtractor(
            layout=KeyboardLayout(),
            sigma=cfg["data"]["key_proximity_sigma"],
            use_key_proximity=cfg["features"]["use_key_proximity"],
            use_velocity=cfg["features"]["use_velocity"],
        )

        # Create model
        model = BLSTMCTCModel(
            input_size=feature_extractor.n_features,
            hidden_size=cfg["model"]["hidden_size"],
            num_layers=cfg["model"]["num_layers"],
            num_classes=cfg["model"]["num_classes"],
            dropout=cfg["model"]["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        return cls(model, feature_extractor, device)

    @torch.no_grad()
    def decode(self, gesture: np.ndarray) -> str:
        """Decode a single gesture to a word.

        Args:
            gesture: (seq_len, 3) raw gesture with [x, y, dt].

        Returns:
            Decoded word string.
        """
        features = self.feature_extractor(gesture)
        features_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
        logits = self.model(features_t)
        return self.model.decode_greedy(logits)[0]

    @torch.no_grad()
    def decode_batch(self, gestures: List[np.ndarray]) -> List[str]:
        """Decode a batch of gestures.

        Args:
            gestures: List of (seq_len, 3) raw gesture arrays.

        Returns:
            List of decoded word strings.
        """
        if not gestures:
            return []

        features_list = [self.feature_extractor(g) for g in gestures]
        features = np.stack(features_list, axis=0)
        features_t = torch.from_numpy(features).to(self.device)
        logits = self.model(features_t)
        return self.model.decode_greedy(logits)

    @torch.no_grad()
    def decode_with_confidence(
        self, gesture: np.ndarray
    ) -> tuple[str, float]:
        """Decode a gesture and return confidence score.

        Args:
            gesture: (seq_len, 3) raw gesture with [x, y, dt].

        Returns:
            Tuple of (decoded_word, confidence_score).
            Confidence is mean of max log-probabilities.
        """
        features = self.feature_extractor(gesture)
        features_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
        logits = self.model(features_t)

        # Compute confidence as mean of max log-probs
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        max_log_probs = log_probs.max(dim=-1).values
        confidence = max_log_probs.mean().item()

        word = self.model.decode_greedy(logits)[0]
        return word, confidence
