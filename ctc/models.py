"""BLSTM model for CTC-based gesture decoding."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class BLSTMCTCModel(nn.Module):
    """Bidirectional LSTM for CTC-based gesture-to-character decoding.

    Architecture:
        Input (batch, seq_len, input_size)
            ↓
        Bidirectional LSTM (num_layers)
            ↓
        Linear projection to character logits
            ↓
        Output (batch, seq_len, num_classes)

    The output can be decoded using CTC greedy or beam search decoding.
    """

    BLANK_IDX = 0  # CTC blank token index

    def __init__(
        self,
        input_size: int = 31,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 27,  # 26 letters + blank
        dropout: float = 0.2,
    ):
        """Initialize the BLSTM CTC model.

        Args:
            input_size: Number of input features per timestep.
            hidden_size: LSTM hidden state size (per direction).
            num_layers: Number of LSTM layers.
            num_classes: Number of output classes (26 letters + blank).
            dropout: Dropout probability between LSTM layers.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project bidirectional output (hidden_size * 2) to character logits
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size) input features.

        Returns:
            logits: (batch, seq_len, num_classes) character logits.
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)

        # Project to character space
        logits = self.fc(lstm_out)  # (batch, seq_len, num_classes)

        return logits

    def decode_greedy(self, logits: torch.Tensor) -> List[str]:
        """Greedy CTC decoding.

        Removes blank tokens and collapses repeated characters.

        Args:
            logits: (batch, seq_len, num_classes) character logits.

        Returns:
            List of decoded strings, one per batch element.
        """
        # Get most likely character at each position
        predictions = logits.argmax(dim=-1)  # (batch, seq_len)

        decoded = []
        for pred in predictions:
            # Remove blanks and collapse repeats
            chars = []
            prev = -1
            for idx in pred.tolist():
                if idx != self.BLANK_IDX and idx != prev:
                    # Convert index to character (1=a, 2=b, ..., 26=z)
                    chars.append(chr(ord("a") + idx - 1))
                prev = idx
            decoded.append("".join(chars))

        return decoded

    def get_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for CTC loss.

        Args:
            x: (batch, seq_len, input_size) input features.

        Returns:
            log_probs: (seq_len, batch, num_classes) log probabilities
                in the format expected by nn.CTCLoss.
        """
        logits = self.forward(x)  # (batch, seq_len, num_classes)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # CTC loss expects (T, N, C) format
        return log_probs.permute(1, 0, 2)
