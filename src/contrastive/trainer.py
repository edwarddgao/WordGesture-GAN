"""
Trainer for contrastive gesture encoder.

Trains embeddings using supervised contrastive loss where:
- Same word gestures -> close embeddings
- Different word gestures -> far embeddings
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

from .model import (
    ContrastiveEncoder,
    SupervisedContrastiveLoss,
    ContrastiveConfig,
    DEFAULT_CONTRASTIVE_CONFIG
)


class ContrastiveTrainer:
    """
    Trainer for contrastive gesture encoder.

    Training loop:
    1. Sample batch of N words × K gestures
    2. Encode all gestures to embeddings
    3. Compute supervised contrastive loss
    4. Update encoder weights
    """

    def __init__(
        self,
        config: ContrastiveConfig = DEFAULT_CONTRASTIVE_CONFIG,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.config = config
        self.device = torch.device(device)

        # Initialize model
        self.encoder = ContrastiveEncoder(config).to(self.device)

        # Loss function
        self.criterion = SupervisedContrastiveLoss(temperature=config.temperature)

        # Optimizer
        self.optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999)
        )

        # LR scheduler
        self.scheduler = None  # Set up during fit()

        # Training state
        self.current_epoch = 0
        self.best_recall = 0.0

    def train_step(
        self,
        gestures: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            gestures: Batch of gestures (batch, seq_len, 3)
            labels: Integer word labels (batch,)

        Returns:
            Dictionary of loss values
        """
        self.encoder.train()

        gestures = gestures.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        embeddings = self.encoder(gestures)

        # Compute contrastive loss
        loss = self.criterion(embeddings, labels)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        k_values: Tuple[int, ...] = (1, 5, 10)
    ) -> Dict[str, float]:
        """
        Evaluate encoder using centroid-based recall@k.

        For each query gesture, find the nearest word centroid and check
        if it matches the query's word. This is a more meaningful metric
        for gesture-to-word recognition.

        Args:
            test_loader: Test data loader
            k_values: Values of k for recall@k

        Returns:
            Dictionary with recall@k for each k
        """
        self.encoder.eval()

        all_embeddings = []
        all_words = []

        # Embed all test gestures
        for batch in test_loader:
            gestures, labels, words = batch[0], batch[1], batch[2]
            gestures = gestures.to(self.device)
            embeddings = self.encoder(gestures)
            all_embeddings.append(embeddings.cpu())
            all_words.extend(words)

        embeddings = torch.cat(all_embeddings, dim=0)

        # Compute word centroids (mean embedding per word)
        unique_words = list(set(all_words))
        word_to_idx = {w: i for i, w in enumerate(unique_words)}
        n_words = len(unique_words)

        centroids = torch.zeros(n_words, embeddings.size(1))
        counts = torch.zeros(n_words)

        for i, word in enumerate(all_words):
            idx = word_to_idx[word]
            centroids[idx] += embeddings[i]
            counts[idx] += 1

        centroids = centroids / counts.unsqueeze(1)
        centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)

        # Query: each gesture → nearest centroid
        similarity = embeddings @ centroids.T  # (n_gestures, n_words)

        max_k = min(max(k_values), n_words)
        _, topk_indices = similarity.topk(max_k, dim=1)

        # Compute recall@k: is the correct word in top-k centroids?
        results = {}
        for k in k_values:
            k_actual = min(k, max_k)
            correct = 0
            for i, word in enumerate(all_words):
                word_idx = word_to_idx[word]
                if word_idx in topk_indices[i, :k_actual]:
                    correct += 1
            results[f'recall@{k}'] = correct / len(all_words)

        # Compute top-1 accuracy (same as recall@1 for centroid-based)
        results['accuracy'] = results['recall@1']

        return results

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: Optional[int] = None,
        log_every: int = 50,
        eval_every: int = 5,
        save_callback=None
    ) -> Dict[str, list]:
        """
        Train the encoder.

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Number of epochs (default from config)
            log_every: Log every N batches
            eval_every: Evaluate every N epochs
            save_callback: Optional callback(trainer, epoch, metrics) for saving

        Returns:
            Dictionary of training history
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        # Setup LR scheduler
        total_steps = num_epochs * len(train_loader)
        if self.config.use_cosine_annealing:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.eta_min
            )

        history = defaultdict(list)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                gestures, labels, words = batch[0], batch[1], batch[2]
                metrics = self.train_step(gestures, labels)
                epoch_losses.append(metrics['loss'])

                if self.scheduler is not None:
                    self.scheduler.step()

                if (batch_idx + 1) % log_every == 0:
                    avg_loss = np.mean(epoch_losses[-log_every:])
                    lr = metrics['learning_rate']
                    print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: {avg_loss:.4f} | LR: {lr:.6f}")

            # Record epoch metrics
            avg_epoch_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_epoch_loss)
            print(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")

            # Evaluate
            if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
                eval_metrics = self.evaluate(test_loader)
                for key, value in eval_metrics.items():
                    history[f'test_{key}'].append(value)
                print(f"Evaluation: " + " | ".join(f"{k}: {v:.4f}" for k, v in eval_metrics.items()))

                # Track best model
                if eval_metrics['recall@1'] > self.best_recall:
                    self.best_recall = eval_metrics['recall@1']
                    if save_callback is not None:
                        save_callback(self, epoch, eval_metrics)
                        print(f"New best recall@1: {self.best_recall:.4f}")

        return dict(history)

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'epoch': self.current_epoch,
            'best_recall': self.best_recall
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_recall = checkpoint.get('best_recall', 0.0)

    def get_encoder(self) -> ContrastiveEncoder:
        """Get the trained encoder."""
        return self.encoder
