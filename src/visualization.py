"""
Visualization utilities for WordGesture-GAN.

Provides functions for plotting:
- Gestures on keyboard layout
- Training curves
- Comparison between real and generated gestures
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Optional, Tuple, Dict
import os

from .keyboard import QWERTYKeyboard
from .config import DEFAULT_KEYBOARD_CONFIG


def draw_keyboard(
    ax: plt.Axes,
    keyboard: QWERTYKeyboard,
    alpha: float = 0.3
) -> None:
    """
    Draw keyboard layout on matplotlib axes.

    Args:
        ax: Matplotlib axes
        keyboard: QWERTYKeyboard instance
        alpha: Transparency for key boxes
    """
    for row_idx, row in enumerate(keyboard.config.rows):
        for key_idx, key in enumerate(row):
            x, y = keyboard.key_centers[key]

            # Convert from [-1, 1] to plot coordinates
            # Key width and height in normalized coordinates
            key_w = keyboard.config.key_width * 2 * 0.9  # Slightly smaller for gap
            key_h = 2.0 / len(keyboard.config.rows) * 0.9

            # Draw key rectangle
            rect = Rectangle(
                (x - key_w / 2, y - key_h / 2),
                key_w, key_h,
                linewidth=1,
                edgecolor='gray',
                facecolor='lightgray',
                alpha=alpha
            )
            ax.add_patch(rect)

            # Draw key label
            ax.text(x, y, key.upper(), ha='center', va='center',
                   fontsize=8, fontweight='bold', color='black')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # y increases downward


def plot_gesture(
    gesture: np.ndarray,
    ax: Optional[plt.Axes] = None,
    keyboard: Optional[QWERTYKeyboard] = None,
    color: str = 'blue',
    label: Optional[str] = None,
    show_points: bool = True,
    alpha: float = 0.8,
    linewidth: float = 2
) -> plt.Axes:
    """
    Plot a single gesture.

    Args:
        gesture: Gesture array of shape (seq_length, 3)
        ax: Matplotlib axes (creates new if None)
        keyboard: Optional keyboard to draw in background
        color: Line color
        label: Optional legend label
        show_points: Whether to show touch points
        alpha: Line transparency
        linewidth: Line width

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if keyboard is not None:
        draw_keyboard(ax, keyboard)

    x = gesture[:, 0]
    y = gesture[:, 1]

    # Plot gesture line
    ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth, label=label)

    # Plot touch points (evenly spaced in time)
    if show_points:
        ax.scatter(x, y, c=color, s=20, alpha=alpha * 0.7, zorder=5)

    # Mark start and end points
    ax.scatter([x[0]], [y[0]], c='green', s=100, marker='o', zorder=10, label='Start')
    ax.scatter([x[-1]], [y[-1]], c='red', s=100, marker='s', zorder=10, label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    return ax


def plot_gesture_comparison(
    real_gesture: np.ndarray,
    fake_gesture: np.ndarray,
    prototype: Optional[np.ndarray] = None,
    word: Optional[str] = None,
    keyboard: Optional[QWERTYKeyboard] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison between real and generated gesture.

    Args:
        real_gesture: Real gesture array
        fake_gesture: Generated gesture array
        prototype: Optional word prototype
        word: Optional word label
        keyboard: Optional keyboard for background
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if keyboard is None:
        keyboard = QWERTYKeyboard()

    # Plot real gesture
    plot_gesture(real_gesture, axes[0], keyboard, color='orange', label='Real')
    axes[0].set_title(f'Real Gesture{" - " + word if word else ""}')
    axes[0].legend()

    # Plot generated gesture
    plot_gesture(fake_gesture, axes[1], keyboard, color='blue', label='Generated')
    axes[1].set_title(f'Generated Gesture{" - " + word if word else ""}')
    axes[1].legend()

    # Plot overlay
    plot_gesture(real_gesture, axes[2], keyboard, color='orange', label='Real', alpha=0.6)
    plot_gesture(fake_gesture, axes[2], None, color='blue', label='Generated', alpha=0.6)
    if prototype is not None:
        plot_gesture(prototype, axes[2], None, color='green', label='Prototype',
                    alpha=0.4, show_points=False, linewidth=1)
    axes[2].set_title('Overlay')
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_multiple_gestures(
    gestures: List[np.ndarray],
    keyboard: Optional[QWERTYKeyboard] = None,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple gestures on the same axes.

    Args:
        gestures: List of gesture arrays
        keyboard: Optional keyboard for background
        colors: Optional list of colors
        labels: Optional list of labels
        title: Optional plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if keyboard is None:
        keyboard = QWERTYKeyboard()

    draw_keyboard(ax, keyboard)

    if colors is None:
        cmap = plt.cm.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(gestures))]

    for i, gesture in enumerate(gestures):
        color = colors[i] if i < len(colors) else colors[i % len(colors)]
        label = labels[i] if labels and i < len(labels) else None
        plot_gesture(gesture, ax, None, color=color, label=label, alpha=0.7)

    if title:
        ax.set_title(title)

    if labels:
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_curves(
    training_history: List[Dict],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training loss curves.

    Args:
        training_history: List of epoch loss dictionaries
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    epochs = [h['epoch'] for h in training_history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Discriminator losses
    if 'd1_loss' in training_history[0]:
        d1_losses = [h['d1_loss'] for h in training_history]
        d2_losses = [h['d2_loss'] for h in training_history]
        axes[0, 0].plot(epochs, d1_losses, label='D1 Loss')
        axes[0, 0].plot(epochs, d2_losses, label='D2 Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Discriminator Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

    # Generator cycle losses
    if 'cycle1_total' in training_history[0]:
        c1_losses = [h['cycle1_total'] for h in training_history]
        c2_losses = [h['cycle2_total'] for h in training_history]
        axes[0, 1].plot(epochs, c1_losses, label='Cycle 1 Total')
        axes[0, 1].plot(epochs, c2_losses, label='Cycle 2 Total')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Generator Cycle Losses')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    # Combined losses over time
    all_losses = []
    for h in training_history:
        total = sum(v for k, v in h.items() if k != 'epoch' and isinstance(v, (int, float)))
        all_losses.append(total)

    axes[1, 0].plot(epochs, all_losses, label='Total Loss', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Total Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Hide unused subplot
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_velocity_acceleration(
    gesture: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot velocity and acceleration profiles for a gesture.

    Args:
        gesture: Gesture array of shape (seq_length, 3)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    from .evaluation import compute_velocity_acceleration

    velocity, acceleration = compute_velocity_acceleration(gesture)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Velocity
    axes[0].plot(velocity, color='blue')
    axes[0].set_xlabel('Point Index')
    axes[0].set_ylabel('Velocity')
    axes[0].set_title('Velocity Profile')
    axes[0].grid(True)

    # Acceleration
    axes[1].plot(acceleration, color='red')
    axes[1].set_xlabel('Point Index')
    axes[1].set_ylabel('Acceleration')
    axes[1].set_title('Acceleration Profile')
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_gesture_grid(
    gestures_dict: Dict[str, List[np.ndarray]],
    n_words: int = 4,
    n_samples: int = 3,
    keyboard: Optional[QWERTYKeyboard] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a grid of gesture visualizations.

    Args:
        gestures_dict: Dict mapping word -> list of gestures
        n_words: Number of words to show
        n_samples: Number of samples per word
        keyboard: Optional keyboard for background
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if keyboard is None:
        keyboard = QWERTYKeyboard()

    words = list(gestures_dict.keys())[:n_words]

    fig, axes = plt.subplots(n_words, n_samples, figsize=(4 * n_samples, 4 * n_words))

    if n_words == 1:
        axes = [axes]
    if n_samples == 1:
        axes = [[ax] for ax in axes]

    cmap = plt.cm.get_cmap('viridis')

    for i, word in enumerate(words):
        gestures = gestures_dict[word][:n_samples]
        for j, gesture in enumerate(gestures):
            ax = axes[i][j]
            draw_keyboard(ax, keyboard, alpha=0.2)
            plot_gesture(gesture, ax, None, color=cmap(j / n_samples), show_points=True)
            if j == 0:
                ax.set_ylabel(f'"{word}"', fontsize=12, fontweight='bold')
            ax.set_title(f'Sample {j + 1}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
