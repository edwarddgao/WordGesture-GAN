"""
Gesture visualization for W&B logging.
Creates paper-style figures (Figures 1, 5-8) with keyboard layout.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Optional, Tuple

from .config import KeyboardConfig, DEFAULT_KEYBOARD_CONFIG


# Paper colors
COLOR_REAL = '#E67E22'  # Orange for user-drawn
COLOR_FAKE = '#3498DB'  # Blue for generated
COLOR_PROTO = '#2ECC71'  # Green for prototype


def draw_keyboard(ax: plt.Axes, config: KeyboardConfig = DEFAULT_KEYBOARD_CONFIG):
    """
    Draw QWERTY keyboard grid centered at origin.

    Gestures are normalized to [-1, 1] with keyboard centered at 0.
    y is inverted: y=-1 is top of keyboard, y=+1 is bottom.
    """
    rows = config.rows
    row_offsets = [0.0, 0.05, 0.15]  # Indent for each row

    n_rows = len(rows)
    key_h = 1.8 / n_rows  # Total height ~1.8, leaving margin

    for row_idx, (row, offset) in enumerate(zip(rows, row_offsets)):
        num_keys = len(row)
        # y: row 0 at top (y ~ +0.7), row 2 at bottom (y ~ -0.7)
        y_center = 0.7 - row_idx * (1.4 / (n_rows - 1)) if n_rows > 1 else 0

        # Calculate key width to fit row centered in [-0.9, 0.9]
        row_span = 1.8 - offset  # Available width for this row
        key_w = row_span / num_keys * 0.95

        for key_idx, key in enumerate(row):
            # Center the row, accounting for offset
            row_start = -0.9 + offset / 2
            x_center = row_start + (key_idx + 0.5) * (row_span / num_keys)

            rect = Rectangle(
                (x_center - key_w / 2, y_center - key_h / 2),
                key_w, key_h,
                fill=False, edgecolor='#BDC3C7', linewidth=0.5
            )
            ax.add_patch(rect)
            ax.text(x_center, y_center, key.upper(),
                    ha='center', va='center', fontsize=6, color='#7F8C8D')


def plot_gesture(
    ax: plt.Axes,
    gesture: np.ndarray,
    color: str = COLOR_FAKE,
    alpha: float = 0.8,
    dot_size: int = 15,
    line_width: float = 1.0,
    show_dots: bool = True
):
    """
    Plot a single gesture with time-spaced dots.

    Args:
        ax: Matplotlib axes
        gesture: Array of shape (N, 3) with (x, y, t) or (N, 2) with (x, y)
        color: Gesture color
        alpha: Transparency
        dot_size: Size of time-spaced dots
        line_width: Width of connecting line
        show_dots: Whether to show dots evenly spaced in time
    """
    x, y = gesture[:, 0], -gesture[:, 1]  # Flip y to match keyboard orientation

    # Draw line
    ax.plot(x, y, color=color, alpha=alpha * 0.7, linewidth=line_width, zorder=2)

    # Draw dots evenly spaced in time (if time dimension exists)
    if show_dots:
        if gesture.shape[1] >= 3:
            # Use time dimension - resample to get evenly spaced dots in time
            t = gesture[:, 2]
            n_dots = 32  # Number of dots to show
            t_uniform = np.linspace(t.min(), t.max(), n_dots)
            x_dots = np.interp(t_uniform, t, x)
            y_dots = np.interp(t_uniform, t, y)
        else:
            # No time - sample uniformly along trajectory
            indices = np.linspace(0, len(gesture) - 1, 32, dtype=int)
            x_dots, y_dots = x[indices], y[indices]

        ax.scatter(x_dots, y_dots, c=color, s=dot_size, alpha=alpha, zorder=3)


def plot_gestures_on_keyboard(
    gestures: np.ndarray,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    show_keyboard: bool = True,
    figsize: Tuple[float, float] = (4, 3),
    config: KeyboardConfig = DEFAULT_KEYBOARD_CONFIG
) -> plt.Figure:
    """
    Plot multiple gestures on keyboard layout.

    Args:
        gestures: Array of shape (N, seq_len, 3) or list of gesture arrays
        colors: Per-gesture colors (defaults to blue)
        title: Optional figure title
        show_keyboard: Whether to draw keyboard grid
        figsize: Figure size
        config: Keyboard configuration

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if show_keyboard:
        draw_keyboard(ax, config)

    # Handle single gesture
    if isinstance(gestures, np.ndarray) and gestures.ndim == 2:
        gestures = [gestures]
    elif isinstance(gestures, np.ndarray) and gestures.ndim == 3:
        gestures = [gestures[i] for i in range(len(gestures))]

    if colors is None:
        colors = [COLOR_FAKE] * len(gestures)

    for gesture, color in zip(gestures, colors):
        plot_gesture(ax, gesture, color=color)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    return fig


def create_comparison_figure(
    real_gestures: np.ndarray,
    fake_gestures: np.ndarray,
    words: List[str],
    n_samples: int = 6,
    config: KeyboardConfig = DEFAULT_KEYBOARD_CONFIG
) -> plt.Figure:
    """
    Create side-by-side comparison grid like paper Figure 6.

    Args:
        real_gestures: Array of shape (N, seq_len, 3) - user-drawn gestures
        fake_gestures: Array of shape (N, seq_len, 3) - generated gestures
        words: List of word labels
        n_samples: Number of samples to show
        config: Keyboard configuration

    Returns:
        Matplotlib figure with 2 rows (real/fake) x n_samples columns
    """
    n = min(n_samples, len(real_gestures), len(fake_gestures))
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5))

    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        # Real gesture (top row)
        ax_real = axes[0, i]
        draw_keyboard(ax_real, config)
        plot_gesture(ax_real, real_gestures[i], color=COLOR_REAL)
        ax_real.set_xlim(-1.1, 1.1)
        ax_real.set_ylim(-1.1, 1.1)
        ax_real.set_aspect('equal')
        ax_real.axis('off')
        if i < len(words):
            ax_real.set_title(f'"{words[i]}"', fontsize=9)

        # Fake gesture (bottom row)
        ax_fake = axes[1, i]
        draw_keyboard(ax_fake, config)
        plot_gesture(ax_fake, fake_gestures[i], color=COLOR_FAKE)
        ax_fake.set_xlim(-1.1, 1.1)
        ax_fake.set_ylim(-1.1, 1.1)
        ax_fake.set_aspect('equal')
        ax_fake.axis('off')

    # Row labels
    axes[0, 0].text(-1.5, 0, 'User-drawn', rotation=90, va='center',
                    fontsize=10, fontweight='bold', color=COLOR_REAL)
    axes[1, 0].text(-1.5, 0, 'Generated', rotation=90, va='center',
                    fontsize=10, fontweight='bold', color=COLOR_FAKE)

    plt.tight_layout()
    return fig


def create_overlay_figure(
    real_gestures: np.ndarray,
    fake_gestures: np.ndarray,
    word: str,
    n_samples: int = 5,
    config: KeyboardConfig = DEFAULT_KEYBOARD_CONFIG
) -> plt.Figure:
    """
    Create overlay figure like paper Figure 7 - multiple gestures overlaid.

    Args:
        real_gestures: Array of shape (N, seq_len, 3) - user-drawn gestures
        fake_gestures: Array of shape (N, seq_len, 3) - generated gestures
        word: Word label
        n_samples: Number of gestures to overlay
        config: Keyboard configuration

    Returns:
        Matplotlib figure with overlaid gestures
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    draw_keyboard(ax, config)

    n = min(n_samples, len(real_gestures), len(fake_gestures))

    # Plot real gestures (orange)
    for i in range(n):
        plot_gesture(ax, real_gestures[i], color=COLOR_REAL, alpha=0.6)

    # Plot fake gestures (blue)
    for i in range(n):
        plot_gesture(ax, fake_gestures[i], color=COLOR_FAKE, alpha=0.6)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'"{word}" - Real (orange) vs Generated (blue)', fontsize=10)

    plt.tight_layout()
    return fig
