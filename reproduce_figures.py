"""Animate word-gesture comparisons as a single MP4."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from wordgesture_gan.data.preprocess import load_dataset
from wordgesture_gan.keyboard.qwerty import QWERTY_ROWS, KeyboardLayout
from wordgesture_gan.minjerk.fit import MinJerkFit, fit_distributions
from wordgesture_gan.minjerk.sample import sample_minimum_jerk
from wordgesture_gan.models.wg_gan import Generator
from wordgesture_gan.prototypes import build_word_prototype
from wordgesture_gan.utils import get_device


MODEL_COLORS = {
    "user": "#F28E2B",
    "wg_gan": "#4E79A7",
    "minjerk": "#E15759",
}

MODEL_LABELS = {
    "user": "User-Drawn",
    "wg_gan": "WordGesture-GAN",
    "minjerk": "Minimum Jerk",
}


@dataclass
class Track:
    line: any
    dot: any
    positions_by_scene: List[np.ndarray]


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_bool(value: str) -> bool:
    return value.lower() == "true"


def _select_words(
    words: Sequence[str],
    n_words: int,
    rng: np.random.Generator,
) -> List[str]:
    unique = sorted(set(words))
    if n_words > len(unique):
        raise ValueError(f"Requested {n_words} words, but only {len(unique)} available.")
    return rng.choice(unique, size=n_words, replace=False).tolist()


def _build_word_index(words: Sequence[str]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for idx, word in enumerate(words):
        index.setdefault(word, []).append(idx)
    return index


def _sample_user_gestures(
    word: str,
    gestures: np.ndarray,
    index: Dict[str, List[int]],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if word not in index:
        raise ValueError(f"No user samples found for word '{word}'.")
    choices = index[word]
    replace = len(choices) < n_samples
    selected = rng.choice(choices, size=n_samples, replace=replace)
    return gestures[selected]


def _load_generator(checkpoint: Path, config_path: Path | None) -> Tuple[Generator, Dict]:
    device = get_device()
    cfg = {}
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    ckpt_cfg = checkpoint_data.get("config", {})
    model_cfg = ckpt_cfg.get("model", {}) or cfg.get("model", {})
    latent_dim = int(model_cfg["latent_dim"])
    hidden_size = int(model_cfg["hidden_size"])
    num_layers = int(model_cfg["num_layers"])
    generator = Generator(
        latent_dim=latent_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)
    generator.load_state_dict(checkpoint_data["generator"])
    generator.eval()
    return generator, ckpt_cfg or cfg


def _sample_wg_gan(
    word: str,
    generator: Generator,
    n_samples: int,
    n_points: int,
    layout: KeyboardLayout,
) -> np.ndarray:
    device = next(generator.parameters()).device
    prototype = build_word_prototype(word, n_points, layout)
    proto_batch = np.repeat(prototype[None, ...], n_samples, axis=0)
    proto_tensor = torch.from_numpy(proto_batch).to(device)
    z = torch.randn(n_samples, generator.latent_dim, device=device)
    with torch.no_grad():
        fake = generator(proto_tensor, z).cpu().numpy()
    return fake


def _sample_minjerk(
    word: str,
    fit: MinJerkFit,
    n_samples: int,
    n_points: int,
    layout: KeyboardLayout,
) -> np.ndarray:
    samples = [sample_minimum_jerk(word, fit, n_points, layout) for _ in range(n_samples)]
    return np.stack(samples, axis=0)


def _gesture_duration(gesture: np.ndarray) -> float:
    duration = float(np.sum(gesture[:, 2]))
    if duration <= 1e-6:
        duration = float(len(gesture) - 1)
    return duration


def _resample_gesture(gesture: np.ndarray, frame_times: np.ndarray, time_mode: str) -> np.ndarray:
    xs = gesture[:, 0]
    ys = gesture[:, 1]
    dt = gesture[:, 2]
    t = np.cumsum(dt)
    if t[-1] <= 1e-6:
        t = np.linspace(0.0, 1.0, num=len(xs))
    else:
        t = t - t[0]
    if time_mode == "normalized":
        t = t / max(t[-1], 1e-6)
    times = np.clip(frame_times, t[0], t[-1])
    x_interp = np.interp(times, t, xs)
    y_interp = np.interp(times, t, ys)
    return np.stack([x_interp, y_interp], axis=-1)


def _compute_frame_times(
    gestures: Iterable[np.ndarray],
    time_mode: str,
    fps: int,
    n_points: int,
) -> np.ndarray:
    if time_mode == "normalized":
        return np.linspace(0.0, 1.0, num=n_points)
    max_duration = max(_gesture_duration(g) for g in gestures)
    frames = max(2, int(math.ceil(max_duration * fps)))
    return np.linspace(0.0, max_duration, num=frames)


def _draw_keyboard(ax: plt.Axes, layout: KeyboardLayout) -> None:
    total_width = 10 * layout.key_width
    total_height = 3 * layout.row_spacing

    def to_norm(x: float, y: float) -> Tuple[float, float]:
        x_norm = (x / total_width) * 2.0 - 1.0
        y_norm = (y / total_height) * 2.0 - 1.0
        return x_norm, y_norm

    for row_idx, row in enumerate(QWERTY_ROWS):
        offset = layout.row_offsets[row_idx] * layout.key_width
        for col_idx, ch in enumerate(row):
            center_x = offset + (col_idx + 0.5) * layout.key_width
            center_y = (row_idx + 0.5) * layout.row_spacing
            x0 = center_x - layout.key_width / 2
            y0 = center_y - layout.key_height / 2
            x_norm, y_norm = to_norm(x0, y0)
            w_norm = (layout.key_width / total_width) * 2.0
            h_norm = (layout.key_height / total_height) * 2.0
            rect = Rectangle((x_norm, y_norm), w_norm, h_norm, fill=False, lw=0.6, ec="#999999")
            ax.add_patch(rect)
            tx, ty = to_norm(center_x, center_y)
            ax.text(tx, ty, ch, ha="center", va="center", fontsize=6, color="#777777")

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(1.05, -1.05)
    ax.set_aspect("equal")
    ax.axis("off")


def _grid_dims(n_items: int) -> Tuple[int, int]:
    cols = int(math.ceil(math.sqrt(n_items)))
    rows = int(math.ceil(n_items / cols))
    return rows, cols


def _ensure_pillow() -> None:
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise RuntimeError("Pillow not found. Install with: pip install pillow")


def _add_legend(fig: plt.Figure, models: List[str]) -> None:
    """Add a legend to the figure showing model colors."""
    handles = [
        Line2D([0], [0], color=MODEL_COLORS[model], lw=2, label=MODEL_LABELS[model])
        for model in models
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(models),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate gesture comparisons as a single MP4.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/wg_gan_latest.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/wg_gan.yaml"))
    parser.add_argument("--n_words", type=int, default=1)
    parser.add_argument("--models", type=str, default="user,wg_gan,minjerk")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--collapse_samples", type=str, choices=["true", "false"], default="true")
    parser.add_argument("--out", type=Path, required=True, help="Output GIF file")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--time_mode", choices=["normalized", "real"], default="normalized")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    _set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    collapse_samples = _parse_bool(args.collapse_samples)

    if args.n_words < 1:
        raise ValueError("n_words must be >= 1.")
    if args.n_samples < 1:
        raise ValueError("n_samples must be >= 1.")

    models = [model.strip().lower() for model in args.models.split(",") if model.strip()]
    for model in models:
        if model not in MODEL_COLORS:
            raise ValueError(f"Unsupported model '{model}'. Use user,wg_gan,minjerk.")

    split_path = args.data_dir / "test.npz"
    train_path = args.data_dir / "train.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {split_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {train_path}")

    split_gestures, split_words = load_dataset(split_path)
    train_gestures, train_words = load_dataset(train_path)
    n_points = int(split_gestures.shape[1])

    word_index = _build_word_index(split_words)
    selected_words = _select_words(split_words, args.n_words, rng)
    n_words = len(selected_words)

    layout = KeyboardLayout()

    generator = None
    if "wg_gan" in models:
        generator, _ = _load_generator(args.checkpoint, args.config)

    minjerk_fit = None
    if "minjerk" in models:
        minjerk_fit = fit_distributions(train_gestures, train_words, layout)

    gestures_by_word: Dict[str, Dict[str, np.ndarray]] = {}
    for word in selected_words:
        per_model: Dict[str, np.ndarray] = {}
        if "user" in models:
            per_model["user"] = _sample_user_gestures(word, split_gestures, word_index, args.n_samples, rng)
        if "wg_gan" in models and generator is not None:
            per_model["wg_gan"] = _sample_wg_gan(word, generator, args.n_samples, n_points, layout)
        if "minjerk" in models and minjerk_fit is not None:
            per_model["minjerk"] = _sample_minjerk(word, minjerk_fit, args.n_samples, n_points, layout)
        gestures_by_word[word] = per_model

    if collapse_samples:
        rows, cols = _grid_dims(n_words)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6 + 0.4))
        fig.subplots_adjust(top=0.82)
        _add_legend(fig, models)
        axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        tracks: List[Track] = []

        all_gestures = []
        for word in selected_words:
            for model in models:
                all_gestures.extend(list(gestures_by_word[word][model]))
        frame_times = _compute_frame_times(all_gestures, args.time_mode, args.fps, n_points)

        for idx, word in enumerate(selected_words):
            ax = axes_list[idx]
            _draw_keyboard(ax, layout)
            ax.set_title(word, fontsize=10)
            for model in models:
                color = MODEL_COLORS[model]
                for sample_idx in range(args.n_samples):
                    gesture = gestures_by_word[word][model][sample_idx]
                    positions = _resample_gesture(gesture, frame_times, args.time_mode)
                    line, = ax.plot([], [], color=color, lw=1.6, alpha=0.8)
                    dot, = ax.plot([], [], marker="o", markersize=2.5, color=color, alpha=0.9)
                    tracks.append(Track(line=line, dot=dot, positions_by_scene=[positions]))

        for ax in axes_list[n_words:]:
            ax.axis("off")

        total_frames = len(frame_times)

        def update(frame_idx: int):
            for track in tracks:
                positions = track.positions_by_scene[0]
                idx = min(frame_idx, len(positions) - 1)
                path = positions[: idx + 1]
                track.line.set_data(path[:, 0], path[:, 1])
                track.dot.set_data([path[-1, 0]], [path[-1, 1]])
            return [artist for track in tracks for artist in (track.line, track.dot)]

        anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000 / args.fps, blit=False)
    else:
        # Grid: rows = models, cols = words * n_samples
        # All words animate simultaneously
        rows = len(models)
        cols = n_words * args.n_samples
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6 + 0.4))
        fig.subplots_adjust(top=0.88)
        _add_legend(fig, models)
        axes = np.array(axes).reshape(rows, cols)
        tracks: List[Track] = []

        # Compute frame times over all gestures
        all_gestures = []
        for word in selected_words:
            for model in models:
                all_gestures.extend(list(gestures_by_word[word][model]))
        frame_times = _compute_frame_times(all_gestures, args.time_mode, args.fps, n_points)

        for row_idx, model in enumerate(models):
            for word_idx, word in enumerate(selected_words):
                for sample_idx in range(args.n_samples):
                    col_idx = word_idx * args.n_samples + sample_idx
                    ax = axes[row_idx, col_idx]
                    _draw_keyboard(ax, layout)
                    label = MODEL_LABELS[model]
                    if args.n_samples == 1:
                        ax.set_title(f"{word}\n{label}", fontsize=8)
                    else:
                        ax.set_title(f"{word} #{sample_idx + 1}\n{label}", fontsize=7)
                    gesture = gestures_by_word[word][model][sample_idx]
                    positions = _resample_gesture(gesture, frame_times, args.time_mode)
                    line, = ax.plot([], [], color=MODEL_COLORS[model], lw=1.8, alpha=0.9)
                    dot, = ax.plot([], [], marker="o", markersize=2.5, color=MODEL_COLORS[model], alpha=0.95)
                    tracks.append(Track(line=line, dot=dot, positions_by_scene=[positions]))

        total_frames = len(frame_times)

        def update(frame_idx: int):
            for track in tracks:
                positions = track.positions_by_scene[0]
                idx = min(frame_idx, len(positions) - 1)
                path = positions[: idx + 1]
                track.line.set_data(path[:, 0], path[:, 1])
                track.dot.set_data([path[-1, 0]], [path[-1, 1]])
            return [artist for track in tracks for artist in (track.line, track.dot)]

        anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000 / args.fps, blit=False)

    _ensure_pillow()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=args.fps)
    anim.save(args.out, writer=writer, dpi=args.dpi)


if __name__ == "__main__":
    main()
