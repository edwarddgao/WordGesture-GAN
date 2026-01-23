"""Parse raw swipe logs into JSONL gesture samples."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass
class GestureSample:
    sentence: str
    word: str
    is_err: int
    keyb_width: int
    keyb_height: int
    timestamps: List[int]
    xs: List[float]
    ys: List[float]

    def to_dict(self) -> Dict:
        return asdict(self)


def _parse_header(header: str) -> Dict[str, int]:
    cols = header.strip().split()
    return {name: idx for idx, name in enumerate(cols)}


def iter_log_gestures(path: Path) -> Iterator[GestureSample]:
    with path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line:
            return
        col_idx = _parse_header(header_line)
        required = [
            "sentence",
            "timestamp",
            "keyb_width",
            "keyb_height",
            "event",
            "x_pos",
            "y_pos",
            "word",
            "is_err",
        ]
        for col in required:
            if col not in col_idx:
                raise ValueError(f"Missing column '{col}' in {path}")

        current: Optional[GestureSample] = None
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < len(col_idx):
                continue

            try:
                event = parts[col_idx["event"]]
                word = parts[col_idx["word"]].lower()
                sentence = parts[col_idx["sentence"]]
                is_err = int(parts[col_idx["is_err"]])
                ts = int(parts[col_idx["timestamp"]])
                keyb_w = int(parts[col_idx["keyb_width"]])
                keyb_h = int(parts[col_idx["keyb_height"]])
                x_pos = float(parts[col_idx["x_pos"]])
                y_pos = float(parts[col_idx["y_pos"]])
            except (ValueError, IndexError):
                # Skip malformed rows
                continue

            if event == "touchstart":
                if current is not None:
                    yield current
                current = GestureSample(
                    sentence=sentence,
                    word=word,
                    is_err=is_err,
                    keyb_width=keyb_w,
                    keyb_height=keyb_h,
                    timestamps=[],
                    xs=[],
                    ys=[],
                )

            if current is None:
                continue

            if current.word != word:
                yield current
                current = GestureSample(
                    sentence=sentence,
                    word=word,
                    is_err=is_err,
                    keyb_width=keyb_w,
                    keyb_height=keyb_h,
                    timestamps=[],
                    xs=[],
                    ys=[],
                )

            current.is_err = max(current.is_err, is_err)
            current.keyb_width = keyb_w
            current.keyb_height = keyb_h
            current.timestamps.append(ts)
            current.xs.append(x_pos)
            current.ys.append(y_pos)

            if event == "touchend":
                yield current
                current = None

        if current is not None:
            yield current


def write_jsonl(samples: Iterable[GestureSample], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict()) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse swipe log files into JSONL.")
    parser.add_argument("--log_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    log_paths = sorted(args.log_dir.rglob("*.log"))
    if args.max_files is not None:
        log_paths = log_paths[: args.max_files]

    from tqdm import tqdm

    manifest = []
    for log_path in tqdm(log_paths, desc="Parsing logs"):
        rel = log_path.relative_to(args.log_dir)
        out_path = args.out_dir / rel.with_suffix(".jsonl")
        count = write_jsonl(iter_log_gestures(log_path), out_path)
        manifest.append({"log": str(rel), "jsonl": str(out_path), "count": count})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
