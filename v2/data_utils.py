"""Small dataset helpers shared by the prep, training and evaluation scripts."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_for(img_path: Path) -> Path:
    """Ultralytics convention: .../images/<split>/x.jpg -> .../labels/<split>/x.txt"""
    parts = list(img_path.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_yolo(label_path: Path, w: int, h: int, cls: int | None = 0) -> np.ndarray:
    """YOLO txt -> (N,4) xyxy pixel boxes. Missing file = no boxes (negative image)."""
    if not label_path.exists():
        return np.zeros((0, 4), np.float32)
    rows = []
    for line in label_path.read_text().splitlines():
        v = line.split()
        if len(v) < 5 or (cls is not None and int(float(v[0])) != cls):
            continue
        cx, cy, bw, bh = map(float, v[1:5])
        rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, np.float32).reshape(-1, 4)


def write_yolo(label_path: Path, boxes_xyxy, w: int, h: int, cls: int = 0) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for x1, y1, x2, y2 in boxes_xyxy:
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw > 0 and bh > 0:
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def image_size(path: Path) -> tuple[int, int]:
    """(w, h) without decoding the full image when PIL is available."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        from classical import load_bgr
        h, w = load_bgr(path).shape[:2]
        return w, h


def phash64(gray: np.ndarray) -> int:
    """Perceptual hash (same recipe as imagehash.phash): 32x32 DCT, 8x8 low band."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    bits = (low > np.median(low)).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def hamming_matrix_rows(hashes: np.ndarray, start: int, stop: int) -> np.ndarray:
    x = hashes[start:stop, None] ^ hashes[None, :]
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8), axis=-1).reshape(*x.shape, 64).sum(-1)


# ----------------------------------------------------------------- splits
def load_split(dataset_root: Path, split: str) -> list[Path]:
    """Images of one split of the frozen dataset (train/val/test/neg_test)."""
    folder = Path(dataset_root) / "images" / split
    return list_images(folder) if folder.exists() else []


def write_data_yaml(dataset_root: Path) -> Path:
    """(Re)write data.yaml with this machine's absolute path (Colab vs Windows)."""
    root = Path(dataset_root).resolve()
    y = root / "data.yaml"
    y.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\n"
                 f"test: images/test\nnames:\n  0: pothole\n")
    return y


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
