from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_image(path: str | Path, img: np.ndarray) -> None:
    path = str(path)
    cv2.imwrite(path, img)


def draw_boxes(bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out
