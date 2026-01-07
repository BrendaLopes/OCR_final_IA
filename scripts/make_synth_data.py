#!/usr/bin/env python3
"""
Build handwritten character dataset from a folder downloaded from Google Drive.

Input (examples supported):
- data/raw/handwritten/mayusculas/A/*.png
- data/raw/handwritten/minusculas/a/*.png
- data/raw/handwritten/numeros/6/*.jpg
- data/raw/handwritten/Datasets OCR otros años/u01204/Mayúsculas/*.png
  (and similar for Minúsculas / Números)
- Filenames like: A_Adrian_Garcia.png, 6_Brenda_Lopes.jpg, or even a.png

Output:
- <output>/images/<label>_<id>.png   (32x32 normalized)
- <output>/labels.csv               (filename,label,source_path)

No OCR libraries are used. Only image preprocessing (OpenCV) to normalize.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Optional, List, Dict

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_to_32x32(img_bgr: np.ndarray) -> np.ndarray:
    """
    Normalize a handwritten character image into a 32x32 grayscale image.

    Steps:
    - to gray
    - blur
    - Otsu threshold (binary invert)
    - crop around ink bbox
    - pad to square
    - resize to 32x32
    - invert back to black ink on white background (common for classifiers)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # small blur to reduce sensor noise
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # binary invert: ink -> 255, background -> 0
    _, bw = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ys, xs = np.where(bw > 0)
    if len(xs) == 0 or len(ys) == 0:
        # no ink found -> just resize grayscale
        out = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        return out

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = bw[y0 : y1 + 1, x0 : x1 + 1]

    h, w = crop.shape
    side = max(h, w)

    pad_y = side - h
    pad_x = side - w
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left

    square = cv2.copyMakeBorder(
        crop, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0
    )

    out = cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA)

    # switch to black ink on white background
    out = 255 - out
    return out


def normalize_unicode_label(lbl: str) -> str:
    """
    Keep labels as single-character classes (A-Z, a-z, 0-9, Ñ/ñ).
    Normalize common variants.
    """
    lbl = lbl.strip()
    if lbl == "ñ":
        return "ñ"
    if lbl == "Ñ":
        return "Ñ"
    return lbl


def infer_label_from_path(img_path: Path) -> Optional[str]:
    """
    Infer label using:
    1) parent folder name if it is exactly a class: A, b, 6, Ñ, ñ
    2) filename prefix: A_..., 6-..., b ..., etc.
    3) filename is exactly a single char: a.png, 6.png
    """
    parent = img_path.parent.name.strip()

    # If parent folder is the class itself
    if re.fullmatch(r"[A-ZÑ]|[a-zñ]|[0-9]", parent):
        return normalize_unicode_label(parent)

    stem = img_path.stem.strip()

    # Filenames starting with "<label>_" or "<label>-"
    m = re.match(r"^([A-ZÑ]|[a-zñ]|[0-9])([_\-\s]).+", stem)
    if m:
        return normalize_unicode_label(m.group(1))

    # Filenames like "a.png" or "6.png"
    if re.fullmatch(r"[A-ZÑ]|[a-zñ]|[0-9]", stem):
        return normalize_unicode_label(stem)

    return None


def walk_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if is_image(p)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build handwritten dataset (32x32 images + labels.csv) from Drive folders"
    )
    ap.add_argument("--input", required=True, help="Input root folder (downloaded from Drive)")
    ap.add_argument("--output", required=True, help="Output folder (e.g., data/processed/handwritten_chars)")
    ap.add_argument(
        "--copy-originals",
        action="store_true",
        help="Copy original images into <output>/originals/ for traceability",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit of processed images (0 = no limit). Useful for quick tests.",
    )
    args = ap.parse_args()

    in_root = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()

    if not in_root.exists():
        raise FileNotFoundError(f"Input folder not found: {in_root}")

    images_out = out_root / "images"
    originals_out = out_root / "originals"

    safe_mkdir(images_out)
    if args.copy_originals:
        safe_mkdir(originals_out)

    all_imgs = sorted(walk_images(in_root))

    rows: List[Dict[str, str]] = []
    skipped_no_label = 0
    skipped_read_fail = 0
    processed = 0

    for i, img_path in enumerate(all_imgs):
        if args.limit and processed >= args.limit:
            break

        label = infer_label_from_path(img_path)
        if label is None:
            skipped_no_label += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped_read_fail += 1
            continue

        norm = normalize_to_32x32(img)

        out_name = f"{label}_{processed:06d}.png"
        out_rel = f"images/{out_name}"
        out_path = images_out / out_name
        cv2.imwrite(str(out_path), norm)

        if args.copy_originals:
            # Keep original filename for traceability
            orig_name = f"{label}_{processed:06d}__{img_path.name}"
            shutil.copy2(img_path, originals_out / orig_name)

        rows.append(
            {
                "filename": out_rel,
                "label": label,
                "source_path": str(img_path),
            }
        )
        processed += 1

    safe_mkdir(out_root)
    labels_csv = out_root / "labels.csv"
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "source_path"])
        writer.writeheader()
        writer.writerows(rows)

    # Quick stats by class
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:15]

    print("=== Handwritten dataset build ===")
    print(f"Input:  {in_root}")
    print(f"Output: {out_root}")
    print(f"Processed: {processed}")
    print(f"Skipped (no label): {skipped_no_label}")
    print(f"Skipped (read fail): {skipped_read_fail}")
    print(f"labels.csv: {labels_csv}")
    print("Top classes (up to 15):")
    for k, v in top:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
