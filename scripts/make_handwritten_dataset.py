##scripts/make_handwritten_dataset.py
#!/usr/bin/env python3
"""
Build handwritten character dataset from a folder structure downloaded from Google Drive.

Input examples supported:
- data/raw/handwritten/mayusculas/A/*.png
- data/raw/handwritten/minusculas/a/*.png
- data/raw/handwritten/numeros/6/*.jpg
- data/raw/handwritten/Datasets OCR otros años/u01204/Mayúsculas/*.png
  (and similar for Minúsculas / Números)

Output:
- <output>/images/<label>_<id>.png   (32x32 normalized, black ink on white)
- <output>/labels.csv               (filename,label,source_path)
- (optional) <output>/originals/    (copy of original images)
- (optional) <output>/rejected/     (QC-rejected originals for debugging)

No OCR libraries are used. Only image preprocessing (OpenCV) to normalize.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------
# Utils
# ---------------------------

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_unicode_label(lbl: str) -> str:
    """Keep labels as single-character classes (A-Z, a-z, 0-9, Ñ/ñ)."""
    lbl = lbl.strip()
    if lbl in {"ñ", "Ñ"}:
        return lbl
    return lbl


def infer_label_from_path(img_path: Path) -> Optional[str]:
    """
    Infer label using:
    1) parent folder name if it is exactly a class: A, b, 6, Ñ, ñ
    2) filename prefix: A_..., 6-..., b ..., etc.
    3) filename is exactly a single char: a.png, 6.png
    """
    parent = img_path.parent.name.strip()

    if re.fullmatch(r"[A-ZÑ]|[a-zñ]|[0-9]", parent):
        return normalize_unicode_label(parent)

    stem = img_path.stem.strip()

    m = re.match(r"^([A-ZÑ]|[a-zñ]|[0-9])([_\-\s]).+", stem)
    if m:
        return normalize_unicode_label(m.group(1))

    if re.fullmatch(r"[A-ZÑ]|[a-zñ]|[0-9]", stem):
        return normalize_unicode_label(stem)

    return None


def walk_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if is_image(p)]


# ---------------------------
# Preprocessing + QC
# ---------------------------

@dataclass
class QCResult:
    ok: bool
    reason: str


def _binarize_choose(gray: np.ndarray) -> np.ndarray:
    """
    Create a foreground mask where ink=255 and background=0.
    Try both Otsu polarities and pick the one with a plausible ink ratio.
    This avoids cases where shadows/background become 'ink'.
    """
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Candidate A: THRESH_BINARY_INV (ink -> 255)
    _, bw_inv = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Candidate B: THRESH_BINARY (ink -> 0) then invert to ink -> 255
    _, bw = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw2 = 255 - bw

    candidates = [bw_inv, bw2]

    best = bw_inv
    best_score = 1e9

    for c in candidates:
        ink_ratio = float((c > 0).mean())

        # Only accept plausible ratios
        if 0.003 < ink_ratio < 0.45:
            # Prefer around 12% ink (typical character coverage)
            score = abs(ink_ratio - 0.12)
            if score < best_score:
                best_score = score
                best = c

    return best


def _select_component(fg255: np.ndarray) -> np.ndarray:
    """
    Keep the most 'character-like' connected component:
    large enough, relatively centered, and not a huge background blob.
    """
    bin_img = (fg255 > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    if n <= 1:
        return fg255

    h, w = bin_img.shape
    cx0, cy0 = w / 2.0, h / 2.0
    diag = float((cx0**2 + cy0**2) ** 0.5)

    best_i = None
    best_score = -1e9

    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        cx, cy = centroids[i]

        # Discard huge components (likely shadow/background)
        if area > 0.5 * h * w:
            continue
        # Discard tiny noise
        if area < 12:
            continue

        touches_border = (x == 0) or (y == 0) or (x + ww >= w) or (y + hh >= h)
        border_penalty = 0.35 if touches_border else 0.0

        dist = float((((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5) / max(1e-6, diag))
        area_norm = float(area / (h * w))

        # Score: larger + centered, penalize border
        score = area_norm - 0.6 * dist - border_penalty

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None:
        return fg255

    mask = (labels == best_i).astype(np.uint8) * 255
    return mask


def _qc(mask255: np.ndarray) -> QCResult:
    """QC on foreground mask (ink=255)."""
    ink_ratio = float((mask255 > 0).mean())
    if ink_ratio < 0.003:
        return QCResult(False, f"too_little_ink({ink_ratio:.4f})")
    if ink_ratio > 0.45:
        return QCResult(False, f"too_much_ink({ink_ratio:.4f})")

    ys, xs = np.where(mask255 > 0)
    if len(xs) == 0:
        return QCResult(False, "no_ink")

    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    if bw < 3 or bh < 3:
        return QCResult(False, f"bbox_too_small({bw}x{bh})")

    return QCResult(True, "ok")


def normalize_to_32x32(img_bgr: np.ndarray) -> Tuple[np.ndarray, QCResult, np.ndarray]:
    """
    Normalize a handwritten character into a 32x32 grayscale image.

    Returns:
      out32: 32x32 uint8, black ink on white background
      qc: quality status
      debug_mask: foreground mask (ink=255) for optional debugging
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Robust binarization
    fg = _binarize_choose(gray)  # ink=255

    # 2) Keep main connected component
    fg = _select_component(fg)

    # 3) QC
    qc = _qc(fg)

    # If QC fails -> fallback grayscale resize (prevents black blocks)
    if not qc.ok:
        out = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        return out, qc, fg

    # 4) Crop bbox around ink
    ys, xs = np.where(fg > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = fg[y0:y1 + 1, x0:x1 + 1]

    # 5) Add small margin (10% of max side)
    h, w = crop.shape
    m = int(0.10 * max(h, w))
    if m > 0:
        crop = cv2.copyMakeBorder(crop, m, m, m, m, cv2.BORDER_CONSTANT, value=0)

    # 6) Pad to square
    h, w = crop.shape
    side = max(h, w)
    pad_y = side - h
    pad_x = side - w

    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left

    square = cv2.copyMakeBorder(
        crop, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    # 7) Resize to 32x32
    out = cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA)

    # 8) Output: black ink on white background
    out = 255 - out

    return out, qc, fg


# ---------------------------
# Main
# ---------------------------

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
        "--save-rejected",
        action="store_true",
        help="Save QC-rejected ORIGINAL images into <output>/rejected/ for debugging",
    )
    ap.add_argument(
        "--reject-on-qc",
        action="store_true",
        help="If set, QC-failed samples are SKIPPED (recommended). If not set, they fallback to grayscale 32x32.",
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
    rejected_out = out_root / "rejected"

    safe_mkdir(images_out)
    if args.copy_originals:
        safe_mkdir(originals_out)
    if args.save_rejected:
        safe_mkdir(rejected_out)

    all_imgs = sorted(walk_images(in_root))

    rows: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}

    skipped_no_label = 0
    skipped_read_fail = 0
    skipped_qc = 0
    processed = 0

    for img_path in all_imgs:
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

        out32, qc, _fg = normalize_to_32x32(img)

        if (not qc.ok) and args.reject_on_qc:
            skipped_qc += 1
            if args.save_rejected:
                # Save ORIGINAL (not the processed) for easy inspection
                rej_name = f"{label}__REJECT__{qc.reason}__{img_path.name}"
                shutil.copy2(img_path, rejected_out / rej_name)
            continue

        out_name = f"{label}_{processed:06d}.png"
        out_rel = f"images/{out_name}"
        out_path = images_out / out_name
        cv2.imwrite(str(out_path), out32)

        if args.copy_originals:
            orig_name = f"{label}_{processed:06d}__{img_path.name}"
            shutil.copy2(img_path, originals_out / orig_name)

        rows.append(
            {
                "filename": out_rel,
                "label": label,
                "source_path": str(img_path),
            }
        )
        counts[label] = counts.get(label, 0) + 1
        processed += 1

    safe_mkdir(out_root)
    labels_csv = out_root / "labels.csv"
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "source_path"])
        writer.writeheader()
        writer.writerows(rows)

    top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:15]

    print("=== Handwritten dataset build ===")
    print(f"Input:    {in_root}")
    print(f"Output:   {out_root}")
    print(f"Found images:         {len(all_imgs)}")
    print(f"Processed:            {processed}")
    print(f"Skipped (no label):   {skipped_no_label}")
    print(f"Skipped (read fail):  {skipped_read_fail}")
    print(f"Skipped (QC reject):  {skipped_qc}")
    print(f"labels.csv: {labels_csv}")
    print("Top classes (up to 15):")
    for k, v in top:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

