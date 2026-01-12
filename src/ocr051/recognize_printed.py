from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _ensure_dir(p: str | Path) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _binarize_choose(gray: np.ndarray) -> np.ndarray:
    """
    Devuelve máscara con tinta=255, fondo=0.
    Prueba Otsu con ambas polaridades y elige la que tiene ratio de tinta razonable.
    """
    g = cv2.GaussianBlur(gray, (3, 3), 0)

    _, inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # tinta=255
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw  # tinta=255

    def score(mask: np.ndarray) -> float:
        r = float((mask > 0).mean())
        if r < 0.003 or r > 0.50:
            return 1e9
        return abs(r - 0.12)

    return inv if score(inv) <= score(bw) else bw


def _cleanup(mask255: np.ndarray) -> np.ndarray:
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, k1, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2, iterations=1)
    return m


def _extract_boxes(mask255: np.ndarray, min_box_area: int) -> List[tuple[int, int, int, int]]:
    bin_img = (mask255 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    H, W = bin_img.shape

    boxes: List[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_box_area:
            continue
        if area > 0.65 * H * W:
            continue
        if w < 2 or h < 2:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    # orden aproximado por línea
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def _group_lines(boxes: List[tuple[int, int, int, int]]) -> List[List[tuple[int, int, int, int]]]:
    if not boxes:
        return []

    centers_y = np.array([y + h / 2 for (_, y, _, h) in boxes], dtype=np.float32)
    med_h = float(np.median([h for (_, _, _, h) in boxes]))
    thr = 0.60 * med_h

    order = np.argsort(centers_y)
    boxes_sorted = [boxes[i] for i in order]
    cy_sorted = [centers_y[i] for i in order]

    lines: List[List[tuple[int, int, int, int]]] = []
    cur = [boxes_sorted[0]]
    cur_y = cy_sorted[0]

    for b, cy in zip(boxes_sorted[1:], cy_sorted[1:]):
        if abs(cy - cur_y) <= thr:
            cur.append(b)
            cur_y = (cur_y * (len(cur) - 1) + cy) / len(cur)
        else:
            cur.sort(key=lambda t: t[0])
            lines.append(cur)
            cur = [b]
            cur_y = cy

    cur.sort(key=lambda t: t[0])
    lines.append(cur)

    # ordena líneas de arriba a abajo
    lines.sort(key=lambda ln: float(np.mean([y + h / 2 for (_, y, _, h) in ln])))
    return lines


def _insert_spaces(line: List[tuple[int, int, int, int]]) -> List[tuple[tuple[int, int, int, int], bool]]:
    if not line:
        return []
    widths = [w for (_, _, w, _) in line]
    med_w = float(np.median(widths)) if widths else 10.0

    out: List[tuple[tuple[int, int, int, int], bool]] = [(line[0], False)]
    prev = line[0]
    for b in line[1:]:
        gap = b[0] - (prev[0] + prev[2])
        out.append((b, gap > 1.2 * med_w))
        prev = b
    return out


def _crop(mask255: np.ndarray, box: tuple[int, int, int, int], pad: int = 2) -> np.ndarray:
    x, y, w, h = box
    H, W = mask255.shape
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return mask255[y0:y1, x0:x1]


def _norm32(mask_crop255: np.ndarray) -> np.ndarray:
    """mask_crop: tinta=255 -> devuelve 32x32 float tinta=1 fondo=0"""
    ys, xs = np.where(mask_crop255 > 0)
    if len(xs) == 0:
        return np.zeros((32, 32), dtype=np.float32)

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    ink = mask_crop255[y0:y1 + 1, x0:x1 + 1]

    h, w = ink.shape
    side = max(h, w)

    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left

    sq = cv2.copyMakeBorder(ink, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    sq = cv2.resize(sq, (32, 32), interpolation=cv2.INTER_AREA)
    return (sq.astype(np.float32) / 255.0)


def _classes_default() -> List[str]:
    cls = list("0123456789")
    cls += [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    cls += [chr(i) for i in range(ord("a"), ord("z") + 1)]
    return cls


def _render_template(ch: str, font_face: int, font_scale: float, thickness: int) -> np.ndarray:
    canvas = np.zeros((64, 64), dtype=np.uint8)
    cv2.putText(canvas, ch, (8, 52), font_face, font_scale, 255, thickness, cv2.LINE_AA)
    _, m = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY)
    return _norm32(m)


def build_template_bank(classes: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Bank: class -> (K,1024) con K variantes (fuente/escala/grosor)
    """
    if classes is None:
        classes = _classes_default()

    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    ]
    scales = [1.6, 1.8, 2.0]
    thicks = [1, 2]

    bank: Dict[str, List[np.ndarray]] = {c: [] for c in classes}
    for c in classes:
        for ff in fonts:
            for sc in scales:
                for th in thicks:
                    t = _render_template(c, ff, sc, th).reshape(-1).astype(np.float32)
                    t = t / (float(np.linalg.norm(t)) + 1e-8)
                    bank[c].append(t)

    return {c: np.stack(bank[c], axis=0) for c in classes}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def predict_char_template(x32: np.ndarray, bank: Dict[str, np.ndarray], temperature: float = 12.0) -> Tuple[str, float]:
    v = x32.reshape(-1).astype(np.float32)
    v = v / (float(np.linalg.norm(v)) + 1e-8)

    classes = list(bank.keys())
    sims = np.zeros((len(classes),), dtype=np.float32)

    for i, c in enumerate(classes):
        sims[i] = float(np.max(bank[c] @ v))

    probs = _softmax(sims * float(temperature))
    j = int(np.argmax(probs))
    return classes[j], float(probs[j])


def _load_or_build_bank(cache_path: Path) -> Dict[str, np.ndarray]:
    if cache_path.exists():
        data = np.load(str(cache_path), allow_pickle=True)
        return {k: data[k] for k in data.files}

    bank = build_template_bank()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), **bank)
    return bank


def ocr_printed(
    image_path: str,
    debug_dir: str | Path = "outputs/debug_print_last",
    min_box_area: int = 80,
    min_conf: float = 0.0,
    low_conf_char: str = "?",
    cache_templates_path: str | Path = "models/print_templates.npz",
    ignore_spaces: bool = False,
) -> tuple[str, float, list[tuple[str, float]]]:
    """
    OCR de imprenta (sin librerías OCR): segmentación por componentes conectados + matching contra templates.
    """
    debug_dir = _ensure_dir(debug_dir)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"No puedo leer la imagen: {image_path}")

    gray = _to_gray(bgr)
    mask = _cleanup(_binarize_choose(gray))

    cv2.imwrite(str(debug_dir / "01_gray.png"), gray)
    cv2.imwrite(str(debug_dir / "02_mask.png"), mask)

    boxes = _extract_boxes(mask, min_box_area=min_box_area)
    if not boxes:
        return "", 0.0, []

    dbg = bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imwrite(str(debug_dir / "03_boxes.png"), dbg)

    bank = _load_or_build_bank(Path(cache_templates_path))

    lines = _group_lines(boxes)
    char_dir = _ensure_dir(debug_dir / "chars")

    parts: List[str] = []
    pairs: List[tuple[str, float]] = []
    confs: List[float] = []

    idx = 0
    for li, line in enumerate(lines):
        for box, add_space in _insert_spaces(line):
            if add_space and not ignore_spaces:
                parts.append(" ")

            crop = _crop(mask, box, pad=2)
            x32 = _norm32(crop)

            ch, p = predict_char_template(x32, bank)
            if p < min_conf:
                ch = low_conf_char

            parts.append(ch)
            pairs.append((ch, p))
            confs.append(p)

            cv2.imwrite(str(char_dir / f"{idx:04d}_{ch}_{p:.2f}.png"), (x32 * 255).astype(np.uint8))
            idx += 1

        if li < len(lines) - 1 and not ignore_spaces:
            parts.append("\n")

    text = "".join(parts).strip("\n")
    global_conf = float(np.mean(confs)) if confs else 0.0
    return text, global_conf, pairs
