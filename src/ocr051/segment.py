from __future__ import annotations

import cv2
import numpy as np


def find_char_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Encuentra cajas (x,y,w,h) de posibles caracteres a partir de la máscara.
    Heurísticas simples: filtrar por tamaño y proporciones.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape[:2]

    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        area = w * h
        if area < 20:
            continue
        if w < 3 or h < 8:
            continue
        if w > 0.9 * W or h > 0.9 * H:
            continue

        # Evita columnas/ruido muy vertical tipo borde
        aspect = w / max(1, h)
        if aspect < 0.05:
            continue

        boxes.append((x, y, w, h))

    # Orden inicial por x
    boxes.sort(key=lambda b: b[0])
    return boxes


def group_lines(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """
    Agrupa cajas por líneas usando el centro Y.
    Sirve para una o varias líneas.
    """
    if not boxes:
        return []

    hs = sorted([h for (_, _, _, h) in boxes])
    med_h = hs[len(hs) // 2]
    thresh = max(10, int(0.8 * med_h))

    items = []
    for b in boxes:
        x, y, w, h = b
        cy = y + h / 2
        items.append((cy, b))
    items.sort(key=lambda t: t[0])

    lines: list[list[tuple[int, int, int, int]]] = []
    current: list[tuple[int, int, int, int]] = []
    current_cy = None

    for cy, b in items:
        if current_cy is None:
            current = [b]
            current_cy = cy
            continue

        if abs(cy - current_cy) <= thresh:
            current.append(b)
            current_cy = (current_cy * (len(current) - 1) + cy) / len(current)
        else:
            current.sort(key=lambda bb: bb[0])
            lines.append(current)
            current = [b]
            current_cy = cy

    current.sort(key=lambda bb: bb[0])
    lines.append(current)

    # Orden de líneas por y
    lines.sort(key=lambda line: min([b[1] for b in line]))
    return lines


def estimate_spaces(line: list[tuple[int, int, int, int]]) -> list[int]:
    """
    Devuelve los gaps entre cajas consecutivas para estimar espacios.
    """
    gaps = []
    for i in range(1, len(line)):
        x0, _, w0, _ = line[i - 1]
        x1, _, _, _ = line[i]
        gap = x1 - (x0 + w0)
        gaps.append(gap)
    return gaps


def build_text_from_line(chars: list[str], line_boxes: list[tuple[int, int, int, int]]) -> str:
    """
    Inserta espacios en base a gaps grandes entre caracteres.
    """
    if not chars:
        return ""

    gaps = estimate_spaces(line_boxes)
    if not gaps:
        return "".join(chars)

    # Umbral simple: gap mayor que 1.6x mediana -> espacio
    g_sorted = sorted(gaps)
    med = g_sorted[len(g_sorted) // 2]
    space_thresh = max(10, int(1.6 * med))

    out = [chars[0]]
    for i, g in enumerate(gaps, start=1):
        if g > space_thresh:
            out.append(" ")
        out.append(chars[i])
    return "".join(out)
