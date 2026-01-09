from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class PreprocessResult:
    gray: np.ndarray          # uint8
    mask: np.ndarray          # uint8, ink=255, bg=0


def to_gray(bgr: np.ndarray) -> np.ndarray:
    if len(bgr.shape) == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def build_ink_mask(gray: np.ndarray) -> np.ndarray:
    """
    Devuelve máscara con tinta=255 y fondo=0.
    Robusto para fotos: blur + Otsu + limpieza morfológica.
    """
    g = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu: probamos ambas polaridades y elegimos la que da un ratio de tinta razonable
    _, m_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # tinta blanca
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m2 = 255 - m

    def score(mask: np.ndarray) -> float:
        r = float((mask > 0).mean())
        # penaliza ratios absurdos: muy poca o demasiada tinta
        if r < 0.002 or r > 0.60:
            return 1e9
        return abs(r - 0.10)  # preferimos ~10% tinta

    mask = m_inv if score(m_inv) <= score(m2) else m2

    # Limpieza: quita puntitos y une trazos
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    return mask


def preprocess_page(bgr: np.ndarray) -> PreprocessResult:
    gray = to_gray(bgr)
    mask = build_ink_mask(gray)
    return PreprocessResult(gray=gray, mask=mask)


def normalize_char_to_32(gray_crop: np.ndarray, mask_crop: np.ndarray) -> np.ndarray:
    """
    Entrada:
      - gray_crop: recorte en gris
      - mask_crop: recorte máscara (tinta=255)
    Salida:
      - img32: uint8 (32x32), tinta negra sobre fondo blanco
    """
    # bbox ajustada al contenido
    ys, xs = np.where(mask_crop > 0)
    if len(xs) == 0 or len(ys) == 0:
        out = cv2.resize(gray_crop, (32, 32), interpolation=cv2.INTER_AREA)
        return out

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    crop = mask_crop[y0:y1 + 1, x0:x1 + 1]

    h, w = crop.shape
    side = max(h, w)

    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left

    square = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    img32 = cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA)

    # Queremos salida: negro=ink, blanco=bg
    img32 = 255 - img32
    return img32
