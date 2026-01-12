from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2

from .preprocess import preprocess_page, normalize_char_to_32
from .segment import find_char_boxes, group_lines, build_text_from_line
from .recognize_handwritten import pick_device, load_model, predict_char_32
from .io_utils import ensure_dir, save_image, draw_boxes
from .recognize_printed import ocr_printed


@dataclass
class OCRChar:
    ch: str
    conf: float           # 0..1
    box: tuple[int, int, int, int]


@dataclass
class OCRLine:
    text: str
    chars: list[OCRChar]
    mean_conf: float      # 0..1
    min_conf: float       # 0..1
    score: float          # heuristic for main line selection


def _line_score(line_boxes: list[tuple[int, int, int, int]]) -> float:
    """
    Heurística para escoger la línea principal:
    - favorece líneas con más caracteres
    - favorece líneas con cajas grandes (menos ruido)
    """
    if not line_boxes:
        return -1e9
    area_sum = sum(w * h for (_, _, w, h) in line_boxes)
    count = len(line_boxes)
    return area_sum + 2000.0 * count


def _dominant_is_upper(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    up = sum(1 for c in letters if c.isupper())
    return (up / len(letters)) >= 0.6


def _apply_case_heuristic(text: str) -> str:
    """
    Si la frase parece mayoritariamente en MAYÚSCULAS, sube todo a mayúsculas.
    """
    if _dominant_is_upper(text):
        return text.upper()
    return text


def ocr_handwritten(
    image_path: str,
    model_path: str,
    classes_path: str,
    debug_dir: str | None = None,
    force_cpu: bool = False,
    main_line_only: bool = True,
    min_box_area: int = 80,
    min_conf: float = 0.00,
    low_conf_char: str = "?",
) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Devuelve:
      - texto_final
      - confianza_global (0..1)
      - lista [(char, conf)] SOLO de la línea final (sin espacios)
    """

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"No puedo leer la imagen: {image_path}")

    prep = preprocess_page(bgr)

    # 1) Segmentación inicial
    boxes = find_char_boxes(prep.mask)

    # 2) Filtro básico por área (reduce polvo/suciedad)
    boxes = [b for b in boxes if (b[2] * b[3]) >= min_box_area]

    # 3) Agrupa en líneas
    lines_boxes = group_lines(boxes)

    device = pick_device(force_cpu=force_cpu)
    model, classes = load_model(model_path, classes_path, device)

    if debug_dir:
        d = ensure_dir(debug_dir)
        save_image(d / "01_gray.png", prep.gray)
        save_image(d / "02_mask.png", prep.mask)
        save_image(d / "03_boxes.png", draw_boxes(bgr, boxes))

    # 4) Reconoce cada línea y calcula score
    ocr_lines: list[OCRLine] = []
    crop_idx = 0

    for line in lines_boxes:
        chars: list[OCRChar] = []
        raw_chars: list[str] = []

        for (x, y, w, h) in line:
            gray_crop = prep.gray[y:y + h, x:x + w]
            mask_crop = prep.mask[y:y + h, x:x + w]

            img32 = normalize_char_to_32(gray_crop, mask_crop)
            ch, p = predict_char_32(model, classes, img32, device)

            chars.append(OCRChar(ch=ch, conf=p, box=(x, y, w, h)))
            raw_chars.append(ch)

            if debug_dir:
                d = Path(debug_dir)
                save_image(d / f"crop_{crop_idx:03d}_{ch}_{p:.2f}.png", img32)
            crop_idx += 1

        # texto con espacios (heurística por gaps)
        line_text = build_text_from_line(raw_chars, line)

        confs = [c.conf for c in chars] if chars else [0.0]
        mean_conf = float(sum(confs) / max(1, len(confs)))
        min_conf_line = float(min(confs)) if confs else 0.0
        score = _line_score(line)

        ocr_lines.append(OCRLine(
            text=line_text,
            chars=chars,
            mean_conf=mean_conf,
            min_conf=min_conf_line,
            score=score
        ))

    if not ocr_lines:
        return "", 0.0, []

    # 5) Escoger línea principal o devolver todas
    if main_line_only:
        chosen_lines = [max(ocr_lines, key=lambda L: L.score)]
    else:
        chosen_lines = sorted(ocr_lines, key=lambda L: -L.score)

    # 6) Post-proceso (manteniendo espacios)
    final_text_lines: list[str] = []
    final_pairs: list[tuple[str, float]] = []
    final_confs: list[float] = []

    for L in chosen_lines:
        clean_chars: list[str] = []
        for c in L.chars:
            out_ch = c.ch if c.conf >= min_conf else low_conf_char
            clean_chars.append(out_ch)
            final_pairs.append((out_ch, c.conf))
            final_confs.append(c.conf)

        # 🔥 IMPORTANTE: reconstruye texto con espacios igual que antes, pero con chars "limpios"
        line_text_clean = build_text_from_line(clean_chars, [cc.box for cc in L.chars])

        line_text_clean = _apply_case_heuristic(line_text_clean)
        final_text_lines.append(line_text_clean)

    final_text = "\n".join(final_text_lines).strip()
    global_conf = float(sum(final_confs) / max(1, len(final_confs)))

    return final_text, global_conf, final_pairs


def ocr_printed_text(
    image_path: str,
    debug_dir: str | None = None,
    min_box_area: int = 80,
    min_conf: float = 0.00,
    low_conf_char: str = "?",
    ignore_spaces: bool = False,
    cache_templates_path: str = "models/print_templates.npz",
) -> tuple[str, float, list[tuple[str, float]]]:
    """
    OCR de imprenta por templates (sin librerías OCR externas).
    """
    return ocr_printed(
        image_path=image_path,
        debug_dir=debug_dir or "outputs/debug_print_last",
        min_box_area=min_box_area,
        min_conf=min_conf,
        low_conf_char=low_conf_char,
        ignore_spaces=ignore_spaces,
        cache_templates_path=cache_templates_path,
    )


def ocr_auto(
    image_path: str,
    model_path: str,
    classes_path: str,
    debug_dir: str | None = None,
    force_cpu: bool = False,
    main_line_only: bool = True,
    min_box_area: int = 80,
    min_conf: float = 0.00,
    low_conf_char: str = "?",
    ignore_spaces: bool = False,
    cache_templates_path: str = "models/print_templates.npz",
) -> tuple[str, float, list[tuple[str, float]], str]:
    """
    Ejecuta PRINT + HANDWRITTEN y elige el que tenga mayor confianza global.
    Devuelve: (text, conf, pairs, chosen_mode)
    """
    dbg_root = Path(debug_dir or "outputs/debug_auto_last")
    dbg_root.mkdir(parents=True, exist_ok=True)

    text_p, conf_p, pairs_p = ocr_printed_text(
        image_path=image_path,
        debug_dir=str(dbg_root / "print"),
        min_box_area=min_box_area,
        min_conf=min_conf,
        low_conf_char=low_conf_char,
        ignore_spaces=ignore_spaces,
        cache_templates_path=cache_templates_path,
    )

    text_h, conf_h, pairs_h = ocr_handwritten(
        image_path=image_path,
        model_path=model_path,
        classes_path=classes_path,
        debug_dir=str(dbg_root / "handwritten"),
        force_cpu=force_cpu,
        main_line_only=main_line_only,
        min_box_area=min_box_area,
        min_conf=min_conf,
        low_conf_char=low_conf_char,
    )

    if conf_p >= conf_h:
        return text_p, conf_p, pairs_p, "print"
    return text_h, conf_h, pairs_h, "handwritten"
