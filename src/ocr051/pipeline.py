from __future__ import annotations

from pathlib import Path
import cv2

from .preprocess import preprocess_page, normalize_char_to_32
from .segment import find_char_boxes, group_lines, build_text_from_line
from .recognize_handwritten import pick_device, load_model, predict_char_32
from .io_utils import ensure_dir, save_image, draw_boxes


def ocr_handwritten(
    image_path: str,
    model_path: str,
    classes_path: str,
    debug_dir: str | None = None,
    force_cpu: bool = False,
) -> str:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"No puedo leer la imagen: {image_path}")

    prep = preprocess_page(bgr)
    boxes = find_char_boxes(prep.mask)
    lines = group_lines(boxes)

    device = pick_device(force_cpu=force_cpu)
    model, classes = load_model(model_path, classes_path, device)

    if debug_dir:
        d = ensure_dir(debug_dir)
        save_image(d / "01_gray.png", prep.gray)
        save_image(d / "02_mask.png", prep.mask)
        save_image(d / "03_boxes.png", draw_boxes(bgr, boxes))

    final_lines = []
    crop_idx = 0

    for li, line in enumerate(lines):
        chars = []
        for (x, y, w, h) in line:
            gray_crop = prep.gray[y:y + h, x:x + w]
            mask_crop = prep.mask[y:y + h, x:x + w]

            img32 = normalize_char_to_32(gray_crop, mask_crop)
            ch, p = predict_char_32(model, classes, img32, device)
            chars.append(ch)

            if debug_dir:
                d = Path(debug_dir)
                save_image(d / f"crop_{crop_idx:03d}_{ch}_{p:.2f}.png", img32)
            crop_idx += 1

        line_text = build_text_from_line(chars, line)
        final_lines.append(line_text)

    return "\n".join(final_lines).strip()
