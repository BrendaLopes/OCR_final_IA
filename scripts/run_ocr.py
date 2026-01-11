#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ocr051.pipeline import ocr_handwritten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Ruta de la foto (jpg/png/...)")
    ap.add_argument("--model", default="models/handwritten_char_cnn_v3/model.pt")
    ap.add_argument("--classes", default="models/handwritten_char_cnn_v3/classes.json")
    ap.add_argument("--debug-dir", default="outputs/debug_last", help="Carpeta para guardar pasos intermedios")
    ap.add_argument("--cpu", action="store_true", help="Forzar CPU")
    ap.add_argument("--all-lines", action="store_true", help="Si lo activas, no filtra a una sola línea principal")
    ap.add_argument("--min-box-area", type=int, default=80, help="Filtra cajas pequeñas (ruido). Sube si hay suciedad.")
    ap.add_argument("--min-conf", type=float, default=0.00, help="Por debajo de esto, reemplaza por '?' (ej 0.30)")
    args = ap.parse_args()

    text, conf, pairs = ocr_handwritten(
        image_path=args.image,
        model_path=args.model,
        classes_path=args.classes,
        debug_dir=args.debug_dir,
        force_cpu=args.cpu,
        main_line_only=not args.all_lines,
        min_box_area=args.min_box_area,
        min_conf=args.min_conf,
        low_conf_char="?",
    )

    print("\n=== OCR RESULT (HANDWRITTEN) ===")
    if text.strip() == "":
        print("(vacío) -> no se detectó texto")
    else:
        print("Texto:", text)
        print(f"Confianza global: {conf*100:.1f}%")

        # por carácter (sin espacios)
        if pairs:
            detail = " ".join([f"{ch}({p*100:.0f}%)" for ch, p in pairs])
            print("Detalle:", detail)

    print("===============================")
    print("Debug guardado en:", args.debug_dir)
    print()


if __name__ == "__main__":
    main()
