#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

# Para poder importar src/ sin instalar paquete
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ocr051.pipeline import ocr_handwritten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Ruta de la foto (jpg/png/...)")
    ap.add_argument("--mode", default="handwritten", choices=["handwritten"], help="Modo OCR")
    ap.add_argument("--model", default="models/handwritten_char_cnn_v3/model.pt")
    ap.add_argument("--classes", default="models/handwritten_char_cnn_v3/classes.json")
    ap.add_argument("--debug-dir", default="outputs/debug_last", help="Carpeta para guardar pasos intermedios")
    ap.add_argument("--cpu", action="store_true", help="Forzar CPU")
    args = ap.parse_args()

    if args.mode == "handwritten":
        text = ocr_handwritten(
            image_path=args.image,
            model_path=args.model,
            classes_path=args.classes,
            debug_dir=args.debug_dir,
            force_cpu=args.cpu,
        )
    else:
        raise SystemExit("Modo no implementado")

    print("\n=== OCR RESULT ===")
    print(text)
    print("==================\n")
    print("Debug guardado en:", args.debug_dir)


if __name__ == "__main__":
    main()
