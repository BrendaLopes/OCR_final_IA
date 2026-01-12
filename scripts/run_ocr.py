#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Permite importar src/ sin instalar paquete
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ocr051.pipeline import ocr_handwritten, ocr_printed_text, ocr_auto


def resolve_model_paths(model_dir: str | None, model_path: str, classes_path: str) -> tuple[str, str]:
    """
    Prioridad:
      1) --handwritten-model-dir (carpeta con model.pt y classes.json)
      2) --model y --classes (paths directos)
    """
    if model_dir:
        d = Path(model_dir)
        return str(d / "model.pt"), str(d / "classes.json")
    return model_path, classes_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Ruta de la foto (jpg/png/...)")

    ap.add_argument("--mode", choices=["auto", "handwritten", "print"], default="auto",
                    help="auto = elige entre print y handwritten por confianza")

    ap.add_argument(
        "--handwritten-model-dir",
        default="models/handwritten_char_cnn",
        help="Carpeta del modelo manuscrito (model.pt + classes.json). "
             "Recomendado por tus tests: models/handwritten_char_cnn",
    )
    ap.add_argument("--model", default="models/handwritten_char_cnn/model.pt", help="Path a model.pt (fallback)")
    ap.add_argument("--classes", default="models/handwritten_char_cnn/classes.json", help="Path a classes.json (fallback)")

    ap.add_argument("--debug-dir", default="outputs/debug_last", help="Carpeta para guardar pasos intermedios")
    ap.add_argument("--cpu", action="store_true", help="Forzar CPU")
    ap.add_argument("--all-lines", action="store_true", help="Si lo activas, no filtra a una sola línea principal")
    ap.add_argument("--min-box-area", type=int, default=120, help="Filtra cajas pequeñas (ruido). Sube si hay suciedad.")
    ap.add_argument("--min-conf", type=float, default=0.00, help="Por debajo de esto, reemplaza por '?' (ej 0.30)")

    # print (templates)
    ap.add_argument("--print-template-cache", default="models/print_templates.npz",
                    help="Cache de templates para imprenta (se crea solo la 1ª vez)")
    ap.add_argument("--ignore-spaces", action="store_true",
                    help="Solo para pruebas (si lo activas, PRINT no inserta espacios)")

    args = ap.parse_args()

    img = Path(args.image)
    if not img.exists():
        raise FileNotFoundError(f"No existe imagen: {img}")

    model_path, classes_path = resolve_model_paths(args.handwritten_model_dir, args.model, args.classes)
    mp = Path(model_path)
    cp = Path(classes_path)

    if args.mode in ("auto", "handwritten"):
        if not mp.exists():
            raise FileNotFoundError(f"No existe model.pt: {mp}")
        if not cp.exists():
            raise FileNotFoundError(f"No existe classes.json: {cp}")

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Image: {img.resolve()}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Debug dir: {debug_dir.resolve()}")

    if args.mode == "handwritten":
        print(f"[INFO] Using handwritten model: {mp.resolve()}")
        print(f"[INFO] Using classes: {cp.resolve()}\n")

        text, conf, pairs = ocr_handwritten(
            image_path=str(img),
            model_path=str(mp),
            classes_path=str(cp),
            debug_dir=str(debug_dir / "handwritten"),
            force_cpu=args.cpu,
            main_line_only=not args.all_lines,
            min_box_area=args.min_box_area,
            min_conf=args.min_conf,
            low_conf_char="?",
        )
        title = "HANDWRITTEN"

    elif args.mode == "print":
        print(f"[INFO] Print template cache: {Path(args.print_template_cache).resolve()}\n")

        text, conf, pairs = ocr_printed_text(
            image_path=str(img),
            debug_dir=str(debug_dir / "print"),
            min_box_area=args.min_box_area,
            min_conf=args.min_conf,
            low_conf_char="?",
            ignore_spaces=args.ignore_spaces,
            cache_templates_path=args.print_template_cache,
        )
        title = "PRINT"

    else:
        print(f"[INFO] Using handwritten model: {mp.resolve()}")
        print(f"[INFO] Using classes: {cp.resolve()}")
        print(f"[INFO] Print template cache: {Path(args.print_template_cache).resolve()}\n")

        text, conf, pairs, chosen = ocr_auto(
            image_path=str(img),
            model_path=str(mp),
            classes_path=str(cp),
            debug_dir=str(debug_dir / "auto"),
            force_cpu=args.cpu,
            main_line_only=not args.all_lines,
            min_box_area=args.min_box_area,
            min_conf=args.min_conf,
            low_conf_char="?",
            ignore_spaces=args.ignore_spaces,
            cache_templates_path=args.print_template_cache,
        )
        title = f"AUTO -> {chosen.upper()}"

    print(f"=== OCR RESULT ({title}) ===")
    if text.strip() == "":
        print("(vacío) -> no se detectó texto")
    else:
        print("Texto:", text)
        print(f"Confianza global: {conf*100:.1f}%")
        if pairs:
            detail = " ".join([f"{ch}({p*100:.0f}%)" for ch, p in pairs])
            print("Detalle:", detail)
    print("===============================")
    print("Debug guardado en:", debug_dir)
    print()


if __name__ == "__main__":
    main()
