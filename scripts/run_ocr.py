#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Permite importar src/ sin instalar paquete
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ocr051.pipeline import ocr_handwritten


def resolve_model_paths(model_dir: str | None, model_path: str, classes_path: str) -> tuple[str, str]:
    """
    Prioridad:
      1) --handwritten-model-dir (carpeta con model.pt y classes.json)
      2) --model y --classes (paths directos)
    """
    if model_dir:
        d = Path(model_dir)
        mp = d / "model.pt"
        cp = d / "classes.json"
        return str(mp), str(cp)
    return model_path, classes_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Ruta de la foto (jpg/png/...)")

    # ✅ opción limpia para elegir modelo por carpeta
    ap.add_argument(
        "--handwritten-model-dir",
        default=None,
        help="Carpeta del modelo manuscrito (debe contener model.pt y classes.json). "
             "Ej: models/handwritten_char_cnn_v3",
    )

    # fallback (compatibilidad con tu versión actual)
    ap.add_argument("--model", default="models/handwritten_char_cnn_v3/model.pt", help="Path a model.pt (fallback)")
    ap.add_argument("--classes", default="models/handwritten_char_cnn_v3/classes.json", help="Path a classes.json (fallback)")

    ap.add_argument("--debug-dir", default="outputs/debug_last", help="Carpeta para guardar pasos intermedios")
    ap.add_argument("--cpu", action="store_true", help="Forzar CPU")
    ap.add_argument("--all-lines", action="store_true", help="Si lo activas, no filtra a una sola línea principal")
    ap.add_argument("--min-box-area", type=int, default=80, help="Filtra cajas pequeñas (ruido). Sube si hay suciedad.")
    ap.add_argument("--min-conf", type=float, default=0.00, help="Por debajo de esto, reemplaza por '?' (ej 0.30)")
    args = ap.parse_args()

    model_path, classes_path = resolve_model_paths(args.handwritten_model_dir, args.model, args.classes)

    # Validaciones claras (para que el profe pueda ejecutar y entender errores)
    mp = Path(model_path)
    cp = Path(classes_path)
    if not mp.exists():
        raise FileNotFoundError(f"No existe model.pt: {mp}")
    if not cp.exists():
        raise FileNotFoundError(f"No existe classes.json: {cp}")

    # Asegura debug dir
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Image: {Path(args.image).resolve()}")
    print(f"[INFO] Using handwritten model: {mp.resolve()}")
    print(f"[INFO] Using classes: {cp.resolve()}")
    print(f"[INFO] Debug dir: {debug_dir.resolve()}\n")

    text, conf, pairs = ocr_handwritten(
        image_path=args.image,
        model_path=str(mp),
        classes_path=str(cp),
        debug_dir=str(debug_dir),
        force_cpu=args.cpu,
        main_line_only=not args.all_lines,
        min_box_area=args.min_box_area,
        min_conf=args.min_conf,
        low_conf_char="?",
    )

    print("=== OCR RESULT (HANDWRITTEN) ===")
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
