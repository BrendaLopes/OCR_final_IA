#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def norm_text(s: str, ignore_spaces: bool) -> str:
    s = (s or "").strip()
    if ignore_spaces:
        s = re.sub(r"\s+", "", s)
    return s


def cer(gt: str, pred: str) -> float:
    """Character Error Rate = edit_distance / len(gt)"""
    gt = gt or ""
    pred = pred or ""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    a, b = gt, pred
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # delete
                dp[j - 1] + 1,  # insert
                prev + cost     # substitute
            )
            prev = cur
    return dp[m] / max(1, n)


@dataclass
class Sample:
    sample_id: str   # e.g., t01
    gt: str          # e.g., BUENAS
    filename: str    # e.g., t01.jpg OR t01 (sin extensión)


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_set = set(cols)
    for c in candidates:
        if c in cols_set:
            return c
    return None


def read_samples_csv(csv_path: Path) -> List[Sample]:
    """
    Acepta:
      - id, filename, gt (o equivalentes)
      - file, gt  (como el tuyo)
    Si no hay id -> genera t01, t02...
    """
    out: List[Sample] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"CSV vacío o sin cabecera: {csv_path}")

        cols = [c.strip() for c in reader.fieldnames]

        id_col = _pick_col(cols, ["id", "sample_id", "sample", "name"])
        file_col = _pick_col(cols, ["filename", "file", "image", "img", "path"])
        gt_col = _pick_col(cols, ["gt", "text", "label", "truth", "target"])

        if not file_col or not gt_col:
            raise SystemExit(
                "El CSV debe tener al menos columnas file,gt (o equivalentes).\n"
                f"Cabeceras encontradas: {cols}"
            )

        for i, r in enumerate(reader, start=1):
            sid = (r.get(id_col, "") if id_col else "").strip()
            if sid == "":
                sid = f"t{i:02d}"

            fn = str(r.get(file_col, "")).strip()
            gt = str(r.get(gt_col, "")).strip()

            if fn == "":
                raise SystemExit(f"Fila {i}: filename/file vacío en {csv_path}")

            out.append(Sample(sample_id=sid, filename=fn, gt=gt))

    return out


def resolve_image_path(samples_dir: Path, filename: str) -> Path:
    """
    Si filename tiene extensión -> usa directo.
    Si no tiene -> prueba extensiones típicas.
    """
    p = samples_dir / filename

    # Si ya existe tal cual, perfecto
    if p.exists():
        return p

    # Si no tiene extensión, probamos con extensiones
    if Path(filename).suffix == "":
        for ext in IMG_EXTS:
            cand = samples_dir / (filename + ext)
            if cand.exists():
                return cand

    # Si tiene extensión pero no existe, también probamos otras por si acaso
    stem = Path(filename).stem
    for ext in IMG_EXTS:
        cand = samples_dir / (stem + ext)
        if cand.exists():
            return cand

    raise FileNotFoundError(f"No existe sample: {p} (ni con extensiones {IMG_EXTS})")


def parse_run_ocr_output(stdout: str) -> Tuple[str, float]:
    """
    Espera lines tipo:
      Texto: BUENAS
      Confianza global: 96.5%
    """
    text = ""
    conf = 0.0

    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("Texto:"):
            text = line.split("Texto:", 1)[1].strip()
        elif line.startswith("Confianza global:"):
            v = line.split("Confianza global:", 1)[1].strip().replace("%", "")
            try:
                conf = float(v) / 100.0
            except ValueError:
                conf = 0.0

    return text, conf


def run_one(
    image_path: Path,
    model_dir: Path,
    min_box_area: int,
    ignore_spaces: bool,
    debug_dir: Path,
) -> Tuple[str, float]:
    cmd = [
        "python3",
        "scripts/run_ocr.py",
        "--image", str(image_path),
        "--min-box-area", str(min_box_area),
        "--handwritten-model-dir", str(model_dir),
        "--debug-dir", str(debug_dir),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Error running OCR:\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{p.stdout}\n"
            f"STDERR:\n{p.stderr}"
        )

    pred_text, conf = parse_run_ocr_output(p.stdout)
    pred_text = norm_text(pred_text, ignore_spaces)
    return pred_text, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-csv", required=True, help="CSV con columnas file,gt (id opcional)")
    ap.add_argument("--samples-dir", required=True, help="Carpeta donde están las imágenes")
    ap.add_argument("--min-box-area", type=int, default=120)
    ap.add_argument("--ignore-spaces", action="store_true")
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Lista de carpetas de modelo (cada una con model.pt y classes.json)",
    )
    ap.add_argument("--out-csv", default="outputs/model_compare_samples.csv")
    ap.add_argument("--debug-root", default="outputs/compare_debug")
    args = ap.parse_args()

    samples_csv = Path(args.samples_csv)
    samples_dir = Path(args.samples_dir)
    out_csv = Path(args.out_csv)
    debug_root = Path(args.debug_root)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)

    samples = read_samples_csv(samples_csv)

    results_rows: List[Dict[str, str]] = []

    for model_dir_str in args.models:
        model_dir = Path(model_dir_str)
        model_name = model_dir.name

        print(f"\n=== Testing model: {model_name} ({model_dir}) ===")

        exact_ok = 0
        cer_list: List[float] = []
        conf_list: List[float] = []

        for s in samples:
            img_path = resolve_image_path(samples_dir, s.filename)

            gt_norm = norm_text(s.gt, args.ignore_spaces)

            dbg = debug_root / model_name / s.sample_id
            dbg.mkdir(parents=True, exist_ok=True)

            pred_norm, conf = run_one(
                image_path=img_path,
                model_dir=model_dir,
                min_box_area=args.min_box_area,
                ignore_spaces=args.ignore_spaces,
                debug_dir=dbg,
            )

            ok = int(pred_norm == gt_norm)
            exact_ok += ok

            c = cer(gt_norm, pred_norm)
            cer_list.append(c)
            conf_list.append(conf)

            print(f"  {s.sample_id}: GT='{gt_norm}'  PRED='{pred_norm}'  ok={ok}  CER={c:.3f}")

            results_rows.append({
                "model": model_name,
                "sample_id": s.sample_id,
                "gt": gt_norm,
                "pred": pred_norm,
                "ok": str(ok),
                "cer": f"{c:.6f}",
                "conf": f"{conf:.6f}",
                "image": str(img_path),
                "debug_dir": str(dbg),
            })

        acc = exact_ok / max(1, len(samples))
        avg_cer = sum(cer_list) / max(1, len(cer_list))
        avg_conf = sum(conf_list) / max(1, len(conf_list))

        print(f"\n[SUMMARY {model_name}]")
        print(f"  Samples: {len(samples)}")
        print(f"  Exact (norm) acc: {acc*100:.1f}%")
        print(f"  Avg CER: {avg_cer*100:.1f}%")
        print(f"  Avg global conf: {avg_conf*100:.1f}%")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "sample_id", "gt", "pred", "ok", "cer", "conf", "image", "debug_dir"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results_rows)

    print(f"\nSaved report CSV: {out_csv}")
    print(f"Debug dirs: {debug_root}/<model>/<sample>/\n")


if __name__ == "__main__":
    main()
