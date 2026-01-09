##scripts/qa_hard_examples.py
#!/usr/bin/env python3
import argparse
import json
import random
import subprocess
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def pick_device(force_cpu: bool = False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise SystemExit("Expected checkpoint dict with key 'model_state'.")
    return ckpt


def load_classes(classes_path: str | None, ckpt: dict):
    if classes_path:
        return json.loads(Path(classes_path).read_text(encoding="utf-8"))
    if "classes" in ckpt:
        return ckpt["classes"]
    raise SystemExit("No classes found. Provide --classes or ensure checkpoint contains 'classes'.")


def build_model(ckpt: dict, classes: list[str], device: torch.device):
    model = SimpleCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_image_tensor(img_path: str, device: torch.device):
    # images are already 32x32, but keep safe
    img = Image.open(img_path).convert("L").resize((32, 32))
    arr = torch.from_numpy((1.0 - (torch.tensor(list(img.getdata()), dtype=torch.float32) / 255.0)).numpy())
    x = arr.view(1, 32, 32).unsqueeze(0).to(device)  # (1,1,32,32)
    return x


@torch.no_grad()
def predict_topk(model, classes, x, k=5):
    logits = model(x).squeeze(0)
    probs = F.softmax(logits, dim=0)
    vals, idxs = torch.topk(probs, k=min(k, probs.numel()))
    out = []
    for v, i in zip(vals, idxs):
        out.append((classes[int(i.item())], float(v.item())))
    return out


def open_image(path: str):
    subprocess.run(["open", path], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/processed/handwritten_chars/labels.csv")
    ap.add_argument("--model", default="models/handwritten_char_cnn/model.pt")
    ap.add_argument("--classes", default=None, help="Optional path to classes.json (otherwise uses checkpoint)")
    ap.add_argument("--k", type=int, default=20, help="How many examples to show")
    ap.add_argument("--topk", type=int, default=5, help="Top-k predictions to print")
    ap.add_argument("--only-wrong", action="store_true", help="Show only misclassified examples")
    ap.add_argument("--per-class", action="store_true", help="Show k hardest per class (slower)")
    ap.add_argument("--open", action="store_true", help="Open each image (macOS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)

    device = pick_device(force_cpu=args.cpu)
    ckpt = load_checkpoint(args.model)
    classes = load_classes(args.classes, ckpt)
    model = build_model(ckpt, classes, device)

    df = pd.read_csv(args.labels)
    if not {"filename", "label"}.issubset(df.columns):
        raise SystemExit("labels.csv must contain columns: filename,label")

    base_dir = Path(args.labels).parent
    df["img_path"] = df["filename"].apply(lambda p: str((base_dir / p).resolve()))

    # Score each sample by p(gt)
    scored = []
    for _, r in df.iterrows():
        gt = str(r["label"])
        img_path = r["img_path"]

        x = load_image_tensor(img_path, device)
        logits = model(x).squeeze(0)
        probs = F.softmax(logits, dim=0)

        pred_idx = int(torch.argmax(probs).item())
        pred = classes[pred_idx]
        p_pred = float(probs[pred_idx].item())

        gt_idx = classes.index(gt) if gt in classes else None
        p_gt = float(probs[gt_idx].item()) if gt_idx is not None else 0.0

        ok = (pred == gt)
        scored.append((gt, pred, ok, p_gt, p_pred, img_path))

    # sort by hardness: lowest p(gt) first
    scored.sort(key=lambda t: t[3])

    def show_one(t):
        gt, pred, ok, p_gt, p_pred, img_path = t
        print("=" * 42)
        print(f"GT (real): {gt}")
        print(f"PRED:     {pred}   ok={ok}   p(gt)={p_gt:.4f}  p(pred)={p_pred:.4f}")
        print(f"IMG: {img_path}")
        top = predict_topk(model, classes, load_image_tensor(img_path, device), k=args.topk)
        print("Top predictions:")
        for c, p in top:
            print(f"  {c:>2}  prob={p:.4f}")
        if args.open:
            open_image(img_path)
        input("Pulsa una tecla para el siguiente...")

    if args.per_class:
        by_class = {}
        for t in scored:
            by_class.setdefault(t[0], []).append(t)

        for gt in sorted(by_class.keys()):
            shown = 0
            for t in by_class[gt]:
                if args.only_wrong and t[2]:
                    continue
                show_one(t)
                shown += 1
                if shown >= args.k:
                    break
    else:
        shown = 0
        for t in scored:
            if args.only_wrong and t[2]:
                continue
            show_one(t)
            shown += 1
            if shown >= args.k:
                break


if __name__ == "__main__":
    main()
