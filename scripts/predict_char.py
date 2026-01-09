##scripts/predict_char.py
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import cv2
except Exception as e:
    raise SystemExit("Missing dependency: opencv-python (cv2).") from e


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


def load_image_32(img_path: Path) -> torch.Tensor:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    if img.shape != (32, 32):
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

    x = img.astype(np.float32) / 255.0
    x = 1.0 - x  # ink -> 1, background -> 0
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # [1,1,32,32]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to a char image (32x32 recommended)")
    ap.add_argument("--model", default="models/handwritten_char_cnn/model.pt")
    ap.add_argument("--classes", default="models/handwritten_char_cnn/classes.json")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    classes = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.cpu else "cpu")

    ckpt = torch.load(args.model, map_location="cpu")
    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)

    x = load_image_32(Path(args.image)).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = np.argsort(-probs)[: args.topk]
    print("Top predictions:")
    for i in top_idx:
        print(f"  {classes[i]}  prob={probs[i]:.4f}")


if __name__ == "__main__":
    main()
