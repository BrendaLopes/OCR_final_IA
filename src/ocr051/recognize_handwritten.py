from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


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


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_path: str, classes_path: str, device: torch.device) -> tuple[nn.Module, list[str]]:
    classes = json.loads(Path(classes_path).read_text(encoding="utf-8"))
    ckpt = torch.load(model_path, map_location="cpu")

    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model, classes


@torch.no_grad()
def predict_char_32(model: nn.Module, classes: list[str], img32: np.ndarray, device: torch.device) -> tuple[str, float]:
    """
    img32: uint8 (32x32), negro=tinta, blanco=fondo
    """
    x = img32.astype(np.float32) / 255.0
    x = 1.0 - x  # tinta->1
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,32,32]

    logits = model(t)
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    return classes[idx], float(probs[idx].item())
