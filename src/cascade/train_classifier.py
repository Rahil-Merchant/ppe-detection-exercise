"""Arm 2 stage 2: bare-head vs helmet crop classifier (DINOv2 ViT-S backbone).

Crop source: GT head/helmet boxes from the arm1 (cleaned) train/valid variant,
plus optional detector-proposal crops (jittered boxes) to match deploy distribution.
Backbone frozen by default (linear probe); --unfreeze for full fine-tune.

Usage:
  python src/cascade/train_classifier.py data/variants/arm1 runs/arm2_cls --epochs 10
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_coco

JITTER = 0.15  # box jitter fraction to simulate detector proposals


def crop_padded(im, bbox, jitter=0.0, pad=0.5):
    x, y, w, h = bbox
    if jitter:
        x += random.uniform(-jitter, jitter) * w
        y += random.uniform(-jitter, jitter) * h
        w *= 1 + random.uniform(-jitter, jitter)
        h *= 1 + random.uniform(-jitter, jitter)
    px, py = w * pad, h * pad
    return im.crop((max(0, x - px), max(0, y - py),
                    min(im.width, x + w + px), min(im.height, y + h + py)))


class CropDS(Dataset):
    def __init__(self, root: Path, split: str, tfm, train: bool, pad: float = 0.5):
        self.samples = []  # (img_path, bbox, label)
        self.tfm, self.train, self.pad = tfm, train, pad
        _, imgs, cats, anns_by_img = load_coco(root / split / "_annotations.coco.json")
        for iid, anns in anns_by_img.items():
            for a in anns:
                name = cats[a["category_id"]].lower()
                if name in ("head", "helmet"):
                    self.samples.append((root / split / imgs[iid]["file_name"],
                                         a["bbox"], 0 if name == "head" else 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, bbox, y = self.samples[i]
        im = Image.open(p).convert("RGB")
        crop = crop_padded(im, bbox, JITTER if self.train else 0.0, self.pad)
        return self.tfm(crop), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--unfreeze", action="store_true")
    ap.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--pad", type=float, default=0.5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(vars(a), default=str, indent=2))

    device = "cuda"
    model = timm.create_model(a.model, pretrained=True, num_classes=2).to(device)
    if not a.unfreeze:
        for n, p in model.named_parameters():
            if "head" not in n:  # timm classifier attr is `head`
                p.requires_grad = False
    cfg = timm.data.resolve_model_data_config(model)
    tfm_train = timm.data.create_transform(**cfg, is_training=True)
    tfm_eval = timm.data.create_transform(**cfg, is_training=False)

    dl_tr = DataLoader(CropDS(a.dataset, "train", tfm_train, True, a.pad), batch_size=a.batch,
                       shuffle=True, num_workers=8, pin_memory=True)
    dl_va = DataLoader(CropDS(a.dataset, "valid", tfm_eval, False, a.pad), batch_size=a.batch,
                       num_workers=8, pin_memory=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = nn.CrossEntropyLoss()
    best = 0.0
    log = []
    for ep in range(a.epochs):
        model.train()
        tr_loss = 0.0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(y)
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dl_va:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item()
                total += len(y)
        acc = correct / total
        log.append({"epoch": ep, "train_loss": round(tr_loss / len(dl_tr.dataset), 4),
                    "val_acc": round(acc, 4)})
        print(log[-1], flush=True)
        if acc > best:
            best = acc
            torch.save({"model": a.model, "state_dict": model.state_dict()},
                       a.out / "best.pth")
    (a.out / "log.json").write_text(json.dumps(log, indent=2))
    print(f"best val acc {best:.4f}")


if __name__ == "__main__":
    main()
