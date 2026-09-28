"""Train RF-DETR (DINOv2 ViT backbone) on a dataset variant.

Usage:
  python src/train/train_rfdetr.py data/variants/arm1 runs/arm1 --model small --res 728 --epochs 30
"""
import argparse
import json
from pathlib import Path

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRNano, RFDETRSmall

MODELS = {"nano": RFDETRNano, "small": RFDETRSmall, "base": RFDETRBase, "large": RFDETRLarge}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="small", choices=MODELS)
    ap.add_argument("--res", type=int, default=728)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(vars(a), default=str, indent=2))

    model = MODELS[a.model](resolution=a.res)
    model.train(
        dataset_dir=str(a.dataset),
        output_dir=str(a.out),
        epochs=a.epochs,
        batch_size=a.batch,
        grad_accum_steps=a.grad_accum,
        lr=a.lr,
        early_stopping=True,
    )


if __name__ == "__main__":
    main()
