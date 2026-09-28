"""Audit step 4: head<->helmet label-flip detection via DINOv2 + cleanlab.

1. Crop every GT head/helmet box (2x context padding, min 32px resize).
2. Embed with DINOv2 ViT-S (timm, frozen).
3. 5-fold logistic regression -> out-of-fold probs.
4. cleanlab find_label_issues -> flagged crops, per split/scale bucket.
5. Contact sheets of flagged crops for human review -> runs/audit/flags/.

Usage: python src/audit/crop_confusion.py data/dataset_root
Writes runs/audit/crop_confusion.json + flag sheets.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
from cleanlab.filter import find_label_issues
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bucket_of, load_coco

SPLITS = ["train", "valid", "test"]
PAD = 0.5  # context padding fraction per side
BATCH = 256


def crop_box(im: Image.Image, bbox):
    x, y, w, h = bbox
    px, py = w * PAD, h * PAD
    return im.crop((max(0, x - px), max(0, y - py),
                    min(im.width, x + w + px), min(im.height, y + h + py)))


@torch.no_grad()
def embed(model, tfm, crops, device):
    feats = []
    for i in range(0, len(crops), BATCH):
        batch = torch.stack([tfm(c) for c in crops[i:i + BATCH]]).to(device)
        feats.append(model(batch).cpu().numpy())
        print(f"  embedded {i + len(batch)}/{len(crops)}", flush=True)
    return np.concatenate(feats)


def main(root: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0).eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    tfm = timm.data.create_transform(**cfg, is_training=False)

    meta, crops = [], []
    for split in SPLITS:
        ann = root / split / "_annotations.coco.json"
        if not ann.exists():
            continue
        _, imgs, cats, anns_by_img = load_coco(ann)
        for iid, anns in anns_by_img.items():
            keep = [a for a in anns if cats[a["category_id"]].lower() in ("head", "helmet")]
            if not keep:
                continue
            im = Image.open(root / split / imgs[iid]["file_name"]).convert("RGB")
            for a in keep:
                crops.append(crop_box(im, a["bbox"]))
                meta.append({"split": split, "img": imgs[iid]["file_name"], "ann_id": a["id"],
                             "label": cats[a["category_id"]].lower(),
                             "h": a["bbox"][3], "bucket": bucket_of(a["bbox"][3])})
    print(f"{len(crops)} head/helmet crops")

    X = embed(model, tfm, crops, device)
    y = np.array([0 if m["label"] == "head" else 1 for m in meta])
    probs = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), X, y, cv=5,
                              method="predict_proba", n_jobs=4)
    issues = find_label_issues(y, probs, return_indices_ranked_by="self_confidence")

    flagged = [dict(meta[i], prob_helmet=round(float(probs[i, 1]), 3), rank=r)
               for r, i in enumerate(issues.tolist())]
    by_bucket = Counter((meta[i]["label"], meta[i]["bucket"]) for i in issues)
    totals = Counter((m["label"], m["bucket"]) for m in meta)
    rates = {f"{lbl}|{bkt}": {"flagged": by_bucket[(lbl, bkt)], "total": t,
                              "rate": round(by_bucket[(lbl, bkt)] / t, 4)}
             for (lbl, bkt), t in sorted(totals.items())}

    out_dir = Path("runs/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crop_confusion.json").write_text(json.dumps(
        {"n_crops": len(crops), "n_flagged": len(flagged),
         "flag_rate": round(len(flagged) / len(crops), 4),
         "rates_by_label_bucket": rates, "flagged": flagged}, indent=2))
    print(f"flagged {len(flagged)}/{len(crops)}; rates: {json.dumps(rates, indent=1)}")

    # contact sheets: 10x10 grids of flagged crops, filename = claimed label
    sheets = Path("runs/audit/flags")
    sheets.mkdir(parents=True, exist_ok=True)
    cell = 96
    for start in range(0, min(len(issues), 600), 100):
        grid = Image.new("RGB", (cell * 10, cell * 10), "white")
        for k, idx in enumerate(issues[start:start + 100].tolist()):
            c = crops[idx].resize((cell, cell))
            grid.paste(c, ((k % 10) * cell, (k // 10) * cell))
        grid.save(sheets / f"flags_{start:04d}_labels.png")
        (sheets / f"flags_{start:04d}_labels.json").write_text(json.dumps(
            [{"pos": k, **{kk: flagged_i[kk] for kk in ("label", "prob_helmet", "img", "split", "ann_id", "bucket")}}
             for k, flagged_i in enumerate(flagged[start:start + 100])], indent=1))
    print(f"contact sheets -> {sheets}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
