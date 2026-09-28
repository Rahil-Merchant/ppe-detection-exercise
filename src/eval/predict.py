"""Generate violation predictions on a split for either arm.

Arm 1 (direct):  --arm direct  --det runs/arm1/checkpoint_best_total.pth
    violation score = confidence of `head` class detection.
Arm 2 (cascade): --arm cascade --det runs/arm2_det/checkpoint_best_total.pth --cls runs/arm2_cls/best.pth
    violation score = det_conf * P(bare_head | crop).

Output: JSON list [{"img", "bbox":[x,y,w,h], "score"}] with a low floor threshold
(0.05) so the eval script can sweep thresholds itself.

Usage:
  python src/eval/predict.py data/variants/arm1/test --arm direct --det ... --out runs/arm1/preds_test.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import timm
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FLOOR = 0.05
# rfdetr emits 0-based class indices in category order:
# arm1 variant: 0=head, 1=helmet; arm2_det variant: 0=head_region
# verify per checkpoint with --probe before trusting (prints class-id histogram)


def load_detector(ckpt, res):
    from rfdetr import RFDETRSmall
    return RFDETRSmall(pretrain_weights=str(ckpt), resolution=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir", type=Path)
    ap.add_argument("--arm", choices=["direct", "cascade"], required=True)
    ap.add_argument("--det", type=Path, required=True)
    ap.add_argument("--cls", type=Path)
    ap.add_argument("--target-id", type=int, required=True,
                    help="0-based class index of the violation/region class in this checkpoint")
    ap.add_argument("--res", type=int, default=736)
    ap.add_argument("--pad", type=float, default=0.5, help="crop context padding, must match classifier training")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--regions-out", type=Path, default=None,
                    help="also dump ALL detections (any class) as class-agnostic proposals")
    a = ap.parse_args()

    det = load_detector(a.det, a.res)
    clf = tfm = None
    if a.arm == "cascade":
        ck = torch.load(a.cls, map_location="cuda", weights_only=False)
        clf = timm.create_model(ck["model"], pretrained=False, num_classes=2)
        clf.load_state_dict(ck["state_dict"])
        clf = clf.eval().cuda()
        cfg = timm.data.resolve_model_data_config(clf)
        tfm = timm.data.create_transform(**cfg, is_training=False)

    imgs = sorted([p for p in a.split_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    preds = []
    all_regions = []
    t0 = time.time()
    for i, p in enumerate(imgs):
        im = Image.open(p).convert("RGB")
        d = det.predict(im, threshold=FLOOR)
        boxes = [([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(s), int(c))
                 for (x1, y1, x2, y2), s, c in zip(d.xyxy, d.confidence, d.class_id)]
        all_regions.extend({"img": p.name, "bbox": bbox, "score": s, "cls": c}
                       for bbox, s, c in boxes)
        if a.arm == "direct":
            for bbox, s, c in boxes:
                if c == a.target_id:
                    preds.append({"img": p.name, "bbox": bbox, "score": s})
        else:
            regions = [(bbox, s) for bbox, s, c in boxes if c == a.target_id]
            if regions:
                crops = []
                for (x, y, w, h), _ in regions:
                    px, py = w * a.pad, h * a.pad
                    crops.append(tfm(im.crop((max(0, x - px), max(0, y - py),
                                              min(im.width, x + w + px),
                                              min(im.height, y + h + py)))))
                with torch.no_grad():
                    prob = torch.softmax(clf(torch.stack(crops).cuda()), dim=1)[:, 0]  # class 0 = bare head
                for (bbox, s), pb in zip(regions, prob.cpu().tolist()):
                    preds.append({"img": p.name, "bbox": bbox, "score": float(s * pb)})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(imgs)} ({(time.time() - t0) / (i + 1) * 1000:.0f} ms/img)", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(preds))
    if a.regions_out:
        a.regions_out.write_text(json.dumps(all_regions))
    dt = time.time() - t0
    print(f"{len(preds)} preds on {len(imgs)} imgs in {dt:.1f}s ({dt / len(imgs) * 1000:.0f} ms/img e2e)")


if __name__ == "__main__":
    main()
