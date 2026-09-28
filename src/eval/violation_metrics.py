"""Violation-level comparison: missed-violation rate vs false-alarm rate.

GT violation = `head` box in trusted test annotations (category name 'head').
Prediction file: JSON list of {"img": file_name, "bbox": [x,y,w,h], "score": float}.
Match: IoU >= 0.3 (alert localization, not tight box; heads are 15-30px).

For each threshold: TP = alert matched to an unclaimed GT head; FP = unmatched alert;
miss = GT head with no alert. Reports curve + operating points at FA budgets.

Usage:
  python src/eval/violation_metrics.py data/variants/arm1/test/_annotations.coco.json \
      --preds runs/arm1/preds_test.json --name arm1 [--out runs/compare]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bucket_of, iou, load_coco

IOU_MATCH = 0.3
FA_BUDGETS = [0.5, 1.0, 2.0]  # false alarms per 100 images


def evaluate(gt_heads, preds_by_img, n_images, thresholds):
    """gt_heads: {img: [bbox,...]}; preds_by_img: {img: [(score,bbox),...]}."""
    rows = []
    for t in thresholds:
        tp = fp = 0
        matched_total = 0
        n_gt = sum(len(v) for v in gt_heads.values())
        miss_by_bucket = defaultdict(lambda: [0, 0])  # bucket -> [missed, total]
        for img, gts in gt_heads.items():
            alerts = [pb for s, pb in sorted(preds_by_img.get(img, []),
                                             key=lambda sp: -sp[0]) if s >= t]
            claimed = [False] * len(gts)
            for ab in alerts:
                best, best_iou = -1, IOU_MATCH
                for gi, gb in enumerate(gts):
                    if not claimed[gi] and iou(ab, gb) >= best_iou:
                        best, best_iou = gi, iou(ab, gb)
                if best >= 0:
                    claimed[best] = True
                    tp += 1
                else:
                    fp += 1
            matched_total += sum(claimed)
            for gi, gb in enumerate(gts):
                b = bucket_of(gb[3])
                miss_by_bucket[b][1] += 1
                if not claimed[gi]:
                    miss_by_bucket[b][0] += 1
        # FPs on images that contain no GT heads at all
        for img, ps in preds_by_img.items():
            if img not in gt_heads:
                fp += sum(1 for s, _ in ps if s >= t)
        miss_rate = 1 - matched_total / max(n_gt, 1)
        fa_per_100 = 100 * fp / max(n_images, 1)
        rows.append({"threshold": round(float(t), 4), "tp": tp, "fp": fp,
                     "miss_rate": round(miss_rate, 4), "fa_per_100_imgs": round(fa_per_100, 3),
                     "precision": round(tp / max(tp + fp, 1), 4),
                     "miss_by_bucket": {k: {"missed": v[0], "total": v[1],
                                            "rate": round(v[0] / max(v[1], 1), 4)}
                                        for k, v in sorted(miss_by_bucket.items())}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", type=Path)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/compare"))
    ap.add_argument("--exclude", type=Path, default=None,
                    help="JSON list of 'test/<file>' names to drop (leaky images)")
    a = ap.parse_args()

    _, imgs, cats, anns_by_img = load_coco(a.gt)
    if a.exclude and a.exclude.exists():
        drop = {Path(x).name for x in json.loads(a.exclude.read_text())}
        removed_ids = [iid for iid, im in imgs.items() if im["file_name"] in drop]
        for iid in removed_ids:
            anns_by_img.pop(iid, None)
            imgs.pop(iid)
        print(f"excluded {len(removed_ids)} leaky images")
    head_ids = [cid for cid, n in cats.items() if n.lower() == "head"]
    gt_heads = {}
    for iid, anns in anns_by_img.items():
        boxes = [x["bbox"] for x in anns if x["category_id"] in head_ids]
        if boxes:
            gt_heads[imgs[iid]["file_name"]] = boxes

    kept_files = {im["file_name"] for im in imgs.values()}
    preds = json.loads(a.preds.read_text())
    preds_by_img = defaultdict(list)
    for p in preds:
        if p["img"] in kept_files:  # drop preds on excluded (leaky) images
            preds_by_img[p["img"]].append((p["score"], p["bbox"]))

    thresholds = np.round(np.arange(0.05, 1.0, 0.025), 4)
    rows = evaluate(gt_heads, dict(preds_by_img), len(imgs), thresholds)

    ops = {}
    for budget in FA_BUDGETS:
        ok = [r for r in rows if r["fa_per_100_imgs"] <= budget]
        ops[f"fa<={budget}/100"] = min(ok, key=lambda r: r["miss_rate"]) if ok else None

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.name}_violation_eval.json").write_text(json.dumps(
        {"n_images": len(imgs), "n_gt_violations": sum(len(v) for v in gt_heads.values()),
         "iou_match": IOU_MATCH, "curve": rows, "operating_points": ops}, indent=2))
    print(f"{a.name}: {sum(len(v) for v in gt_heads.values())} GT violations / {len(imgs)} imgs")
    for k, r in ops.items():
        if r:
            print(f"  {k}: miss={r['miss_rate']:.3f} fa/100={r['fa_per_100_imgs']:.2f} thr={r['threshold']}")
        else:
            print(f"  {k}: unreachable")


if __name__ == "__main__":
    main()
