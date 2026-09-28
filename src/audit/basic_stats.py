"""Audit step 1: dataset composition stats per split/class/scale bucket.

Usage: python src/audit/basic_stats.py data/dataset_root
Expects <root>/{train,valid,test}/_annotations.coco.json (Roboflow COCO export).
Writes runs/audit/basic_stats.json and prints a markdown summary.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bucket_of, load_coco

SPLITS = ["train", "valid", "test"]


def main(root: Path):
    out = {}
    for split in SPLITS:
        ann = root / split / "_annotations.coco.json"
        if not ann.exists():
            print(f"!! missing {ann}")
            continue
        _, imgs, cats, anns_by_img = load_coco(ann)
        cls_count = Counter()
        cls_bucket = defaultdict(Counter)
        boxes_per_img = []
        img_sizes = Counter()
        degenerate = 0
        for im in imgs.values():
            img_sizes[(im["width"], im["height"])] += 1
        for img_id, anns in anns_by_img.items():
            boxes_per_img.append(len(anns))
            for a in anns:
                name = cats[a["category_id"]]
                w, h = a["bbox"][2], a["bbox"][3]
                if w <= 1 or h <= 1:
                    degenerate += 1
                cls_count[name] += 1
                cls_bucket[name][bucket_of(h)] += 1
        n = len(imgs)
        out[split] = {
            "images": n,
            "boxes": sum(cls_count.values()),
            "class_counts": dict(cls_count),
            "class_by_scale": {k: dict(v) for k, v in cls_bucket.items()},
            "imgs_with_zero_boxes": sum(1 for b in boxes_per_img if b == 0),
            "mean_boxes_per_img": round(sum(boxes_per_img) / max(n, 1), 2),
            "degenerate_boxes": degenerate,
            "img_size_top5": {f"{w}x{h}": c for (w, h), c in img_sizes.most_common(5)},
        }
    dst = Path("runs/audit/basic_stats.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    # markdown summary
    for split, s in out.items():
        print(f"\n## {split}: {s['images']} imgs, {s['boxes']} boxes, "
              f"{s['imgs_with_zero_boxes']} empty imgs, {s['degenerate_boxes']} degenerate")
        print(f"classes: {s['class_counts']}")
        for cls, buckets in s["class_by_scale"].items():
            print(f"  {cls}: {buckets}")
        print(f"  top sizes: {s['img_size_top5']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
