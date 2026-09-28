"""Shared helpers: COCO loading, scale buckets, matching."""
import json
from pathlib import Path

SCALE_BUCKETS = [(0, 16, "tiny(<16px)"), (16, 32, "small(16-32)"), (32, 96, "med(32-96)"), (96, 1e9, "large(>96)")]


def bucket_of(box_h: float) -> str:
    for lo, hi, name in SCALE_BUCKETS:
        if lo <= box_h < hi:
            return name
    return SCALE_BUCKETS[-1][2]


def load_coco(ann_path: str | Path):
    with open(ann_path) as f:
        c = json.load(f)
    imgs = {im["id"]: im for im in c["images"]}
    cats = {ct["id"]: ct["name"] for ct in c["categories"]}
    anns_by_img: dict[int, list] = {i: [] for i in imgs}
    for a in c["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    return c, imgs, cats, anns_by_img


def iou(a, b):
    """xywh boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0
