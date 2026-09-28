"""Build the missed-annotation adjudication queue for the test split.

Takes union of high-confidence detections from multiple predictors, keeps those
unmatched by any GT head/helmet box (IoU < 0.3), cross-source NMS, renders zoomed
review panels (numbered) for human adjudication.

Usage:
  python src/audit/make_addition_queue.py data/variants/arm1/test \
      --preds runs/arm1/preds_test_regions.json runs/arm2_det/preds_test_regions.json \
      --conf 0.5 --out runs/audit/additions
Preds format: [{"img", "bbox":[x,y,w,h], "score"}] (class-agnostic region proposals).
Writes queue.json + panel_*.png (16 per panel).
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import iou, load_coco

CELL, CAP = 224, 18
PAD = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir", type=Path)
    ap.add_argument("--preds", nargs="+", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    _, imgs, cats, anns_by_img = load_coco(a.split_dir / "_annotations.coco.json")
    gt_by_file = {}
    for iid, anns in anns_by_img.items():
        gt_by_file[imgs[iid]["file_name"]] = [x["bbox"] for x in anns]

    cands = {}  # img -> [(score, bbox)]
    for pf in a.preds:
        for p in json.loads(pf.read_text()):
            if p["score"] < a.conf:
                continue
            if any(iou(p["bbox"], g) >= 0.3 for g in gt_by_file.get(p["img"], [])):
                continue
            cands.setdefault(p["img"], []).append((p["score"], p["bbox"]))

    # cross-source NMS (IoU 0.5)
    queue = []
    for img, lst in sorted(cands.items()):
        lst.sort(key=lambda x: -x[0])
        kept = []
        for s, b in lst:
            if all(iou(b, kb) < 0.5 for _, kb in kept):
                kept.append((s, b))
        for s, b in kept:
            queue.append({"idx": len(queue), "img": img, "bbox": [round(v, 1) for v in b],
                          "score": round(s, 3)})

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "queue.json").write_text(json.dumps(queue, indent=1))
    print(f"{len(queue)} unmatched proposals across {len(cands)} images")

    for start in range(0, len(queue), 16):
        batch = queue[start:start + 16]
        grid = Image.new("RGB", (CELL * 4, (CELL + CAP) * 4), "white")
        draw = ImageDraw.Draw(grid)
        for k, q in enumerate(batch):
            im = Image.open(a.split_dir / q["img"]).convert("RGB")
            x, y, w, h = q["bbox"]
            px, py = w * PAD, h * PAD
            crop = im.crop((max(0, x - px), max(0, y - py),
                            min(im.width, x + w + px), min(im.height, y + h + py)))
            sx, sy = CELL / crop.width, CELL / crop.height
            crop = crop.resize((CELL, CELL), Image.LANCZOS)
            gx, gy = (k % 4) * CELL, (k // 4) * (CELL + CAP)
            grid.paste(crop, (gx, gy))
            bx = (x - max(0, x - px)) * sx
            by = (y - max(0, y - py)) * sy
            draw.rectangle([gx + bx, gy + by, gx + bx + w * sx, gy + by + h * sy],
                           outline="lime", width=2)
            draw.text((gx + 3, gy + CELL + 2),
                      f"#{q['idx']} s={q['score']} {int(h)}px", fill="black")
        grid.save(a.out / f"panel_{start:03d}.png")
    print(f"panels -> {a.out}")


if __name__ == "__main__":
    main()
