"""Render zoomed review panels for cleanlab-flagged crops.

4x4 grids, each cell 224px crop (2x context) + caption: index | claimed label | P(helmet) | split.
Usage: python src/audit/make_flag_panels.py data/raw runs/audit/crop_confusion.json runs/audit/panels
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

CELL, CAP = 224, 18
PAD = 0.75


def main(root: Path, conf_json: Path, out_dir: Path):
    flagged = json.loads(conf_json.read_text())["flagged"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # need ann bbox: reload from split anns
    bboxes = {}
    for split in ("train", "valid", "test"):
        ann = root / split / "_annotations.coco.json"
        if not ann.exists():
            continue
        coco = json.loads(ann.read_text())
        imgs = {im["id"]: im["file_name"] for im in coco["images"]}
        for a in coco["annotations"]:
            bboxes[(split, a["id"])] = (imgs[a["image_id"]], a["bbox"])

    per = 16
    for start in range(0, len(flagged), per):
        batch = flagged[start:start + per]
        grid = Image.new("RGB", (CELL * 4, (CELL + CAP) * 4), "white")
        draw = ImageDraw.Draw(grid)
        for k, f in enumerate(batch):
            fname, bbox = bboxes[(f["split"], f["ann_id"])]
            im = Image.open(root / f["split"] / fname).convert("RGB")
            x, y, w, h = bbox
            px, py = w * PAD, h * PAD
            crop = im.crop((max(0, x - px), max(0, y - py),
                            min(im.width, x + w + px), min(im.height, y + h + py)))
            crop = crop.resize((CELL, CELL), Image.LANCZOS)
            gx, gy = (k % 4) * CELL, (k // 4) * (CELL + CAP)
            grid.paste(crop, (gx, gy))
            # box outline of the labeled region inside the padded crop
            sx, sy = CELL / crop.width if False else CELL / (2 * px + w), CELL / (2 * py + h)
            draw.rectangle([gx + px * sx, gy + py * sy, gx + (px + w) * sx, gy + (py + h) * sy],
                           outline="lime", width=2)
            draw.text((gx + 3, gy + CELL + 2),
                      f"#{start + k} {f['label']} P(hel)={f['prob_helmet']} {f['split']} {f['bucket']}",
                      fill="black")
        grid.save(out_dir / f"panel_{start:03d}.png")
    print(f"{(len(flagged) + per - 1) // per} panels -> {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
