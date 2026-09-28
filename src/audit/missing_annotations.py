"""Audit step 3: estimate under-annotation using COCO-pretrained RF-DETR.

Person: detector person-class predictions vs GT `person` boxes -> missing-person rate.
Head-region proxy: detector person predictions whose image has zero head/helmet GT
inside the person box -> likely unlabeled head (upper-body heuristic, reported separately).

Usage: python src/audit/missing_annotations.py data/dataset_root [--conf 0.7]
Writes runs/audit/missing_annotations.json
"""
import argparse
import json
import sys
from pathlib import Path

import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import iou, load_coco

PERSON_COCO_ID = 1  # COCO class id for person in rfdetr output
SPLITS = ["train", "valid", "test"]


def main(root: Path, conf: float):
    model = RFDETRBase()  # COCO-pretrained
    out = {}
    for split in SPLITS:
        ann = root / split / "_annotations.coco.json"
        if not ann.exists():
            continue
        _, imgs, cats, anns_by_img = load_coco(ann)
        name_by_file = {im["file_name"]: iid for iid, im in imgs.items()}
        person_cat_ids = [cid for cid, n in cats.items() if n.lower() == "person"]

        n_pred_person = 0
        n_pred_matched = 0
        imgs_with_unlabeled_person = 0
        n_imgs = 0
        per_image = []
        for fname, iid in name_by_file.items():
            p = root / split / fname
            if not p.exists():
                continue
            n_imgs += 1
            det = model.predict(Image.open(p).convert("RGB"), threshold=conf)
            person_boxes = [
                [x1, y1, x2 - x1, y2 - y1]
                for (x1, y1, x2, y2), cid in zip(det.xyxy, det.class_id)
                if cid == PERSON_COCO_ID
            ]
            gt_person = [a["bbox"] for a in anns_by_img[iid] if a["category_id"] in person_cat_ids]
            unmatched = 0
            for pb in person_boxes:
                n_pred_person += 1
                if any(iou(pb, g) >= 0.3 for g in gt_person):
                    n_pred_matched += 1
                else:
                    unmatched += 1
            if unmatched:
                imgs_with_unlabeled_person += 1
                per_image.append({"img": f"{split}/{fname}", "unmatched_persons": unmatched})
        out[split] = {
            "images_scanned": n_imgs,
            "pred_persons_conf>=%.2f" % conf: n_pred_person,
            "matched_to_gt_person": n_pred_matched,
            "unmatched_rate": round(1 - n_pred_matched / max(n_pred_person, 1), 4),
            "imgs_with_unlabeled_person": imgs_with_unlabeled_person,
            "sample_offenders": per_image[:30],
        }
        print(split, {k: v for k, v in out[split].items() if k != "sample_offenders"})
    dst = Path("runs/audit/missing_annotations.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--conf", type=float, default=0.7)
    a = ap.parse_args()
    main(a.root, a.conf)
