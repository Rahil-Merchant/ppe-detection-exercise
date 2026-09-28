"""Run COCO-pretrained YOLO11-pose on a split; dump per-image persons with head keypoints.

Usage: .venv/bin/python src/eval/pose_keypoints.py data/raw/test runs/pose/test_keypoints.json
Output: {img: [{"box":[x,y,w,h], "conf": c, "head_kpts": [[x,y,conf]x5]}]}  (kpts 0-4 = nose,eyes,ears)
"""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

split, out = Path(sys.argv[1]), Path(sys.argv[2])
model = YOLO("yolo11x-pose.pt")
imgs = sorted(p for p in split.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
res = {}
for i in range(0, len(imgs), 16):
    batch = imgs[i:i + 16]
    for p, r in zip(batch, model([str(x) for x in batch], conf=0.15, verbose=False)):
        persons = []
        if r.keypoints is not None and r.boxes is not None:
            kpts = r.keypoints.data.cpu().numpy()  # (n,17,3)
            for box, conf, kp in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy(), kpts):
                cx, cy, w, h = box.tolist()
                persons.append({"box": [cx - w / 2, cy - h / 2, w, h], "conf": round(float(conf), 3),
                                "head_kpts": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                                              for x, y, c in kp[:5].tolist()]})
        res[p.name] = persons
    if (i // 16) % 10 == 0:
        print(f"{i + len(batch)}/{len(imgs)}", flush=True)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res))
print(f"done: {sum(len(v) for v in res.values())} persons in {len(res)} imgs")
