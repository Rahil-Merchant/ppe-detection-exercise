"""Pose-gated violation predictions: suppress alerts without human-keypoint support.

Gate variants per violation box:
  kpt   — some person has >=1 head keypoint (conf>=kc) inside the (slightly expanded) box
  kpt+box — kpt support OR box center falls in the top-35% band of a person box
            (rescues back-of-head views where face keypoints are invisible)

Usage:
  python src/eval/pose_gate.py runs/arm1/preds_test.json runs/pose/test_keypoints.json runs/pose
Writes gated preds JSONs for each variant/param combo.
"""
import json
import sys
from pathlib import Path

KC = [0.3, 0.5]          # keypoint confidence
EXPAND = 0.25            # box expansion fraction per side for kpt containment
PERSON_CONF = 0.25


def inside(px, py, box, expand=0.0):
    x, y, w, h = box
    ex, ey = w * expand, h * expand
    return (x - ex) <= px <= (x + w + ex) and (y - ey) <= py <= (y + h + ey)


def main(preds_path, kpts_path, out_dir):
    preds = json.load(open(preds_path))
    kpts = json.load(open(kpts_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for kc in KC:
        for variant in ("kpt", "kptbox"):
            kept, dropped = [], 0
            for p in preds:
                persons = [q for q in kpts.get(p["img"], []) if q["conf"] >= PERSON_CONF]
                support = any(
                    inside(kx, ky, p["bbox"], EXPAND)
                    for q in persons for kx, ky, c in q["head_kpts"] if c >= kc
                )
                if not support and variant == "kptbox":
                    bx = p["bbox"][0] + p["bbox"][2] / 2
                    by = p["bbox"][1] + p["bbox"][3] / 2
                    for q in persons:
                        x, y, w, h = q["box"]
                        if x <= bx <= x + w and y <= by <= y + 0.35 * h:
                            support = True
                            break
                if support:
                    kept.append(p)
                else:
                    dropped += 1
            name = f"gated_{variant}_kc{kc}"
            (out_dir / f"{name}.json").write_text(json.dumps(kept))
            print(f"{name}: kept {len(kept)}, dropped {dropped}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
