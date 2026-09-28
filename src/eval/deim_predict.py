"""DEIMv2 test-split predictions -> violation preds JSON (head class).

Run from inside the deimv2 repo dir (imports its `engine` package):
  python ../repo/src/eval/deim_predict.py -c configs/deimv2/deimv2_hardhat_s.yml \
      -r ../repo/runs/deimv2_s_arm1/best_stg1.pth \
      --split-dir ../repo/data/variants/arm1/test \
      --out ../repo/runs/deimv2_s_arm1/preds_test.json
Labels: dataset ids (1=head, 2=helmet) since remap_mscoco_category=False.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, ".")
from engine.core import YAMLConfig

HEAD_ID = 1
FLOOR = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-r", "--resume", required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cfg = YAMLConfig(a.config, resume=a.resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(a.resume, map_location="cpu", weights_only=False)
    state = ckpt.get("ema", {}).get("module", ckpt.get("model"))
    cfg.model.load_state_dict(state)

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.post = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            return self.post(self.model(images), orig_sizes)

    model = Wrap().eval().cuda()
    tfm = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    imgs = sorted(p for p in a.split_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    preds = []
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(imgs):
            im = Image.open(p).convert("RGB")
            size = torch.tensor([[im.width, im.height]], device="cuda")
            labels, boxes, scores = model(tfm(im)[None].cuda(), size)
            for lb, bx, sc in zip(labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()):
                if sc >= FLOOR and int(lb) == HEAD_ID:
                    x1, y1, x2, y2 = bx
                    preds.append({"img": p.name, "bbox": [x1, y1, x2 - x1, y2 - y1],
                                  "score": float(sc)})
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(imgs)}", flush=True)
    a.out.write_text(json.dumps(preds))
    dt = time.time() - t0
    print(f"{len(preds)} preds, {dt / len(imgs) * 1000:.0f} ms/img e2e")


if __name__ == "__main__":
    main()
