"""Latency benchmark for deployment note: batch-1 and batched throughput.

Measures detector forward (and cascade classifier overhead per N crops).
Usage: python src/eval/bench_latency.py --det runs/arm1/checkpoint_best_total.pth --res 728 \
           [--cls runs/arm2_cls/best.pth] [--crops-per-img 4]
"""
import argparse
import time

import numpy as np
import timm
import torch
from PIL import Image


def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--res", type=int, default=728)
    ap.add_argument("--cls")
    ap.add_argument("--crops-per-img", type=int, default=4)
    a = ap.parse_args()

    from rfdetr import RFDETRSmall
    det = RFDETRSmall(pretrain_weights=a.det, resolution=a.res)
    import torch as _t
    det.inference(dtype=_t.float16)
    img = Image.fromarray(np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8))

    ms = timeit(lambda: det.predict(img, threshold=0.3))
    print(f"detector batch-1 @res{a.res}: {ms:.1f} ms/img -> {1000 / ms:.1f} img/s")
    print(f"12 streams @ 1 frame/2s each = 6 img/s budget -> "
          f"{'OK' if 1000 / ms >= 6 else 'INSUFFICIENT'} "
          f"(headroom {1000 / ms / 6:.1f}x)")

    if a.cls:
        ck = torch.load(a.cls, map_location="cuda", weights_only=False)
        clf = timm.create_model(ck["model"], pretrained=False, num_classes=2)
        clf.load_state_dict(ck["state_dict"])
        clf = clf.eval().cuda()
        cfg = timm.data.resolve_model_data_config(clf)
        size = cfg["input_size"][1]
        x = torch.randn(a.crops_per_img, 3, size, size, device="cuda")
        with torch.no_grad():
            ms_c = timeit(lambda: clf(x))
        print(f"classifier {a.crops_per_img} crops: {ms_c:.1f} ms -> cascade e2e ~{ms + ms_c:.1f} ms/img")


if __name__ == "__main__":
    main()
