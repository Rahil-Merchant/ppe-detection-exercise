"""Audit step 2: exact + near-duplicate images, cross-split leakage.

Usage: python src/audit/find_dupes.py data/dataset_root
Writes runs/audit/dupes.json. Exact = md5; near = pHash hamming <= 4.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

SPLITS = ["train", "valid", "test"]
EXTS = {".jpg", ".jpeg", ".png"}
PHASH_THRESH = 4


def main(root: Path):
    files = []  # (split, path)
    for split in SPLITS:
        for p in sorted((root / split).glob("*")):
            if p.suffix.lower() in EXTS:
                files.append((split, p))
    print(f"{len(files)} images")

    md5s = defaultdict(list)
    phashes = []
    for split, p in files:
        md5s[hashlib.md5(p.read_bytes()).hexdigest()].append((split, p.name))
        phashes.append((split, p.name, imagehash.phash(Image.open(p))))

    exact_groups = [v for v in md5s.values() if len(v) > 1]

    # near-dupes: full pairwise hamming via numpy popcount (7k imgs is cheap)
    bits = np.array([h.hash.flatten() for _, _, h in phashes], dtype=bool)
    packed = np.packbits(bits, axis=1)  # (N, 8) uint8
    lut = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)
    popcnt = lut[packed[:, None, :] ^ packed[None, :, :]].sum(axis=2, dtype=np.uint16)
    ii, jj = np.where(np.triu(popcnt <= PHASH_THRESH, k=1))
    near_pairs = []
    for i, j in zip(ii.tolist(), jj.tolist()):
        s1, n1, _ = phashes[i]
        s2, n2, _ = phashes[j]
        near_pairs.append({"a": f"{s1}/{n1}", "b": f"{s2}/{n2}", "dist": int(popcnt[i, j])})

    cross_exact = [g for g in exact_groups if len({s for s, _ in g}) > 1]
    cross_near = [p for p in near_pairs if p["a"].split("/")[0] != p["b"].split("/")[0]]

    out = {
        "exact_dupe_groups": len(exact_groups),
        "exact_dupe_images": sum(len(g) for g in exact_groups),
        "cross_split_exact_groups": len(cross_exact),
        "cross_split_exact_detail": [[f"{s}/{n}" for s, n in g] for g in cross_exact],
        "near_dupe_pairs": len(near_pairs),
        "cross_split_near_pairs": len(cross_near),
        "cross_split_near_detail": cross_near,
    }
    dst = Path("runs/audit/dupes.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("detail")}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
