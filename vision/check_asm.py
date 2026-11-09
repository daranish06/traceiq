#!/usr/bin/env python3
"""
Check the Stage 2 (assembled-board) dataset before training:
  - images and labels per split, images without a label file
  - defect instances per class per split
  - image sizes (decides whether tiling is needed)
  - exact duplicate images across splits (SHA-256) -> train/test leakage
  - optional: --dedup removes test/val images that also appear in train (and their labels)
Writes datasets/asm.yaml with absolute paths.

  python vision/check_asm.py --root datasets/asm_raw/final_solder
"""
from __future__ import annotations

import argparse, hashlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["insufficient_solder", "excess_solder", "solder_bridge", "solder_void", "missing_component",
         "shifted_component", "burn", "physical_damage", "oxidation"]
SPLITS = ["train", "val", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "datasets/asm_raw/final_solder"))
    ap.add_argument("--dedup", action="store_true", help="delete val/test images duplicated in train")
    a = ap.parse_args()
    root = Path(a.root).resolve()

    hashes: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for s in SPLITS:
        imgs = sorted(p for p in (root / "images" / s).iterdir() if p.suffix.lower() in IMG_EXT)
        inst, sizes, no_label, empty = Counter(), Counter(), 0, 0
        for i, p in enumerate(imgs):
            lab = root / "labels" / s / (p.stem + ".txt")
            if not lab.exists():
                no_label += 1
            else:
                lines = [l.split() for l in lab.read_text().splitlines() if l.strip()]
                empty += not lines
                for l in lines:
                    inst[int(l[0])] += 1
            if i % 10 == 0:  # sample sizes
                im = cv2.imread(str(p))
                if im is not None:
                    h, w = im.shape[:2]
                    sizes[f"{w}x{h}"] += 1
            hashes[sha(p)][s].append(p)
        print(f"\n== {s}: {len(imgs)} images, {no_label} without label file, {empty} with no defects")
        for c, n in enumerate(NAMES):
            print(f"   {n:22s} {inst[c]}")
        print("   sizes (1 in 10 sampled):", ", ".join(f"{k} ({v})" for k, v in sizes.most_common(6)))

    dup_in_train = {s: [p for h in hashes.values() if "train" in h for p in h.get(s, [])] for s in ["val", "test"]}
    within = sum(len(v["train"]) - 1 for v in hashes.values() if len(v.get("train", [])) > 1)
    print(f"\nexact duplicates: val in train {len(dup_in_train['val'])}, test in train {len(dup_in_train['test'])}, "
          f"repeated inside train {within}")
    if a.dedup:
        for s, ps in dup_in_train.items():
            for p in ps:
                p.unlink()
                (root / "labels" / s / (p.stem + ".txt")).unlink(missing_ok=True)
        print("removed duplicated val/test images; rerun without --dedup to confirm 0")

    yaml = ROOT / "datasets/asm.yaml"
    yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    print("wrote", yaml)


if __name__ == "__main__":
    main()
