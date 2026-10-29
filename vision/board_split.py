#!/usr/bin/env python3
"""Hold out whole PCB board designs: test/valid boards are never seen in training."""
import argparse, shutil, yaml
from pathlib import Path
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--test", nargs="+", default=["08", "12"])
ap.add_argument("--valid", nargs="+", default=["11"])
a = ap.parse_args()
src = Path("~/traceiq/datasets/pku_raw").expanduser()
dst = Path("~/traceiq/datasets/pku_board").expanduser()
names = yaml.safe_load((src / "data.yaml").read_text())["names"]
counts = {s: Counter() for s in ["train", "valid", "test"]}

for split in ["train", "valid", "test"]:
    for ip in (src / split / "images").iterdir():
        board = ip.name.split("_")[0]
        out = "test" if board in a.test else "valid" if board in a.valid else "train"
        lp = src / split / "labels" / (ip.stem + ".txt")
        (dst / out / "images").mkdir(parents=True, exist_ok=True)
        (dst / out / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy(ip, dst / out / "images" / ip.name)
        if lp.exists():
            shutil.copy(lp, dst / out / "labels" / lp.name)
            first = lp.read_text().split()
            if first:
                counts[out][names[int(first[0])]] += 1

shutil.copy(src / "data.yaml", dst / "data.yaml")
print(f"test boards {a.test}, valid boards {a.valid}")
print(f"{'class':16s} {'train':>6s} {'valid':>6s} {'test':>6s}")
for n in names:
    print(f"{n:16s} {counts['train'][n]:6d} {counts['valid'][n]:6d} {counts['test'][n]:6d}")
print(f"{'TOTAL':16s} {sum(counts['train'].values()):6d} {sum(counts['valid'].values()):6d} {sum(counts['test'].values()):6d}")
