#!/usr/bin/env python3
"""Re-split PKU photos 80/10/10 per defect class so every class is in every split."""
import random, shutil, yaml
from pathlib import Path
from collections import defaultdict

src = Path("~/traceiq/datasets/pku_raw").expanduser()
dst = Path("~/traceiq/datasets/pku_strat").expanduser()
random.seed(42)

by_class = defaultdict(list)
for split in ["train", "valid", "test"]:
    for ip in (src / split / "images").iterdir():
        lp = src / split / "labels" / (ip.stem + ".txt")
        lines = lp.read_text().split() if lp.exists() else []
        cls = int(lines[0]) if lines else -1          # PKU: one defect type per photo
        by_class[cls].append((ip, lp))

names = yaml.safe_load((src / "data.yaml").read_text())["names"]
for cls, items in sorted(by_class.items()):
    random.shuffle(items)
    n = len(items); n_te = max(1, round(n * 0.10)); n_va = max(1, round(n * 0.10))
    parts = {"test": items[:n_te], "valid": items[n_te:n_te + n_va], "train": items[n_te + n_va:]}
    for split, its in parts.items():
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for ip, lp in its:
            shutil.copy(ip, dst / split / "images" / ip.name)
            if lp.exists():
                shutil.copy(lp, dst / split / "labels" / lp.name)
    label = names[cls] if cls >= 0 else "no-label"
    print(f"{label:16s} total {n:3d} -> train {len(parts['train'])}, valid {len(parts['valid'])}, test {len(parts['test'])}")

shutil.copy(src / "data.yaml", dst / "data.yaml")
print("done:", dst)
