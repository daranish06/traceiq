#!/usr/bin/env python3
"""
Choose a confidence threshold per class on the VALIDATION sets (never the test sets):
for each class, sweep thresholds 0.05..0.95 and keep the one with the best F1 (balance of
misses vs false alarms). Writes models/class_thresholds.json, used by app/inspect_board.py.

  python vision/tune_thresholds.py --model models/pcb_combined_y11n.pt
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAL = [ROOT / "datasets/pku_board_tiles/valid", ROOT / "datasets/dspcbsd_v3/valid"]


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/pcb_combined_y11n.pt"))
    ap.add_argument("--val", nargs="+", default=[str(p) for p in DEFAULT_VAL])
    ap.add_argument("--out", default=str(ROOT / "models/class_thresholds.json"))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min_thr", type=float, default=0.25, help="never go below this")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model, task="detect")
    names = model.names
    preds = defaultdict(list)          # cls -> list of (conf, is_true_positive)
    n_gt = defaultdict(int)
    images = []
    for d in a.val:
        images += sorted((Path(d) / "images").glob("*.jpg"))
    print(f"{len(images)} validation images")

    for i in range(0, len(images), 32):
        batch = images[i:i + 32]
        for ip, r in zip(batch, model.predict([str(p) for p in batch], imgsz=640, conf=0.05, verbose=False)):
            h, w = r.orig_shape
            gts = []
            lp = ip.parent.parent / "labels" / (ip.stem + ".txt")
            if lp.exists():
                for line in lp.read_text().split("\n"):
                    p = line.split()
                    if len(p) == 5:
                        c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
                        gts.append((c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
                        n_gt[c] += 1
            used = set()
            boxes = r.boxes
            order = np.argsort(-boxes.conf.cpu().numpy()) if len(boxes) else []
            for j in order:
                c = int(boxes.cls[j]); cf = float(boxes.conf[j]); b = boxes.xyxy[j].tolist()
                best, bi = 0.0, -1
                for k, (gc, gb) in enumerate(gts):
                    if gc == c and k not in used:
                        v = iou(b, gb)
                        if v > best:
                            best, bi = v, k
                tp = best >= a.iou
                if tp:
                    used.add(bi)
                preds[c].append((cf, tp))

    result = {}
    print(f"{'class':30s} {'thr':>5s} {'P':>6s} {'R':>6s} {'F1':>6s}  (at 0.25: P / R)")
    for c, name in names.items():
        if n_gt[c] == 0:
            result[name] = 0.5
            print(f"{name:30s}  no validation labels -> 0.50")
            continue
        arr = np.array(preds[c], dtype=float) if preds[c] else np.zeros((0, 2))
        best = (0, a.min_thr, 0, 0)
        stats = {}
        for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
            sel = arr[arr[:, 0] >= t] if len(arr) else arr
            tp = sel[:, 1].sum() if len(sel) else 0
            p = tp / len(sel) if len(sel) else 0
            rcl = tp / n_gt[c]
            f1 = 2 * p * rcl / (p + rcl) if p + rcl else 0
            stats[t] = (p, rcl)
            if t >= a.min_thr and f1 > best[0]:
                best = (f1, t, p, rcl)
        f1, t, p, rcl = best
        result[name] = float(t)
        p25, r25 = stats.get(0.25, (0, 0))
        print(f"{name:30s} {t:5.2f} {p:6.3f} {rcl:6.3f} {f1:6.3f}  ({p25:.3f} / {r25:.3f})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
