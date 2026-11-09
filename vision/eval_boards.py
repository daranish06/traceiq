#!/usr/bin/env python3
"""
Board-level evaluation of the full pipeline (tiling + detection + merge + per-class thresholds)
on FULL test images, against their ground-truth labels. Reports what the operator experiences:
defects found vs missed, false alarms per board, and boards with every defect found.

  python vision/eval_boards.py --images datasets/pku_board/test/images
  python vision/eval_boards.py --images datasets/dspcbsd_v3/test/images
"""
import argparse, json, sys, time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from inspect_board import STAGES, Detector, label_path, list_images, stage_paths  # noqa: E402


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--model", default=None, help="override the stage's model")
    ap.add_argument("--thresholds", default=None, help="override the stage's thresholds")
    ap.add_argument("--tiling", choices=["on", "off", "auto"], default=None)
    ap.add_argument("--imgsz", type=int, default=None, help="tile / input size (default: the stage's)")
    ap.add_argument("--iou", type=float, default=0.3, help="match IoU (boxes on full images)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None, help="name for the output file (default: from the folder)")
    ap.add_argument("--out", default=str(ROOT / "bench"))
    a = ap.parse_args()

    model, thr_default = stage_paths(a.stage)
    thr_file = Path(a.thresholds or thr_default)
    thr = json.loads(thr_file.read_text()) if thr_file.exists() else {}
    det = Detector(a.model or model, class_thresholds=thr, tile=a.imgsz or STAGES[a.stage]["imgsz"],
                   tiling=a.tiling or STAGES[a.stage]["tiling"])
    names = det.names
    imgs = list_images(a.images)[: a.limit or None]
    tp, fp, fn = Counter(), Counter(), Counter()
    boards_all_found = boards_no_fa = 0
    times = []
    for ip in imgs:
        img = cv2.imread(str(ip)); h, w = img.shape[:2]
        gts = []
        lp = label_path(ip)
        for line in (lp.read_text().split("\n") if lp.exists() else []):
            p = line.split()
            if len(p) == 5:
                c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
                gts.append((names[c], [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
        t0 = time.perf_counter(); dets, _ = det(img); times.append((time.perf_counter() - t0) * 1e3)
        used, b_fp = set(), 0
        for name, conf, box in sorted(dets, key=lambda d: -d[1]):
            best, bi = 0, -1
            for k, (gn, gb) in enumerate(gts):
                if gn == name and k not in used and iou(box, gb) > best:
                    best, bi = iou(box, gb), k
            if best >= a.iou:
                used.add(bi); tp[name] += 1
            else:
                fp[name] += 1; b_fp += 1
        for k, (gn, _) in enumerate(gts):
            if k not in used:
                fn[gn] += 1
        boards_all_found += len(used) == len(gts)
        boards_no_fa += b_fp == 0

    n = len(imgs)
    print(f"\n{n} boards from {a.images}")
    print(f"{'class':30s} {'found':>6s} {'missed':>7s} {'false':>6s} {'P':>6s} {'R':>6s}")
    rows = []
    for name in sorted(set(tp) | set(fp) | set(fn)):
        P = tp[name] / (tp[name] + fp[name]) if tp[name] + fp[name] else 0
        R = tp[name] / (tp[name] + fn[name]) if tp[name] + fn[name] else 0
        rows.append({"class": name, "found": tp[name], "missed": fn[name], "false_alarms": fp[name],
                     "precision": round(P, 3), "recall": round(R, 3)})
        print(f"{name:30s} {tp[name]:6d} {fn[name]:7d} {fp[name]:6d} {P:6.3f} {R:6.3f}")
    T, F, M = sum(tp.values()), sum(fp.values()), sum(fn.values())
    summary = {"boards": n, "defects_found": T, "defects_missed": M, "false_alarms": F,
               "precision": round(T / (T + F), 3) if T + F else 0, "recall": round(T / (T + M), 3) if T + M else 0,
               "false_alarms_per_board": round(F / n, 2) if n else 0,
               "boards_all_defects_found": f"{boards_all_found}/{n}",
               "boards_without_false_alarm": f"{boards_no_fa}/{n}",
               "mean_vision_ms": round(sum(times) / n, 1) if n else 0}
    print("\n" + "\n".join(f"{k:28s} {v}" for k, v in summary.items()))
    Path(a.out).mkdir(exist_ok=True)
    tag = a.tag or Path(a.images).parent.parent.name + "_" + Path(a.images).parent.name
    (Path(a.out) / f"board_eval_{tag}.json").write_text(json.dumps({"summary": summary, "per_class": rows}, indent=2))
    print("wrote", Path(a.out) / f"board_eval_{tag}.json")


if __name__ == "__main__":
    main()
