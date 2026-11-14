#!/usr/bin/env python3
"""
Find the sharpest fixed focus for the bench (camera fixed, board in place, do not touch it).
Coarse sweep 1..1023, then a fine sweep around the best; saves pi/camera.env (TRACEIQ_FOCUS=...),
which the dashboard uses at start, and focus_best.jpg.
  python pi/focus_sweep.py
"""
import subprocess, time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
for _ in range(30):
    cap.read()
subprocess.run("v4l2-ctl -d /dev/video0 -c focus_automatic_continuous=0", shell=True)


def score(f):
    subprocess.run(f"v4l2-ctl -d /dev/video0 -c focus_absolute={f}", shell=True)
    time.sleep(0.6)
    for _ in range(5):
        cap.read()
    ok, im = cap.read()
    h, w = im.shape[:2]
    g = cv2.cvtColor(im[h // 4:3 * h // 4, w // 4:3 * w // 4], cv2.COLOR_BGR2GRAY)
    s = cv2.Laplacian(g, cv2.CV_64F).var()
    print(f"focus {f:4d}  sharpness {s:7.1f}", flush=True)
    return s, im


best = max(((score(f), f) for f in range(1, 1024, 50)), key=lambda x: x[0][0])
c = best[1]
fine = max(((score(f), f) for f in range(max(1, c - 45), min(1023, c + 46), 10)), key=lambda x: x[0][0])
(s, im), f = max([best, fine], key=lambda x: x[0][0])
cv2.imwrite(str(ROOT / "focus_best.jpg"), im)
(ROOT / "pi/camera.env").write_text(f"TRACEIQ_FOCUS={f}\n")
print(f"BEST focus {f}  sharpness {s:.1f}  -> saved pi/camera.env and focus_best.jpg")
