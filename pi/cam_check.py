#!/usr/bin/env python3
"""
Camera check for setting up the inspection bench (height, focus, light).
Takes a photo every few seconds with the same camera code as the dashboard, saves it as
cam_check.jpg and prints brightness (aim: 90-170) and sharpness (higher = sharper; compare positions).
  python pi/cam_check.py            # 5 shots, 3 s apart; Ctrl+C to stop early
"""
import argparse, sys, time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from ui import Camera  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shots", type=int, default=5)
ap.add_argument("--every", type=float, default=3.0)
a = ap.parse_args()

cam = Camera()
for i in range(1, a.shots + 1):
    t = time.time()
    f = cam.capture_bgr()
    if f is None:
        sys.exit("No camera found")
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    bright, sharp = g.mean(), cv2.Laplacian(g, cv2.CV_64F).var()
    hint = "too dark: more light / open lens cover" if bright < 70 else "too bright: less light / glare" \
        if bright > 190 else "ok"
    out = ROOT / "cam_check.jpg"
    cv2.imwrite(str(out), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[{i}] {cam.kind} {f.shape[1]}x{f.shape[0]}  brightness {bright:5.1f} ({hint})  "
          f"sharpness {sharp:7.1f}  capture {time.time() - t:.1f}s  -> {out.name}", flush=True)
    if i < a.shots:
        time.sleep(a.every)
