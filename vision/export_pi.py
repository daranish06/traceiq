#!/usr/bin/env python3
"""
Export the trained YOLO model for the Raspberry Pi CPU (run on the laptop).
  python vision/export_pi.py --model models/pcb_combined_y11n.pt
Creates models/pcb_combined_y11n_ncnn_model/ (fastest on the Pi CPU) and models/pcb_combined_y11n.onnx
(for an NCNN-vs-ONNX comparison in the report), then sanity-checks that both give the same detections
as the PyTorch model on a few validation tiles.
"""
import argparse, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/pcb_combined_y11n.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--check", default=str(ROOT / "datasets/pku_board_tiles/valid/images"))
    a = ap.parse_args()
    from ultralytics import YOLO

    pt = YOLO(a.model)
    outs = {}
    for fmt in ("ncnn", "onnx"):
        outs[fmt] = pt.export(format=fmt, imgsz=a.imgsz, half=False, simplify=(fmt == "onnx"))
        print(f"{fmt}: {outs[fmt]}")

    imgs = sorted(Path(a.check).glob("*.jpg"))[:20]
    ref = [len(r.boxes) for r in pt.predict([str(p) for p in imgs], imgsz=a.imgsz, conf=0.25, verbose=False)]
    for fmt, path in outs.items():
        m = YOLO(path, task="detect")
        got = [len(m.predict(str(p), imgsz=a.imgsz, conf=0.25, verbose=False)[0].boxes) for p in imgs]
        same = sum(x == y for x, y in zip(ref, got))
        print(f"{fmt}: same number of detections as PyTorch on {same}/{len(imgs)} tiles")
    shutil.copy(ROOT / "models/class_thresholds.json", Path(outs["ncnn"]) / "class_thresholds.json") \
        if (ROOT / "models/class_thresholds.json").exists() else None


if __name__ == "__main__":
    main()
