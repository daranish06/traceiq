#!/usr/bin/env python3
"""
TraceIQ inspection pipeline:
  board image -> 640x640 overlapping tiles -> YOLO -> merge boxes (NMS) on the full image
  -> defect records + rule-based verdict -> Llama advisor -> annotated image + log (CSV + JSONL)

Works on the laptop (PyTorch .pt model) and on the Raspberry Pi (NCNN model folder).

Examples
  python app/inspect_board.py --source datasets/pku_board/test/images --limit 5
  python app/inspect_board.py --source board.jpg --backend native
  python app/inspect_board.py --source board.jpg --model models/pcb_combined_ncnn_model --backend llama_main
"""
from __future__ import annotations

import argparse, csv, json, sys, time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "advisor"))
import advisor as A  # noqa: E402

COLORS = {"critical": (0, 0, 255), "major": (0, 140, 255), "minor": (0, 200, 255)}  # BGR
VERDICT_COLORS = {"PASS": (0, 160, 0), "REVIEW": (0, 170, 255), "FAIL": (0, 0, 220)}


# ---------------------------------------------------------------- tiling + merge
def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:
        s.append(length - tile)
    return s


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float, contain: float = 0.6) -> list[int]:
    """Greedy NMS that also drops a box mostly inside a stronger one (partial boxes cut by tile edges)."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1, yy1 = np.maximum(x1[i], x1[rest]), np.maximum(y1[i], y1[rest])
        xx2, yy2 = np.minimum(x2[i], x2[rest]), np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ov_iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        ov_small = inter / (np.minimum(areas[i], areas[rest]) + 1e-9)
        order = rest[(ov_iou <= iou) & (ov_small <= contain)]
    return keep


class Detector:
    def __init__(self, model_path: str, conf: float = 0.25, tile: int = 640, overlap: float = 0.2,
                 iou: float = 0.5, batch: int = 8, class_thresholds: dict | None = None):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task="detect")
        self.names = self.model.names
        self.thr = class_thresholds or {}
        # run the model at the lowest threshold in use, then filter per class
        self.conf = min([conf] + list(self.thr.values()))
        self.default_conf = conf
        self.tile, self.iou, self.batch = tile, iou, batch
        self.stride = int(tile * (1 - overlap))

    def __call__(self, img: np.ndarray) -> tuple[list[tuple[str, float, list[float]]], dict]:
        H, W = img.shape[:2]
        t0 = time.perf_counter()
        crops, offsets = [], []
        for ty in tile_starts(H, self.tile, self.stride):
            for tx in tile_starts(W, self.tile, self.stride):
                crops.append(img[ty:ty + self.tile, tx:tx + self.tile])
                offsets.append((tx, ty))
        t1 = time.perf_counter()
        boxes, scores, classes = [], [], []
        for i in range(0, len(crops), self.batch):
            results = self.model.predict(crops[i:i + self.batch], imgsz=self.tile, conf=self.conf,
                                         verbose=False)
            for r, (ox, oy) in zip(results, offsets[i:i + self.batch]):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                b = r.boxes.xyxy.cpu().numpy() + np.array([ox, oy, ox, oy])
                boxes.append(b)
                scores.append(r.boxes.conf.cpu().numpy())
                classes.append(r.boxes.cls.cpu().numpy().astype(int))
        t2 = time.perf_counter()
        dets = []
        if boxes:
            B, S, C = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
            for c in np.unique(C):  # class-wise NMS removes duplicates from overlapping tiles
                idx = np.where(C == c)[0]
                name = self.names[int(c)]
                idx = idx[S[idx] >= self.thr.get(name, self.default_conf)]
                if idx.size == 0:
                    continue
                for k in nms(B[idx], S[idx], self.iou):
                    j = idx[k]
                    dets.append((name, float(S[j]), B[j].tolist()))
        t3 = time.perf_counter()
        timing = {"tiles": len(crops), "tile_ms": round((t1 - t0) * 1e3, 1),
                  "infer_ms": round((t2 - t1) * 1e3, 1), "merge_ms": round((t3 - t2) * 1e3, 1)}
        return dets, timing


# ---------------------------------------------------------------- drawing + logging
def annotate(img: np.ndarray, insp: A.Inspection) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    th = max(1, round(min(H, W) / 400))
    for d in insp.defects:
        x1, y1, x2, y2 = d.bbox_px
        pad = 6 * th
        col = COLORS[d.severity]
        cv2.rectangle(out, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), col, 2 * th)
        label = f"{A.KNOWLEDGE[d.defect_class]['label']} {d.confidence:.2f}"
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * th, th)
        lx = int(min(max(0, x1 - pad), W - tw - 2))          # keep label inside the image
        ly = int(y1 - pad - 4) if y1 - pad - 4 > tht + 2 else int(y2 + pad + tht + 4)
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * th, col, th, cv2.LINE_AA)
    bar = 34 * th  # banner above the image so it never hides defects
    canvas = np.zeros((H + bar, W, 3), np.uint8)
    canvas[:bar] = VERDICT_COLORS[insp.verdict]
    canvas[bar:] = out
    cv2.putText(canvas, f"{insp.board_id}  {insp.verdict}  ({len(insp.defects)} defects)", (10, int(bar * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * th, (255, 255, 255), 2 * th, cv2.LINE_AA)
    out = canvas
    return out


def append_logs(insp: A.Inspection, timing: dict, out_dir: Path, image_path: str, annotated: str):
    row = {"timestamp": insp.timestamp, "board_id": insp.board_id, "verdict": insp.verdict,
           "n_defects": len(insp.defects),
           "defects": "; ".join(A.defect_summary(insp)), "log_line": insp.log_line,
           "advisor_backend": insp.advisor_backend, "advisor_s": insp.advisor_seconds,
           **timing, "image": image_path, "annotated": annotated}
    csv_path = out_dir / "inspections.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    with open(out_dir / "inspections.jsonl", "a") as f:
        f.write(json.dumps({**asdict(insp), "defects": [asdict(d) for d in insp.defects],
                            "timing": timing, "image": image_path}) + "\n")


# ---------------------------------------------------------------- main
def inspect_one(img_path: Path, det: Detector, backend: str, cfg: dict, out_dir: Path, board_id: str):
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"cannot read {img_path}")
    H, W = img.shape[:2]
    t0 = time.perf_counter()
    dets, timing = det(img)
    recs = [A.make_record(c, p, b, W, H) for c, p, b in dets if c in A.KNOWLEDGE]
    insp = A.Inspection(board_id, time.strftime("%Y-%m-%dT%H:%M:%S"), A.verdict_for(recs), recs)
    timing["vision_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    A.advise(insp, backend, cfg)
    timing["advisor_ms"] = round(insp.advisor_seconds * 1e3, 1)
    ann_path = out_dir / "annotated" / f"{board_id}_{img_path.stem}.jpg"
    cv2.imwrite(str(ann_path), annotate(img, insp), [cv2.IMWRITE_JPEG_QUALITY, 90])
    append_logs(insp, timing, out_dir, str(img_path), str(ann_path))
    return insp, timing


def main():
    ap = argparse.ArgumentParser(description="TraceIQ board inspection")
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--model", default=str(ROOT / "models/pcb_combined_y11n.pt"))
    ap.add_argument("--backend", choices=["template", "native", "llama_main"], default="template")
    ap.add_argument("--config", default=str(ROOT / "advisor/advisor_config.json"))
    ap.add_argument("--conf", type=float, default=0.25, help="used for classes without a tuned threshold")
    ap.add_argument("--thresholds", default=str(ROOT / "models/class_thresholds.json"),
                    help="per-class thresholds from vision/tune_thresholds.py (skipped if missing)")
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--limit", type=int, default=0, help="max images from a folder (0 = all)")
    a = ap.parse_args()

    out_dir = Path(a.out)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
    cfg = {}
    if a.backend != "template":
        cfg = {k: str(Path(v).expanduser()) if isinstance(v, str) and ("/" in v or v.startswith("~")) else v
               for k, v in json.loads(Path(a.config).read_text())[a.backend].items()}
    src = Path(a.source)
    images = sorted(p for p in (src.iterdir() if src.is_dir() else [src])
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if a.limit:
        images = images[:a.limit]

    thr_path = Path(a.thresholds)
    thresholds = json.loads(thr_path.read_text()) if thr_path.exists() else {}
    print("class thresholds:", thresholds or f"none (using {a.conf} for all)")
    det = Detector(a.model, conf=a.conf, class_thresholds=thresholds)
    det(np.zeros((640, 640, 3), np.uint8))  # warm-up so the first board's timing is fair
    start_n = sum(1 for _ in open(out_dir / "inspections.csv")) - 1 if (out_dir / "inspections.csv").exists() else 0
    for n, p in enumerate(images, start=start_n + 1):
        insp, t = inspect_one(p, det, a.backend, cfg, out_dir, f"B-{n:04d}")
        print(f"\n[{insp.board_id}] {p.name}  {insp.verdict}  | {t['tiles']} tiles, vision {t['vision_ms']} ms, "
              f"advisor {t['advisor_ms']} ms ({insp.advisor_backend})")
        print("  " + insp.advice.replace("\n", "\n  "))
    print(f"\nlog: {out_dir / 'inspections.csv'}   images: {out_dir / 'annotated'}")


if __name__ == "__main__":
    main()
