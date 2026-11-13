#!/usr/bin/env python3
"""
TraceIQ inspection pipeline:
  board image -> 640x640 overlapping tiles -> YOLO -> merge boxes (NMS) on the full image
  -> defect records + rule-based verdict -> Llama advisor -> annotated image + log (CSV + JSONL)

Works on the laptop (PyTorch .pt model) and on the Raspberry Pi (NCNN model folder).

Examples
  python app/inspect_board.py --source datasets/pku_board/test/images --limit 5
  python app/inspect_board.py --source board.jpg --backend native
  python app/inspect_board.py --stage 2 --source datasets/asm_raw/final_solder/images/test --limit 5
  python app/inspect_board.py --source board.jpg --pi --backend llama_main          # Raspberry Pi
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time

# Everything runs on-device. Stop library telemetry so nothing is sent even when a network is present.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "advisor"))
import advisor as A  # noqa: E402

COLORS = {"critical": (0, 0, 255), "major": (0, 140, 255), "minor": (0, 200, 255)}  # BGR
VERDICT_COLORS = {"PASS": (0, 160, 0), "REVIEW": (0, 170, 255), "FAIL": (0, 0, 220)}

# Two inspection stages, each with its own model, thresholds and tiling mode.
#   Stage 1 (bare board): small defects on large board photos -> always tile.
#   Stage 2 (assembled board): trained on whole close-up photos -> tile only very large images.
STAGES = {
    1: {"name": "Stage 1: bare board", "model": "models/pcb_combined_y11n.pt",
        "pi_model": "models/pcb_combined_y11n_ncnn_model",
        "thresholds": "models/class_thresholds.json", "tiling": "on", "imgsz": 640, "area_frac": 0.002},
    2: {"name": "Stage 2: assembled board", "model": "models/pcb_asm_y11n.pt",
        "pi_model": "models/pcb_asm_y11n_ncnn_model",
        "thresholds": "models/class_thresholds_asm.json", "tiling": "auto", "imgsz": 640, "area_frac": None},
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def label_path(img_path: Path) -> Path:
    """YOLO convention: .../images/<split>/x.jpg -> .../labels/<split>/x.txt (last 'images' part replaced)."""
    parts = list(img_path.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def stage_paths(stage: int, pi: bool = False) -> tuple[str, str]:
    s = STAGES[stage]
    model = ROOT / (s["pi_model"] if pi else s["model"])
    return str(model), str(ROOT / s["thresholds"])


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
                 iou: float = 0.5, batch: int = 8, class_thresholds: dict | None = None,
                 tiling: str = "on"):
        """tiling: 'on' = always cut into tile x tile crops; 'off' = whole image resized to tile;
        'auto' = whole image unless it is more than twice the tile size on its long side."""
        from ultralytics import YOLO
        try:  # no usage events to Ultralytics' servers
            from ultralytics import settings
            if settings.get("sync", False):
                settings.update({"sync": False})
        except Exception:  # noqa: BLE001
            pass
        self.model = YOLO(model_path, task="detect")
        self.names = self.model.names
        self.thr = class_thresholds or {}
        # run the model at the lowest threshold in use, then filter per class
        self.conf = min([conf] + list(self.thr.values()))
        self.default_conf = conf
        self.tile, self.iou, self.batch, self.tiling = tile, iou, batch, tiling
        self.stride = int(tile * (1 - overlap))

    def use_tiles(self, H: int, W: int) -> bool:
        return self.tiling == "on" or (self.tiling == "auto" and max(H, W) > 2 * self.tile)

    def __call__(self, img: np.ndarray) -> tuple[list[tuple[str, float, list[float]]], dict]:
        H, W = img.shape[:2]
        t0 = time.perf_counter()
        crops, offsets = [], []
        if not self.use_tiles(H, W):
            crops, offsets = [img], [(0, 0)]
        else:
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
    placed: list[tuple[int, int, int, int]] = []  # label rectangles already drawn (x1, y1, x2, y2)

    def free(r):
        return all(r[2] <= p[0] or r[0] >= p[2] or r[3] <= p[1] or r[1] >= p[3] for p in placed)

    for d in insp.defects:
        x1, y1, x2, y2 = d.bbox_px
        pad = 6 * th
        col = COLORS[d.severity]
        cv2.rectangle(out, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), col, 2 * th)
        label = f"{A.KNOWLEDGE[d.defect_class]['label']} {d.confidence:.2f}"
        (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * th, th)
        lh = tht + base + 2
        lx = int(min(max(0, x1 - pad), W - tw - 2))          # keep label inside the image
        # candidate baselines: above the box, below it, then further up/down until the label is clear of others
        cands = [y1 - pad - 4, y2 + pad + tht + 4] + [y1 - pad - 4 - k * lh for k in range(1, 6)] + \
                [y2 + pad + tht + 4 + k * lh for k in range(1, 6)]
        cands = [y for y in cands if tht + 2 <= y <= H - base - 2] or [max(tht + 2, min(H - 2, y1))]
        ly = next((y for y in cands if free((lx, y - tht - 2, lx + tw, y + base))), cands[0])
        placed.append((lx, ly - tht - 2, lx + tw, ly + base))
        cv2.rectangle(out, (lx - 1, ly - tht - 2), (lx + tw + 1, ly + base), (30, 30, 30), -1)  # dark backing so every colour is readable
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
def inspect_one(img_path: Path, det: Detector, backend: str, cfg: dict, out_dir: Path, board_id: str,
                area_frac: float | None = 0.002):
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"cannot read {img_path}")
    H, W = img.shape[:2]
    t0 = time.perf_counter()
    dets, timing = det(img)
    recs = [A.make_record(c, p, b, W, H, area_frac) for c, p, b in dets if c in A.KNOWLEDGE]
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
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1, help="1 = bare board, 2 = assembled board")
    ap.add_argument("--pi", action="store_true", help="use the stage's NCNN model (Raspberry Pi)")
    ap.add_argument("--model", default=None, help="override the stage's model")
    ap.add_argument("--tiling", choices=["on", "off", "auto"], default=None, help="override the stage's tiling")
    ap.add_argument("--backend", choices=["template", "native", "inproc", "llama_main"], default="template")
    ap.add_argument("--config", default=str(ROOT / "advisor/advisor_config.json"))
    ap.add_argument("--conf", type=float, default=0.25, help="used for classes without a tuned threshold")
    ap.add_argument("--thresholds", default=None,
                    help="override the stage's per-class thresholds (vision/tune_thresholds.py; skipped if missing)")
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--limit", type=int, default=0, help="max images from a folder (0 = all)")
    a = ap.parse_args()

    out_dir = Path(a.out)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
    cfg = {}
    if a.backend != "template":
        cfg = {k: str(Path(v).expanduser()) if isinstance(v, str) and ("/" in v or v.startswith("~")) else v
               for k, v in json.loads(Path(a.config).read_text())["native" if a.backend == "inproc" else a.backend].items()}
    src = Path(a.source)
    images = list_images(src) if src.is_dir() else [src]
    if a.limit:
        images = images[:a.limit]

    model, thr_default = stage_paths(a.stage, a.pi)
    model = a.model or model
    thr_path = Path(a.thresholds or thr_default)
    thresholds = json.loads(thr_path.read_text()) if thr_path.exists() else {}
    print(f"{STAGES[a.stage]['name']} | model {model}")
    print("class thresholds:", thresholds or f"none (using {a.conf} for all)")
    det = Detector(model, conf=a.conf, class_thresholds=thresholds, tile=STAGES[a.stage]["imgsz"],
                   tiling=a.tiling or STAGES[a.stage]["tiling"])
    det(np.zeros((640, 640, 3), np.uint8))  # warm-up so the first board's timing is fair
    start_n = sum(1 for _ in open(out_dir / "inspections.csv")) - 1 if (out_dir / "inspections.csv").exists() else 0
    for n, p in enumerate(images, start=start_n + 1):
        insp, t = inspect_one(p, det, a.backend, cfg, out_dir, f"B-{n:04d}", STAGES[a.stage]["area_frac"])
        print(f"\n[{insp.board_id}] {p.name}  {insp.verdict}  | {t['tiles']} tiles, vision {t['vision_ms']} ms, "
              f"advisor {t['advisor_ms']} ms ({insp.advisor_backend})")
        print("  " + insp.advice.replace("\n", "\n  "))
    print(f"\nlog: {out_dir / 'inspections.csv'}   images: {out_dir / 'annotated'}")


if __name__ == "__main__":
    main()
