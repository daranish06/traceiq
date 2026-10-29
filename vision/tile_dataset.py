#!/usr/bin/env python3
"""TraceIQ: cut full-size PCB photos + YOLO labels into overlapping tiles."""
import argparse, random
from pathlib import Path
import cv2, yaml

def load_labels(p):
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                c, x, y, w, h = parts
                out.append((int(c), float(x), float(y), float(w), float(h)))
    return out

def starts(length, tile, stride):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:
        s.append(length - tile)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min_vis", type=float, default=0.6)   # keep box if >=60% visible in tile
    ap.add_argument("--bg_keep", type=float, default=0.15)  # fraction of empty TRAIN tiles kept
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    src, dst = Path(a.src).expanduser(), Path(a.dst).expanduser()
    stride = int(a.tile * (1 - a.overlap))

    for split in ["train", "valid", "test"]:
        idir, ldir = src / split / "images", src / split / "labels"
        if not idir.exists():
            continue
        oi, ol = dst / split / "images", dst / split / "labels"
        oi.mkdir(parents=True, exist_ok=True); ol.mkdir(parents=True, exist_ok=True)
        n_img = n_tiles = n_box = n_bg = 0
        for ip in sorted(idir.iterdir()):
            if ip.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = cv2.imread(str(ip))
            if img is None:
                continue
            H, W = img.shape[:2]; n_img += 1
            boxes = [(c, (x - w/2)*W, (y - h/2)*H, (x + w/2)*W, (y + h/2)*H)
                     for c, x, y, w, h in load_labels(ldir / (ip.stem + ".txt"))]
            for ty in starts(H, a.tile, stride):
                for tx in starts(W, a.tile, stride):
                    tw, th = min(a.tile, W - tx), min(a.tile, H - ty)
                    lines = []
                    for c, x1, y1, x2, y2 in boxes:
                        ix1, iy1 = max(x1, tx), max(y1, ty)
                        ix2, iy2 = min(x2, tx + tw), min(y2, ty + th)
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        vis = (ix2 - ix1) * (iy2 - iy1) / max((x2 - x1) * (y2 - y1), 1e-6)
                        if vis < a.min_vis:
                            continue
                        lines.append(f"{c} {((ix1+ix2)/2 - tx)/tw:.6f} {((iy1+iy2)/2 - ty)/th:.6f} "
                                     f"{(ix2-ix1)/tw:.6f} {(iy2-iy1)/th:.6f}")
                    if not lines:
                        if split == "train" and random.random() > a.bg_keep:
                            continue
                        n_bg += 1
                    name = f"{ip.stem}_{tx}_{ty}"
                    cv2.imwrite(str(oi / (name + ".jpg")), img[ty:ty+th, tx:tx+tw],
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (ol / (name + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
                    n_tiles += 1; n_box += len(lines)
        print(f"{split}: {n_img} photos -> {n_tiles} tiles ({n_box} boxes, {n_bg} background tiles)")

    names = yaml.safe_load((src / "data.yaml").read_text())["names"]
    (dst / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(dst), "train": "train/images", "val": "valid/images", "test": "test/images",
         "names": {i: n for i, n in enumerate(names)}}, sort_keys=False))
    print("wrote", dst / "data.yaml")

if __name__ == "__main__":
    main()
