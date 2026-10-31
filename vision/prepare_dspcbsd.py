#!/usr/bin/env python3
"""
Build the DsPCBSD+ leakage-safer V3 split (train/valid/test) in YOLO format with TraceIQ's
unified 10-class list, and write data.yaml files for training PKU + DsPCBSD+ together.

Inputs (in --src):
  Data_YOLO/images/{train,val}/*.jpg, Data_YOLO/labels/{train,val}/*.txt   (original DsPCBSD+)
  Data_COCO/annotations/instances_train2017.json                            (class names / order)
  Supplementary_Data_1_DsPCBSDplus_V3_split.csv  (filename, split, image_sha256)

Every image is matched to the CSV by SHA-256 checksum, so a wrong or changed file is never used.
"""
import argparse, csv, hashlib, json, shutil
from collections import Counter
from pathlib import Path

import yaml

# TraceIQ unified classes: PKU-Market-PCB's 6 first (same ids as our PKU models), then DsPCBSD+ extras
UNIFIED = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper",
           "hole_breakout", "conductor_scratch", "conductor_foreign_object", "base_material_foreign_object"]
DS_TO_UNIFIED = {"SH": "short", "SP": "spur", "SC": "spurious_copper", "OP": "open_circuit",
                 "MB": "mouse_bite", "HB": "hole_breakout", "CS": "conductor_scratch",
                 "CFO": "conductor_foreign_object", "BMFO": "base_material_foreign_object"}
SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/traceiq/datasets/dspcbsd_raw")
    ap.add_argument("--dst", default="~/traceiq/datasets/dspcbsd_v3")
    ap.add_argument("--csv", default="Supplementary_Data_1_DsPCBSDplus_V3_split.csv")
    ap.add_argument("--pku", default="~/traceiq/datasets/pku_board_tiles",
                    help="tiled PKU dataset to combine with (board-held-out by default)")
    a = ap.parse_args()
    src, dst, pku = (Path(p).expanduser() for p in (a.src, a.dst, a.pku))

    # class order: COCO category id k  <->  YOLO class k-1  (standard COCO->YOLO conversion)
    coco = json.loads((src / "Data_COCO/annotations/instances_train2017.json").read_text())
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    ds_names = [c["name"] for c in cats]
    remap = {i: UNIFIED.index(DS_TO_UNIFIED[n]) for i, n in enumerate(ds_names)}
    print("DsPCBSD+ classes (YOLO id -> unified id):",
          {f"{i}:{n}": f"{remap[i]}:{UNIFIED[remap[i]]}" for i, n in enumerate(ds_names)})

    # sanity check: YOLO train label counts per class must equal COCO train annotation counts
    coco_counts = Counter(a_["category_id"] - 1 for a_ in coco["annotations"])
    yolo_counts = Counter()
    for lp in (src / "Data_YOLO/labels/train").glob("*.txt"):
        for line in lp.read_text().split("\n"):
            if line.strip():
                yolo_counts[int(line.split()[0])] += 1
    if coco_counts != yolo_counts:
        raise SystemExit(f"Class-order check FAILED.\n COCO: {dict(coco_counts)}\n YOLO: {dict(yolo_counts)}")
    print("class-order check OK (YOLO and COCO train counts match per class)")

    # sha256 -> split from the leakage-safer V3 manifest
    by_sha = {}
    with open(src / a.csv) as f:
        for row in csv.DictReader(f):
            by_sha[row["image_sha256"]] = (row["split"], row["filename"])
    print("manifest rows:", len(by_sha))

    for s in SPLIT_DIR.values():
        (dst / s / "images").mkdir(parents=True, exist_ok=True)
        (dst / s / "labels").mkdir(parents=True, exist_ok=True)

    placed, boxes, unmatched, no_label = Counter(), {s: Counter() for s in SPLIT_DIR.values()}, [], 0
    for orig in ["train", "val"]:
        for ip in sorted((src / "Data_YOLO/images" / orig).glob("*.jpg")):
            hit = by_sha.get(sha256(ip))
            if hit is None:
                unmatched.append(ip.name)
                continue
            split = SPLIT_DIR[hit[0]]
            lp = src / "Data_YOLO/labels" / orig / (ip.stem + ".txt")
            out = []
            if lp.exists():
                for line in lp.read_text().split("\n"):
                    p = line.split()
                    if len(p) == 5:
                        c = remap[int(p[0])]
                        out.append(" ".join([str(c)] + p[1:]))
                        boxes[split][UNIFIED[c]] += 1
            else:
                no_label += 1
            shutil.copy(ip, dst / split / "images" / ip.name)
            (dst / split / "labels" / (ip.stem + ".txt")).write_text("\n".join(out) + ("\n" if out else ""))
            placed[split] += 1

    print(f"placed: train {placed['train']}, valid {placed['valid']}, test {placed['test']} "
          f"(expected 7537 / 876 / 1846); unmatched: {len(unmatched)}; images without label file: {no_label}")
    if unmatched:
        print("first unmatched:", unmatched[:5])
    print(f"{'class':30s} {'train':>6s} {'valid':>6s} {'test':>6s}")
    for n in UNIFIED:
        print(f"{n:30s} {boxes['train'][n]:6d} {boxes['valid'][n]:6d} {boxes['test'][n]:6d}")

    names = {i: n for i, n in enumerate(UNIFIED)}
    yamls = {
        "dspcbsd_only.yaml": {"train": [str(dst / "train/images")], "val": [str(dst / "valid/images")],
                              "test": [str(dst / "test/images")]},
        "combined_pku_dspcbsd.yaml": {
            "train": [str(pku / "train/images"), str(dst / "train/images")],
            "val": [str(pku / "valid/images"), str(dst / "valid/images")],
            "test": [str(pku / "test/images"), str(dst / "test/images")]},
        "test_pku_only.yaml": {"train": [str(pku / "train/images")], "val": [str(pku / "valid/images")],
                               "test": [str(pku / "test/images")]},
    }
    for fn, d in yamls.items():
        d["names"] = names
        (dst / fn).write_text(yaml.safe_dump(d, sort_keys=False))
        print("wrote", dst / fn)


if __name__ == "__main__":
    main()
