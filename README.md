# TraceIQ

**On-device PCB defect inspection with a Llama advisor, running on a Raspberry Pi 5.**
Entry for the **Bharat AI-SoC Challenge 2026–27**, Problem Statement 5: *Real-time industrial defect detection & quality-control assistant.*

A camera above an inspection bench captures a PCB. A small YOLO model running on the Pi's Arm CPU detects and classifies defects. Each detection becomes a structured defect record (class, severity, location, confidence). A **Llama 3.2 1B Instruct** model, deployed on-device with **ExecuTorch**, turns those records into plain-language explanations, likely causes and corrective actions for the operator, and writes the inspection log. Everything runs offline.

```
Camera → Pre-process (OpenCV) → Vision model (YOLO) → Defect record (JSON) → Llama + ExecuTorch → Operator UI + log
```

Two inspection stages are planned:

| Stage | What it inspects | Status |
|---|---|---|
| 1. Bare board | PCB after fabrication, before assembly (shorts, opens, spurs, mouse bites, missing holes, spurious copper) | Baseline model trained |
| 2. Assembled board | PCB after soldering (solder bridges, missing/shifted components, insufficient/excess solder) | Planned |

## Status

- [x] Vision baseline: YOLO11n on PKU-Market-PCB, tiled to 640×640
- [x] Honest evaluation splits: stratified by class, and board-held-out (test boards never seen in training)
- [x] Baseline evaluated: test mAP50 0.970 on unseen boards, 0.974 stratified
- [x] Stage 1 upgrade: added DsPCBSD+ (real factory defects), 10-class combined model
- [ ] Stage 2: assembled-board model
- [ ] Advisor: Llama 3.2 1B Instruct exported with ExecuTorch, running on Pi 5
- [ ] Pipeline, operator UI, logging, GPIO alert
- [ ] Benchmarks on Pi 5 (FPS, latency, memory, temperature)

## Results so far: Stage 1 baseline (YOLO11n, 2.6 M parameters, 640×640 tiles)

Trained on an RTX 5050 Laptop GPU, 100 epochs. Scores are on held-out **test** sets that were never used for training or model selection.

| Class | Unseen boards (08 + 12): P / R / mAP50 | Stratified: P / R / mAP50 |
|---|---|---|
| missing_hole | 0.993 / 1.000 / 0.995 | 0.928 / 1.000 / 0.968 |
| mouse_bite | 0.899 / 0.963 / 0.954 | 0.858 / 0.945 / 0.984 |
| open_circuit | 0.975 / 0.967 / 0.965 | 0.991 / 0.976 / 0.995 |
| short | 0.931 / 0.988 / 0.960 | 0.936 / 0.945 / 0.939 |
| spur | 0.910 / 0.925 / 0.975 | 0.912 / 0.976 / 0.973 |
| spurious_copper | 0.945 / 0.964 / 0.972 | 0.969 / 0.960 / 0.985 |
| **All** | **0.942 / 0.968 / 0.970** (mAP50-95 0.502) | **0.933 / 0.967 / 0.974** (mAP50-95 0.554) |

- **Unseen boards**: whole board designs 08 and 12 held out (120 photos, 2,940 tiles, 975 defects). Shows the model generalises to board layouts it has never seen.
- **Stratified**: 12 test photos per defect class (72 photos, 1,697 tiles, 517 defects).
- Weakest classes: mouse_bite and spur precision (thin defects that resemble normal copper edges).
- These are tile-level scores on PKU-Market-PCB, whose defects were added digitally. Real-board accuracy will be measured separately.

## Results: combined model (PKU-Market-PCB + DsPCBSD+, 10 classes, YOLO11n)

Trained 100 epochs on PKU board-held-out tiles + DsPCBSD+ leakage-safer V3 split. Test-set scores per source:

| Test set | Images | Defects | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| PKU-Market-PCB, unseen boards 08 + 12 (synthetic defects) | 2,940 tiles | 975 | 0.906 | 0.939 | **0.964** | 0.492 |
| DsPCBSD+ V3 test (real factory AOI defects) | 1,846 | 4,052 | 0.724 | 0.704 | **0.746** | 0.417 |
| Both together | 4,786 | 5,027 | 0.699 | 0.759 | 0.809 | 0.449 |

DsPCBSD+ test, per class (P / R / mAP50): mouse_bite 0.666 / 0.640 / 0.669 · open_circuit 0.650 / 0.823 / 0.786 · short 0.790 / 0.867 / 0.881 · spur 0.711 / 0.642 / 0.727 · spurious_copper 0.754 / 0.520 / 0.664 · hole_breakout 0.871 / 0.961 / 0.978 · conductor_scratch 0.667 / 0.623 / 0.670 · conductor_foreign_object 0.659 / 0.490 / 0.558 · base_material_foreign_object 0.746 / 0.768 / 0.783.

Notes:
- Adding real factory data kept PKU accuracy almost unchanged (0.964 vs 0.970 for the PKU-only model) while adding 4 new defect types.
- 0.746 mAP50 on DsPCBSD+ is the realistic figure for real defects; thin defects (mouse bite, spur, spurious copper, scratches, conductor foreign objects) are the weakest.
- missing_hole precision is low on the combined test because DsPCBSD+ does not annotate that class: correct detections of holes in DsPCBSD+ images count as false positives. This is a label-space mismatch between datasets, reported as a limitation.

## Repository layout

```
vision/
  tile_dataset.py       cut full-size PCB photos + YOLO labels into overlapping 640×640 tiles
  stratified_split.py   80/10/10 split per defect class (every class in every split)
  board_split.py        hold out whole board designs for valid/test
run_overnight.sh        train + test both evaluation setups back to back
```

Datasets, training runs and model weights are not stored in git (see `.gitignore`).

## Reproduce the vision baseline (Ubuntu + NVIDIA GPU)

```bash
conda create -yn vision python=3.11 && conda activate vision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics

# 1. Download PKU-Market-PCB in YOLOv8 format from Roboflow Universe
#    (pcbdefect-eftc9/pcb-defect-djz4r) and unzip into ~/traceiq/datasets/pku_raw

# 2. Build the two evaluation splits and tile them
python vision/stratified_split.py
python vision/board_split.py --test 08 12 --valid 11
python vision/tile_dataset.py --src ~/traceiq/datasets/pku_strat --dst ~/traceiq/datasets/pku_strat_tiles
python vision/tile_dataset.py --src ~/traceiq/datasets/pku_board --dst ~/traceiq/datasets/pku_board_tiles

# 3. Train and test both models (~5 h on an RTX 5050 Laptop GPU)
nohup bash run_overnight.sh > overnight.log 2>&1 &
```

## Advisor model

- **Llama 3.2 1B Instruct** by Meta, obtained via `unsloth/Llama-3.2-1B-Instruct` on Hugging Face, used under the Llama 3.2 Community License. **Built with Llama.**
- `advisor/convert_hf_to_meta.py` converts the Hugging Face checkpoint to Meta's original format (`consolidated.00.pth`, `params.json`, `tokenizer.model`) for ExecuTorch's Llama export. Weights are not stored in this repo.

## Datasets and credit

- **DsPCBSD+**: S. Lv et al., *A dataset for deep learning based detection of printed circuit board surface defect*, Scientific Data (2024); Figshare DOI 10.6084/m9.figshare.24970329, CC BY 4.0.
- **DsPCBSD+ leakage-safer V3 split**: Yihong Ma (2026), Figshare DOI 10.6084/m9.figshare.34355967 (split manifest only).

- **PKU-Market-PCB**: W. Huang and P. Wei, *A PCB Dataset for Defects Detection and Classification*, arXiv:1901.08204 (2019), Peking University Open Lab on Human Robot Interaction. YOLO-format copy from Roboflow Universe, CC BY 4.0. Defects in this dataset were added digitally to photos of about 10 board designs; see Limitations.

## Known limitations (tracked)

1. PKU defects are synthetic and cover ~10 board designs, so dataset scores are optimistic. Mitigation: board-held-out test, real-factory data (DsPCBSD+), and a separate test on our own Pi-camera photos.
2. Stage 1 covers bare boards only. Assembled boards are Stage 2.
3. Pi 5 speed with tiling is not yet measured.

## Team

PSG Institute of Technology and Applied Research, Coimbatore.
