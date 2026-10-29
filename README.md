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
- [ ] Stage 1 upgrade: add DsPCBSD+ (real factory defects)
- [ ] Stage 2: assembled-board model
- [ ] Advisor: Llama 3.2 1B Instruct exported with ExecuTorch, running on Pi 5
- [ ] Pipeline, operator UI, logging, GPIO alert
- [ ] Benchmarks on Pi 5 (FPS, latency, memory, temperature)

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

## Datasets and credit

- **PKU-Market-PCB**: W. Huang and P. Wei, *A PCB Dataset for Defects Detection and Classification*, arXiv:1901.08204 (2019), Peking University Open Lab on Human Robot Interaction. YOLO-format copy from Roboflow Universe, CC BY 4.0. Defects in this dataset were added digitally to photos of about 10 board designs; see Limitations.

## Known limitations (tracked)

1. PKU defects are synthetic and cover ~10 board designs, so dataset scores are optimistic. Mitigation: board-held-out test, real-factory data (DsPCBSD+), and a separate test on our own Pi-camera photos.
2. Stage 1 covers bare boards only. Assembled boards are Stage 2.
3. Pi 5 speed with tiling is not yet measured.

## Team

PSG Institute of Technology and Applied Research, Coimbatore.
