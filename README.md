# TraceIQ

**On-device PCB defect inspection with a Llama advisor, running on a Raspberry Pi 5.**
**Demo video:** https://youtu.be/6X447fa0w_k

Entry for the **Bharat AI-SoC Challenge 2026–27**, Problem Statement 5: *Real-time industrial defect detection & quality-control assistant.*

A camera above an inspection bench captures a PCB. A small YOLO model running on the Pi's Arm CPU detects and classifies defects. Each detection becomes a structured defect record (class, severity, location, confidence). A **Llama 3.2 1B Instruct** model, deployed on-device with **ExecuTorch**, turns those records into plain-language explanations, likely causes and corrective actions for the operator, and writes the inspection log. Everything runs offline.

```
Camera → Pre-process (OpenCV) → Vision model (YOLO) → Defect record (JSON) → Llama + ExecuTorch → Operator UI + log
```

Two inspection stages are planned:

| Stage | What it inspects | Status |
|---|---|---|
| 1. Bare board | PCB after fabrication, before assembly (shorts, opens, spurs, mouse bites, missing holes, spurious copper) | Baseline model trained |
| 2. Assembled board | PCB after soldering (solder bridges, missing/shifted components, insufficient/excess solder) | Model trained |

## Status

- [x] Vision baseline: YOLO11n on PKU-Market-PCB, tiled to 640×640
- [x] Honest evaluation splits: stratified by class, and board-held-out (test boards never seen in training)
- [x] Baseline evaluated: test mAP50 0.970 on unseen boards, 0.974 stratified
- [x] Stage 1 upgrade: added DsPCBSD+ (real factory defects), 10-class combined model
- [x] Model comparison (YOLO11n, YOLO11n + stronger augmentation, YOLO26n, YOLO11s); YOLO11n + stronger augmentation chosen
- [x] Advisor: Llama 3.2 1B Instruct exported with ExecuTorch (XNNPACK, 8-bit dynamic activations / 4-bit weights), with rule-based verdict, safety check and template fallback
- [x] Pipeline, operator dashboard (Gradio), CSV/JSONL logging, GPIO alert code
- [x] Vision model exported to NCNN and ONNX for the Arm CPU
- [x] Stage 2: assembled-board model (9 solder/component classes), selectable in the dashboard
- [x] Full pipeline deployed on Raspberry Pi 5 (8 GB, CPU only): camera, YOLO (NCNN), rule verdict, Llama (ExecuTorch), dashboard, log
- [x] Benchmarks on Pi 5 (latency, FPS, memory, temperature, Llama time-to-first-token and tokens/s)
- [x] Real-board test: 21 field-returned faulty boards under our own camera on the Pi (see below)
- [x] Offline test (Wi-Fi off, Ethernet unplugged), physical alert (LED + buzzer), auto-start at boot

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

## Model comparison (combined 10-class data, 100 epochs, test sets)

| Model | Params | PKU unseen boards P / R / mAP50 | DsPCBSD+ test P / R / mAP50 |
|---|---|---|---|
| YOLO11n (baseline) | 2.6 M | 0.906 / 0.939 / 0.964 | 0.724 / 0.704 / 0.746 |
| **YOLO11n + stronger augmentation (chosen)** | 2.6 M | **0.948 / 0.962 / 0.965** | **0.726 / 0.728 / 0.762** |
| YOLO26n | 2.4 M | 0.897 / 0.968 / 0.949 | 0.720 / 0.720 / 0.753 |
| YOLO11s | 9.4 M | 0.860 / 0.968 / 0.970 | 0.733 / 0.730 / 0.769 |

YOLO11s is only slightly better on DsPCBSD+ but is about 3.6× larger, which costs too much speed on a CPU-only Pi 5.
YOLO11n with stronger augmentation gives the best balance, so it is the deployed model.

## Results: full-board pipeline (what the operator sees)

Full pipeline on whole test images: 640×640 tiling, chosen YOLO11n model, merge across tiles,
per-class confidence thresholds tuned on the validation sets only (`vision/tune_thresholds.py`).
Evaluated with `vision/eval_boards.py` (match IoU 0.3).

| | PKU-Market-PCB, unseen boards 08 + 12 | DsPCBSD+ V3 test (real factory defects) |
|---|---|---|
| Boards / images | 120 | 1,846 |
| Recall (defects found) | **0.997** (599 / 601) | **0.705** (2,855 / 4,052) |
| Precision (alarms that are real) | **0.874** | **0.876** |
| False alarms per board | 0.72 | 0.22 |
| Boards with every defect found | 118 / 120 | 1,118 / 1,846 |
| Boards with no false alarm | 47 / 120 | 1,515 / 1,846 |
| Vision time per board, laptop GPU | 91 ms (24 tiles) | 4.4 ms (1 tile) |

Per class, DsPCBSD+ recall: hole breakout 0.93, open circuit 0.83, base-material foreign object 0.78,
short 0.78, spur 0.72, scratch 0.62, spurious copper 0.59, mouse bite 0.58, conductor foreign object 0.47.

Weak points:
- On PKU, 61 of the 86 false alarms are `short` (precision 0.62 for that class). On real DsPCBSD+ boards `short`
  precision is 0.92, so this mostly affects PKU's digitally edited images.
- `missing_hole` now has no false alarms (tuned threshold 0.6); the earlier model flagged SMD pads that have no hole by design.
- On DsPCBSD+, thin and low-contrast defects (conductor foreign objects, mouse bites, spurious copper, scratches)
  have the lowest recall.
- Planned improvement: golden-board comparison (compare against a known-good board of the same design).

## Results: Stage 2, assembled boards (YOLO11n, 9 classes)

Data: PCB solder and component defect photos (ParsaVictor, Hugging Face), 3,836 train / 687 val / 686 test images,
checked for exact duplicates across splits (none found, `vision/check_asm.py`). Images are close-ups, so the whole
image is used (no tiling).

Model choice (test set, YOLO validation):

| Model | P | R | mAP50 | mAP50-95 | Laptop GPU time |
|---|---|---|---|---|---|
| **YOLO11n @ 640 (chosen)** | 0.909 | **0.766** | **0.861** | 0.558 | **3.4 ms** |
| YOLO11n @ 960 | 0.924 | 0.740 | 0.858 | 0.576 | 8.7 ms |

960 px input gave no gain in mAP50 and is about 2.5× slower, so the 640 model is deployed.

Full pipeline with per-class thresholds tuned on val only (`vision/eval_boards.py --stage 2`, match IoU 0.3):

| Test set: 686 images (427 with no defect) | |
|---|---|
| Recall (defects found) | **0.828** (549 / 663) |
| Precision (alarms that are real) | **0.896** |
| False alarms per image | 0.09 |
| Images with every defect found | 611 / 686 |
| Images with no false alarm | 634 / 686 |

Per class, recall / precision: shifted component 1.00 / 0.89 · solder bridge 0.87 / 1.00 · insufficient solder 0.86 / 0.85 ·
missing component 0.84 / 0.85 · excess solder 0.83 / 0.95 · physical damage 0.75 / 0.86 · solder void 0.57 / 0.64 ·
burn 0.47 / 0.75.

Weak points: burn marks and solder voids (few test examples: 32 and 28). `oxidation` has no val/test examples in this
dataset, so it is trained but not measured.

## Results: on-device speed, Raspberry Pi 5 (8 GB, CPU only, no accelerator)

Measured with `bench/bench_pi.py` (browser closed, active cooler, official 27 W supply).

Vision (YOLO11n, NCNN on the Arm CPU):

| | Stage 1, bare board | Stage 2, assembled board |
|---|---|---|
| Vision per board, mean / p95 | 1,388 / 2,752 ms (15.1 tiles on average) | **83.7 / 122.7 ms** (1 tile) |
| Throughput | 0.71 boards/s (42 / min) | **10.6 FPS** (635 / min) |
| Pre-process / inference / post-process | 0.1 / 1,387.8 / 0.3 ms | 0.0 / 83.5 / 0.2 ms |
| Peak system RAM | 2.47 GB | 2.36 GB |
| Temperature start → end, throttling | 50.5 → 59.8 °C, none | 54.9 → 59.3 °C, none |

Advisor (Llama 3.2 1B Instruct, ExecuTorch XNNPACK, 4-bit weights, `llama_main` on the Arm CPU), mean of 3 runs:

| | |
|---|---|
| Model load | 2.5 s (8.8 s on a cold start) |
| Time to first token (304-token prompt) | **3.1 s** (prefill ≈ 100 tokens/s) |
| Generation speed | **12.0 tokens/s** |
| Response time excluding load (93 tokens) | **10.8 s** |
| Peak system RAM with vision model and Llama | **3.39 GB** of 8 GB |

Inference is over 99% of vision time, so model size and input resolution are the levers for speed. First real-board test:
a hand-assembled board under the Kreo Owl Lite webcam (1080p, fixed focus found with `pi/focus_sweep.py`) was
flagged FAIL for insufficient solder on unsoldered joints, with Llama advice accepted by the safety check.

## Results: real boards under our own camera (Raspberry Pi 5, 22 Nov 2026)

21 field-returned faulty boards (student-built through-hole boards, two bare boards, a red relay module, a blue
Arduino shield), 38 inspections including retakes and large boards in parts. Faults were reported as "not working",
so each alarm was judged visually from the captured images (not verified electrically); per-box judgements are in
`bench/realboard_review.csv`.

| Judgement of the 37 alarms | Count |
|---|---|
| Correct defect type | 2 |
| Real problem, wrong type (blobs/rework called "bridge"; empty IC footprints called "insufficient solder") | 10 |
| False alarm (mostly single normal joints called "solder bridge", confidence 0.30–0.46) | 18 |
| Unclear (hidden or blurred) | 7 |

Clean bare boards passed correctly; messy solder areas were found, but about half of the alarms were false and burn
marks and red/blue boards were missed. The public training data (close-ups of mostly factory boards) differ from
student-made boards under a desk webcam; fine-tuning on our own bench images is the Phase 2 fix. Llama advice on these
boards: 16 of 20 answers accepted by the safety check, about 16 s per board in live use.

## Repository layout

```
vision/   dataset prep and splits, tiling, threshold tuning, full-board evaluation, NCNN/ONNX export
advisor/  Llama prompt, safety check, template fallback, knowledge file, HF->Meta conversion, export config, advice eval
app/      inspection pipeline (inspect_board.py) and operator dashboard (ui.py)
pi/       Pi setup, copy script, auto-start service, camera check and focus sweep
bench/    Pi benchmark script, real-board review
run_*.sh  training runs (Stage 1, model comparison, Stage 2)
```

## Run on a Raspberry Pi 5

```bash
# on the laptop: copy code, models and the exported Llama to the Pi
bash pi/copy_to_pi.sh traceiq@<pi-address>
# on the Pi
bash ~/traceiq/pi/setup_pi.sh          # packages, Python env, auto-start at boot
python ~/traceiq/pi/focus_sweep.py     # board under the camera: lock the sharpest focus
bash ~/traceiq/pi/run_traceiq.sh       # then open http://localhost:7860 (or http://<pi-address>:7860)
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
- `advisor/convert_hf_to_meta.py` converts the Hugging Face checkpoint to Meta's original format (`consolidated.00.pth`, `params.json`, `tokenizer.model`) for ExecuTorch's Llama export. Exported with XNNPACK, 8-bit dynamic activations and 4-bit weights (group size 32). Weights are not stored in this repo.
- The verdict (PASS / REVIEW / FAIL) is decided by rules from the defect severities, never by the LLM. The LLM only explains.
- The prompt gives the detected defects plus one reference cause and action per type (`advisor/defect_knowledge.json`) and a one-shot example. Stage 2 examples have a single defect of a type that was not detected, so the 1B model has nothing extra to copy.
- Safety check on every answer: Problem / Cause / Action format, every detected type mentioned, no undetected type, no invented location, no denial. Clauses about an undetected type in the Cause/Action lines are removed first; if the answer still fails, a template built from the reference text is shown instead.

Advice check on 25 test boards per stage (`advisor/eval_advice.py`, Llama running in-process on the laptop CPU):

| | LLM answer used | Fallback to template | Mean advice time (laptop CPU) |
|---|---|---|---|
| Stage 1, bare boards | **96%** (23 / 24) | 1 | 30 s |
| Stage 2, assembled boards | **95%** (21 / 22) | 1 | 17 s |

Advice times vary with laptop CPU load; Pi 5 times will be measured separately. The rating sheets
(`bench/advice_eval*.csv`) have columns for a human score of correctness, grounding and usefulness.

## Datasets and credit

- **PCB solder defect detection** (Stage 2): ParsaVictor, Hugging Face `ParsaVictor/pcb-solder-defect-detection`, merged and re-annotated from several real-world sources; released for research, educational and internal engineering use. Not redistributed here.

- **DsPCBSD+**: S. Lv et al., *A dataset for deep learning based detection of printed circuit board surface defect*, Scientific Data (2024); Figshare DOI 10.6084/m9.figshare.24970329, CC BY 4.0.
- **DsPCBSD+ leakage-safer V3 split**: Yihong Ma (2026), Figshare DOI 10.6084/m9.figshare.34355967 (split manifest only).

- **PKU-Market-PCB**: W. Huang and P. Wei, *A PCB Dataset for Defects Detection and Classification*, arXiv:1901.08204 (2019), Peking University Open Lab on Human Robot Interaction. YOLO-format copy from Roboflow Universe, CC BY 4.0. Defects in this dataset were added digitally to photos of about 10 board designs; see Limitations.

## Known limitations (tracked)

1. PKU defects are synthetic and cover ~10 board designs, so dataset scores are optimistic. Mitigation: board-held-out test, real-factory data (DsPCBSD+), and a separate test on our own Pi-camera photos.
2. Stage 2 (assembled boards) is trained on public close-up photos; accuracy on whole boards seen by our own camera is still to be measured.
3. Stage 1 on full-board photos takes about 1.4 s per board on the Pi CPU (about 15 tiles per board); fine for a place-and-inspect station, not for a fast conveyor.
4. PKU-Market-PCB and DsPCBSD+ annotate different defect types (DsPCBSD+ has no `missing_hole` labels), so combined-test precision for some classes is understated.

## Team

Balaharish M (Team Lead), Daranish K, Hari Ragavan B — PSG Institute of Technology and Applied Research, Coimbatore.
