#!/bin/bash
# Train YOLO11n on PKU-Market-PCB (board-held-out tiles) + DsPCBSD+ (leakage-safer V3), 10 classes,
# then test on each source separately and on both together.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision
cd ~/traceiq
R=/home/daranish/traceiq/runs
D=datasets/dspcbsd_v3
M=$R/pcb_combined/weights/best.pt

echo "=== TRAIN combined  $(date) ==="
yolo detect train model=yolo11n.pt data=$D/combined_pku_dspcbsd.yaml imgsz=640 epochs=100 batch=16 \
  patience=30 flipud=0.5 workers=8 device=0 exist_ok=True project=$R name=pcb_combined

for T in combined_pku_dspcbsd test_pku_only dspcbsd_only; do
  echo "=== TEST  $T  $(date) ==="
  yolo detect val model=$M data=$D/$T.yaml split=test device=0 project=$R name=test_combined_on_$T exist_ok=True
done
echo "=== ALL DONE  $(date) ==="
