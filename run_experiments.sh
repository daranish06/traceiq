#!/bin/bash
# Overnight model comparison on the combined PKU + DsPCBSD+ data (10 classes).
# Baseline already trained: runs/pcb_combined (YOLO11n, default augmentation).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision
cd ~/traceiq
R=/home/daranish/traceiq/runs
D=datasets/dspcbsd_v3
BASE="imgsz=640 epochs=100 batch=16 patience=30 flipud=0.5 workers=8 device=0 exist_ok=True"
STRONG="hsv_h=0.02 hsv_s=0.8 hsv_v=0.5 degrees=5 scale=0.5 mixup=0.1"

run () {  # $1=name  $2=model  $3=extra args
  echo "=== TRAIN $1  $(date) ==="
  yolo detect train model=$2 data=$D/combined_pku_dspcbsd.yaml $BASE $3 project=$R name=$1
  for T in test_pku_only dspcbsd_only; do
    echo "=== TEST  $1 on $T  $(date) ==="
    yolo detect val model=$R/$1/weights/best.pt data=$D/$T.yaml split=test device=0 project=$R name=test_${1}_$T exist_ok=True
  done
}

run cmp_y11n_strong yolo11n.pt "$STRONG"   # same model, stronger augmentation (robustness to camera/lighting)
run cmp_y26n        yolo26n.pt ""          # newer model designed for faster CPU inference
run cmp_y11s        yolo11s.pt ""          # bigger model: accuracy vs speed trade-off
echo "=== ALL DONE  $(date) ==="
