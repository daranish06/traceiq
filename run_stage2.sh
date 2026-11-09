#!/bin/bash
# Stage 2 (assembled boards, 9 solder/component classes) training.
# Run 1: YOLO11n at 640 (fast on the Pi). Run 2: YOLO11n at 960 (small solder joints keep more detail).
# Both use the stronger augmentation chosen for Stage 1. Note: 'oxidation' has no val/test examples,
# so its score cannot be measured; it is still trained.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision
cd ~/traceiq
R=/home/daranish/traceiq/runs
DATA=datasets/asm.yaml
BASE="epochs=100 patience=30 flipud=0.5 workers=8 device=0 exist_ok=True"
STRONG="hsv_h=0.02 hsv_s=0.8 hsv_v=0.5 degrees=5 scale=0.5 mixup=0.1"

run () {  # $1=name  $2=imgsz  $3=batch
  echo "=== TRAIN $1  $(date) ==="
  yolo detect train model=yolo11n.pt data=$DATA imgsz=$2 batch=$3 $BASE $STRONG project=$R name=$1
  echo "=== TEST  $1  $(date) ==="
  yolo detect val model=$R/$1/weights/best.pt data=$DATA imgsz=$2 split=test device=0 project=$R name=test_$1 exist_ok=True
}

run asm_y11n_640 640 16
run asm_y11n_960 960 8
echo "=== ALL DONE  $(date) ==="
