#!/bin/bash
# Resume the model comparison after a restart: finish YOLO26n from its last saved epoch,
# test it, then train + test YOLO11s. (cmp_y11n_strong is already done.)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision
cd ~/traceiq
R=/home/daranish/traceiq/runs
D=datasets/dspcbsd_v3

test_model () {
  for T in test_pku_only dspcbsd_only; do
    echo "=== TEST  $1 on $T  $(date) ==="
    yolo detect val model=$R/$1/weights/best.pt data=$D/$T.yaml split=test device=0 project=$R name=test_${1}_$T exist_ok=True
  done
}

echo "=== RESUME cmp_y26n  $(date) ==="
yolo detect train resume model=$R/cmp_y26n/weights/last.pt
test_model cmp_y26n

echo "=== TRAIN cmp_y11s  $(date) ==="
yolo detect train model=yolo11s.pt data=$D/combined_pku_dspcbsd.yaml imgsz=640 epochs=100 batch=16 patience=30 \
  flipud=0.5 workers=8 device=0 exist_ok=True project=$R name=cmp_y11s
test_model cmp_y11s
echo "=== ALL DONE  $(date) ==="
