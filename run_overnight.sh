#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision
cd ~/traceiq
R=/home/daranish/traceiq/runs
COMMON="imgsz=640 epochs=100 batch=16 patience=30 flipud=0.5 workers=8 device=0 exist_ok=True"

echo "=== [1/4] TRAIN board-held-out  $(date) ==="
yolo detect train model=yolo11n.pt data=datasets/pku_board_tiles/data.yaml $COMMON project=$R name=pcb_board
echo "=== [2/4] TEST  board-held-out  $(date) ==="
yolo detect val model=$R/pcb_board/weights/best.pt data=datasets/pku_board_tiles/data.yaml split=test device=0 project=$R name=test_board exist_ok=True

echo "=== [3/4] TRAIN stratified  $(date) ==="
yolo detect train model=yolo11n.pt data=datasets/pku_strat_tiles/data.yaml $COMMON project=$R name=pcb_strat
echo "=== [4/4] TEST  stratified  $(date) ==="
yolo detect val model=$R/pcb_strat/weights/best.pt data=datasets/pku_strat_tiles/data.yaml split=test device=0 project=$R name=test_strat exist_ok=True

echo "=== ALL DONE  $(date) ==="
