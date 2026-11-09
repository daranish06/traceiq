#!/bin/bash
# After run_stage2.sh: install the chosen Stage 2 model, tune its thresholds on val,
# evaluate it on the test set, and export it for the Pi.
#   bash finish_stage2.sh asm_y11n_640 640      (or: asm_y11n_960 960)
set -e
RUN=${1:?usage: finish_stage2.sh <run name> <imgsz>}
IMGSZ=${2:?usage: finish_stage2.sh <run name> <imgsz>}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision
cd ~/traceiq
A=datasets/asm_raw/final_solder

cp runs/$RUN/weights/best.pt models/pcb_asm_y11n.pt
python vision/tune_thresholds.py --model models/pcb_asm_y11n.pt --val $A/images/val --imgsz $IMGSZ \
       --out models/class_thresholds_asm.json
python vision/eval_boards.py --stage 2 --images $A/images/test --imgsz $IMGSZ --tag asm_test
python vision/export_pi.py --model models/pcb_asm_y11n.pt --imgsz $IMGSZ --check $A/images/val
echo "Stage 2 model installed: $RUN (imgsz $IMGSZ)"
