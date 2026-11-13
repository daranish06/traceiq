#!/bin/bash
# Start the full TraceIQ pipeline on the Pi: camera -> YOLO (NCNN, Arm CPU) -> rule verdict ->
# Llama 3.2 1B (ExecuTorch, Arm CPU) -> buzzer/LEDs -> CSV/JSONL log. Open http://<pi-address>:7860
source ~/tiq/bin/activate
cd ~/traceiq
exec python app/ui.py --pi --backend llama_main --gpio "$@"
