#!/bin/bash
# One-time setup on the Raspberry Pi 5 (Raspberry Pi OS 64-bit, Trixie). Run on the Pi:
#   bash ~/traceiq/pi/setup_pi.sh
set -e
echo "== system packages"
sudo apt update
sudo apt install -y python3-venv python3-picamera2 python3-gpiozero python3-opencv git rsync
echo "== python environment (uses system picamera2/gpiozero/opencv)"
python3 -m venv --system-site-packages ~/tiq
source ~/tiq/bin/activate
pip install --upgrade pip
pip install ultralytics ncnn gradio pandas
echo "== Llama runner library path"
cd ~/traceiq/pi_deploy && chmod +x llama_main
echo "== checks"
echo -n "GLIBC: "; ldd --version | head -1
echo -n "RAM: "; free -h | awk '/Mem:/{print $2}'
vcgencmd measure_temp; vcgencmd get_throttled
python -c "import ultralytics, ncnn, gradio; print('python packages OK')"
LD_LIBRARY_PATH=$PWD ./llama_main --help > /dev/null && echo "llama_main runs OK"
echo "Setup done. Activate later with: source ~/tiq/bin/activate"
