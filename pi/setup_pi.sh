#!/bin/bash
# One-time setup on the Raspberry Pi 5 (Raspberry Pi OS 64-bit, Trixie). Run on the Pi:
#   bash ~/traceiq/pi/setup_pi.sh
set -e
echo "== keep the board's bootloader (EEPROM) untouched (borrowed Pi)"
sudo systemctl mask rpi-eeprom-update || true
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
echo "== start TraceIQ automatically at boot (stop: sudo systemctl disable --now traceiq@$USER)"
chmod +x ~/traceiq/pi/run_traceiq.sh
sudo cp ~/traceiq/pi/traceiq@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable traceiq@$USER
echo "Setup done. Start now: sudo systemctl start traceiq@$USER   (logs: journalctl -u traceiq@$USER -f)"
echo "Manual run instead: bash ~/traceiq/pi/run_traceiq.sh"
