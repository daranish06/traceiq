#!/bin/bash
# Run on the LAPTOP. Copies code, models and the Llama runner to the Pi.
#   bash pi/copy_to_pi.sh <pi-user>@traceiq.local
set -e
PI=${1:-pi@traceiq.local}
cd ~/traceiq
rsync -av --progress --exclude datasets --exclude runs --exclude output --exclude llm --exclude .git \
  ./ "$PI":~/traceiq/
echo "copied. Next on the Pi: bash ~/traceiq/pi/setup_pi.sh"
