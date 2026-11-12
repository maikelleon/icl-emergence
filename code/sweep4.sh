#!/bin/bash
# Transience campaign (paper section: stability/transience): 100k-step overtraining
echo $$ > "$(dirname "$0")/sweep4.pid"
# at boundary diversities. Constant post-warmup LR; curriculum ramp fixed at 8000
# steps (matching the phase-diagram runs). Resumable: skips finished cells.
cd "$(dirname "$0")"

run() {
  size=$1; L=$2; dm=$3; h=$4; M=$5; seed=$6
  tag=${size}_M${M}_s${seed}_long
  if [ -f results/${tag}.jsonl ] && grep -q '"step": 100000' results/${tag}.jsonl; then
    echo "=== skip ${tag} (done) ==="; return
  fi
  echo "=== $(date +%H:%M:%S) starting ${tag} ==="
  python3 icl_pilot.py --M $M --layers $L --dmodel $dm --heads $h \
    --steps 100000 --eval-every 4000 --curr --curr-ramp 8000 --schedule const \
    --seed $seed --tag $tag --auto-resume
}

# priority order: boundary first, medium then large, then replication/wings
run m 3 64 4 16 0
run l 4 128 4 16 0
run m 3 64 4 32 0
run m 3 64 4 8 0
run m 3 64 4 64 0
run m 3 64 4 16 1
run m 3 64 4 32 1
run s 2 32 2 8 0
run s 2 32 2 16 0
run s 2 32 2 32 0

echo "=== TRANSIENCE SWEEP COMPLETE $(date +%H:%M:%S) ==="
