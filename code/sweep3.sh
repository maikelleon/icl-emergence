#!/bin/bash
# Paper-grade overnight batch. Priority-ordered; resumable (skips finished cells).
cd "$(dirname "$0")"

run() {
  size=$1; L=$2; dm=$3; h=$4; M=$5; seed=$6
  tag=${size}_M${M}_s${seed}
  if [ -f results/${tag}.jsonl ] && grep -q '"step": 20000' results/${tag}.jsonl; then
    echo "=== skip ${tag} (done) ==="; return
  fi
  echo "=== $(date +%H:%M:%S) starting ${tag} ==="
  python3 icl_pilot.py --M $M --layers $L --dmodel $dm --heads $h \
    --steps 20000 --eval-every 2000 --curr --seed $seed --tag $tag
}

# 1) medium fine grid, seed 0 (fills the transition)
for M in 4 8 32 64 256; do run m 3 64 4 $M 0; done
# 2) small fine grid, seed 0
for M in 4 8 32 64 256; do run s 2 32 2 $M 0; done
# 3) large row core, seed 0 (third capacity)
for M in 2 16 128 2048 0; do run l 4 128 4 $M 0; done
# 4) medium seeds 1-2, transition + endpoints
for seed in 1 2; do for M in 2 8 16 32 128 2048; do run m 3 64 4 $M $seed; done; done
# 5) small seeds 1-2
for seed in 1 2; do for M in 2 8 16 32 128 2048; do run s 2 32 2 $M $seed; done; done
# 6) large extras in the transition
for M in 8 32; do run l 4 128 4 $M 0; done

echo "=== OVERNIGHT SWEEP COMPLETE $(date +%H:%M:%S) ==="
