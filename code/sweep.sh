#!/bin/bash
cd "$(dirname "$0")"
for spec in "m 3 64 4" "s 2 32 2"; do
  set -- $spec
  size=$1; layers=$2; dm=$3; heads=$4
  for M in 0 2 16 128 2048; do
    echo "=== $(date +%H:%M:%S) starting ${size} M=${M} ==="
    python3 icl_pilot.py --M $M --layers $layers --dmodel $dm --heads $heads \
      --steps 20000 --eval-every 2000 --curr --tag ${size}_M${M}
  done
done
echo "=== SWEEP COMPLETE $(date +%H:%M:%S) ==="
