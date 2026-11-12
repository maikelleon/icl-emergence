#!/bin/bash
# Master completion queue. Waits for the transience sweep, then runs, in order:
#   A) mechanism runs (checkpointed) + probes     B) M=4 anomaly seeds
#   C) d-sweep for the boundary law               D) held-out sparse family
#   E) large-row seed replication
# Every phase is resumable; finished cells are skipped.
echo $$ > "$(dirname "$0")/sweep5.pid"
cd "$(dirname "$0")"

until grep -q "TRANSIENCE SWEEP COMPLETE" sweep4*.log 2>/dev/null; do
  sleep 120
done
echo "=== sweep4 complete; starting completion queue $(date +%H:%M:%S) ==="

run() { # size layers dm heads M seed steps tag extra...
  local size=$1 L=$2 dm=$3 h=$4 M=$5 seed=$6 steps=$7 tag=$8; shift 8
  if [ -f results/${tag}.jsonl ] && grep -q "\"step\": ${steps}" results/${tag}.jsonl; then
    echo "=== skip ${tag} (done) ==="; return
  fi
  echo "=== $(date +%H:%M:%S) starting ${tag} ==="
  python3 icl_pilot.py --M $M --layers $L --dmodel $dm --heads $h \
    --steps $steps --eval-every 2000 --curr --seed $seed --tag $tag --auto-resume "$@"
}

# A) mechanism: checkpoint every 500 steps
for M in 2 16 2048; do
  run m 3 64 4 $M 0 20000 mech_m_M${M} --save-ckpt-every 500
done
for M in 2 16 2048; do
  if [ ! -f results/mech_mech_m_M${M}.jsonl ]; then
    echo "=== $(date +%H:%M:%S) probing mech_m_M${M} ==="
    python3 mech_probe.py --tag mech_m_M${M} --M $M
  fi
done

# B) M=4 anomaly seeds
for seed in 1 2; do run m 3 64 4 4 $seed 20000 m_M4_s${seed}; done

# C) d-sweep (medium): boundary versus dimension
for M in 2 4 8 16 32; do run m 3 64 4 $M 0 20000 d2_m_M${M} --dim 2; done
for M in 4 8 16 32 64 128; do run m 3 64 4 $M 0 20000 d6_m_M${M} --dim 6; done

# D) held-out sparse family (preregistered in prereg.md)
for seed in 0 1; do
  for M in 2 4 8 16 32 128; do
    run m 3 64 4 $M $seed 20000 sp_m_M${M}_s${seed} --family sparse
  done
done

# E) large-row seed replication
for seed in 1 2; do
  for M in 2 8 16 32 128 2048; do
    run l 4 128 4 $M $seed 20000 l_M${M}_s${seed}
  done
done

echo "=== FULL PIPELINE COMPLETE $(date +%H:%M:%S) ==="
