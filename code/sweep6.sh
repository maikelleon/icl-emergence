#!/bin/bash
# Campaign 6: (a) curriculum-free instrumented controls, (b) pool-draw replication.
cd "$(dirname "$0")"
echo $$ > sweep6.pid

run() { # tag steps extra-args...
  local tag=$1 steps=$2; shift 2
  if [ -f results/${tag}.jsonl ] && grep -q "\"step\": ${steps}" results/${tag}.jsonl; then
    echo "=== skip ${tag} (done) ==="; return
  fi
  echo "=== $(date +%H:%M:%S) starting ${tag} ==="
  python3 icl_pilot.py --steps ${steps} --tag ${tag} --auto-resume "$@"
}

# --- stage 1: curriculum-free instrumented controls (medium, 500-step ckpts)
run mechnc_m_M2048 20000 --M 2048 --eval-every 500 --save-ckpt-every 500
run mechnc_m_M16   20000 --M 16   --eval-every 500 --save-ckpt-every 500

# probes (skip if already done)
for M in 2048 16; do
  if [ ! -f results/mech_mechnc_m_M${M}.jsonl ]; then
    python3 mech_probe.py --tag mechnc_m_M${M} --M ${M}
  fi
done

# --- stage 2: pool replication (medium, main-campaign protocol, new pools)
for M in 2 8 16 32 128; do
  run poolB_m_M${M} 20000 --M ${M} --eval-every 2000 --curr --pool-seed-base 2000
done
for M in 8 16 32; do
  run poolC_m_M${M} 20000 --M ${M} --eval-every 2000 --curr --pool-seed-base 3000
done

echo "=== CAMPAIGN6 COMPLETE $(date +%H:%M:%S) ==="
