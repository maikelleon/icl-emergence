#!/bin/bash
# Campaign 7: (A) SNR-matched sparse family, (B) d-sweep crossing seeds,
#             (C) stability boundary at lower constant LR, (D) no-curr M=inf anchor.
cd "$(dirname "$0")"
echo $$ > sweep7.pid

run() { # tag steps args...
  local tag=$1 steps=$2; shift 2
  if [ -f results/${tag}.jsonl ] && grep -q "\"step\": ${steps}" results/${tag}.jsonl; then
    echo "=== skip ${tag} (done) ==="; return
  fi
  echo "=== $(date +%H:%M:%S) starting ${tag} ==="
  python3 icl_pilot.py --steps ${steps} --tag ${tag} --auto-resume "$@"
}

ST=1.4142135623730951

# A) SNR-matched sparse family (post-hoc control; grid brackets both outcomes)
for seed in 0 1; do
  for M in 4 8 16 32 64; do
    run spm_m_M${M}_s${seed} 20000 --M ${M} --eval-every 2000 --curr --seed ${seed} \
        --family sparse --sparse-tau ${ST}
  done
done

# B) d-sweep crossing seeds
for seed in 1 2; do
  for M in 8 16;  do run d2_m_M${M}_s${seed} 20000 --M ${M} --eval-every 2000 --curr --seed ${seed} --dim 2; done
  for M in 16 32; do run d6_m_M${M}_s${seed} 20000 --M ${M} --eval-every 2000 --curr --seed ${seed} --dim 6; done
done

# C) stability boundary at lower constant LR (transience protocol, lr 3e-4)
for M in 16 32; do
  run m_M${M}_s0_lr3e4_long 100000 --M ${M} --eval-every 4000 --curr --curr-ramp 8000 \
      --schedule const --lr 3e-4 --seed 0
done

# D) curriculum-free infinite-diversity anchor at full budget
run nc_m_M0 20000 --M 0 --eval-every 2000 --seed 0

echo "=== CAMPAIGN7 COMPLETE $(date +%H:%M:%S) ==="
