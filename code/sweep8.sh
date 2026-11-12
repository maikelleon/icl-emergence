#!/bin/bash
# Campaign 8: capacity extension (xlarge, 5 layers / d_model 192 / 6 heads, 2.23M
# params) at the boundary and anchors, plus completion of the remaining grid cells.
cd "$(dirname "$0")"
echo $$ > sweep8.pid

run() { # tag steps args...
  local tag=$1 steps=$2; shift 2
  if [ -f results/${tag}.jsonl ] && grep -q "\"step\": ${steps}" results/${tag}.jsonl; then
    echo "=== skip ${tag} (done) ==="; return
  fi
  echo "=== $(date +%H:%M:%S) starting ${tag} ==="
  python3 icl_pilot.py --steps ${steps} --eval-every 2000 --curr --auto-resume --tag ${tag} "$@"
}

XL="--layers 5 --dmodel 192 --heads 6"
LG="--layers 4 --dmodel 128 --heads 4"
SM="--layers 2 --dmodel 32 --heads 2"

# A) xlarge: boundary first, then anchors
for M in 16 8 32 2 128; do
  run xl_M${M}_s0 20000 --M ${M} ${XL} --seed 0
done

# B) large: missing diversities at seed 0
for M in 4 64 256; do
  run l_M${M}_s0 20000 --M ${M} ${LG} --seed 0
done

# C) medium: missing seeds at M in {64, 256}
for seed in 1 2; do
  for M in 64 256; do
    run m_M${M}_s${seed} 20000 --M ${M} --seed ${seed}
  done
done

# D) small: missing seeds at M in {4, 64, 256}
for seed in 1 2; do
  for M in 4 64 256; do
    run s_M${M}_s${seed} 20000 --M ${M} ${SM} --seed ${seed}
  done
done

echo "=== CAMPAIGN8 COMPLETE $(date +%H:%M:%S) ==="
