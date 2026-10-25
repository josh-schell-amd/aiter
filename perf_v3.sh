#!/bin/bash
# Kernel A/B for the review fixes (after setup_v3.sh). Foreground, ~1.5 h.
set -uo pipefail
V=/projects/val; L=$V/logs/v3; mkdir -p $L
GPU=$(python3 -c "import torch; f=[torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]; b=max(range(len(f)), key=lambda i: f[i]); print(b if f[b] > 280*2**30 else -1)")
[ "$GPU" -ge 0 ] || { echo "no idle GPU, try again later"; exit 1; }
export HIP_VISIBLE_DEVICES=$GPU FLYDSL_RUNTIME_ENABLE_CACHE=0; echo "using GPU $GPU"
for pair in "stock stock2" "stock pra"; do
  echo "===== $pair (37 shapes)"
  python3 $V/ab.py $pair > $L/ab_${pair// /_}.log 2>&1
  grep -E " us |SUMMARY|ABORT|Error|Traceback" $L/ab_${pair// /_}.log
done
R="nt:8x4096x4096 nt:16x4096x4096 nt:32x4096x4096 nt:32x14336x4096 nt:16x28672x4096 nt:2048x2048x2048"
for pair in "pra stock" "prb pra" "pra prb"; do
  echo "===== $pair (decode rows)"
  python3 $V/ab.py $pair $R 2>&1 | tee $L/ab_rows_${pair// /_}.log | grep -E " us |SUMMARY|ABORT|Error|Traceback"
done
