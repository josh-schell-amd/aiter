#!/bin/bash
# ISA and correctness for the review fixes (after setup_v3.sh). Foreground.
set -uo pipefail
V=/projects/val; L=$V/logs/v3; mkdir -p $L
GPU=$(python3 -c "import torch; f=[torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]; b=max(range(len(f)), key=lambda i: f[i]); print(b if f[b] > 40*2**30 else -1)")
[ "$GPU" -ge 0 ] || { echo "no GPU with 40 GiB free, try again later"; exit 1; }
export HIP_VISIBLE_DEVICES=$GPU; echo "using GPU $GPU"
S="32x14336x4096 16x28672x4096 8x4096x4096 2048x2048x2048 4096x4096x8192 1x5120x2880:3 16x5120x2880:3 32x384x7168:8"
echo "== 1. ISA (8 configs x stock/prb/pra; expect '24 1')"
rm -rf /projects/isa3
for s in $S; do for v in stock prb pra; do
  FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/projects/isa3/gemm_a16w16_gfx950_${v}_${s/:/_ks} python3 /projects/dump_isa3.py gemm_a16w16_gfx950_$v $s 2>&1 | grep -c compiled
done; done | sort | uniq -c
echo "-- v0.1.26 vs new"; python3 /projects/isa_full.py gemm_a16w16_gfx950_stock gemm_a16w16_gfx950_pra $S
echo "-- previous PR head vs new"; python3 /projects/isa_full.py gemm_a16w16_gfx950_prb gemm_a16w16_gfx950_pra $S
echo "-- store-path diff vs v0.1.26 (drain region, non-metadata lines)"
python3 - <<'EOF'
import difflib, glob, re
norm = lambda l: re.sub(r"\b([vsa])\[?\d+(:\d+)?\]?", r"\1", l.split(";")[0].strip())
def drain(v, s):
    t = open(glob.glob(f"/projects/isa3/gemm_a16w16_gfx950_{v}_{s}/**/*final_isa.s", recursive=True)[0]).read().splitlines()
    body = [l for l in t if (l.startswith((" ", "\t")) and not l.strip().startswith((".", "-"))) or l.startswith(".LBB")]
    labels = {l.rstrip(":"): i for i, l in enumerate(body) if l.startswith(".LBB")}
    loops = [(labels[m.group(1)], i) for i, l in enumerate(body) for m in [re.search(r"s_cbranch_\w+\s+(\.LBB\w+)", l)] if m and m.group(1) in labels and labels[m.group(1)] < i]
    _, le = max(loops, key=lambda x: x[1] - x[0])
    return [norm(l) for l in body[le + 1:]]
for s in ("8x4096x4096", "16x28672x4096", "4096x4096x8192"):
    A, B = drain("stock", s), drain("pra", s)
    d = [l for l in difflib.unified_diff(A, B, n=0, lineterm="") if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    print(f"{s}: drain {len(A)} -> {len(B)} instrs, {len(d)} diff lines, vector {sum(1 for l in d if l[1:].startswith('v_'))}")
    for l in d[:14]: print("    ", l)
EOF
cd /projects/aiter-pr; echo "test checkout: $(git log --oneline -1)"
run() {
  local name=$1; shift
  timeout 3600 "$@" > $L/$name.log 2>&1; local rc=$?
  echo "$name: exit $rc  passed~ $(grep -c 'passed~' $L/$name.log)  failed! $(grep -c 'failed!' $L/$name.log)  $(grep -E '[0-9]+ (passed|failed|error)|^OK|^FAILED' $L/$name.log | tail -1 | cut -c1-80)"
  grep -E '^FAILED|AssertionError|OutOfMemory|Error:' $L/$name.log | grep -v 'failed!' | cut -c1-160 | sort | uniq -c | head -4
  echo "   non-asm failed!: $(grep 'failed!' $L/$name.log | grep -vc asm-)"
}
echo "== 2. tests"
run flydsl_hgemm python3 op_tests/test_flydsl_hgemm.py
run flydsl_a16w16 python3 op_tests/flydsl_tests/test_gemm_a16w16.py
run lookup python3 -m unittest op_tests.tuning_tests.test_a16w16_tuned_config_lookup -v
run gemm_a16w16 python3 op_tests/test_gemm_a16w16.py
echo "== 3. tgemm repro (N=6144, K=1280) and memory held"
cd /app
for m in 360000 768000 1677721 1677722; do
  AITER_LOG_TUNED_CONFIG=1 timeout 900 python3 $V/repro.py $m > $L/repro_$m.log 2>&1; rc=$?
  echo "M=$m exit=$rc $(grep -o -m1 'libtype is [a-z]*' $L/repro_$m.log) skipped=$(grep -c 'does not support' $L/repro_$m.log) $(grep -o 'max_err=.*' $L/repro_$m.log) $(grep -o -m1 -E 'Memory access fault|HSA_STATUS_ERROR_[A-Z_]*|OutOfMemory' $L/repro_$m.log)"
done
for a in "flydsl 340000" "flydsl 768000"; do python3 $V/mem.py $a 2>&1 | grep '^MEM'; done
