#!/bin/bash
# Test the PR on AITER main (editable source install) inside jschell_aiter_main. Foreground.
# Usage: bash main_test.sh install | before | after
set -uo pipefail
V=/projects/val; L=$V/logs/main; mkdir -p $L
SRC=/projects/aiter-main-test
MAIN=a24724ef8   # main commit merged into the PR head
PR=bf9caa742     # PR head
pick_gpu() {
  python3 -c "import torch; f=[torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]; b=max(range(len(f)), key=lambda i: f[i]); print(b if f[b] > 40*2**30 else -1)"
}
run() {
  local name=$1; shift
  timeout 5400 "$@" > $L/$name.log 2>&1; local rc=$?
  echo "$name: exit $rc  passed~ $(grep -c 'passed~' $L/$name.log)  failed! $(grep -c 'failed!' $L/$name.log)  $(grep -E '[0-9]+ (passed|failed|error)|^OK$|^FAILED' $L/$name.log | tail -1 | cut -c1-80)"
  grep -E '^FAILED|AssertionError|OutOfMemory|Error:|not honoured' $L/$name.log | grep -v 'failed!' | cut -c1-170 | sort | uniq -c | head -4
  echo "   non-asm failed!: $(grep 'failed!' $L/$name.log | grep -vc asm-)"
}
repro() {
  for m in 340000 360000 768000; do
    AITER_LOG_TUNED_CONFIG=1 timeout 1800 python3 $V/repro.py $m > $L/repro_$1_$m.log 2>&1; rc=$?
    echo "M=$m exit=$rc $(grep -o -m1 'libtype is [a-z]*' $L/repro_$1_$m.log) $(grep -o 'max_err=.*' $L/repro_$1_$m.log) $(grep -o -m1 -E 'Memory access fault|HSA_STATUS_ERROR_[A-Z_]*|illegal memory access' $L/repro_$1_$m.log)"
  done
}
case "${1:-}" in
  install)
    rm -rf $SRC
    git clone -q --recursive -b fix/flydsl-hgemm-int64-offsets https://github.com/josh-schell-amd/aiter.git $SRC || exit 1
    cd $SRC && git checkout -q $PR && git log --oneline -1
    pip uninstall -y amd-aiter aiter >/dev/null 2>&1
    AITER_USE_SYSTEM_TRITON=1 python3 -m pip install -q -e . 2>&1 | tail -3
    cd /app && python3 -c "import aiter, flydsl, os; print('aiter from', os.path.dirname(aiter.__file__)); print('flydsl', flydsl.__version__)"
    ;;
  before)
    cd $SRC && git checkout -q $MAIN && echo "== AITER main $(git log --oneline -1 | cut -c1-70)"
    export HIP_VISIBLE_DEVICES=$(pick_gpu) FLYDSL_RUNTIME_ENABLE_CACHE=0
    [ "$HIP_VISIBLE_DEVICES" -ge 0 ] || { echo "no GPU with 40 GiB free"; exit 1; }
    cd /app && rm -rf /tmp/aiter_configs && repro main
    cd $SRC && git checkout -q $PR
    ;;
  after)
    cd $SRC && git checkout -q $PR && echo "== AITER main + PR $(git log --oneline -1 | cut -c1-70)"
    export HIP_VISIBLE_DEVICES=$(pick_gpu) FLYDSL_RUNTIME_ENABLE_CACHE=0
    [ "$HIP_VISIBLE_DEVICES" -ge 0 ] || { echo "no GPU with 40 GiB free"; exit 1; }
    cd /app && rm -rf /tmp/aiter_configs && repro pr
    for args in "flydsl 340000" "flydsl 768000"; do python3 $V/mem.py $args 2>&1 | grep '^MEM'; done
    cd $SRC
    run flydsl_hgemm python3 op_tests/test_flydsl_hgemm.py
    run flydsl_a16w16 python3 op_tests/flydsl_tests/test_gemm_a16w16.py
    run lookup python3 -m unittest op_tests.tuning_tests.test_a16w16_tuned_config_lookup
    run gemm_a16w16 python3 op_tests/test_gemm_a16w16.py -t normal skinny out
    run aot python3 -m aiter.aot.flydsl.gemm
    grep -E "Compiled: +[0-9]" $L/aot.log | tail -1
    ;;
  *) echo "usage: main_test.sh install | before | after"; exit 1 ;;
esac
