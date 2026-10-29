#!/bin/bash
# Switch the container to FlyDSL 0.4.1 (AITER main's pin) and rebuild the AOT cache. Foreground.
# Undo: pip install --no-deps flydsl==0.4.0, then rerun the AOT step below.
set -euo pipefail
SP=/usr/local/lib/python3.12/dist-packages; V=/projects/val
echo "before: $(pip show flydsl 2>/dev/null | grep Version)"
pip install -q --no-deps "flydsl==0.4.1"
echo "after:  $(pip show flydsl 2>/dev/null | grep Version)"
C=$SP/aiter/jit/flydsl_cache; rm -rf $V/flydsl_cache_prev && mkdir -p $V/flydsl_cache_prev
mv $C/gemm_a16w16_gfx950_* $V/flydsl_cache_prev/ 2>/dev/null || true
FLYDSL_RUNTIME_CACHE_DIR=$C python3 -m aiter.aot.flydsl.gemm --csv /tmp/aiter_configs/bf16_tuned_gemm.csv 2>&1 | grep -E "FAIL|Compiled: +[0-9]"
