#!/bin/bash
# Install v0.1.26 + the review-fix branch's diff, rebuild the AOT cache. Foreground.
set -euo pipefail
V=/projects/val; SP=/usr/local/lib/python3.12/dist-packages; R=/projects/aiter-wip; D=$SP/aiter/ops/flydsl/kernels
BRANCH=wip/a16w16-review-fixes; HEAD=3fa2515f1; BASE_SUBJECT="Retune gfx950 preshuffled AFP4WFP4"
F="aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py aiter/ops/flydsl/gemm_kernels.py aiter/tuned_gemm.py"
mkdir -p $V/logs && echo "$F" > $V/files
git -C $R fetch -q --depth 12 origin +$BRANCH:refs/remotes/origin/pra
[ "$(git -C $R rev-parse --short=9 origin/pra)" = $HEAD ] || { echo "branch head is not $HEAD"; exit 1; }
BASE=$(git -C $R rev-parse origin/pra~8)
git -C $R log -1 --format=%s $BASE | grep -q "$BASE_SUBJECT" || { echo "origin/pra~8 is not the PR base"; exit 1; }
# Keep the previous PR kernel as module "prb" for the ISA comparison.
cp $D/gemm_a16w16_gfx950_utils_pra.py $D/gemm_a16w16_gfx950_utils_prb.py
sed "s/gemm_a16w16_gfx950_utils_pra import/gemm_a16w16_gfx950_utils_prb import/" $D/gemm_a16w16_gfx950_pra.py > $D/gemm_a16w16_gfx950_prb.py
echo "module prb (previous PR head): c_tile=$(grep -c 'c_tile' $D/gemm_a16w16_gfx950_prb.py) c_is_contiguous=$(grep -c 'c_is_contiguous' $D/gemm_a16w16_gfx950_prb.py)"
rm -rf $V/pra && mkdir -p $V/pra
git -C $R archive v0.1.26 $F | tar -x -C $V/pra
git -C $R diff $BASE origin/pra -- $F > $V/pr.diff
(cd $V/pra && git apply --verbose $V/pr.diff 2>&1 | grep -E "^Applied|error")
cp $V/pra/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py $D/gemm_a16w16_gfx950_utils_pra.py
sed "s/from \.gemm_a16w16_gfx950_utils import/from .gemm_a16w16_gfx950_utils_pra import/" \
  $V/pra/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py > $D/gemm_a16w16_gfx950_pra.py
echo "module pra (new): c_tile=$(grep -c 'c_tile' $D/gemm_a16w16_gfx950_pra.py) span_check=$(grep -c '_span_bytes' $D/gemm_a16w16_gfx950_pra.py)"
bash $V/use.sh pra
C=$SP/aiter/jit/flydsl_cache; rm -rf $V/flydsl_cache_prev && mkdir -p $V/flydsl_cache_prev
mv $C/gemm_a16w16_gfx950_* $V/flydsl_cache_prev/ 2>/dev/null || true
FLYDSL_RUNTIME_CACHE_DIR=$C python3 -m aiter.aot.flydsl.gemm --csv /tmp/aiter_configs/bf16_tuned_gemm.csv 2>&1 | grep -E "FAIL|Compiled: +[0-9]"
rm -rf /projects/aiter-pr && git clone -q --depth 10 -b $BRANCH https://github.com/josh-schell-amd/aiter.git /projects/aiter-pr
echo "test checkout: $(git -C /projects/aiter-pr log --oneline -1)"
