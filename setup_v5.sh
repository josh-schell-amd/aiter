#!/bin/bash
# Apply the latest PR commit (host-side span-check change) to the installed PR build.
# The previous PR build is kept as module "prb". Foreground.
set -euo pipefail
V=/projects/val; SP=/usr/local/lib/python3.12/dist-packages; D=$SP/aiter/ops/flydsl/kernels
BRANCH=wip/a16w16-review-fixes-4
cp $D/gemm_a16w16_gfx950_utils_pra.py $D/gemm_a16w16_gfx950_utils_prb.py
sed "s/gemm_a16w16_gfx950_utils_pra import/gemm_a16w16_gfx950_utils_prb import/" $D/gemm_a16w16_gfx950_pra.py > $D/gemm_a16w16_gfx950_prb.py
(cd $V/pra && git apply --verbose $V/incr.diff 2>&1 | grep -E "^Applied|error")
sed "s/from \.gemm_a16w16_gfx950_utils import/from .gemm_a16w16_gfx950_utils_pra import/" \
  $V/pra/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py > $D/gemm_a16w16_gfx950_pra.py
echo "module prb: _is_dense_2d=$(grep -c _is_dense_2d $D/gemm_a16w16_gfx950_prb.py)  module pra: _is_dense_2d=$(grep -c _is_dense_2d $D/gemm_a16w16_gfx950_pra.py)"
bash $V/use.sh pra
rm -rf /projects/aiter-pr && git clone -q --depth 4 -b $BRANCH https://github.com/josh-schell-amd/aiter.git /projects/aiter-pr
echo "test checkout: $(git -C /projects/aiter-pr log --oneline -1)"
