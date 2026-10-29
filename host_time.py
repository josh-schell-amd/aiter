# Per-call host time of gemm_a16w16: v0.1.26 (stock), previous PR (prb), latest PR (pra).
import importlib
import statistics
import time

import torch

mods = {
    n: importlib.import_module("aiter.ops.flydsl.kernels.gemm_a16w16_gfx950_" + n)
    for n in ("stock", "prb", "pra")
}
tile = dict(
    block_m=16, block_n=16, block_k=256, stages=4, split_k=1, m_waves=1,
    n_waves=1, k_waves=2, group_m=4, use_half_tile_interleaved=False,
)
m, n, k = 8, 4096, 4096
a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
a_strided = torch.randn(m, k + 64, dtype=torch.bfloat16, device="cuda")[:, :k]
w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
calls, rounds = 2000, 7
cases = {"stock": ("stock", a), "prb": ("prb", a), "pra": ("pra", a), "pra strided A": ("pra", a_strided)}


def per_call(mod, x):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(calls):
        mod.gemm_a16w16(x, w.t(), out=out, user_kwargs=tile, layout="nt")
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / calls * 1e6


for name, x in cases.values():
    for _ in range(50):
        mods[name].gemm_a16w16(x, w.t(), out=out, user_kwargs=tile, layout="nt")
res = {c: [] for c in cases}
for r in range(rounds):
    for c in list(cases) if r % 2 == 0 else list(reversed(cases)):
        res[c].append(per_call(mods[cases[c][0]], cases[c][1]))
for c, v in res.items():
    print(f"HOST {c:14s} median {statistics.median(v):7.2f} us/call  (min {min(v):.2f}, max {max(v):.2f})")
