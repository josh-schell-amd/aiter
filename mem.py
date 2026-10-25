import gc, sys, torch
path, M = sys.argv[1], int(sys.argv[2])
f0, tot = torch.cuda.mem_get_info()
if (tot - f0) / 2**30 > 1.5:
    sys.exit(f"MEM ABORT: {(tot - f0) / 2**30:.1f} GiB already in use on this GPU")
from aiter.tuned_gemm import tgemm
def non_torch():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free - torch.cuda.memory_reserved()) / 2**30
w = torch.randn(6144, 1280, dtype=torch.bfloat16, device="cuda")
b = torch.randn(6144, dtype=torch.bfloat16, device="cuda")
a = torch.randn(M, 1280, dtype=torch.bfloat16, device="cuda")
before = non_torch()
out = tgemm.mm(a, w, b) if path == "flydsl" else torch.nn.functional.linear(a, w, b)
del out, a; gc.collect(); torch.cuda.empty_cache()
print(f"MEM {path:6s} M={M}: held outside torch after free = {non_torch() - before:6.2f} GiB", flush=True)
