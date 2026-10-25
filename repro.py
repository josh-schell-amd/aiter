import sys, torch
from aiter.tuned_gemm import tgemm
M = int(sys.argv[1])
torch.manual_seed(0)
w = torch.randn(6144, 1280, dtype=torch.bfloat16, device="cuda")
b = torch.randn(6144, dtype=torch.bfloat16, device="cuda")
a = torch.randn(M, 1280, dtype=torch.bfloat16, device="cuda")
out = tgemm.mm(a, w, b)
torch.cuda.synchronize()
lin = torch.nn.functional.linear
err = max((out[i:i + 200000] - lin(a[i:i + 200000], w, b)).abs().max().item() for i in range(0, M, 200000))
print(f"RESULT max_err={err}", flush=True)
