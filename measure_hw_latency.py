# measure_hw_latency.py  --  R2.3 hardware validation on a datacentre GPU.
# Measures: (a) per-sample T_core(B) for B in {1,16,64,256} on this GPU,
#           (b) real PCIe host<->device round-trip latency,
#           (c) reproduces the overhead model from measured inputs.
# Runtime: < 5 min. No dataset needed (latency is data-independent for fixed shapes).
import json, time, platform
import torch, torch.nn as nn, torch.nn.functional as F

assert torch.cuda.is_available(), "No CUDA device visible."
dev = torch.device("cuda")
gpu = torch.cuda.get_device_name(0)

# --- encoder dimensions identical to the paper ---
IN, RED, W, T, NCLS = 700, 256, 4096, 100, 20

class INT4Lin(nn.Linear):
    def forward(self, x):
        s = (self.weight.abs().max()/7.0).clamp(min=1e-8)
        w = self.weight + (torch.round(self.weight/s)*s - self.weight).detach()
        return F.linear(x, w, self.bias)

class Enc(nn.Module):
    def __init__(s):
        super().__init__()
        s.sp = INT4Lin(IN, RED); s.tc = nn.Conv1d(RED, RED, 5, padding=2)
        s.f1 = INT4Lin(RED, W); s.f2 = INT4Lin(W, NCLS)
    def forward(s, x):
        b, t, _ = x.shape
        xp = F.relu(s.sp(x.reshape(-1, IN))).view(b, t, -1)
        c = F.relu(s.tc(xp.permute(0,2,1))).permute(0,2,1)
        sp = F.softplus(s.f1(c))
        drive = sp * 0.3
        spk = torch.clamp(drive + (torch.poisson(drive)-drive).detach(), max=1.0)
        d = torch.exp(torch.linspace(-2,0,t,device=x.device))
        return s.f2((spk*d.view(1,t,1)).sum(1))

@torch.no_grad()
def t_core(model, B, reps=50, warm=10):
    x = torch.rand(B, T, IN, device=dev)
    for _ in range(warm): model(x); 
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): model(x)
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/reps/B*1000.0  # ms per sample

@torch.no_grad()
def pcie_rt(nbytes=8, reps=2000, warm=200):
    # round-trip: host->device->host of a tiny payload = real PCIe + driver latency
    h = torch.zeros(nbytes//8, dtype=torch.float64).pin_memory()
    for _ in range(warm):
        d = h.to(dev, non_blocking=True); _ = d.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        d = h.to(dev, non_blocking=True); _ = d.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1000.0)
    ts.sort()
    return {"median_ms": ts[len(ts)//2], "p95_ms": ts[int(0.95*len(ts))]}

model = Enc().to(dev).eval()
tcore = {B: t_core(model, B) for B in (1,16,64,256)}
pcie = pcie_rt()

# overhead model check (Eq. 2) using MEASURED inputs
def overhead(tau, Tc): return tau/(tau+Tc)*100
checks = {B: {
    "T_core_ms": round(tcore[B]*B,4),
    "T_per_sample_ms": round(tcore[B],5),
    "pcie_measured_overhead_pct": round(overhead(pcie["median_ms"], tcore[B]*B),3),
} for B in (1,16,64,256)}

out = {
    "gpu": gpu, "torch": torch.__version__,
    "cuda": torch.version.cuda, "platform": platform.platform(),
    "pcie_roundtrip": pcie,
    "per_sample_T_core_ms": {str(k): round(v,5) for k,v in tcore.items()},
    "overhead_model_check": checks,
}
print(json.dumps(out, indent=2))
open("hw_latency.json","w").write(json.dumps(out, indent=2))
print("\nWROTE hw_latency.json")