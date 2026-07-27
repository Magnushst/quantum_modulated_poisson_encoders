"""
expressivity_probe.py  --  CONTROLLED EXPRESSIVITY ABLATION
===========================================================
Question this script answers:
    "Once the PQC drive is input-conditioned (lambda depends on x) and the
     entangling depth is increased, does the quantum circuit buy ANY accuracy
     that a classical surrogate under a controlled parameter budget cannot?"

The reported separation is normalised by
sqrt(sigma_PQC^2 + sigma_classical^2). With three seeds it is a transparent
effect-size diagnostic, not a formal significance test. A null result means the
PQC remains a workload generator and no accuracy advantage is attributed to it.

OUTPUT:
    expressivity.csv                 (one row per depth x drive x seed)
    expressivity_verdict.txt         (machine-written ablation summary)
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import concurrent.futures

import numpy as np
import torch
import torch.nn.functional as F

# ---- shared constants (mirror benckmark_programme.py) ----------------------
QPU_QUBITS = 8
TIME_STEPS = 100
INPUT_CHANNELS = 700
NUM_CLASSES = 20
REDUCED_DIM = 256
ENCODER_WIDTH = 4096
TARGET_HZ = 12.5
POOL_DIM = QPU_QUBITS          # input is pooled to 8 angles to feed the circuit
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pennylane as qml
HAS_PENNYLANE = True

OUT_CSV = Path("expressivity.csv")
OUT_VERDICT = Path("expressivity_verdict.txt")


# ---------------------------------------------------------------------------
# Data: require SHD for publication reproduction; surrogate only for --smoke
# ---------------------------------------------------------------------------
def surrogate_data():
    g = torch.Generator().manual_seed(0)
    Xtr = (torch.rand(256, TIME_STEPS, INPUT_CHANNELS, generator=g) < 0.03).float()
    ytr = torch.randint(0, NUM_CLASSES, (256,), generator=g)
    Xte = (torch.rand(64, TIME_STEPS, INPUT_CHANNELS, generator=g) < 0.03).float()
    yte = torch.randint(0, NUM_CLASSES, (64,), generator=g)
    return Xtr, ytr, Xte, yte


def load_data(allow_surrogate: bool = False):
    try:
        import h5py
    except ImportError as exc:
        if allow_surrogate:
            print("[data] h5py unavailable; using seeded surrogate for smoke test only.")
            return surrogate_data()
        raise RuntimeError(
            "h5py is required for the publication ablation. "
            "Install it with `pip install h5py`."
        ) from exc

    cache = Path("data")
    train_p, test_p = cache / "shd_train.h5", cache / "shd_test.h5"
    if not (train_p.exists() and test_p.exists()):
        if allow_surrogate:
            print("[data] SHD files absent; using seeded surrogate for smoke test only.")
            return surrogate_data()
        raise FileNotFoundError(
            "SHD files not found. Place shd_train.h5 and shd_test.h5 in ./data/ "
            "before reproducing the publication ablation."
        )

    def proc(fp):
        with h5py.File(fp, "r") as f:
            units = f["spikes"]["units"][:]
            times = f["spikes"]["times"][:]
            labels = f["labels"][:]
        X = torch.zeros((len(labels), TIME_STEPS, INPUT_CHANNELS))
        for i in range(len(labels)):
            if len(units[i]) > 0:
                ti = np.clip((times[i] * TIME_STEPS).astype(int), 0, TIME_STEPS - 1)
                X[i, ti, units[i]] = 1.0
        return X, torch.tensor(labels, dtype=torch.long)

    Xtr, ytr = proc(train_p)
    Xte, yte = proc(test_p)
    print(f"[data] SHD loaded: train={tuple(Xtr.shape)} test={tuple(Xte.shape)}")
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# Input-conditioned drive heads. ALL take pooled angles (B, POOL_DIM) -> (B, T).
# ---------------------------------------------------------------------------
def make_pqc_qnode(depth: int):
    # lightning.qubit delegates the adjoint calculation to a C++ backend
    dev = qml.device("lightning.qubit", wires=QPU_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(angles, weights):
        qml.AngleEmbedding(angles, wires=range(QPU_QUBITS), rotation="X")
        
        for d in range(depth):
            for i in range(QPU_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[QPU_QUBITS - 1, 0])
            for i in range(QPU_QUBITS):
                qml.RY(weights[d, i], wires=i)
                
        return [qml.expval(qml.PauliZ(i)) for i in range(QPU_QUBITS)]

    return circuit


class PQCDriveHead(torch.nn.Module):
    def __init__(self, depth: int):
        super().__init__()
        self.depth = depth
        self.circuit = make_pqc_qnode(depth)
        self.q_weights = torch.nn.Parameter(torch.rand(depth, QPU_QUBITS) * 2 * math.pi)
        basis = torch.stack([torch.sin(torch.linspace(0, math.pi * (k + 1), TIME_STEPS))
                             for k in range(QPU_QUBITS)], dim=0)
        self.register_buffer("basis", basis)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        # Isolate inputs to the CPU for the quantum simulator
        angles_cpu = angles.cpu()
        weights_cpu = self.q_weights.cpu()
        
        # Execute the quantum circuit on the CPU backend
        ev = torch.stack(self.circuit(angles_cpu, weights_cpu), dim=-1)
        
        # Return outputs to the original compute device (GPU)
        ev = ev.to(self.basis.dtype).to(self.basis.device)
        return F.softplus(ev @ self.basis) + 0.1

    def n_params(self) -> int:
        return self.q_weights.numel()


class MLPDriveHead(torch.nn.Module):
    def __init__(self, depth: int):
        super().__init__()
        # Exact drive-head parameter matching at d=2/4; d=1 uses the smallest
        # non-zero two-layer head (16 rather than 8 parameters).
        hidden = max(1, (depth * QPU_QUBITS) // (POOL_DIM + QPU_QUBITS))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(POOL_DIM, hidden, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, QPU_QUBITS, bias=False),
        )
        basis = torch.stack([torch.sin(torch.linspace(0, math.pi * (k + 1), TIME_STEPS))
                             for k in range(QPU_QUBITS)], dim=0)
        self.register_buffer("basis", basis)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        ev = torch.tanh(self.net(angles))
        return F.softplus(ev @ self.basis) + 0.1

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ConstantDriveHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.basis = None

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        b = angles.shape[0]
        return torch.ones(b, TIME_STEPS, device=angles.device) * 0.3

    def n_params(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Shared classical trunk 
# ---------------------------------------------------------------------------
class INT4LinearSTE(torch.nn.Linear):
    def forward(self, x):
        scale = (self.weight.abs().max() / 7.0).clamp(min=1e-8)
        w = self.weight + (torch.round(self.weight / scale) * scale - self.weight).detach()
        return F.linear(x, w, self.bias)


class ConditionedEncoder(torch.nn.Module):
    def __init__(self, drive_head: torch.nn.Module, width: int = ENCODER_WIDTH):
        super().__init__()
        self.spatial_proj = INT4LinearSTE(INPUT_CHANNELS, REDUCED_DIM)
        self.temporal_conv = torch.nn.Conv1d(REDUCED_DIM, REDUCED_DIM, kernel_size=5, padding=2)
        self.fc1 = INT4LinearSTE(REDUCED_DIM, width)
        self.dropout = torch.nn.Dropout(0.4)
        self.fc2 = INT4LinearSTE(width, NUM_CLASSES)
        self.gain = torch.nn.Parameter(torch.tensor(1.0))
        self.drive_head = drive_head
        self._pool = torch.nn.Linear(INPUT_CHANNELS, POOL_DIM, bias=False)

    def pooled_angles(self, x):
        return math.pi * torch.tanh(self._pool(x.mean(dim=1)))

    def forward(self, x: torch.Tensor):
        b, t, _ = x.shape
        angles = self.pooled_angles(x)
        drive_bt = self.drive_head(angles)
        xp = F.relu(self.spatial_proj(x.reshape(-1, x.shape[-1]))).view(b, t, -1)
        conv = F.relu(self.temporal_conv(xp.permute(0, 2, 1))).permute(0, 2, 1)
        spatial = F.softplus(self.fc1(conv))
        drive = spatial * drive_bt.unsqueeze(2) * torch.abs(self.gain)
        spikes = drive + (torch.poisson(drive) - drive).detach()
        spikes = torch.clamp(spikes, max=1.0)
        decay = torch.exp(torch.linspace(-2.0, 0.0, t, device=x.device))
        weighted = (spikes * decay.unsqueeze(0).unsqueeze(2)).sum(dim=1)
        logits = self.fc2(self.dropout(weighted))
        return logits, drive_bt


# ---------------------------------------------------------------------------
# Execution Functions
# ---------------------------------------------------------------------------
@torch.no_grad()
def effective_dimension(head: torch.nn.Module, angles: torch.Tensor, n_theta: int = 30) -> float:
    if head.n_params() == 0:
        return 0.0
    base = [p.detach().clone() for p in head.parameters()]
    feats = []
    for _ in range(n_theta):
        for p, b in zip(head.parameters(), base):
            p.copy_(b + 0.3 * torch.randn_like(b))
        feats.append(head(angles).flatten())
    for p, b in zip(head.parameters(), base):
        p.copy_(b)
    F_mat = torch.stack(feats)
    F_mat = F_mat - F_mat.mean(0, keepdim=True)
    cov = (F_mat @ F_mat.T) / max(1, F_mat.shape[1])
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    if eig.sum() <= 0:
        return 0.0
    p_norm = eig / eig.sum()
    p_norm = p_norm[p_norm > 0]
    entropy = -(p_norm * p_norm.log()).sum().item()
    return float(entropy / math.log(n_theta))


def seed_all(s: int):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_head(kind: str, depth: int):
    if kind == "pqc":
        return PQCDriveHead(depth)
    if kind == "mlp":
        return MLPDriveHead(depth)
    return ConstantDriveHead()


def train_one(kind, depth, seed, Xtr, ytr, Xte, yte, epochs, batch=64):
    seed_all(seed)
    head = make_head(kind, depth).to(DEVICE)
    model = ConditionedEncoder(head).to(DEVICE)
        
    _ = model.pooled_angles(Xtr[:2].to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.14)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.06)
    Xtr, ytr = Xtr.to(DEVICE), ytr.to(DEVICE)
    Xte, yte = Xte.to(DEVICE), yte.to(DEVICE)

    best = 0.0; final = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        for i in range(0, len(Xtr), batch):
            xb, yb = Xtr[i:i + batch], ytr[i:i + batch]
            opt.zero_grad()
            logits, drive_bt = model(xb)
            loss = crit(logits, yb) + ((drive_bt.mean() - TARGET_HZ / 1000.0) ** 2) * 1e5
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            correct = 0
            for i in range(0, len(Xte), batch):
                lg, _ = model(Xte[i:i + batch])
                correct += (lg.argmax(1) == yte[i:i + batch]).sum().item()
            acc = correct / len(Xte) * 100
        best = max(best, acc); final = acc

    with torch.no_grad():
        ang = model.pooled_angles(Xte[:min(64, len(Xte))])
        drive_var = float(model.drive_head(ang).var(dim=0).mean().item())
        eff = effective_dimension(model.drive_head, ang)
    
    n_p = head.n_params()
    
    import gc
    del model
    del head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {
        "depth": depth, "drive_kind": kind, "seed": seed,
        "test_acc": round(best, 3), "final_test_acc": round(final, 3),
        "eff_dim": round(eff, 4), "lambda_input_var": round(drive_var, 6),
        "n_params": n_p, # (Requires saving head.n_params() before deletion)
    }


# Standalone worker function for safe Windows multiprocessing
def worker_task(task_args):
    kind, d, s, Xtr, ytr, Xte, yte, epochs = task_args
    return train_one(kind, d, s, Xtr, ytr, Xte, yte, epochs)


def verdict(rows: list[dict]) -> str:
    import statistics as st
    lines = ["EXPRESSIVITY ABLATION SUMMARY", "=" * 40]
    passed_any = False
    for d in sorted({r["depth"] for r in rows}):
        sub = {k: [r["test_acc"] for r in rows if r["depth"] == d and r["drive_kind"] == k]
               for k in ("pqc", "mlp", "constant")}
        if not sub["pqc"] or not sub["mlp"]:
            continue
        mp, mm = st.mean(sub["pqc"]), st.mean(sub["mlp"])
        mc = st.mean(sub["constant"]) if sub["constant"] else float("nan")
        sp = st.pstdev(sub["pqc"]) if len(sub["pqc"]) > 1 else 0.0
        sm = st.pstdev(sub["mlp"]) if len(sub["mlp"]) > 1 else 0.0
        pooled = math.sqrt(sp ** 2 + sm ** 2)
        sigma = (mp - mm) / pooled if pooled > 1e-6 else float("nan")
        ok = pooled > 1e-6 and (mp - mm) > 2 * pooled and mp > mc
        passed_any = passed_any or (ok and d >= 2)
        lines.append(f"depth={d}: pqc={mp:.2f}  mlp(parameter-count-controlled)={mm:.2f}  "
                     f"const={mc:.2f}  combined-SD separation={sigma:+.2f}  "
                     f"{'threshold met' if ok else 'no advantage'}")
    lines.append("=" * 40)
    if passed_any:
        lines += ["RESULT: the predeclared >2 combined-SD accuracy threshold is met",
                  "against the parameter-count-controlled classical head at depth>=2."]
    else:
        lines += ["RESULT: NULL -- no observed PQC accuracy advantage over the",
                  "parameter-count-controlled classical head at the tested depths and seeds.",
                  "The PQC is retained solely as a workload generator for the hybrid-I/O analysis."]
    return "\n".join(lines)


def main():
    torch.set_num_threads(8)
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not HAS_PENNYLANE:
        return
    if args.smoke:
        args.epochs = min(args.epochs, 15)
    
    Xtr, ytr, Xte, yte = load_data(allow_surrogate=args.smoke)

    # Sequential, single process. The 21 runs are each a full training; the
    # speedup comes from batched PQC evaluation inside each run, not from
    # process parallelism (which deadlocks under spawn + unpicklable QNodes).
    tasks = []
    for d in args.depths:
        for kind in ("constant", "mlp", "pqc"):
            if kind == "constant" and d != args.depths[0]:
                continue
            for s in args.seeds:
                tasks.append((kind, d, s))

    rows = []
    for (kind, d, s) in tasks:
        r = train_one(kind, d, s, Xtr, ytr, Xte, yte, args.epochs)
        rows.append(r)
        print(f"  d={d} {kind:8s} seed={s}: best={r['test_acc']:.2f}% "
              f"final={r['final_test_acc']:.2f}% eff_dim={r['eff_dim']:.3f} "
              f"lambda_var={r['lambda_input_var']:.2e} params={r['n_params']}", flush=True)

    # Replicate constant rows across depths for grouped comparison in verdict().
    const_rows = [r for r in rows if r["drive_kind"] == "constant"]
    for d in args.depths:
        for cr in const_rows:
            if cr["depth"] != d:
                rows.append({**cr, "depth": d})

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    v = verdict(rows)
    OUT_VERDICT.write_text(v)
    print("\n" + v)
    print(f"\nWrote {OUT_CSV} and {OUT_VERDICT}")


if __name__ == "__main__":
    main()
