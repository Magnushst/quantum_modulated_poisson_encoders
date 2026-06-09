"""
e2e_hybrid_validation.py
========================
Validates the additive overhead model (Eq. 2 of the paper) END-TO-END by
executing the full hybrid pipeline as a real system: the 8-qubit PQC is
evaluated INLINE in the B=1 inference path, the whole call is timed with one
wall-clock, and the components (tau_QPU, classical forward) are timed inside.

Validation criterion: the unmodelled residual
        gap = E2E_total - (tau_QPU + T_forward)
should be a small fraction of E2E_total. A small gap means the additive model
T_total = T_core + tau_QPU holds on a real running hybrid system -- which is
exactly the end-to-end validation Reviewer 2 requested. We also compare the
inline forward time against an isolated forward-only measurement to detect
contention between the QPU call and the GPU pipeline.

Three real systems are validated:
  LOCAL : laptop + default.qubit, laptop + lightning.qubit (no cost)
  CLOUD : laptop + ibm_marrakesh inline (--cloud; ~3 jobs of QPU budget)

RUN (on the SAME laptop used for the paper's T_core numbers):
    python e2e_hybrid_validation.py                       # local only
    export IBMQ_TOKEN=...; export IBMQ_CRN="crn:v1:..."   # then:
    python e2e_hybrid_validation.py --cloud --n-cloud 3

OUTPUT: e2e_validation.csv, one row per backend:
    backend, n, e2e_median_ms, tau_median_ms, fwd_median_ms,
    gap_median_ms, gap_pct, fwd_vs_isolated_pct
plus a paste-ready summary in e2e_validation.txt.

Complexity: O(n) timed calls per backend; memory O(B*T*W) for one forward.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QUBITS, T, IN, RED, W, NCLS = 8, 100, 700, 256, 4096, 20
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_CSV, OUT_TXT = Path("e2e_validation.csv"), Path("e2e_validation.txt")


# --- encoder identical in shape to the paper's reference network -------------
class INT4Lin(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = (self.weight.abs().max() / 7.0).clamp(min=1e-8)
        w = self.weight + (torch.round(self.weight / s) * s - self.weight).detach()
        return F.linear(x, w, self.bias)


class Enc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sp = INT4Lin(IN, RED)
        self.tc = nn.Conv1d(RED, RED, 5, padding=2)
        self.f1 = INT4Lin(RED, W)
        self.f2 = INT4Lin(W, NCLS)
        basis = torch.stack(
            [torch.sin(torch.linspace(0, math.pi * (k + 1), T)) for k in range(QUBITS)], 0
        )
        self.register_buffer("basis", basis)  # (8, T)

    def drive_from_ev(self, ev: torch.Tensor) -> torch.Tensor:
        return F.softplus(ev @ self.basis) + 0.1  # (T,)

    def forward(self, x: torch.Tensor, drive_t: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        xp = F.relu(self.sp(x.reshape(-1, IN))).view(b, t, -1)
        c = F.relu(self.tc(xp.permute(0, 2, 1))).permute(0, 2, 1)
        sp = F.softplus(self.f1(c)) * drive_t.view(1, t, 1)
        spk = torch.clamp(sp + (torch.poisson(sp) - sp).detach(), max=1.0)
        d = torch.exp(torch.linspace(-2, 0, t, device=x.device))
        return self.f2((spk * d.view(1, t, 1)).sum(1))


def _sync() -> None:
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def med(v: list[float]) -> float:
    return statistics.median(v)


@torch.no_grad()
def isolated_forward_ms(model: Enc, reps: int = 100, warm: int = 20) -> float:
    """Reference T_core(1): forward with a precomputed drive, no QPU in loop."""
    x = torch.rand(1, T, IN, device=DEV)
    drive = torch.full((T,), 0.3, device=DEV)
    for _ in range(warm):
        model(x, drive)
    _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        model(x, drive)
    _sync()
    return (time.perf_counter() - t0) / reps * 1000.0


@torch.no_grad()
def validate_local(model: Enc, backend: str, n: int, iso_ms: float) -> dict | None:
    try:
        import pennylane as qml
    except ImportError:
        print("[local] PennyLane missing.")
        return None
    try:
        qdev = qml.device(backend, wires=QUBITS)
    except Exception as e:
        print(f"[local] {backend} unavailable: {e}")
        return None

    @qml.qnode(qdev)  # numpy interface: latency-only, no grads needed
    def circ(a, w):
        for i in range(QUBITS):
            qml.RX(a[i], wires=i)
        for i in range(QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[QUBITS - 1, 0])
        for i in range(QUBITS):
            qml.RY(w[i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(QUBITS)]

    ang = np.ones(QUBITS) * 0.7
    wts = np.random.rand(QUBITS) * 2 * math.pi
    x = torch.rand(1, T, IN, device=DEV)
    for _ in range(5):  # warm both stages
        ev = torch.tensor(np.asarray(circ(ang, wts)), dtype=torch.float32, device=DEV)
        model(x, model.drive_from_ev(ev))
    _sync()

    e2e, tau, fwd = [], [], []
    for _ in range(n):
        t0 = time.perf_counter()
        ta = time.perf_counter()
        ev_np = np.asarray(circ(ang, wts))                # tau_QPU (inline)
        tb = time.perf_counter()
        # inter-stage glue (conversion, drive synthesis) is deliberately OUTSIDE
        # the fwd timer so the additive gap captures real unmodelled cost
        ev = torch.tensor(ev_np, dtype=torch.float32, device=DEV)
        drive = model.drive_from_ev(ev)
        _sync()
        tg = time.perf_counter()
        out = model(x, drive)                             # classical stage only
        _sync()
        tc_ = time.perf_counter()
        e2e.append((tc_ - t0) * 1000)
        tau.append((tb - ta) * 1000)
        fwd.append((tc_ - tg) * 1000)
    gap = [e - a - f for e, a, f in zip(e2e, tau, fwd)]
    return _row(backend, n, e2e, tau, fwd, gap, iso_ms)


@torch.no_grad()
def validate_cloud(model: Enc, n: int, iso_ms: float) -> dict | None:
    token, crn = os.environ.get("IBMQ_TOKEN"), os.environ.get("IBMQ_CRN")
    if not (token and crn):
        print("[cloud] IBMQ_TOKEN/IBMQ_CRN not set; skipping.")
        return None
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
    except ImportError:
        print("[cloud] qiskit-ibm-runtime missing; skipping.")
        return None
    svc = QiskitRuntimeService(channel="ibm_cloud", token=token, instance=crn)
    backend = svc.least_busy(operational=True, simulator=False)
    print(f"[cloud] using {backend.name}")

    qc = QuantumCircuit(QUBITS)
    for i in range(QUBITS):
        qc.rx(0.7, i)
    for i in range(QUBITS - 1):
        qc.cx(i, i + 1)
    qc.cx(QUBITS - 1, 0)
    for i in range(QUBITS):
        qc.ry(0.5, i)
    isa = transpile(qc, backend=backend, optimization_level=1)
    obs = [SparsePauliOp("I" * (QUBITS - 1 - i) + "Z" + "I" * i).apply_layout(isa.layout)
           for i in range(QUBITS)]
    est = EstimatorV2(mode=backend)

    x = torch.rand(1, T, IN, device=DEV)
    e2e, tau, fwd = [], [], []
    for k in range(n):
        t0 = time.perf_counter()
        ta = time.perf_counter()
        res = est.run([(isa, obs)]).result()              # submit->queue->exec->result
        tb = time.perf_counter()
        ev = torch.tensor(np.asarray(res[0].data.evs, dtype=np.float32), device=DEV)
        drive = model.drive_from_ev(ev)
        _sync()
        tg = time.perf_counter()
        out = model(x, drive)
        _sync()
        tc_ = time.perf_counter()
        e2e.append((tc_ - t0) * 1000)
        tau.append((tb - ta) * 1000)
        fwd.append((tc_ - tg) * 1000)
        print(f"[cloud] job {k+1}/{n}: e2e {e2e[-1]:.0f} ms (tau {tau[-1]:.0f} ms)")
    gap = [e - a - f for e, a, f in zip(e2e, tau, fwd)]
    return _row(f"cloud ({backend.name})", n, e2e, tau, fwd, gap, iso_ms)


def _row(name: str, n: int, e2e, tau, fwd, gap, iso_ms: float) -> dict:
    r = {
        "backend": name, "n": n,
        "e2e_median_ms": round(med(e2e), 4),
        "tau_median_ms": round(med(tau), 4),
        "fwd_median_ms": round(med(fwd), 4),
        "gap_median_ms": round(med(gap), 4),
        "gap_pct": round(med(gap) / med(e2e) * 100, 3),
        "fwd_vs_isolated_pct": round((med(fwd) - iso_ms) / iso_ms * 100, 2),
        "isolated_fwd_ms": round(iso_ms, 4),
    }
    print(f"[{name}] e2e {r['e2e_median_ms']} ms = tau {r['tau_median_ms']} + "
          f"fwd {r['fwd_median_ms']} + gap {r['gap_median_ms']} "
          f"(gap {r['gap_pct']}%; fwd vs isolated {r['fwd_vs_isolated_pct']:+}%)")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-local", type=int, default=200)
    ap.add_argument("--n-cloud", type=int, default=3)
    ap.add_argument("--cloud", action="store_true")
    args = ap.parse_args()

    model = Enc().to(DEV).eval()
    iso = isolated_forward_ms(model)
    print(f"[ref] isolated T_core(1) on this machine: {iso:.3f} ms")

    rows: list[dict] = []
    for be in ("default.qubit", "lightning.qubit"):
        r = validate_local(model, be, args.n_local, iso)
        if r:
            rows.append(r)
    if args.cloud:
        r = validate_cloud(model, args.n_cloud, iso)
        if r:
            rows.append(r)
    if not rows:
        print("No validations produced.")
        return

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    worst = max(abs(r["gap_pct"]) for r in rows)
    txt = (
        "E2E HYBRID VALIDATION SUMMARY\n" + "=" * 40 + "\n"
        + "\n".join(f"{r['backend']}: e2e={r['e2e_median_ms']}ms, "
                    f"additive gap={r['gap_pct']}%, "
                    f"inline-vs-isolated forward={r['fwd_vs_isolated_pct']:+}%"
                    for r in rows)
        + f"\n\nWorst additive-model error across systems: {worst}%\n"
          "PAPER SENTENCE NUMBER: 'within "
          f"{math.ceil(worst)}\\%' (use the worst-case, rounded up).\n"
    )
    OUT_TXT.write_text(txt)
    print("\n" + txt + f"Wrote {OUT_CSV} and {OUT_TXT}")


if __name__ == "__main__":
    main()
