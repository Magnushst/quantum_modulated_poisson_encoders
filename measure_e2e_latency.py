"""
measure_e2e_latency.py
======================
End-to-end hybrid QPU round-trip measurement for the revision of CAL-2026-05-0136.

Purpose (addresses Reviewer 2, comment 3 "validate the model with real
measurements"): replace the hand-typed LATENCY_BENCHMARKS_MS constants used for
the cloud row of Table II with at least one *measured* end-to-end
submit -> queue -> execute -> retrieve round-trip, plus measured local-backend
round-trips, with proper statistics (median, p95, IQR) rather than a bare mean.

It writes `e2e_latency.csv` with one row per backend:
    backend, n_calls, mean_ms, median_ms, p95_ms, std_ms, source

Three measurement tiers, in decreasing order of fidelity. The script runs every
tier it can and records which one succeeded so the manuscript can cite the
highest-fidelity number available to you.

  TIER 1  Real cloud QPU/simulator queue (IBM Quantum via qiskit-ibm-runtime).
          This is a genuine network + queue + execution round-trip and is the
          number Reviewer 2 actually wants. Requires a free IBM Quantum token
          in the env var IBMQ_TOKEN. Uses the *least-busy* backend or the hosted
          statevector simulator if no hardware is free, and reports which.

  TIER 2  Local PennyLane backends (default.qubit, lightning.qubit). Always
          available; gives the local-simulation floor already in the paper but
          now with median/p95, not just mean.

  TIER 3  Loopback PCIe-class proxy: a localhost TCP round-trip carrying an
          8-float payload, isolating the serialisation + syscall + scheduler
          component of I/O on the actual host. This is an honest *measured*
          lower bound on any off-die interconnect on your machine, and lets you
          state the PCIe row as "measured loopback proxy" rather than assumed.

Complexity: all timing loops are O(n_calls); negligible compute. Memory O(1).

Usage:
    # Tier 2 + Tier 3 only (no account needed):
    python measure_e2e_latency.py

    # Add Tier 1 (recommended for the rebuttal):
    export IBMQ_TOKEN=<your_ibm_quantum_token>
    python measure_e2e_latency.py --cloud --shots 1024 --n-cloud 20

Notes on honesty:
  * Cloud queue time is dominated by scheduling, not physics. We therefore
    report BOTH the full wall-clock round-trip AND the reported per-job
    execution time when the provider exposes it, and we say so in the CSV
    `source` column. Do not conflate the two in the paper.
  * n_cloud is deliberately small (jobs are slow/limited); we report median and
    p95 over whatever n you can afford, and flag n<10 as "indicative".
"""
from __future__ import annotations

import argparse
import csv
import os
import socket
import statistics
import struct
import threading
import time
from pathlib import Path
from typing import Callable

QUBITS = 8
OUT = Path("e2e_latency.csv")


def _stats(samples_ms: list[float], source: str, backend: str) -> dict:
    """Reduce a latency sample to robust statistics. O(n log n) for the sort."""
    n = len(samples_ms)
    s = sorted(samples_ms)
    p95 = s[min(n - 1, int(round(0.95 * (n - 1))))] if n else float("nan")
    return {
        "backend": backend,
        "n_calls": n,
        "mean_ms": round(statistics.fmean(samples_ms), 4) if n else float("nan"),
        "median_ms": round(statistics.median(samples_ms), 4) if n else float("nan"),
        "p95_ms": round(p95, 4),
        "std_ms": round(statistics.pstdev(samples_ms), 4) if n > 1 else 0.0,
        "source": source + (" [n<10: indicative]" if n < 10 else ""),
    }


def time_loop(fn: Callable[[], None], n: int, warmup: int = 5) -> list[float]:
    """Call fn n times, return per-call wall-clock in ms. O(n)."""
    for _ in range(warmup):
        fn()
    out: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


# ----------------------------------------------------------------------------
# TIER 2: local PennyLane backends
# ----------------------------------------------------------------------------
def tier2_local(n_calls: int) -> list[dict]:
    try:
        import pennylane as qml
        import numpy as np
    except ImportError:
        print("[tier2] PennyLane not installed; skipping local backends.")
        return []

    rows: list[dict] = []
    weights = np.random.rand(QUBITS) * 2 * np.pi
    inputs = np.ones(QUBITS)

    for backend in ("default.qubit", "lightning.qubit"):
        try:
            dev = qml.device(backend, wires=QUBITS)

            @qml.qnode(dev)
            def circuit(inp, w):
                for i in range(QUBITS):
                    qml.RX(inp[i], wires=i)
                for i in range(QUBITS - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[QUBITS - 1, 0])
                for i in range(QUBITS):
                    qml.RY(w[i], wires=i)
                return [qml.expval(qml.PauliZ(i)) for i in range(QUBITS)]

            samples = time_loop(lambda: circuit(inputs, weights), n_calls)
            rows.append(_stats(samples, f"measured local ({backend})", backend))
            print(f"[tier2] {backend}: median {rows[-1]['median_ms']} ms")
        except Exception as exc:
            print(f"[tier2] {backend} unavailable: {exc}")
    return rows


# ----------------------------------------------------------------------------
# TIER 3: localhost loopback PCIe-class proxy
# ----------------------------------------------------------------------------
def tier3_loopback(n_calls: int) -> list[dict]:
    """
    Measure a localhost TCP round-trip of an 8xfloat64 payload. This captures
    the syscall + scheduler + serialisation cost that any off-die interconnect
    inherits, on the user's actual host. Honest measured lower bound, not a QPU.
    """
    payload = struct.pack("8d", *([0.5] * QUBITS))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()

    def echo_server() -> None:
        conn, _ = srv.accept()
        with conn:
            while True:
                data = conn.recv(64)
                if not data:
                    break
                conn.sendall(data)

    t = threading.Thread(target=echo_server, daemon=True)
    t.start()

    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect((host, port))
    cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def one_rt() -> None:
        cli.sendall(payload)
        _ = cli.recv(64)

    samples = time_loop(one_rt, n_calls)
    cli.close()
    srv.close()
    row = _stats(samples, "measured localhost loopback proxy", "PCIe-class loopback")
    print(f"[tier3] loopback proxy: median {row['median_ms']} ms")
    return [row]


# ----------------------------------------------------------------------------
# TIER 1: real cloud QPU / hosted simulator round-trip (IBM Quantum)
# ----------------------------------------------------------------------------
def tier1_cloud(n_calls: int, shots: int) -> list[dict]:
    token = os.environ.get("IBMQ_TOKEN")
    if not token:
        print("[tier1] IBMQ_TOKEN not set; skipping cloud measurement.")
        return []
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError:
        print("[tier1] qiskit-ibm-runtime not installed; skipping.")
        return []

    # Channel renamed mid-2025: 'ibm_quantum' -> 'ibm_quantum_platform'.
    # New channel requires the CRN instance string in addition to the token.
    crn = os.environ.get("IBMQ_CRN")
    if not crn:
        print("[tier1] IBMQ_CRN not set; the new platform requires your instance CRN. Skipping.")
        return []
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=crn,
    )
    try:
        backend = service.least_busy(operational=True, simulator=False)
        kind = f"real QPU ({backend.name})"
    except Exception:
        backend = service.backends()[0]
        kind = f"fallback backend ({backend.name})"
    print(f"[tier1] using {kind}")

    qc = QuantumCircuit(QUBITS)
    for i in range(QUBITS):
        qc.rx(1.0, i)
    for i in range(QUBITS - 1):
        qc.cx(i, i + 1)
    qc.cx(QUBITS - 1, 0)
    for i in range(QUBITS):
        qc.ry(0.5, i)
    qc.measure_all()

    # New runtime requires circuits transpiled to the backend's ISA.
    qc = transpile(qc, backend=backend, optimization_level=1)

    sampler = SamplerV2(mode=backend)
    wall, exec_ms = [], []
    for k in range(n_calls):
        t0 = time.perf_counter()
        job = sampler.run([qc], shots=shots)
        _ = job.result()
        wall.append((time.perf_counter() - t0) * 1000.0)
        try:
            md = job.metrics()
            if md and "usage" in md and "quantum_seconds" in md["usage"]:
                exec_ms.append(md["usage"]["quantum_seconds"] * 1000.0)
        except Exception:
            pass
        print(f"[tier1] job {k+1}/{n_calls}: wall {wall[-1]:.1f} ms")

    rows = [_stats(wall, f"measured cloud wall-clock, {kind}", "Cloud Quantum (measured)")]
    if exec_ms:
        rows.append(_stats(exec_ms, f"measured cloud exec-only, {kind}",
                           "Cloud Quantum exec-only (measured)"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", action="store_true", help="run Tier 1 cloud measurement")
    ap.add_argument("--n-local", type=int, default=100)
    ap.add_argument("--n-loopback", type=int, default=1000)
    ap.add_argument("--n-cloud", type=int, default=20)
    ap.add_argument("--shots", type=int, default=1024)
    args = ap.parse_args()

    rows: list[dict] = []
    if args.cloud:
        rows += tier1_cloud(args.n_cloud, args.shots)
    rows += tier2_local(args.n_local)
    rows += tier3_loopback(args.n_loopback)

    if not rows:
        print("No measurements produced.")
        return

    fields = ["backend", "n_calls", "mean_ms", "median_ms", "p95_ms", "std_ms", "source"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT.resolve()} with {len(rows)} rows.")
    print("Use the highest-fidelity row available (Tier 1 cloud > Tier 2 local > Tier 3 loopback)")
    print("as the 'Measured cloud queue' / PCIe rows in Table II of the revision.")


if __name__ == "__main__":
    main()
