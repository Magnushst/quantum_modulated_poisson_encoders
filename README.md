# Quantum-Modulated Poisson Encoder for Hybrid QPU Workloads

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reference implementation and benchmarking harness for the paper:

> **Quantum-Modulated Poisson Encoders for Hybrid QPU Workloads: A
> Closed-Form Hardware-Selection Frontier for Edge Inference**
> Submitted to *IEEE Computer Architecture Letters*, 2026.

## Overview

Hybrid classical-quantum systems suffer severe I/O latency bottlenecks
when classical accelerators communicate with Quantum Processing Units
(QPUs). The figures conventionally reported in the literature are
measured at saturated batch sizes, which mask the latency-bound regime
that motivates edge deployment.

This repository provides a deterministic, reproducible workload
generator — a non-homogeneous Poisson encoder modulated by a fixed,
offline Parametrised Quantum Circuit (PQC) — and a benchmarking harness
that characterises QPU-classical synchronisation overhead across five
candidate physical integration paradigms. The PQC functions purely as a
workload generator; a drive-ablation experiment confirms that the
analysis isolates the cost of *querying* a quantum device, not the
expressivity of the queried circuit.

## Key Results

- **Cloud-API quantum integration imposes a 50.9% synchronisation
  penalty at saturated batch (B=256), rising to 98.3% at B=1.**
- **Closed-form hardware-selection frontier:** the maximum tolerable
  QPU round-trip for an overhead budget *p* is
  τ*(B; p) = (p/(1−p)) · T_core(B). At p=1%, only Monolithic
  Through-Silicon Via (TSV) integration meets the budget across the
  full batch range B ∈ [1, 256]; even MCM-class chiplet integration
  falls just outside the threshold at saturated batch.
- **Measured PQC round-trip latencies** on two PennyLane simulator
  backends (`default.qubit` 5.22 ms; `lightning.qubit` 1.67 ms)
  bracket the latency of *local* quantum simulation as a realistic
  upper bound on best-case quantum integration.

## Components

- **Spatiotemporal proxy network:** a 1D temporal convolution
  (`C_out=256`, `k=5`) over a dense INT4-quantised spatial projection,
  followed by a Straight-Through-Estimator Poisson sampling layer and
  an exponentially decayed temporal readout.
- **PQC modulator:** an 8-qubit hardware-efficient ansatz (PennyLane;
  `default.qubit` and `lightning.qubit` backends) used to generate a
  fixed, offline λ(t) waveform. A classical sinusoidal surrogate is
  used as a fallback when PennyLane is unavailable.
- **Two-stage initialisation:** variance stabilisation followed by a
  one-shot multiplicative gain calibration anchoring the encoder to a
  target rate of 12.5 Hz; a rate-penalty regulariser maintains the rate
  during training.
- **Latency-overhead analysis:** five paradigms (Cloud, PCIe, MCM,
  CPO, TSV) plus two measured PennyLane backends, evaluated at
  saturated batch and across an inference batch sweep.

## Reproducibility

All experiments are deterministic with `torch.manual_seed`,
`numpy.random.seed`, `random.seed`, and
`torch.backends.cudnn.deterministic=True`. The reference run uses seed
42; multi-seed statistics are computed across seeds {1, 2, 3, 4, 5}.

Hardware fingerprint of the reference run: NVIDIA RTX 4070 Laptop GPU
(8.59 GB VRAM), CUDA 12.4, PyTorch 2.5.1, Python 3.12.13, Windows 11.

## Requirements

```bash
pip install torch          # 2.5.1 used in the reference run
pip install pennylane      # for the PQC; optional - falls back to sinusoid
pip install pennylane-lightning  # for the lightning.qubit backend
pip install h5py numpy matplotlib
```

The Spiking Heidelberg Digits (SHD) dataset is downloaded automatically
on first run from `compneuro.net/datasets/`. If `h5py` is unavailable,
a small random surrogate is used (only suitable for code testing).

## Running the experiments

```bash
MODE=main         python benchmark_programme.py   # ~5 minutes
MODE=seeds        python benchmark_programme.py   # ~25 minutes
MODE=ablation     python benchmark_programme.py   # ~15 minutes
MODE=sweep        python benchmark_programme.py   # ~60 minutes
MODE=pqc_latency  python benchmark_programme.py   # ~10 seconds
MODE=all          python benchmark_programme.py   # full reproduction
```

Wall-clock times above are measured on the reference hardware. The
sensitivity sweep dominates total runtime; on lower-VRAM GPUs the
`W=8192, B=256` configuration may suffer thermal/paging contamination
and should be excluded.

## Outputs

All artefacts are written to `publication_results/`:

| File                                | Source experiment      |
|-------------------------------------|------------------------|
| `fig1_latent_drive.{png,pdf,…}`     | `main`                 |
| `raster_epoch_100.{…}`              | `main`                 |
| `latency_overhead.{…}`              | `main`                 |
| `latency_overhead_batch.{…}`        | `sweep`                |
| `breakeven_scaling.{…}`             | `sweep`                |
| `energy_breakdown.{…}`              | `main`                 |
| `confusion_matrix.{…}`              | `main`                 |
| `energy_pareto.{…}`                 | `sweep`                |
| `training_metrics.csv`              | `main`                 |
| `seed_summary.csv`                  | `seeds`                |
| `drive_ablation.csv`                | `ablation`             |
| `sensitivity_sweep.csv`             | `sweep`                |
| `pqc_round_trip.csv`                | `pqc_latency`          |
| `per_class_accuracy.csv`            | `main`                 |
| `hardware_fingerprint.csv`          | always                 |

## Limitations

- The PQC is fixed and offline. The drive-ablation experiment in the
  paper makes this fully transparent: the analysis isolates I/O cost,
  not quantum expressivity.
- Energy estimates use order-of-magnitude per-operation constants
  (0.4 pJ MAC, 2.5 pJ event routing) and assume a hypothetical
  fully-INT4 deployment; the constants are not from a primary source
  and are clearly flagged as such in the paper.
- Per-sample inference latency at B=1 is dominated by Python loop
  overhead and CUDA kernel launches; a production inference runtime
  would lower this floor.

## License

MIT.
