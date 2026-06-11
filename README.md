# Quantum-Modulated Poisson Encoders for Hybrid QPU Workloads

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official benchmark implementation and measurement suite for the paper:
**"Quantum-Modulated Poisson Encoders for Hybrid QPU Workloads: A Closed-Form
Hardware-Selection Frontier for Edge Inference"** (IEEE Computer Architecture
Letters, under review).

## Overview

Hybrid classical-quantum inference is bottlenecked by the QPU-classical I/O
interface, and the saturated-batch overhead figures common in the literature
systematically understate the cost at the single-query ($B{=}1$) operating
point. This repository provides a deterministic, reproducible workload --- a
non-homogeneous Poisson encoder modulated by an 8-qubit Parametrised Quantum
Circuit (PQC) --- that is **deliberately engineered so the quantum circuit is
non-contributory to accuracy** (verified by a controlled, parameter-matched
ablation). This circuit-independence is what allows the measured latency
frontier to characterise the *interface* rather than any particular circuit.

Key results reproduced by this repository:

* Cloud-API synchronisation penalty of **50.9%** at saturated batch
  ($B{=}256$) rising to **98.3%** at $B{=}1$ on the Spiking Heidelberg Digits
  (SHD) task.
* A closed-form hardware-selection frontier
  $\tau^{*}_{\mathrm{QPU}}(B;p) = (p/(1-p))\,T_{\mathrm{core}}(B)$.
* A controlled drive ablation (PQC vs. parameter-matched classical head vs.
  constant, depths $d\in\{1,2,4\}$, three seeds): **no quantum accuracy
  advantage exists on this task**, by design and by measurement.
* End-to-end validation of the additive overhead model on three real systems
  (two local simulator backends and the `ibm_marrakesh` superconducting QPU):
  unmodelled residual $\leq 0.53$ ms locally and $0.009\%$ on hardware.
* A first-order cryogenic thermal-budget analysis showing monolithic TSV
  integration --- the only latency-feasible paradigm at $B{=}1$ --- violates
  the mK cooling budget by five to six orders of magnitude.

## Repository layout

| File | Purpose | Output |
|---|---|---|
| `benckmark_programme.py` | Reference training/benchmark run (SHD, $W{=}4096$, $B{=}256$, 100 epochs) | `training_metrics.csv`, `seed_summary.csv`, `drive_ablation.csv`, figures |
| `expressivity_probe.py` | Controlled ablation: input-conditioned PQC vs. parameter-matched classical head vs. constant, depths 1/2/4 | `expressivity.csv`, `expressivity_verdict.txt` |
| `e2e_hybrid_validation.py` | End-to-end validation of the additive model with the PQC inline at $B{=}1$ (local backends + IBM hardware) | `e2e_validation.csv` |
| `measure_e2e_latency.py` | Round-trip latency statistics: local PennyLane backends, loopback proxy, IBM cloud queue | `e2e_latency.csv` |
| `measure_hw_latency.py` | Datacentre measurements (H100 PCIe): per-sample $T_{\mathrm{core}}(B)$ and PCIe DMA round-trip floor | `hw_latency.json` |
| `thermal_model.py` | First-order cryogenic heat-balance vs. stage cooling budgets | `thermal_budget.csv` + chart |

## Measured vs. estimated quantities (disclosure)

Measured: local simulator round-trips (`default.qubit`, `lightning.qubit`),
IBM open-plan cloud round-trips (queue-dominated; reported as such), PCIe DMA
floor (H100 Gen5), all $T_{\mathrm{core}}(B)$ values, end-to-end hybrid
residuals. **Estimated** (first-principles; labelled as such in the paper):
warm cloud-API (50 ms), loaded PCIe (5 ms), MCM (0.5 ms), CPO (0.05 ms),
TSV (0.5 us) --- no such hybrid hardware exists to measure. The end-to-end
validation ensures these estimates feed a verified additive model.

## Requirements

Python 3.10+:

```bash
pip install torch pennylane numpy matplotlib h5py
# optional, for the cloud measurements:
pip install qiskit qiskit-ibm-runtime
```

SHD dataset: download `shd_train.h5` / `shd_test.h5` from the University of
Heidelberg repository into `./data/`.

## Reproducing the paper

```bash
# Reference workload and ablation (seeds 1-5 / 42, deterministic):
python benckmark_programme.py

# Controlled expressivity ablation (Table/Section III-B-1):
python expressivity_probe.py --depths 1 2 4 --seeds 1 2 3 --epochs 60

# Latency statistics (local; add --cloud with IBMQ_TOKEN/IBMQ_CRN set):
python measure_e2e_latency.py

# End-to-end additive-model validation (Section III-C):
python e2e_hybrid_validation.py --n-local 200
python e2e_hybrid_validation.py --cloud --n-cloud 3   # requires IBM credentials

# Thermal budget (Section III-F):
python thermal_model.py
```

Hardware fingerprint of the reference runs: NVIDIA RTX 4070 Laptop GPU
(8.59 GB, CUDA 12.4), PyTorch 2.5.1, `cudnn.deterministic=True`. Datacentre
measurements: NVIDIA H100 PCIe (CUDA 12.8, PyTorch 2.7.0). Quantum hardware:
`ibm_marrakesh` (Heron r2) via the IBM Quantum open plan.

## Honesty notes

* The drive ablation is a **null result by design**: no quantum accuracy
  advantage is claimed, and none should be inferred from this codebase.
* IBM open-plan round-trips are scheduling-dominated and are reported as
  queue-bound upper bounds, not deployed-service latencies.
* The energy figures produced by `benckmark_programme.py` are
  order-of-magnitude deployment estimates, not measured device energies.

## Acknowledgements

SHD dataset: University of Heidelberg neuromorphic data repository. Quantum
simulations: PennyLane. Quantum hardware access: IBM Quantum services
(`ibm_marrakesh`); the views expressed are those of the authors and do not
reflect the official policy or position of IBM or the IBM Quantum team.
Datacentre GPU measurements were performed on a Lambda Cloud H100 instance.

## Licence

MIT.
