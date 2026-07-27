# Quantum-Modulated Poisson Encoders for Hybrid QPU Workloads

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official benchmark implementation and measurement suite for the paper:
**"Quantum-Modulated Poisson Encoders for Hybrid QPU Workloads: A Closed-Form
Hardware-Selection Frontier for Edge Inference"** (IEEE Computer Architecture
Letters).

## Overview

Hybrid classical-quantum inference can be bottlenecked by the QPU-classical I/O
interface, and saturated-batch overhead figures can materially understate the
cost at the single-query ($B{=}1$) operating
point. This repository provides a deterministic, reproducible workload--a
non-homogeneous Poisson encoder modulated by an 8-qubit Parametrised Quantum
Circuit (PQC)--that is **deliberately engineered so the quantum circuit is
non-contributory to accuracy** (verified by a controlled classical-head and
constant-drive ablation). The PQC and classical drive heads have matching
parameter counts at depths 2 and 4; at depth 1, the smallest non-zero
two-layer classical head has 16 parameters versus 8 for the PQC. This
controlled design lets the latency frontier characterise the *interface* for
circuits with comparable aggregate round-trips, without attributing an
accuracy benefit to the tested PQC.

Key results reproduced by this repository:

* Under the paper's deliberately favourable illustrative 50 ms cloud-API
  scenario, a synchronisation penalty of **50.9%** at saturated batch
  ($B{=}256$) rises to **98.3%** at $B{=}1$ on the Spiking Heidelberg Digits
  (SHD) task.
* A closed-form hardware-selection frontier
  $\tau^{*}_{\mathrm{QPU}}(B;p) = (p/(1-p))\,T_{\mathrm{core}}(B)$.
* A controlled drive ablation (PQC vs. classical head vs. constant, depths
  $d\in\{1,2,4\}$, three seeds): **no quantum accuracy advantage exists on
  this task**, by design and by measurement. Drive-head parameter counts match
  at depths 2 and 4; the depth-1 classical head is the smallest non-zero
  two-layer comparator.
* End-to-end validation on two local simulator configurations and the
  `ibm_marrakesh` superconducting QPU: median per-call residuals of
  0.35/0.52 ms (7.8/17.7%) locally and 1.91 ms (0.009%) on hardware.
* Under the paper's illustrative 1% criterion, a first-order cryogenic
  thermal-budget analysis shows that monolithic TSV --- the only
  latency-feasible $B{=}1$ scenario --- violates the mK cooling budget by five
  to six orders of magnitude.

## Repository layout

| File | Purpose | Output |
|---|---|---|
| `benckmark_programme.py` | Reference training/benchmark run (SHD, $W{=}4096$, $B{=}256$, 100 epochs) | `training_metrics.csv`, `seed_summary.csv`, `drive_ablation.csv`, `pqc_round_trip.csv`, `tcore_batch_latency.csv`, figures |
| `expressivity_probe.py` | Controlled ablation: input-conditioned PQC vs. classical head vs. constant, depths 1/2/4; exact parameter matching at depths 2/4 | `expressivity.csv`, `expressivity_verdict.txt` |
| `e2e_hybrid_validation.py` | End-to-end validation of the additive model with the PQC inline at $B{=}1$ (local backends + IBM hardware) | `e2e_validation.csv` |
| `measure_e2e_latency.py` | Round-trip latency statistics: local PennyLane backends, loopback proxy, IBM cloud queue | `e2e_latency.csv` |
| `measure_hw_latency.py` | Datacentre measurements (H100 PCIe): per-sample $T_{\mathrm{core}}(B)$ and PCIe DMA round-trip floor | `hw_latency.json` |

## Measured vs. estimated quantities (disclosure)

Measured: local simulator round-trips (`default.qubit`, `lightning.qubit`),
IBM open-plan cloud round-trips (queue-dominated; reported as such), PCIe DMA
floor (H100 Gen5), all $T_{\mathrm{core}}(B)$ values, end-to-end hybrid
residuals. The exact warm-stage batch latencies used by the paper are archived
in `tcore_batch_latency.csv`: $B\leq64$ comes from the sensitivity sweep, while
$B=256$ comes from a separate paging-free isolated forward loop.
**Illustrative scenario values (not measurements or forecasts):** warm
cloud-API (50 ms), loaded PCIe (5 ms), MCM (0.5 ms), CPO (0.05 ms), and TSV
(0.5 us). No such complete hybrid systems exist to measure. The end-to-end
hardware run supports the additive decomposition used to evaluate these
scenarios. Table 2's local simulator rows use the PyTorch-interface means in
`pqc_round_trip.csv`; `e2e_latency.csv` is a separate NumPy-interface probe.

## Requirements

Python 3.10+:

```bash
pip install torch pennylane numpy matplotlib h5py
# optional, for the cloud measurements:
pip install qiskit qiskit-ibm-runtime
```

SHD dataset: download
[`shd_train.h5.zip`](https://compneuro.net/datasets/shd_train.h5.zip) and
[`shd_test.h5.zip`](https://compneuro.net/datasets/shd_test.h5.zip), extract
them, and place `shd_train.h5` and `shd_test.h5` in `./data/`.

## Reproducing the paper

```bash
# Reference workload and ablation (seeds 1-5 / 42, seeded configuration):
python benckmark_programme.py

# Controlled expressivity ablation (Section 3.1.1):
python expressivity_probe.py --depths 1 2 4 --seeds 1 2 3 --epochs 60

# Latency statistics (local; add --cloud with IBMQ_TOKEN/IBMQ_CRN set):
python measure_e2e_latency.py

# End-to-end additive-model validation (Section 3.2):
python e2e_hybrid_validation.py --n-local 200
python e2e_hybrid_validation.py --cloud --n-cloud 3   # requires IBM credentials

```

The publication artefacts are archived under `results_csv/`. Fresh runs do not
overwrite that archive: `benckmark_programme.py` writes to
`publication_results/`, while the other measurement scripts write their named
outputs in the repository root. Move or compare fresh results deliberately.

Publication reproduction requires the SHD files and the listed dependencies.
`expressivity_probe.py --smoke` is the only mode that may use a small seeded
random surrogate when SHD is absent; smoke-test output is not publication
evidence.

The thermal subsection is the first-order budget comparison stated in the
manuscript using its cited cooling-power sources; no unreported co-integration
measurement or missing simulation output is claimed.

Hardware fingerprint of the reference runs: NVIDIA RTX 4070 Laptop GPU
(8.59 GB, CUDA 12.4), PyTorch 2.5.1, `cudnn.deterministic=True`. Datacentre
measurements: NVIDIA H100 PCIe (CUDA 12.8, PyTorch 2.7.0). Quantum hardware:
`ibm_marrakesh` (Heron r2) via the IBM Quantum open plan.
The scripts use seeded deterministic settings, but exact bitwise equality
across different hardware, drivers, and library versions is not asserted.

## Honesty notes

* The drive ablation is a **null result by design**: no quantum accuracy
  advantage is claimed, and none should be inferred from this codebase.
* IBM open-plan round-trips are scheduling-dominated observations, not
  deployed-service latencies or bounds on dedicated-service performance.
* The energy figures produced by `benckmark_programme.py` are
  order-of-magnitude deployment estimates, not measured device energies.

## Acknowledgements

SHD dataset: University of Heidelberg neuromorphic data repository. Quantum
simulations: PennyLane. Quantum hardware access: IBM Quantum services
(`ibm_marrakesh`); the views expressed are those of the author and do not
reflect the official policy or position of IBM or the IBM Quantum team.
Datacentre GPU measurements were performed on a Lambda Cloud H100 instance.

## Licence

MIT.
