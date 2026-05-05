# Quantum-Modulated Poisson Encoder — Benchmark Harness

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Drop-in replacement for `benckmark_programme.py`. Fixes two bugs from
the previous revision (swapped energy assignments, hard-coded latency
baseline) and adds the experimental machinery to broaden the paper's
contribution without overclaiming.

## Bugs fixed since the previous version

1. **Energy assignments were swapped.** Previous code had
   `energy_mac = events × 0.4 pJ` and `energy_route = mac_ops × 2.5 pJ`
   — both wrong. Now correctly: `energy_mac = mac_ops × 0.4 pJ` and
   `energy_route = events × 2.5 pJ`.
2. **`benchmark_latency_sweep(3.9454)` was hard-coded.** Now driven by
   `median_epoch_s` measured with `cuda_timer()` (CUDA-synchronised),
   so the latency table is consistent with the artefact's actual
   runtime on whatever hardware runs it.

## Run modes

Set the `MODE` environment variable (default `"all"`):

| `MODE`        | What it does                                                         | Approx. wall time     |
|---------------|----------------------------------------------------------------------|-----------------------|
| `main`        | Single-seed reference run + all reference figures                    | 3–5 min on a 30-series GPU |
| `seeds`       | Five-seed reproducibility study                                      | 15–25 min             |
| `ablation`    | PQC vs sinusoidal vs constant drive                                   | 10–15 min             |
| `sweep`       | (encoder_width × batch_size) sensitivity sweep, short epochs          | 25–35 min             |
| `pqc_latency` | Real PennyLane round-trip on default + lightning backends            | 10 s                  |
| `all`         | All of the above (default)                                            | 60–90 min             |

Examples:
```bash
MODE=main python benchmark_programme.py
MODE=seeds python benchmark_programme.py
```

## Outputs (in `publication_results/`)

| File                                | Source experiment   | Purpose in the paper                              |
|-------------------------------------|---------------------|---------------------------------------------------|
| `fig1_latent_drive.{png,pdf,…}`     | `main`              | Figure 1 (existing)                               |
| `raster_epoch_100.{…}`              | `main`              | Figure 2 (existing)                               |
| `latency_overhead.{…}`              | `main`              | Figure 3 (existing)                               |
| `energy_breakdown.{…}`              | `main`              | Figure 4 (existing)                               |
| `confusion_matrix.{…}`              | `main`              | Optional Section 3.1 figure                       |
| `latency_overhead_batch.{…}`        | `sweep`             | **New Figure**: overhead vs batch                 |
| `breakeven_scaling.{…}`             | `sweep`             | **New Figure**: hardware-selection frontier       |
| `energy_pareto.{…}`                 | `sweep`             | **New Figure**: encoder-width Pareto              |
| `training_metrics.csv`              | `main`              | Per-epoch loss / accuracy / rate / wall-clock     |
| `per_class_accuracy.csv`            | `main`              | Per-class SHD breakdown                           |
| `seed_summary.csv`                  | `seeds`             | Mean ± std across 5 seeds                         |
| `drive_ablation.csv`                | `ablation`          | PQC vs sinusoid vs constant table                 |
| `sensitivity_sweep.csv`             | `sweep`             | (W, B) grid: accuracy, runtime, energy            |
| `pqc_round_trip.csv`                | `pqc_latency`       | Measured PennyLane round-trip (ms)                |
| `hardware_fingerprint.csv`          | always              | GPU / CUDA / PyTorch versions                     |
| `latency_table.csv`                 | `main`              | Closed-form overhead at default config            |

## Mapping experiments to manuscript edits

See `manuscript_additions.tex` for the LaTeX blocks to paste in. The
mapping:

- Bullet [2] (Reproducibility paragraph) ← `hardware_fingerprint.csv`
- Bullet [3] (Multi-seed table) ← `seed_summary.csv`
- Bullet [4] (Drive ablation table) ← `drive_ablation.csv`
- Bullet [5] (Batch-size scaling subsection + figure) ← `latency_overhead_batch.png`
- Bullet [6] (Hardware-selection frontier + figure) ← `breakeven_scaling.png`
- Bullet [7] (Measured PQC latency rows) ← `pqc_round_trip.csv`
- Bullet [8] (Energy Pareto figure) ← `energy_pareto.png`
- Bullet [9] (Confusion matrix figure, optional) ← `confusion_matrix.png`

## Why these specific extensions

The previous draft was structurally honest but narrow: a closed-form
overhead calculation at one batch size with one encoder width. The
additions broaden the contribution along three axes that are credibly
defensible and require no new architecture work:

1. **Reproducibility credibility.** Five-seed runs + cuDNN determinism
   + hardware fingerprint kill the "single-seed cherry-pick" reviewer
   suspicion.
2. **Functional ablation.** Showing that PQC ≈ sinusoid ≈ constant on
   accuracy proves the paper is what it says it is — an I/O analysis,
   not a hidden expressivity claim. Reviewers respect this kind of
   self-honesty.
3. **A genuinely new closed-form result.** The hardware-selection
   frontier `τ*(B; p) = (p/(1-p)) · T_core(B)` is the new substantive
   contribution. It's two lines of algebra, but it reframes the
   conclusion from "TSV is fastest" (uninteresting) to "TSV is the
   only paradigm meeting a 1% budget across the full edge-inference
   batch range, derived analytically" (useful selection criterion).
4. **Batch-scaling dataset.** The sensitivity sweep produces real
   measured `T_core(B)` for B ∈ {1, 16, 64, 256}, so the new figures
   are not synthetic.

## Known limitations to disclose

- The energy section still uses order-of-magnitude constants
  (0.4 pJ MAC, 2.5 pJ route). These are not from a primary source;
  the manuscript correctly flags this.
- The PQC is offline and one-shot. The drive ablation makes this
  fully transparent.
- Cuda determinism with `Conv1d` may still produce small numerical
  differences across GPU generations (Ampere vs Hopper); the
  per-seed std captures this.
- `lightning.qubit` may not be installed; the script handles this
  gracefully and reports `nan`.
