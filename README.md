# Macroscopic Cox-Process Synthesis (MCPS) for Hybrid QSNNs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official benchmark implementation for the theoretical framework introduced in the paper: **"Overcoming the Quantum-Classical Bottleneck: Macroscopic Cox-Process Synthesis for Ultra-Low Latency Quantum Spiking Neural Networks."**

## Overview

Hybrid Quantum Spiking Neural Networks (QSNNs) struggle with severe I/O latency bottlenecks when classical simulators attempt to communicate with Noisy Intermediate-Scale Quantum (NISQ) devices. Traditional methods require dense $O(N^2 \cdot T)$ recurrent Euler integration, making real-time execution impossible.

This repository provides a Python-based classical proxy simulator that implements **Macroscopic Cox-Process Synthesis (MCPS)** modulated by a Parametrised Quantum Circuit (PQC). By shifting from microscopic Euler integration to Poisson-distributed activations, this architecture establishes the theoretical limits of QSNN efficiency and provides a framework for evaluating classical-quantum synchronisation overhead.

## Key Features

*   **Spatiotemporal Convolutional Proxy:** Utilises a 1D Temporal Convolution layer to explicitly capture sequential dynamics in neuromorphic data without collapsing the temporal dimension prematurely.
*   **PQC Modulation (PennyLane):** Integrates an 8-qubit Parametrised Quantum Circuit that synthesises a macroscopic population drive ($\lambda(t)$) to modulate the classical network.
*   **Cox-Process Synthesis:** Bypasses iterative Euler integration in favour of $O(N \cdot T)$ independent Poisson sampling based on the PQC-modulated intensity.
*   **Biologically Plausible Normalisation:** Employs a two-stage data-driven tuning algorithm (Variance Stabilisation + Global Target-Locking) to anchor the network in the Asynchronous Irregular (AI) critical regime (target: 10–15 Hz).
*   **Hardware Overhead Extrapolation:** Calculates arithmetic latency extrapolations across distinct hardware paradigms (Cloud Quantum, PCIe, CPO, Monolithic TSV) based on core execution times.
*   **Empirical Energy Estimation:** Provides INT4 equivalent Synaptic Operation (SynOp) counts and projects energy payloads using estimates inspired by sub-10nm asynchronous neuromorphic hardware (e.g., Intel 4).

## Requirements

The simulation requires Python 3.8+ and the following dependencies:
```bash
pip install torch torchvision torchaudio
pip install pennylane
pip install numpy matplotlib
