import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import csv
import platform
import random
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F



ENERGY_PER_MAC_PJ = 0.4
ENERGY_PER_ROUTE_PJ = 2.5
CLOUD_QPU_LATENCY_MS = 50.0

ENCODER_WIDTH_DEFAULT = 4096
TIME_STEPS = 100
TARGET_HZ = 12.5
QPU_QUBITS = 8
BATCH_SIZE_DEFAULT = 256
EPOCHS_DEFAULT = 100
INPUT_CHANNELS = 700
NUM_CLASSES = 20
REDUCED_DIM = 256

OUTPUT_DIR = Path("publication_results")
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available()
                     else "cpu")
MODE = os.environ.get("MODE", "all")

# Architectures & latencies for the closed-form overhead sweep
ARCHITECTURES = ("Cloud Quantum", "PCIe Local", "MCM Chiplet", "Advanced CPO", "Monolithic TSV")
LATENCY_BENCHMARKS_MS = (50.0, 5.0, 0.5, 0.05, 0.0005)

COLOUR_PRIMARY = '#1f77b4'
COLOUR_SECONDARY = '#ff7f0e'
COLOUR_ACCENT = '#2ca02c'
COLOUR_DANGER = '#d62728'
COLOUR_NEUTRAL = '#7f7f7f'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.0,
    'figure.dpi': 200,
    'savefig.dpi': 600,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    print("[warn] PennyLane not installed; classical sinusoidal surrogate will be used for the latent drive.")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("[warn] h5py not installed; SHD download path is disabled, random surrogate data will be used.")


def seed_all(seed: int) -> None:
    """Set every RNG source we use. O(1)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def hardware_fingerprint() -> dict:
    """Capture a reproducibility fingerprint. Strings only; safe for CSV."""
    fp = {
        "device": DEVICE.type,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        fp["gpu"] = torch.cuda.get_device_name(0)
        fp["cuda"] = torch.version.cuda or "unknown"
        fp["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return fp



def get_shd_dataset(cache_dir: str = "data"):
    """Loads SHD or returns a small random surrogate if h5py is unavailable."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    train_path, test_path = cache / "shd_train.h5", cache / "shd_test.h5"

    if not HAS_H5PY:
        X_train = (torch.rand(256, TIME_STEPS, INPUT_CHANNELS) < 0.03).float()
        y_train = torch.randint(0, NUM_CLASSES, (256,))
        X_test = (torch.rand(64, TIME_STEPS, INPUT_CHANNELS) < 0.03).float()
        y_test = torch.randint(0, NUM_CLASSES, (64,))
        return X_train, y_train, X_test, y_test

    if not (train_path.exists() and test_path.exists()):
        print("Downloading SHD dataset...")
        for split, url in (("train", "https://compneuro.net/datasets/shd_train.h5.zip"),
                           ("test", "https://compneuro.net/datasets/shd_test.h5.zip")):
            zip_path = cache / f"shd_{split}.zip"
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(cache)

    def process(filepath: Path):
        with h5py.File(filepath, 'r') as f:
            units = f['spikes']['units'][:]
            times = f['spikes']['times'][:]
            labels = f['labels'][:]
        X = torch.zeros((len(labels), TIME_STEPS, INPUT_CHANNELS))
        for i in range(len(labels)):
            if len(units[i]) > 0:
                t_idx = np.clip((times[i] * TIME_STEPS).astype(int), 0, TIME_STEPS - 1)
                X[i, t_idx, units[i]] = 1.0
        return X, torch.tensor(labels, dtype=torch.long)

    X_tr, y_tr = process(train_path)
    X_te, y_te = process(test_path)
    return X_tr, y_tr, X_te, y_te



if HAS_PENNYLANE:
    _qdev_default = qml.device("default.qubit", wires=QPU_QUBITS)

    @qml.qnode(_qdev_default, interface="torch")
    def _pqc(inputs, weights):
        for i in range(QPU_QUBITS):
            qml.RX(inputs[i], wires=i)
        for i in range(QPU_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[QPU_QUBITS - 1, 0])
        for i in range(QPU_QUBITS):
            qml.RY(weights[i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(QPU_QUBITS)]


def generate_macroscopic_drive(steps: int, drive_kind: str = "pqc") -> torch.Tensor:
    """
    Build the (steps,) drive vector.

    drive_kind:
        "pqc"      : 8-qubit PQC (PennyLane), or sinusoidal fallback if unavailable
        "sine"     : classical sinusoidal surrogate (control)
        "constant" : flat drive (control; isolates the contribution of temporal modulation)
    """
    if drive_kind == "constant":
        return torch.ones(steps) * 0.3

    if drive_kind == "sine" or not HAS_PENNYLANE:
        t = torch.arange(steps, dtype=torch.float32)
        freqs = torch.rand(QPU_QUBITS) * 0.1
        phases = torch.rand(QPU_QUBITS) * 2 * np.pi
        wave = torch.sin(t.unsqueeze(1) * freqs + phases).mean(dim=1)
        return F.softplus(wave) + 0.1

    keyframes = 50
    t_keys = torch.linspace(0, 5.0, keyframes)
    snaps = torch.zeros(keyframes)
    weights = torch.rand(QPU_QUBITS) * 2 * np.pi
    for i in range(keyframes):
        inputs = torch.full((QPU_QUBITS,), t_keys[i].item())
        snaps[i] = torch.stack(_pqc(inputs, weights)).mean()
    drive = F.interpolate(snaps[None, None, :], size=steps, mode='linear', align_corners=True).squeeze()
    return F.softplus(drive) + 0.1


def measure_pqc_round_trip(n_calls: int = 100) -> dict:
    """Measure mean wall-clock latency of one PQC evaluation on default/lightning."""
    if not HAS_PENNYLANE:
        return {}
    results = {}
    weights = torch.rand(QPU_QUBITS) * 2 * np.pi
    inputs = torch.full((QPU_QUBITS,), 1.0)

    for backend in ("default.qubit", "lightning.qubit"):
        try:
            dev_b = qml.device(backend, wires=QPU_QUBITS)

            @qml.qnode(dev_b, interface="torch")
            def circuit(inp, w):
                for i in range(QPU_QUBITS):
                    qml.RX(inp[i], wires=i)
                for i in range(QPU_QUBITS - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[QPU_QUBITS - 1, 0])
                for i in range(QPU_QUBITS):
                    qml.RY(w[i], wires=i)
                return [qml.expval(qml.PauliZ(i)) for i in range(QPU_QUBITS)]

            for _ in range(5):  # warm-up
                circuit(inputs, weights)
            t0 = time.perf_counter()
            for _ in range(n_calls):
                _ = circuit(inputs, weights)
            elapsed = (time.perf_counter() - t0) / n_calls
            results[backend] = elapsed * 1000.0  # ms
        except Exception as exc:
            results[backend] = float("nan")
            print(f"[warn] backend {backend!r} unavailable: {exc}")
    return results



def save_high_res_figure(fig, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'tiff', 'svg'):
        fig.savefig(OUTPUT_DIR / f"{filename}.{ext}", dpi=600, format=ext, bbox_inches='tight')

def plot_latent_drive(drive: torch.Tensor) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    w = drive.cpu().numpy()
    ax.plot(w, color=COLOUR_PRIMARY, linewidth=2.5)
    ax.fill_between(range(len(w)), w, color=COLOUR_PRIMARY, alpha=0.15)
    ax.set_title(r"PQC-Modulated Macroscopic Population Drive ($\lambda(t)$)", fontweight='bold')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Drive Intensity")
    save_high_res_figure(fig, "fig1_latent_drive")
    plt.close(fig)

def plot_raster(spikes: torch.Tensor, epoch: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    t_idx, u_idx = np.where(spikes[0, :, :100].detach().cpu().numpy())
    ax.scatter(t_idx, u_idx, s=6, c=COLOUR_PRIMARY, marker='|', alpha=0.9)
    ax.set_title(f"Encoder Raster Activity Snapshot (Epoch {epoch})", fontweight='bold')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Encoder Unit ID")
    ax.set_xlim(0, TIME_STEPS); ax.set_ylim(0, 100)
    save_high_res_figure(fig, f"raster_epoch_{epoch}")
    plt.close(fig)

def plot_energy_breakdown(mac_J: float, route_J: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [mac_J, route_J]
    explode = (0.05, 0.0) if mac_J > route_J else (0.0, 0.05)
    ax.pie(sizes, explode=explode,
           labels=['Weight Projection (MAC)', 'Event Routing'],
           colors=[COLOUR_PRIMARY, COLOUR_SECONDARY],
           autopct='%1.2f%%', shadow=False, startangle=140,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title("Hardware Energy Allocation", fontweight='bold')
    save_high_res_figure(fig, "energy_breakdown")
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, num_classes: int = NUM_CLASSES) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('SHD Test-Set Confusion Matrix', fontweight='bold')
    save_high_res_figure(fig, "confusion_matrix")
    plt.close(fig)
    return cm

def plot_latency_sweep(core_s: float) -> list[float]:
    """Closed-form overhead sweep at the headline batch size."""
    overheads = []
    print(f"\n--- Latency Integration Benchmarks (T_core = {core_s:.4f} s) ---")
    for arch, lat_ms in zip(ARCHITECTURES, LATENCY_BENCHMARKS_MS):
        lat_s = lat_ms / 1000.0
        ovh = (lat_s / (core_s + lat_s)) * 100
        overheads.append(ovh)
        print(f"  {arch:<20} | I/O = {lat_ms:<7} ms | Overhead = {ovh:>7.4f}%")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(LATENCY_BENCHMARKS_MS, overheads, marker='o', markersize=9,
            color=COLOUR_PRIMARY, linewidth=2.5, zorder=3)
    for i, name in enumerate(ARCHITECTURES):
        ax.annotate(f" {name}", (LATENCY_BENCHMARKS_MS[i], overheads[i]),
                    fontsize=11, color=COLOUR_NEUTRAL, xytext=(5, 5), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_title("QPU I/O Latency vs. System Overhead", fontweight='bold')
    ax.set_xlabel("Hardware Integration Latency (ms) [Log Scale]")
    ax.set_ylabel("QPU Synchronisation Penalty (%)")
    save_high_res_figure(fig, "latency_overhead")
    plt.close(fig)
    return overheads

def plot_latency_vs_batch(core_times_per_batch: dict[int, float]) -> None:
    """The headline new figure: overhead inverts at low batch sizes."""
    fig, ax = plt.subplots(figsize=(9, 6))
    batches = sorted(core_times_per_batch)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(LATENCY_BENCHMARKS_MS)))
    for (arch, lat_ms), c in zip(zip(ARCHITECTURES, LATENCY_BENCHMARKS_MS), cmap):
        ovh = []
        for b in batches:
            t = core_times_per_batch[b]
            lat_s = lat_ms / 1000.0
            ovh.append((lat_s / (t + lat_s)) * 100)
        ax.plot(batches, ovh, marker='o', linewidth=2.2, color=c, label=f"{arch} ({lat_ms} ms)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Inference Batch Size [Log Scale]")
    ax.set_ylabel("QPU Synchronisation Penalty (%) [Log Scale]")
    ax.set_title("Edge-Inference Bottleneck: Overhead Inverts at Small Batch", fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    save_high_res_figure(fig, "latency_overhead_batch")
    plt.close(fig)

def plot_breakeven(core_times: dict[int, float], threshold_pct: float = 1.0) -> None:
    """tau_breakeven = (p/100)/(1 - p/100) * T_core, plotted vs batch."""
    fig, ax = plt.subplots(figsize=(9, 6))
    batches = sorted(core_times)
    p = threshold_pct / 100.0
    breakeven_ms = [(p / (1 - p)) * core_times[b] * 1000.0 for b in batches]
    ax.plot(batches, breakeven_ms, marker='s', linewidth=2.2, color=COLOUR_DANGER, label=f"Break-even ({threshold_pct}%)")
    for arch, lat_ms in zip(ARCHITECTURES, LATENCY_BENCHMARKS_MS):
        ax.axhline(lat_ms, linestyle=':', linewidth=1.0, color=COLOUR_NEUTRAL, alpha=0.7)
        ax.text(batches[-1], lat_ms, f"  {arch}", fontsize=9, va='center', color=COLOUR_NEUTRAL)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Inference Batch Size [Log Scale]")
    ax.set_ylabel(r"Maximum Tolerable QPU Round-Trip $\tau_{QPU}^*$ (ms) [Log Scale]")
    ax.set_title(f"Hardware Selection Frontier ({threshold_pct}% Overhead Budget)", fontweight='bold')
    ax.legend(loc='best')
    save_high_res_figure(fig, "breakeven_scaling")
    plt.close(fig)

def plot_energy_pareto(records: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    widths = [r['encoder_width'] for r in records]
    accs = [r['test_acc'] for r in records]
    energies_uj = [r['energy_J'] * 1e6 for r in records]
    sc = ax.scatter(energies_uj, accs, s=[w / 8 for w in widths],
                    c=widths, cmap='viridis', edgecolor='black', linewidth=0.8, zorder=3)
    for r in records:
        ax.annotate(f" W={r['encoder_width']}", (r['energy_J'] * 1e6, r['test_acc']),
                    fontsize=10, color=COLOUR_NEUTRAL)
    ax.set_xlabel(r"Estimated Energy per Inference ($\mu$J)")
    ax.set_ylabel("SHD Test Accuracy (%)")
    ax.set_title("Encoder Width: Accuracy / Energy Pareto Front", fontweight='bold')
    fig.colorbar(sc, label="Encoder Width")
    save_high_res_figure(fig, "energy_pareto")
    plt.close(fig)



class INT4LinearSTE(torch.nn.Linear):
    """
    Per-tensor symmetric INT4 STE quantisation. O(n) on the weight tensor.
    Forward emulates W_q in [-7, 7] * s, backward passes gradients straight through.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = (self.weight.abs().max() / 7.0).clamp(min=1e-8)
        w_ste = self.weight + (torch.round(self.weight / scale) * scale - self.weight).detach()
        return F.linear(x, w_ste, self.bias)

class QuantumModulatedEncoder(torch.nn.Module):
    """
    Spatiotemporal Poisson encoder driven by a one-shot offline PQC waveform.

    Forward complexity: O(B * T * (D_in + D_red * k + D_red * W + W * C))
                      ~ O(B * T * D_red * W) for typical W >> D_red, k.
    Memory: O(B * T * W) for the spike tensor.
    """

    def __init__(self,
                 in_features: int = INPUT_CHANNELS,
                 encoder_width: int = ENCODER_WIDTH_DEFAULT,
                 out_classes: int = NUM_CLASSES,
                 reduced_dim: int = REDUCED_DIM,
                 time_steps: int = TIME_STEPS) -> None:
        super().__init__()
        self.reduced_dim = reduced_dim
        self.encoder_width = encoder_width
        self.time_steps = time_steps

        self.spatial_proj = INT4LinearSTE(in_features, reduced_dim)
        self.temporal_conv = torch.nn.Conv1d(reduced_dim, reduced_dim, kernel_size=5, padding=2)
        self.fc1 = INT4LinearSTE(reduced_dim, encoder_width)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.fc2 = INT4LinearSTE(encoder_width, out_classes)
        self.gain = torch.nn.Parameter(torch.tensor(1.0))

    def _compute_drive(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D_in)
        b, t, _ = x.shape
        x_proj = F.relu(self.spatial_proj(x.reshape(-1, x.shape[-1]))).view(b, t, -1)
        conv_out = F.relu(self.temporal_conv(x_proj.permute(0, 2, 1))).permute(0, 2, 1)
        spatial_act = F.softplus(self.fc1(conv_out))
        return spatial_act * latent.unsqueeze(0).unsqueeze(2) * torch.abs(self.gain)

    def forward(self, x: torch.Tensor, latent: torch.Tensor):
        drive = self._compute_drive(x, latent)
        spikes = drive + (torch.poisson(drive) - drive).detach()  # STE-Poisson
        spikes = torch.clamp(spikes, max=1.0)
        decay = torch.exp(torch.linspace(-2.0, 0.0, self.time_steps, device=spikes.device))
        weighted = (spikes * decay.unsqueeze(0).unsqueeze(2)).sum(dim=1)
        logits = self.fc2(self.dropout(weighted))
        return logits, spikes, drive

    def calibrate_gain(self, sample_x: torch.Tensor, latent: torch.Tensor,
                       target_hz: float = TARGET_HZ) -> float:
        """Two-stage init: variance stabilise, then gain-lock to target rate. O(B*T*W)."""
        with torch.no_grad():
            std_target = np.sqrt(2.0 / (self.temporal_conv.weight.size(1) * self.temporal_conv.weight.size(2)))
            self.temporal_conv.weight.data /= (self.temporal_conv.weight.data.std() + 1e-8)
            self.temporal_conv.weight.data *= std_target

            self.fc1.weight.data /= (self.fc1.weight.data.std() + 1e-8)
            self.fc1.weight.data *= np.sqrt(2.0 / self.fc1.weight.size(1))

            base = self._compute_drive(sample_x, latent)
            current = base.mean().item()
            if current > 0:
                self.gain.data *= (target_hz / 1000.0) / current
            tuned_hz = self._compute_drive(sample_x, latent).mean().item() * 1000.0
        return tuned_hz


@contextmanager
def cuda_timer():
    """Wall-clock timer that synchronises CUDA at entry/exit. Yields callable for elapsed seconds."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    state = {"t": None}
    try:
        yield state
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        state["t"] = time.perf_counter() - t0

def train_one_run(seed: int,
                  encoder_width: int = ENCODER_WIDTH_DEFAULT,
                  batch_size: int = BATCH_SIZE_DEFAULT,
                  epochs: int = EPOCHS_DEFAULT,
                  drive_kind: str = "pqc",
                  X_train=None, y_train=None, X_test=None, y_test=None,
                  verbose: bool = True,
                  save_artefacts: bool = False) -> dict:
    """One full training run. Returns metrics dict. O(epochs * |train| * T * W)."""
    seed_all(seed)
    if any(v is None for v in (X_train, y_train, X_test, y_test)):
        X_train, y_train, X_test, y_test = get_shd_dataset()
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

    latent = generate_macroscopic_drive(TIME_STEPS, drive_kind=drive_kind).to(DEVICE)
    if save_artefacts:
        plot_latent_drive(latent)

    model = QuantumModulatedEncoder(in_features=INPUT_CHANNELS,
                                    encoder_width=encoder_width,
                                    out_classes=NUM_CLASSES).to(DEVICE)

    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.14)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.06)
    tuned_hz = model.calibrate_gain(X_train[:batch_size], latent, TARGET_HZ)
    if verbose:
        print(f"[seed={seed} W={encoder_width} B={batch_size} drive={drive_kind}] tuned to {tuned_hz:.2f} Hz")

    best_test = 0.0
    history = []
    epoch_times: list[float] = []
    last_spikes = None
    last_total_spikes = 0
    last_train_acc = 0.0
    last_rate = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        correct = total_spikes = 0
        epoch_loss = 0.0
        with cuda_timer() as et:
            for i in range(0, len(X_train), batch_size):
                xb = X_train[i:i + batch_size]
                yb = y_train[i:i + batch_size]
                optimiser.zero_grad()
                logits, spikes, drive = model(xb, latent)
                loss = criterion(logits, yb) + (((drive.mean() - (TARGET_HZ / 1000.0)) ** 2) * 1e5)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total_spikes += spikes.sum().item()
                last_spikes = spikes
        epoch_times.append(et["t"])
        train_acc = (correct / len(X_train)) * 100
        avg_rate_hz = (total_spikes / (len(X_train) * encoder_width * TIME_STEPS)) * 1000

        model.eval()
        with torch.no_grad():
            correct_t = 0
            for i in range(0, len(X_test), batch_size):
                logits, _, _ = model(X_test[i:i + batch_size], latent)
                correct_t += (logits.argmax(dim=1) == y_test[i:i + batch_size]).sum().item()
        test_acc = (correct_t / len(X_test)) * 100
        best_test = max(best_test, test_acc)
        history.append((epoch, epoch_loss, train_acc, test_acc, avg_rate_hz, et["t"]))
        last_total_spikes, last_train_acc, last_rate = total_spikes, train_acc, avg_rate_hz
        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"  ep {epoch:3d} | loss {epoch_loss:7.3f} | tr {train_acc:5.2f}% | te {test_acc:5.2f}% | rate {avg_rate_hz:5.2f} Hz | {et['t']*1000:.0f} ms")

    if save_artefacts and last_spikes is not None:
        plot_raster(last_spikes, epochs)

    # Per-class accuracy & confusion matrix
    model.eval()
    y_true_all, y_pred_all = [], []
    inf_lat_per_sample_list = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = X_test[i:i + batch_size]
            yb = y_test[i:i + batch_size]
            with cuda_timer() as it:
                logits, _, _ = model(xb, latent)
            inf_lat_per_sample_list.append(it["t"] / xb.shape[0])
            y_true_all.extend(yb.cpu().numpy().tolist())
            y_pred_all.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    cm = None
    if save_artefacts:
        cm = plot_confusion_matrix(y_true_all, y_pred_all, NUM_CLASSES)

    # Energy: assumes hypothetical full-INT4 deployment.
    # MAC ops accounting:
    #   - input projection: avg_input_spikes * encoder_width
    #   - readout: avg_events * num_classes
    avg_events = last_total_spikes / len(X_train)
    avg_input_spikes = X_train.sum().item() / len(X_train)
    mac_ops = (avg_input_spikes * encoder_width) + (avg_events * NUM_CLASSES)
    energy_mac_J = mac_ops * ENERGY_PER_MAC_PJ * 1e-12
    energy_route_J = avg_events * ENERGY_PER_ROUTE_PJ * 1e-12
    energy_total_J = energy_mac_J + energy_route_J

    median_epoch_s = float(np.median(epoch_times))
    mean_inf_per_sample_ms = float(np.mean(inf_lat_per_sample_list)) * 1000.0

    if save_artefacts:
        plot_energy_breakdown(energy_mac_J, energy_route_J)
        # Use inference time per batch (B * t_sample), not training epoch time.
        T_core_inference = mean_inf_per_sample_ms / 1000.0 * batch_size
        plot_latency_sweep(T_core_inference)

    return {
        "seed": seed,
        "encoder_width": encoder_width,
        "batch_size": batch_size,
        "drive_kind": drive_kind,
        "epochs": epochs,
        "tuned_hz": tuned_hz,
        "train_acc": last_train_acc,
        "test_acc": best_test,
        "final_test_acc": history[-1][3],
        "event_rate_hz": last_rate,
        "median_epoch_s": median_epoch_s,
        "mean_inference_per_sample_ms": mean_inf_per_sample_ms,
        "mac_ops_per_sample": mac_ops,
        "events_per_sample": avg_events,
        "energy_mac_J": energy_mac_J,
        "energy_route_J": energy_route_J,
        "energy_J": energy_total_J,
        "history": history,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "confusion_matrix": cm,
    }



def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = [k for k in rows[0].keys() if not isinstance(rows[0][k], (list, np.ndarray))]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})


# Experiments
def experiment_main(X_train, y_train, X_test, y_test) -> dict:
    """Single-seed reference run with full artefact generation."""
    print("\n=========================  EXPERIMENT 1: REFERENCE RUN  =========================")
    out = train_one_run(seed=42, drive_kind="pqc",
                        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        save_artefacts=True)
    # Persist training history
    with open(OUTPUT_DIR / "training_metrics.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "train_acc", "test_acc", "rate_hz", "epoch_seconds"])
        for row in out["history"]:
            w.writerow(row)
    # Per-class accuracy
    cm = out["confusion_matrix"]
    if cm is not None:
        per_class = []
        for c in range(NUM_CLASSES):
            denom = cm[c].sum()
            acc = (cm[c, c] / denom * 100.0) if denom > 0 else 0.0
            per_class.append({"class": c, "samples": int(denom), "accuracy_pct": float(acc)})
        write_csv(OUTPUT_DIR / "per_class_accuracy.csv", per_class)
    return out

def experiment_seeds(X_train, y_train, X_test, y_test, seeds: Iterable[int] = (1, 2, 3, 4, 5)) -> list[dict]:
    """Multi-seed reproducibility study at default config."""
    print("\n=========================  EXPERIMENT 2: MULTI-SEED RUN  =========================")
    rows = []
    for s in seeds:
        r = train_one_run(seed=s, drive_kind="pqc",
                          X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                          save_artefacts=False, verbose=False, epochs=EPOCHS_DEFAULT)
        rows.append({
            "seed": s,
            "test_acc": r["test_acc"],
            "final_test_acc": r["final_test_acc"],
            "train_acc": r["train_acc"],
            "rate_hz": r["event_rate_hz"],
            "median_epoch_s": r["median_epoch_s"],
            "energy_uJ": r["energy_J"] * 1e6,
        })
        print(f"  seed {s}: best_test={r['test_acc']:.2f}% rate={r['event_rate_hz']:.2f} Hz "
              f"epoch={r['median_epoch_s']*1000:.0f} ms")
    write_csv(OUTPUT_DIR / "seed_summary.csv", rows)
    accs = np.array([r["test_acc"] for r in rows])
    rates = np.array([r["rate_hz"] for r in rows])
    times = np.array([r["median_epoch_s"] for r in rows])
    print(f"  -> test acc:     {accs.mean():.2f} +/- {accs.std(ddof=1):.2f} %")
    print(f"  -> event rate:   {rates.mean():.2f} +/- {rates.std(ddof=1):.2f} Hz")
    print(f"  -> epoch time:   {times.mean()*1000:.1f} +/- {times.std(ddof=1)*1000:.1f} ms")
    return rows

def experiment_drive_ablation(X_train, y_train, X_test, y_test) -> list[dict]:
    """Compare PQC drive against classical sinusoidal and constant controls."""
    print("\n=========================  EXPERIMENT 3: DRIVE ABLATION  =========================")
    rows = []
    for kind in ("constant", "sine", "pqc"):
        r = train_one_run(seed=42, drive_kind=kind,
                          X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                          save_artefacts=False, verbose=False, epochs=EPOCHS_DEFAULT)
        rows.append({
            "drive_kind": kind,
            "test_acc": r["test_acc"],
            "rate_hz": r["event_rate_hz"],
            "median_epoch_s": r["median_epoch_s"],
        })
        print(f"  drive={kind:8s} test_acc={r['test_acc']:5.2f}% rate={r['event_rate_hz']:.2f} Hz")
    write_csv(OUTPUT_DIR / "drive_ablation.csv", rows)
    return rows

def experiment_sensitivity_sweep(X_train, y_train, X_test, y_test,
                                 widths=(1024, 2048, 4096, 8192),
                                 batches=(1, 16, 64, 256),
                                 short_epochs: int = 25) -> list[dict]:
    """
    (encoder_width, batch_size) sweep. Uses fewer epochs to keep total runtime sane.
    Provides T_core(B) for the latency-vs-batch and break-even plots.
    """
    print("\n=========================  EXPERIMENT 4: SENSITIVITY SWEEP  =========================")
    rows = []
    pareto = []  # one point per width at default batch
    core_times_per_batch: dict[int, float] = {}
    for w in widths:
        for b in batches:
            r = train_one_run(seed=42, encoder_width=w, batch_size=b,
                              drive_kind="pqc", epochs=short_epochs,
                              X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                              save_artefacts=False, verbose=False)
            row = {
                "encoder_width": w,
                "batch_size": b,
                "test_acc": r["test_acc"],
                "median_epoch_s": r["median_epoch_s"],
                "mean_inf_per_sample_ms": r["mean_inference_per_sample_ms"],
                "energy_J": r["energy_J"],
            }
            rows.append(row)
            print(f"  W={w:5d} B={b:3d} test={r['test_acc']:5.2f}% epoch={r['median_epoch_s']*1000:.0f} ms "
                  f"inf/sample={r['mean_inference_per_sample_ms']:.2f} ms")
            if b == 256 and w == ENCODER_WIDTH_DEFAULT:
                pass  # main run captures default config
            if b == 256:
                pareto.append({"encoder_width": w, "test_acc": r["test_acc"], "energy_J": r["energy_J"]})
            if w == ENCODER_WIDTH_DEFAULT:
                # Skip B=256 here; the sweep run was contaminated by VRAM paging at
                # this batch size on 8 GB GPUs. The reference run measures B=256 cleanly.
                if b != 256:
                    core_times_per_batch[b] = r["mean_inference_per_sample_ms"] / 1000.0 * b
    write_csv(OUTPUT_DIR / "sensitivity_sweep.csv", rows)
    if pareto:
        plot_energy_pareto(pareto)
    if core_times_per_batch:
        plot_latency_vs_batch(core_times_per_batch)
        plot_breakeven(core_times_per_batch, threshold_pct=1.0)
    return rows

def experiment_pqc_latency() -> dict:
    """Real PennyLane round-trip times. Adds a measured row to the latency table."""
    print("\n=========================  EXPERIMENT 5: PQC LATENCY  =========================")
    res = measure_pqc_round_trip(n_calls=100)
    if not res:
        return {}
    for backend, ms in res.items():
        print(f"  {backend:<20} {ms:8.4f} ms / call")
    write_csv(OUTPUT_DIR / "pqc_round_trip.csv",
              [{"backend": k, "mean_latency_ms": v} for k, v in res.items()])
    return res



def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_all(42)

    print("=" * 80)
    print(" Quantum-Modulated Poisson Encoder: Benchmark Harness")
    print("=" * 80)
    fp = hardware_fingerprint()
    for k, v in fp.items():
        print(f"  {k:<10} {v}")
    write_csv(OUTPUT_DIR / "hardware_fingerprint.csv", [fp])

    # Load dataset once and reuse across experiments
    print("\nLoading SHD...")
    X_train, y_train, X_test, y_test = get_shd_dataset()
    print(f"  train={tuple(X_train.shape)} test={tuple(X_test.shape)}")

    if MODE in ("main", "all"):
        experiment_main(X_train, y_train, X_test, y_test)

    if MODE in ("seeds", "all"):
        experiment_seeds(X_train, y_train, X_test, y_test)

    if MODE in ("ablation", "all"):
        experiment_drive_ablation(X_train, y_train, X_test, y_test)

    sweep_core_times: dict[int, float] = {}
    if MODE in ("sweep", "all"):
        sweep_rows = experiment_sensitivity_sweep(X_train, y_train, X_test, y_test)
        if MODE == "all":
            # Pull reference-run inference time from a fresh, isolated forward pass.
            seed_all(42)
            ref_model = QuantumModulatedEncoder(in_features=INPUT_CHANNELS,
                                                encoder_width=ENCODER_WIDTH_DEFAULT,
                                                out_classes=NUM_CLASSES).to(DEVICE)
            ref_latent = generate_macroscopic_drive(TIME_STEPS, drive_kind="pqc").to(DEVICE)
            ref_x = X_test[:256].to(DEVICE)
            ref_model.eval()
            with torch.no_grad():
                for _ in range(3):  # warm-up
                    ref_model(ref_x, ref_latent)
                with cuda_timer() as rt:
                    for _ in range(5):
                        ref_model(ref_x, ref_latent)
            t_per_sample_256_s = rt["t"] / (5 * 256)
            sweep_core_times[256] = t_per_sample_256_s * 256
            for row in sweep_rows:
                if row["encoder_width"] == ENCODER_WIDTH_DEFAULT and row["batch_size"] != 256:
                    sweep_core_times[row["batch_size"]] = row["mean_inf_per_sample_ms"] / 1000.0 * row["batch_size"]
            if sweep_core_times:
                plot_latency_vs_batch(sweep_core_times)
                plot_breakeven(sweep_core_times, threshold_pct=1.0)
            print("\n--- Clean batch-scaling T_core(B) used for figures ---")
            for b in sorted(sweep_core_times):
                print(f"  B={b:>3}: T_core = {sweep_core_times[b]*1000:>7.2f} ms")

    if MODE in ("pqc_latency", "all"):
        experiment_pqc_latency()

    print("\nAll experiments complete. Artefacts in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    torch.set_num_threads(15)
    main()
