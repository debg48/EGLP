# Event-Gated Local Plasticity (EGLP): A Biologically Plausible Learning Framework

**Event-Gated Local Plasticity (EGLP)** is a PyTorch-based framework for experimenting with biologically plausible learning rules as an alternative to standard backpropagation.

This project implements a hybrid learning approach where:

1. **Local Plasticity**: Individual layers update their weights based on local pre-synaptic and post-synaptic activity (Hebbian-like learning).
2. **Event Gating**: A global scalar signal ("event"), approximating neuromodulatory release (e.g., dopamine), gates these local updates.

The goal is to explore how sparse, global feedback signals can coordinate local learning rules to solve tasks like MNIST digit classification without the non-local error signal propagation required by backpropagation.

## 🔑 Key Features

- **Biologically Plausible Learning**: Custom `LocalLinear` and `LocalConv2d` layers that implement local update rules, avoiding global gradient chains.
- **Event-Based Modulation**: Supports multiple event generation strategies:
  - **Fixed Rate**: Events occur randomly at a set frequency (e.g., $\epsilon=0.1$).
  - **Threshold-Triggered**: Events are triggered by an `EventController` when the loss exceeds a dynamic threshold.
  - **Always-On**: Pure Hebbian learning baseline.
- **Comprehensive Benchmarking**: Includes a complete experimental suite to compare EGLP against:
  - Standard Backpropagation (performance upper bound).
  - Pure Local / Hebbian Learning (performance lower bound).
- **Analysis Tools**: Built-in modules for ablation studies, linear probing of representation quality, layer sensitivity analysis, and robustness testing on noisy/rotated data.

## 📦 Installation & Requirements

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/EGLP.git
    cd EGLP
    ```

2. **Install Dependencies**:
    Requires Python 3.8+ and PyTorch.

    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage Guide

The framework is driven by a single entry point `main.py` which handles configuration, training, and evaluation.

### Basic Command

```bash
python main.py --experiment <EXPERIMENT_NAME> [OPTIONS]
```

### Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--experiment` | `str` | `all` | The experiment to run. Options: `all`, `backprop`, `pure_local`, `eglp_fixed`, `eglp_triggered`, `ablation_rate`, `linear_probe`, `layer_sensitivity`, `robustness`. |
| `--epochs` | `int` | `20` | Number of training epochs. |
| `--batch_size` | `int` | `128` | Batch size for training and evaluation. |
| `--local_lr` | `float` | `0.0001` | Learning rate for local updates. |
| `--event_budget` | `int` | `1000` | Maximum number of events for the `ThresholdController`. |
| `--event_rate` | `float` | `0.05` | Probability of an event for `FixedRateController`. |
| `--seed` | `int` | `42` | Random seed for reproducibility. |
| `--output_dir` | `str` | `./results`| Directory to save logs, metrics, and plots. |
| `--plot` | `flag` | `False` | If set, generates plots automatically after experiments complete. |
| `--threshold_factor` | `float` | `2.0` | Multiplier for mean loss to set dynamic threshold. |

## 🧪 Reproducible Experiments

### 1. Main Benchmarking (Table 1 Replication)

To compare EGLP against baselines (Backprop and Pure Hebbian):

```bash
# Run standard backpropagation baseline
python main.py --experiment backprop --epochs 20

# Run pure local learning (events always on) - Lower Bound
python main.py --experiment pure_local --epochs 20

# Run EGLP with triggered events (dynamic thresholding) - Main Method
python main.py --experiment eglp_triggered --epochs 20 --event_budget 1000 --threshold_factor 2.0

```

### 2. Fair Comparison (Constrained Architecture)

To compare EGLP against Backprop in a regime where Backprop doesn't trivially solve the task (accuracy ~70-80%), use the restricted architecture with **3 hidden neurons**:

```bash
# Standard Backprop Baseline (Restricted)
python3 main.py --experiment backprop --hidden_dims "3" --epochs 20 --plot

# EGLP with Threshold Triggering (Restricted)
python3 main.py --experiment eglp_triggered --hidden_dims "3" --epochs 20 --plot
```

### 3. Ablation Study: Event Rate

Analyze the trade-off between communication cost (event rate) and accuracy. This sweeps event rates $\epsilon \in [0, 1]$.

```bash
python main.py --experiment ablation_rate --plot
```

### 3. Representation Quality: Linear Probe

Freeze the EGLP-trained backbone and train a linear classifier on top to evaluate the quality of the learned representations.

```bash
python main.py --experiment linear_probe
```

### 4. Robustness: Distribution Shift

Evaluate the model's performance on Noisy (Gaussian noise) and Rotated MNIST sets to test generalization capability.

```bash
python main.py --experiment robustness
```

### 5. Run Full Suite

To execute all experiments and generate a comprehensive report:

```bash
python main.py --experiment all --plot
```

## � Codebase Structure

```
EGLP/
├── main.py                 # Entry point for all experiments
├── eglp/
│   ├── experiment_runner.py # Core logic for running training loops and collecting metrics
│   ├── network.py           # EGLPNetwork wrapper and MLP/CNN factory functions
│   ├── local_layer.py       # Custom autograd functions for local update rules
│   ├── event_controller.py  # Logic for event triggering (Threshold, FixedRate)
│   ├── metrics.py           # Logging, CKA analysis, and FLOPs estimation
│   └── visualization.py     # Plotting utilities
```

## 📝 Citation

If you use this code in your research, please cite:
