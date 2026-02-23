# Event-Gated Local Plasticity (EGLP)

A PyTorch framework for experimenting with biologically plausible learning rules as alternatives to backpropagation.

## Architecture

EGLP employs a hybrid training strategy:

1. **Hidden Layers** — trained via local Hebbian rules (Oja's rule), gated by a global scalar signal
2. **Output Layer** — a linear readout trained via standard supervision (backprop on one layer)
3. **Event Controllers** — decide when to broadcast learning signals (Fixed Rate, Threshold-Triggered, Always-On, Never)

Advanced features include Predictive Coding layers (generative matrix G, lateral inhibition V), Direct Feedback Alignment (DFA), synaptic consolidation, homeostatic scaling, and cosine LR scheduling.

## Installation

```bash
git clone https://github.com/yourusername/EGLP.git
cd EGLP
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
source env/bin/activate

# Run backprop baseline
python3 main.py --experiment backprop --epochs 10 --hidden_dims "256,128" --plot

# Run EGLP-Triggered
python3 main.py --experiment eglp_triggered --epochs 10 --hidden_dims "256,128" --plot

# Run all experiments
python3 main.py --experiment all --epochs 10 --hidden_dims "256,128" --plot
```

## Validation Pipeline

`validate.py` implements a rigorous 6-phase verification protocol:

```bash
# Phase 1: Verify core claims (~7 min)
python3 validate.py --phase phase1 --epochs 10

# Phase 2: Depth/width scaling study (~15 min)
python3 validate.py --phase scaling --epochs 10

# Phase 3: Structural ablations (~2 min)
python3 validate.py --phase ablation --epochs 10

# Phase 4: Robustness tests (~1 min)
python3 validate.py --phase robustness --epochs 10

# Phase 5–6: Scaling laws + report (instant)
python3 validate.py --phase scaling_laws
python3 validate.py --phase report

# Run ALL phases end-to-end (~25 min)
python3 validate.py --phase all --epochs 10
```

Results are saved as JSON to `results/validation/`.

## Experimental Results

### Phase 1 — Verification (MNIST & CIFAR-10, [256,128])

| Test | MNIST | CIFAR-10 | Finding |
| --- | --- | --- | --- |
| Multi-seed stability | PL 86.2%, Trig 86.0% | — | Identical — event gating irrelevant |
| Hard zero event | 87.1% | 33.5% | Zero Event = Random Features. Error signal irrelevant |
| Unsupervised Hebbian | **11.4%** vs Random 87.3% | **14.8%** vs Random 35.3% | **Hebbian destroys initialization** |

### Phase 2 — Scaling (MNIST)

| Depth (width=256) | Accuracy | Width (depth=2) | Accuracy |
| --- | --- | --- | --- |
| 2 layers | 90.2% | 128 neurons | 84.6% |
| 4 layers | 82.7% | 256 neurons | 90.2% |
| 6 layers | 73.7% | 512 neurons | 93.4% |
| 8 layers | 64.1% | 768 neurons | 94.6% |

**Deeper = worse** (credit assignment degrades). **Wider = better** (random projection scaling law).

### Phase 3 — Ablations

| Component Removed | Accuracy | Effect |
| --- | --- | --- |
| Reference EGLP-V2 | 86.3% | — |
| Generative Matrix (G) | 87.1% | G hurts (+0.8%) |
| Lateral Inhibition (V) | 86.3% | No effect |
| Consolidation | 86.3% | No effect |
| kWTA replacement | 82.9% | Sparsity hurts (−3.4%) |

## Codebase Structure

```text
EGLP/
├── main.py                  # Entry point for individual experiments
├── validate.py              # 6-phase validation pipeline
├── requirements.txt
├── eglp/
│   ├── experiment_runner.py # Training loops and experiment configurations
│   ├── network.py           # EGLPNetwork, AdaptiveEGLPNetwork, factory functions
│   ├── local_layer.py       # LocalLinear, LocalConv2d, LocalPredictiveLayer
│   ├── event_controller.py  # Threshold, FixedRate, AlwaysOn, Never controllers
│   ├── metrics.py           # MetricsLogger, CKA, FLOPs estimation
│   └── visualization.py     # Plotting utilities
├── tests/
│   ├── test_no_grad_leak.py # Verify no gradient leakage in hidden layers
│   ├── test_metrics.py      # Test metrics computation
│   └── test_features.py     # Test feature extraction
└── results/
    ├── commands.txt          # All experiment commands
    ├── results.txt           # Latest benchmark results
    └── validation/           # Validation pipeline JSON outputs
```

## Key Findings

The validation pipeline reveals that the current EGLP framework functions as a **random feature network** with a trained linear classifier. The ~91% accuracy on MNIST is entirely explained by random projections through wide hidden layers + a supervised output layer. The Hebbian learning rule (Oja's rule) does not learn useful features — it actively degrades the random initialization.

## Tests

```bash
python3 -m pytest tests/ -v
```
