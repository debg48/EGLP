"""Event-Gated Local Plasticity (EGLP) Framework."""

from .local_layer import LocalLinear, LocalConv2d
from .event_controller import ThresholdController, FixedRateController
from .network import EGLPNetwork
from .experiment_runner import ExperimentRunner
from .metrics import MetricsLogger, compute_cka, estimate_flops
from .visualization import plot_main_results, plot_ablation_results

__all__ = [
    "LocalLinear",
    "LocalConv2d", 
    "ThresholdController",
    "FixedRateController",
    "EGLPNetwork",
    "ExperimentRunner",
    "MetricsLogger",
    "compute_cka",
    "estimate_flops",
]
