"""Metrics and logging utilities for EGLP experiments."""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    events_triggered: int = 0
    flops_estimate: float = 0.0
    cka_scores: Dict[int, float] = field(default_factory=dict)


class MetricsLogger:
    """Tracks and logs training metrics across epochs."""
    
    def __init__(self, experiment_name: str, output_dir: str = "./results"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.epochs: List[EpochMetrics] = []
        self.total_events = 0
        self.total_flops = 0.0
    
    def log_epoch(self, metrics: EpochMetrics) -> None:
        """Log metrics for an epoch."""
        self.epochs.append(metrics)
        self.total_events += metrics.events_triggered
        self.total_flops += metrics.flops_estimate
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.epochs:
            return {}
        
        return {
            "experiment_name": self.experiment_name,
            "total_epochs": len(self.epochs),
            "final_train_accuracy": self.epochs[-1].train_accuracy,
            "final_val_accuracy": self.epochs[-1].val_accuracy,
            "best_val_accuracy": max(
                (e.val_accuracy for e in self.epochs if e.val_accuracy is not None),
                default=None
            ),
            "total_events": self.total_events,
            "total_flops": self.total_flops,
            "communication_cost": self.total_events,  # Alias for clarity
        }
    
    def save(self, filename: Optional[str] = None) -> str:
        """Save metrics to JSON file."""
        if filename is None:
            filename = f"{self.experiment_name}_metrics.json"
        
        filepath = self.output_dir / filename
        
        data = {
            "summary": self.get_summary(),
            "epochs": [asdict(e) for e in self.epochs],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def get_plot_data(self) -> Dict[str, List]:
        """Get data formatted for matplotlib plotting."""
        return {
            "epochs": [e.epoch for e in self.epochs],
            "train_loss": [e.train_loss for e in self.epochs],
            "train_accuracy": [e.train_accuracy for e in self.epochs],
            "val_loss": [e.val_loss for e in self.epochs],
            "val_accuracy": [e.val_accuracy for e in self.epochs],
            "events_triggered": [e.events_triggered for e in self.epochs],
            "cumulative_events": list(
                torch.cumsum(torch.tensor([e.events_triggered for e in self.epochs]), 0).tolist()
            ),
        }


def compute_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    debiased: bool = True,
) -> float:
    """Compute Centered Kernel Alignment (CKA) between two representations.
    
    CKA measures similarity between neural network representations across
    different layers or networks. Used to compare EGLP layers with backprop layers.
    
    Args:
        X: First representation matrix (n_samples, n_features_x)
        Y: Second representation matrix (n_samples, n_features_y)
        debiased: Whether to use debiased estimator
        
    Returns:
        CKA score in [0, 1], where 1 = identical representations.
    """
    # Ensure 2D
    if X.dim() > 2:
        X = X.view(X.size(0), -1)
    if Y.dim() > 2:
        Y = Y.view(Y.size(0), -1)
    
    # Center the representations
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute Gram matrices (linear kernel)
    K = X @ X.T
    L = Y @ Y.T
    
    if debiased:
        # Debiased HSIC estimator
        n = K.size(0)
        
        # Zero out diagonal
        K_no_diag = K.clone()
        L_no_diag = L.clone()
        K_no_diag.fill_diagonal_(0)
        L_no_diag.fill_diagonal_(0)
        
        # Compute HSIC
        term1 = (K_no_diag * L_no_diag).sum()
        term2 = K_no_diag.sum() * L_no_diag.sum() / ((n - 1) * (n - 2))
        term3 = 2 * (K_no_diag.sum(dim=1) * L_no_diag.sum(dim=1)).sum() / (n - 2)
        
        hsic_kl = (term1 + term2 - term3) / (n * (n - 3))
        
        # HSIC(K, K) and HSIC(L, L) for normalization
        term1_k = (K_no_diag * K_no_diag).sum()
        term2_k = K_no_diag.sum() ** 2 / ((n - 1) * (n - 2))
        term3_k = 2 * (K_no_diag.sum(dim=1) ** 2).sum() / (n - 2)
        hsic_kk = (term1_k + term2_k - term3_k) / (n * (n - 3))
        
        term1_l = (L_no_diag * L_no_diag).sum()
        term2_l = L_no_diag.sum() ** 2 / ((n - 1) * (n - 2))
        term3_l = 2 * (L_no_diag.sum(dim=1) ** 2).sum() / (n - 2)
        hsic_ll = (term1_l + term2_l - term3_l) / (n * (n - 3))
        
        cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-10)
    else:
        # Standard CKA
        hsic_kl = (K * L).sum()
        hsic_kk = (K * K).sum()
        hsic_ll = (L * L).sum()
        cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-10)
    
    return cka.item()


def estimate_flops(
    model: nn.Module,
    input_shape: tuple,
    include_backward: bool = False,
) -> float:
    """Estimate FLOPs for a forward (and optionally backward) pass.
    
    This is a rough estimate based on layer types.
    
    Args:
        model: Network to analyze.
        input_shape: Shape of input tensor (excluding batch).
        include_backward: If True, estimate backward pass FLOPs (≈2x forward).
        
    Returns:
        Estimated FLOPs.
    """
    total_flops = 0.0
    
    for module in model.modules():
        if isinstance(module, (nn.Linear,)):
            # Linear: 2 * in_features * out_features (multiply-add)
            flops = 2 * module.in_features * module.out_features
            total_flops += flops
        
        elif isinstance(module, (nn.Conv2d,)):
            # Conv2d: 2 * k^2 * in_channels * out_channels * output_spatial
            # Simplified estimate assuming output spatial ≈ input spatial / stride
            k = module.kernel_size[0]
            flops = 2 * k * k * module.in_channels * module.out_channels
            # Multiply by approximate output size (needs actual input to be precise)
            total_flops += flops
    
    if include_backward:
        total_flops *= 3  # Backward is roughly 2x forward
    
    return total_flops


def compute_communication_cost(events_triggered: int, num_layers: int) -> float:
    """Compute communication cost as events * layers.
    
    This represents the total number of global signals broadcast.
    
    Args:
        events_triggered: Number of events from controller.
        num_layers: Number of layers receiving events.
        
    Returns:
        Communication cost metric.
    """
    return events_triggered * num_layers
