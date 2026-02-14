"""Visualization utilities for EGLP experiments."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot_main_results(
    results: Dict[str, Any],
    output_dir: str = "./results",
    save_name: str = "main_results.png",
) -> None:
    """Generate the main results plot comparing all methods.
    
    Creates a 2x2 grid:
    - Top-left: Accuracy vs. Epoch
    - Top-right: Accuracy vs. Communication Cost
    - Bottom-left: CKA similarity with backprop
    - Bottom-right: Training curve comparison
    """
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = {
        "backprop": "#2ecc71",      # Green
        "pure_local": "#e74c3c",    # Red
        "eglp_fixed": "#3498db",    # Blue
        "eglp_triggered": "#9b59b6", # Purple
    }
    
    labels = {
        "backprop": "Backprop (Upper Bound)",
        "pure_local": "Pure Local (Hebbian)",
        "eglp_fixed": "EGLP-Fixed",
        "eglp_triggered": "EGLP-Triggered",
    }
    
    # ========================
    # Top-left: Accuracy vs Epoch
    # ========================
    ax1 = fig.add_subplot(gs[0, 0])
    
    for name, logger in results.items():
        if hasattr(logger, 'get_plot_data'):
            data = logger.get_plot_data()
            epochs_1indexed = [e + 1 for e in data["epochs"]]
            if name in colors:
                ax1.plot(
                    epochs_1indexed, 
                    data["val_accuracy"],
                    color=colors[name],
                    label=labels[name],
                    linewidth=2,
                    marker='o',
                    markersize=4,
                )
    
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Validation Accuracy", fontsize=12)
    ax1.set_title("Learning Curves", fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # ========================
    # Top-right: Accuracy vs Communication Cost
    # ========================
    ax2 = fig.add_subplot(gs[0, 1])
    
    for name, logger in results.items():
        if hasattr(logger, 'get_summary'):
            summary = logger.get_summary()
            if name in colors and name != "backprop":  # Backprop has no events
                ax2.scatter(
                    summary.get("total_events", 0),
                    summary.get("final_val_accuracy", 0),
                    color=colors[name],
                    label=labels[name],
                    s=150,
                    marker='o',
                    edgecolors='black',
                    linewidth=1.5,
                )
    
    # Add backprop as horizontal line
    if "backprop" in results:
        backprop_acc = results["backprop"].get_summary().get("final_val_accuracy", 0)
        ax2.axhline(y=backprop_acc, color=colors["backprop"], 
                   linestyle='--', label=labels["backprop"], linewidth=2)
    
    ax2.set_xlabel("Total Events (Communication Cost)", fontsize=12)
    ax2.set_ylabel("Final Validation Accuracy", fontsize=12)
    ax2.set_title("Accuracy vs. Communication", fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # ========================
    # Bottom-left: CKA Similarity
    # ========================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Get CKA scores from final epoch
    cka_data = {}
    for name, logger in results.items():
        if hasattr(logger, 'epochs') and logger.epochs:
            final_epoch = logger.epochs[-1]
            if hasattr(final_epoch, 'cka_scores') and final_epoch.cka_scores:
                cka_data[name] = final_epoch.cka_scores
    
    if cka_data:
        x_positions = np.arange(len(list(cka_data.values())[0]))
        width = 0.2
        
        for i, (name, scores) in enumerate(cka_data.items()):
            if name in colors:
                layer_scores = [scores.get(j, 0) for j in sorted(scores.keys())]
                ax3.bar(
                    x_positions + i * width,
                    layer_scores,
                    width,
                    color=colors[name],
                    label=labels[name],
                    alpha=0.8,
                )
        
        ax3.set_xlabel("Layer Index", fontsize=12)
        ax3.set_ylabel("CKA Similarity with Backprop", fontsize=12)
        ax3.set_title("Representation Similarity", fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.set_xticks(x_positions + width)
        ax3.set_xticklabels([f"Layer {i}" for i in range(len(x_positions))])
    else:
        ax3.text(0.5, 0.5, "CKA data not available\n(run backprop baseline first)",
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    
    ax3.grid(True, alpha=0.3)
    
    # ========================
    # Bottom-right: Event Distribution
    # ========================
    ax4 = fig.add_subplot(gs[1, 1])
    
    for name, logger in results.items():
        if hasattr(logger, 'get_plot_data') and name in ["eglp_fixed", "eglp_triggered"]:
            data = logger.get_plot_data()
            epochs_1indexed = [e + 1 for e in data["epochs"]]
            ax4.plot(
                epochs_1indexed,
                data["events_triggered"],
                color=colors[name],
                label=labels[name],
                linewidth=2,
                marker='s',
                markersize=4,
            )
    
    ax4.set_xlabel("Epoch", fontsize=12)
    ax4.set_ylabel("Events Triggered", fontsize=12)
    ax4.set_title("Event Distribution Over Training", fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save
    output_path = Path(output_dir) / save_name
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Main results plot saved to {output_path}")


def plot_ablation_results(
    results: Dict[float, Any],
    output_dir: str = "./results",
    save_name: str = "ablation_rate.png",
) -> None:
    """Plot ablation study: accuracy vs. event rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    rates = sorted(results.keys())
    accuracies = []
    events = []
    
    for rate in rates:
        logger = results[rate]
        summary = logger.get_summary()
        accuracies.append(summary.get("final_val_accuracy", 0))
        events.append(summary.get("total_events", 0))
    
    # Accuracy vs Rate
    ax1.plot(rates, accuracies, 'o-', color='#3498db', linewidth=2, markersize=8)
    ax1.fill_between(rates, accuracies, alpha=0.2, color='#3498db')
    ax1.set_xlabel("Event Rate (ε)", fontsize=12)
    ax1.set_ylabel("Final Validation Accuracy", fontsize=12)
    ax1.set_title("Accuracy vs. Event Rate", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Events vs Rate
    ax2.plot(rates, events, 's-', color='#e74c3c', linewidth=2, markersize=8)
    ax2.fill_between(rates, events, alpha=0.2, color='#e74c3c')
    ax2.set_xlabel("Event Rate (ε)", fontsize=12)
    ax2.set_ylabel("Total Events", fontsize=12)
    ax2.set_title("Communication Cost vs. Event Rate", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / save_name
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Ablation plot saved to {output_path}")


def plot_layer_sensitivity(
    results: Dict[str, Any],
    output_dir: str = "./results",
    save_name: str = "layer_sensitivity.png",
) -> None:
    """Plot layer sensitivity analysis results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = list(results.keys())
    accuracies = []
    
    for config in configs:
        logger = results[config]
        summary = logger.get_summary()
        accuracies.append(summary.get("final_val_accuracy", 0))
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    bars = ax.bar(configs, accuracies, color=colors[:len(configs)], alpha=0.8)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Final Validation Accuracy", fontsize=12)
    ax.set_title("Layer Sensitivity Analysis", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    output_path = Path(output_dir) / save_name
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Layer sensitivity plot saved to {output_path}")


def plot_robustness(
    results: Dict[str, Any],
    output_dir: str = "./results",
    save_name: str = "robustness.png",
) -> None:
    """Plot robustness test results."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    tests = ["clean", "noisy", "rotated"]
    accuracies = []
    
    for test in tests:
        if test in results:
            logger = results[test]
            summary = logger.get_summary()
            accuracies.append(summary.get("final_val_accuracy", 0))
        else:
            accuracies.append(0)
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(tests, accuracies, color=colors, alpha=0.8)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel("Test Condition", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Robustness to Distribution Shift", fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    output_path = Path(output_dir) / save_name
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Robustness plot saved to {output_path}")


def plot_training_curves(
    results: Dict[str, Any],
    output_dir: str = "./results",
    save_name: str = "training_curves.png",
) -> None:
    """Generic function to plot training curves for any set of experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color scheme generator
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for i, (name, logger) in enumerate(results.items()):
        if hasattr(logger, 'get_plot_data'):
            data = logger.get_plot_data()
            epochs_1indexed = [e + 1 for e in data["epochs"]]
            color = colors[i % len(colors)]
            
            # Accuracy
            ax1.plot(
                epochs_1indexed, 
                data["val_accuracy"],
                label=name,
                linewidth=2,
                marker='o',
                markersize=4,
                color=color
            )
            
            # Loss
            ax2.plot(
                epochs_1indexed,
                data["train_loss"], # Using train loss as val loss might be infrequent/noisy
                label=name,
                linewidth=2,
                linestyle='--',
                color=color
            )

    # Styling
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Validation Accuracy", fontsize=12)
    ax1.set_title("Validation Accuracy", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Training Loss", fontsize=12)
    ax2.set_title("Training Loss", fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    output_path = Path(output_dir) / save_name
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training curves saved to {output_path}")
