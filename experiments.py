"""Unified EGLP Experiments Framework.

This script executes all experiments (baselines, representation diagnostics,
scaling laws, structural ablations, and robustness) using a single, 
consolidated source of truth for models, data loaders, and training loops.
"""

import argparse
import time
from pathlib import Path
import json

import torch
from eglp.experiment_runner import ExperimentConfig, ExperimentRunner
from eglp.visualization import (
    plot_main_results, 
    plot_training_curves,
    plot_robustness,
    plot_ablation_results
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified EGLP Experiment Framework")
    
    parser.add_argument(
        "--group", 
        type=str, 
        required=True,
        choices=["baseline", "diagnostics", "scaling", "ablations", "robustness", "all", "report"],
        help="Experiment group to run."
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "both"],
        help="Dataset to use (mnist, cifar10, both)"
    )
    
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--local_lr", type=float, default=1e-5, help="Local learning rate")
    parser.add_argument("--event_budget", type=int, default=1000, help="Event budget")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    
    return parser.parse_args()


# ==============================================================================
# GROUP EXECUTORS
# ==============================================================================

def run_baseline(args, ds_name, base_out_dir):
    """Run core baseline comparisons."""
    print(f"\n--- Running BASELINE group on {ds_name} ---")
    
    config = ExperimentConfig(
        dataset=ds_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        local_lr=args.local_lr,
        event_budget=args.event_budget,
        seed=args.seed,
        output_dir=str(Path(base_out_dir) / "baseline")
    )
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    runner = ExperimentRunner(config)
    results = {}
    
    # 1. Backprop Upper Bound
    results["backprop"] = runner.run_backprop_baseline()
    
    # 2. Pure Local (Hebbian only)
    results["pure_local"] = runner.run_pure_local()
    
    # 3. EGLP Fixed Rate
    results["eglp_fixed"] = runner.run_eglp_fixed()
    
    # 4. EGLP Triggered (Threshold Controller)
    results["eglp_triggered"] = runner.run_eglp_triggered()
    
    # 5. EGLP Baseline (V1)
    results["eglp_baseline"] = runner.run_eglp()
    
    # 6. EGLP V2
    results["eglp_v2"] = runner.run_eglp_v2()
    
    # 7. Linear Probe
    results["linear_probe"] = runner.run_linear_probe()
    
    # Generate Baseline Plots
    print("\nGenerating baseline plots...")
    plot_main_results(results, output_dir=config.output_dir, save_name="baseline_summary.png")
    plot_training_curves(results, output_dir=config.output_dir, save_name="baseline_curves.png")
    
    return results


def run_diagnostics(args, ds_name, base_out_dir):
    """Run representation diagnostics (Zero event, Freeze retrains, Unsupervised)."""
    print(f"\n--- Running DIAGNOSTICS group on {ds_name} ---")
    out_dir = str(Path(base_out_dir) / "diagnostics")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Multi-seed stability
    print("\n[Diagnostic 1/4] Multi-Seed Stability")
    stability_accs = {"pure_local": [], "eglp_triggered": []}
    seeds = [42, 123, 456, 789, 1024]
    
    for seed in seeds:
        print(f"\n  Testing Seed: {seed}")
        config = ExperimentConfig(
            dataset=ds_name, epochs=args.epochs, batch_size=args.batch_size,
            local_lr=args.local_lr, event_budget=args.event_budget,
            seed=seed, output_dir=str(Path(out_dir) / f"seed_{seed}")
        )
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        runner = ExperimentRunner(config)
        
        pl_log = runner.run_pure_local()
        eglp_log = runner.run_eglp_triggered()
        
        stability_accs["pure_local"].append(pl_log.test_metrics["accuracy"])
        stability_accs["eglp_triggered"].append(eglp_log.test_metrics["accuracy"])
    
    # Compute stability stats
    for key, accs in stability_accs.items():
        mean = sum(accs) / len(accs)
        std = (sum((x - mean) ** 2 for x in accs) / len(accs)) ** 0.5
        print(f"  {key} Stability: {mean:.4f} ± {std:.4f}")
        results[f"stability_{key}"] = {"mean": mean, "std": std, "runs": accs}

    
    # Reset config back to main seed for rest of tests
    config = ExperimentConfig(
        dataset=ds_name, epochs=args.epochs, batch_size=args.batch_size,
        local_lr=args.local_lr, event_budget=args.event_budget,
        seed=args.seed, output_dir=out_dir
    )
    runner = ExperimentRunner(config)
    
    # 2. Hard Zero Event
    print("\n[Diagnostic 2/4] Hard Zero Event Test")
    zero_log = runner.run_zero_event()
    results["zero_event"] = zero_log.test_metrics["accuracy"]
    
    # 3. Freeze Representation Retrains
    print("\n[Diagnostic 3/4] Freeze Representation Retrains")
    freeze_res = runner.run_freeze_representation(num_retrains=5)
    results["freeze_representation"] = freeze_res
    
    # 4. Fully Unsupervised Pretraining
    print("\n[Diagnostic 4/4] Fully Unsupervised Pretraining")
    unsup_log = runner.run_unsupervised_pretraining()
    results["unsupervised"] = unsup_log.test_metrics["accuracy"]
    
    # Save diagnostic summary
    summary_path = Path(out_dir) / "diagnostics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDiagnostics summary saved to: {summary_path}")
    
    return results


def run_scaling(args, ds_name, base_out_dir):
    """Run depth and width scaling sweeps."""
    print(f"\n--- Running SCALING group on {ds_name} ---")
    out_dir = str(Path(base_out_dir) / "scaling")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = {"depth_scaling": {}, "width_scaling": {}, "strategic_combos": {}}
    
    # 1. Depth scaling
    print("\n[Scaling 1/3] Depth Sweep")
    depths = [2, 4, 6, 8]
    width = 256
    best_depth, best_acc = 2, 0.0
    
    for d in depths:
        h_dims = [width] * d
        print(f"\n  Testing Depth {d} (Dims: {h_dims})")
        config = ExperimentConfig(
            dataset=ds_name, epochs=args.epochs, batch_size=args.batch_size,
            local_lr=args.local_lr, event_budget=args.event_budget,
            hidden_dims=h_dims, seed=args.seed, output_dir=str(Path(out_dir) / f"depth_{d}")
        )
        runner = ExperimentRunner(config)
        log = runner.run_eglp_triggered()
        
        acc = log.test_metrics["accuracy"]
        results["depth_scaling"][d] = acc
        if acc > best_acc:
            best_depth, best_acc = d, acc
            
    # 2. Width scaling
    print(f"\n[Scaling 2/3] Width Sweep (Fixed Depth={best_depth})")
    widths = [128, 256, 512, 768]
    for w in widths:
        h_dims = [w] * best_depth
        print(f"\n  Testing Width {w} (Dims: {h_dims})")
        config = ExperimentConfig(
            dataset=ds_name, epochs=args.epochs, batch_size=args.batch_size,
            local_lr=args.local_lr, event_budget=args.event_budget,
            hidden_dims=h_dims, seed=args.seed, output_dir=str(Path(out_dir) / f"width_{w}")
        )
        runner = ExperimentRunner(config)
        log = runner.run_eglp_triggered()
        results["width_scaling"][w] = log.test_metrics["accuracy"]
        
    # 3. Strategic combos
    print("\n[Scaling 3/3] Strategic Combinations")
    combos = [(2, 128), (4, 256), (6, 512), (8, 512)]
    for (d, w) in combos:
        h_dims = [w] * d
        print(f"\n  Testing Combo: Depth {d}, Width {w} (Dims: {h_dims})")
        config = ExperimentConfig(
            dataset=ds_name, epochs=args.epochs, batch_size=args.batch_size,
            local_lr=args.local_lr, event_budget=args.event_budget,
            hidden_dims=h_dims, seed=args.seed, output_dir=str(Path(out_dir) / f"combo_{d}x{w}")
        )
        runner = ExperimentRunner(config)
        log = runner.run_eglp_triggered()
        results["strategic_combos"][f"{d}x{w}"] = log.test_metrics["accuracy"]
        
    summary_path = Path(out_dir) / "scaling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_ablations(args, ds_name, base_out_dir):
    """Systematically ablate structural components."""
    print(f"\n--- Running ABLATIONS group on {ds_name} ---")
    out_dir = str(Path(base_out_dir) / "ablations")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    
    ablations = [
        # (Name, kwargs, runner_method)
        ("Base (EGLP v2)", {"use_consolidation": True}, "run_eglp_v2"),
        ("Remove G (Basic EGLP)", {}, "run_eglp_triggered"),
        ("Remove V (Inhib=0)", {"use_consolidation": True, "inhib_lr": 0.0, "inhib_steps": 0}, "run_eglp_v2"),
        ("Replace V with kWTA (SoftWTA)", {"soft_wta_k": 0.3}, "run_eglp_triggered"),
        ("Remove Consolidation", {"use_consolidation": False}, "run_eglp_v2"),
    ]
    
    for name, kwargs, method_name in ablations:
        print(f"\n[Ablation] Testing: {name}")
        
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        config_kwargs = {
            "dataset": ds_name, "epochs": args.epochs, "batch_size": args.batch_size,
            "local_lr": args.local_lr, "event_budget": args.event_budget,
            "seed": args.seed, "output_dir": str(Path(out_dir) / safe_name)
        }
        config_kwargs.update(kwargs)
        
        config = ExperimentConfig(**config_kwargs)
        runner = ExperimentRunner(config)
        
        method = getattr(runner, method_name)
        log = method()
        results[safe_name] = log.test_metrics["accuracy"]
        print(f"  → Accuracy: {results[safe_name]:.4f}")
        
    summary_path = Path(out_dir) / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nGenerating ablation plots...")
    
    # Needs to be a bit custom for ablations because plot_ablation_results expects a rate dictionary, 
    # but we will just provide the simple bar chart.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = list(results.keys())
    accuracies = list(results.values())
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f']
    bars = ax.bar(configs, accuracies, color=colors[:len(configs)], alpha=0.8)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel("Ablation Configuration", fontsize=12)
    ax.set_ylabel("Final Validation Accuracy", fontsize=12)
    ax.set_title("Component Ablation Analysis", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    output_path = Path(out_dir) / "ablation_summary.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Ablation plot saved to {output_path}")
        
    return results


def run_robustness(args, ds_name, base_out_dir):
    """Test model robustness against perturbations."""
    print(f"\n--- Running ROBUSTNESS group on {ds_name} ---")
    out_dir = str(Path(base_out_dir) / "robustness")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Train base model
    config = ExperimentConfig(
        dataset=ds_name, epochs=args.epochs, batch_size=args.batch_size,
        local_lr=args.local_lr, event_budget=args.event_budget,
        seed=args.seed, output_dir=out_dir
    )
    runner = ExperimentRunner(config)
    
    # run_robustness_test is already built to test perturbations after training
    res = runner.run_robustness_test()
    
    # Extract just the accuracies for simple JSON tracking
    simple_res = {k: v.test_metrics["accuracy"] for k, v in res.items()}
    summary_path = Path(out_dir) / "robustness_summary.json"
    with open(summary_path, "w") as f:
        json.dump(simple_res, f, indent=2)
        
    print("\nGenerating robustness plots...")
    plot_robustness(res, output_dir=out_dir, save_name="robustness_chart.png")
        
    return simple_res


def run_report(args, ds_name, base_out_dir):
    """Aggregate results into a Markdown report."""
    print(f"\n--- Running REPORT aggregation on {ds_name} ---")
    
    # Try loading all summary JSONs
    data = {}
    for group in ["diagnostics", "scaling", "ablations", "robustness"]:
        summary_path = Path(base_out_dir) / group / f"{group}_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                data[group] = json.load(f)
                
    if not data:
        print("No group summary files found! Run other groups first.")
        return
        
    report_lines = [
        f"# Unified EGLP Framework Report: {ds_name.upper()}",
        f"\n*Auto-generated based on results in `{base_out_dir}`*\n",
    ]
    
    # Analyze Diagnostics
    if "diagnostics" in data:
        diag = data["diagnostics"]
        report_lines.append("## Representation Diagnostics")
        report_lines.append("")
        
        # Unsupervised test is the key
        unsup = diag.get("unsupervised", 0.0)
        report_lines.append(f"- **Zero Event Accuracy**: {diag.get('zero_event', 0.0):.4f}")
        report_lines.append(f"- **Freeze Representation Mean**: {diag.get('freeze_representation', {}).get('mean', 0.0):.4f}")
        report_lines.append(f"- **Fully Unsupervised + Linear Probe**: {unsup:.4f}")
        report_lines.append("")
        
        if unsup > 0.8:
            report_lines.append("**Conclusion**: Hebbian learning is effectively forming self-organized representations.")
        else:
            report_lines.append("**Conclusion**: The network is acting as a Random Feature Network. Unsupervised Hebbian features do not linearly separate classes.")
        report_lines.append("")

    # Output to screen and file
    report = "\n".join(report_lines)
    print(report)
    
    report_dir = Path(base_out_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "analysis_report.md"
    
    with open(report_path, "w") as f:
        f.write(report)
        
    print(f"\nReport written to {report_path}")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main():
    args = parse_args()
    datasets_to_run = ["mnist", "cifar10"] if args.dataset == "both" else [args.dataset]
    
    start_time = time.time()
    
    for ds in datasets_to_run:
        print(f"\n{'='*72}")
        print(f"EXECUTING FRAMEWORK ON: {ds.upper()}")
        print(f"{'='*72}\n")
        
        base_out_dir = str(Path(args.output_dir) / ds)
        
        if args.group in ["baseline", "all"]:
            run_baseline(args, ds, base_out_dir)
            
        if args.group in ["diagnostics", "all"]:
            run_diagnostics(args, ds, base_out_dir)
            
        if args.group in ["scaling", "all"]:
            run_scaling(args, ds, base_out_dir)
            
        if args.group in ["ablations", "all"]:
            run_ablations(args, ds, base_out_dir)
            
        if args.group in ["robustness", "all"]:
            run_robustness(args, ds, base_out_dir)
            
        if args.group in ["report", "all"]:
            run_report(args, ds, base_out_dir)
            
    elapsed = time.time() - start_time
    print(f"\n{'='*72}")
    print(f"Framework execution complete in {elapsed:.1f}s")
    print(f"{'='*72}")

if __name__ == "__main__":
    main()
