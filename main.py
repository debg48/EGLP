"""Main entry point for EGLP experiments."""

import argparse
from pathlib import Path

from eglp import ExperimentRunner
from eglp.experiment_runner import ExperimentConfig
from eglp.visualization import plot_main_results, plot_ablation_results


def parse_args():
    parser = argparse.ArgumentParser(description="EGLP Experiment Runner")
    
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="all",
        choices=["all", "backprop", "pure_local", "eglp_fixed", "eglp_triggered",
                 "ablation_rate", "linear_probe", "layer_sensitivity", "robustness"],
        help="Which experiment to run"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--local_lr", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--event_budget", type=int, default=1000, help="Event budget")
    parser.add_argument("--event_rate", type=float, default=0.1, help="Fixed event rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Generate plots after experiments")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config
    config = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        local_lr=args.local_lr,
        event_budget=args.event_budget,
        event_rate=args.event_rate,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize runner
    runner = ExperimentRunner(config)
    
    # Run experiments
    results = {}
    
    if args.experiment == "all":
        results = runner.run_all_experiments()
        
        # Also run additional experiments
        print("\n" + "="*60)
        print("Running additional experiment tasks...")
        print("="*60)
        
        results["ablation_rate"] = runner.run_ablation_rate()
        results["linear_probe"] = runner.run_linear_probe()
        results["layer_sensitivity"] = runner.run_layer_sensitivity()
        results["robustness"] = runner.run_robustness_test()
        
    elif args.experiment == "backprop":
        results["backprop"] = runner.run_backprop_baseline()
        
    elif args.experiment == "pure_local":
        results["pure_local"] = runner.run_pure_local()
        
    elif args.experiment == "eglp_fixed":
        results["eglp_fixed"] = runner.run_eglp_fixed()
        
    elif args.experiment == "eglp_triggered":
        results["eglp_triggered"] = runner.run_eglp_triggered()
        
    elif args.experiment == "ablation_rate":
        results["ablation_rate"] = runner.run_ablation_rate()
        
    elif args.experiment == "linear_probe":
        results["linear_probe"] = runner.run_linear_probe()
        
    elif args.experiment == "layer_sensitivity":
        results["layer_sensitivity"] = runner.run_layer_sensitivity()
        
    elif args.experiment == "robustness":
        results["robustness"] = runner.run_robustness_test()
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        try:
            if "backprop" in results or "eglp_triggered" in results:
                plot_main_results(results, args.output_dir)
            if "ablation_rate" in results:
                plot_ablation_results(results["ablation_rate"], args.output_dir)
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    print("\n" + "="*60)
    print("Experiments complete! Results saved to:", args.output_dir)
    print("="*60)


if __name__ == "__main__":
    main()
