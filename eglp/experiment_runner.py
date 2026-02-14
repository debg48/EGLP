"""Experiment runner for EGLP framework.

Implements the full experimental pipeline including:
- Baseline: Standard Backprop (upper bound)
- Baseline: Pure Local (Hebbian only)
- EGLP-Fixed: Events at fixed rate
- EGLP-Triggered: Events via controller
"""

import random
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from .local_layer import LocalLinear
from .network import EGLPNetwork, create_mlp, create_cnn
from .event_controller import (
    EventController, 
    ThresholdController, 
    FixedRateController, 
    AlwaysOnController,
    NeverController
)
from .metrics import (
    MetricsLogger, 
    EpochMetrics, 
    compute_cka, 
    estimate_flops,
    compute_classification_metrics
)


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    # Data
    dataset: str = "mnist"
    batch_size: int = 128
    
    # Network
    input_dim: int = 784  # 28x28 flattened
    hidden_dims: List[int] = None
    output_dim: int = 10
    
    # Training
    epochs: int = 20
    local_lr: float = 0.0001
    backprop_lr: float = 0.001
    
    # EGLP
    event_budget: int = 1000
    event_rate: float = 0.05
    threshold_factor: float = 2.0
    
    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./results"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class ExperimentRunner:
    """Runs EGLP experiments with various configurations."""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self._set_seed(self.config.seed)
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = self._load_data()
        
        # Storage for backprop representations (for CKA)
        self.backprop_representations: Dict[int, torch.Tensor] = {}
    
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load MNIST dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        
        # Split train into train/val
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def _create_backprop_network(self) -> nn.Module:
        """Create standard backprop network for baseline."""
        layers = []
        prev_dim = self.config.input_dim
        
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.config.output_dim))
        
        return nn.Sequential(*layers).to(self.config.device)
    
    def _create_eglp_network(self) -> EGLPNetwork:
        """Create EGLP network with local layers."""
        network = create_mlp(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            lr=self.config.local_lr,
        )
        return network.to(self.config.device)
    
    def _evaluate(
        self, 
        model: nn.Module, 
        loader: DataLoader,
        is_eglp: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate model accuracy and loss."""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                output = model(data)
                
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                
                _, predicted = output.max(1)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_targets.extend(target.cpu().numpy().tolist())
        
        metrics = compute_classification_metrics(all_targets, all_preds)
        
        return total_loss / len(loader.dataset), metrics
    
    # ========================
    # BASELINE: Standard Backprop
    # ========================
    
    def run_backprop_baseline(self) -> MetricsLogger:
        """Run standard backprop training (upper bound baseline)."""
        print("\n" + "="*60)
        print("BASELINE: Standard Backprop")
        print("="*60)
        
        self._set_seed(self.config.seed)
        model = self._create_backprop_network()
        optimizer = optim.Adam(model.parameters(), lr=self.config.backprop_lr)
        criterion = nn.CrossEntropyLoss()
        logger = MetricsLogger("backprop_baseline", self.config.output_dir)
        
        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for data, target in self.train_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
                total += target.size(0)
            
            train_loss = epoch_loss / total
            train_acc = correct / total
            val_loss, val_metrics = self._evaluate(model, self.val_loader)
            
            # Estimate FLOPs (include backward for backprop)
            flops = estimate_flops(model, (self.config.input_dim,), include_backward=True)
            flops *= len(self.train_loader)
            
            logger.log_epoch(EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_metrics["accuracy"],
                val_precision=val_metrics["precision"],
                val_recall=val_metrics["recall"],
                val_f1=val_metrics["f1"],
                events_triggered=0,  # N/A for backprop
                flops_estimate=flops,
            ))
            
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Store representations for CKA comparison
        self._store_backprop_representations(model)
        
        logger.save()
        return logger
    
    def _store_backprop_representations(self, model: nn.Module) -> None:
        """Store layer activations from backprop model for CKA."""
        model.eval()
        
        # Get a batch for representation computation
        data, _ = next(iter(self.val_loader))
        data = data.to(self.config.device)
        
        # Hook to capture activations
        activations = {}
        hooks = []
        layer_idx = 0
        
        def make_hook(idx):
            def hook(module, input, output):
                activations[idx] = output.detach()
            return hook
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(layer_idx)))
                layer_idx += 1
        
        with torch.no_grad():
            _ = model(data)
        
        for hook in hooks:
            hook.remove()
        
        self.backprop_representations = activations
    
    # ========================
    # BASELINE: Pure Local (Hebbian)
    # ========================
    
    def run_pure_local(self) -> MetricsLogger:
        """Run pure local learning (Hebbian only, events always on)."""
        print("\n" + "="*60)
        print("BASELINE: Pure Local (Hebbian)")
        print("="*60)
        
        self._set_seed(self.config.seed)
        model = self._create_eglp_network()
        controller = AlwaysOnController()
        logger = MetricsLogger("pure_local", self.config.output_dir)
        
        return self._run_eglp_training(model, controller, logger, "Pure Local")
    
    # ========================
    # EGLP-Fixed: Fixed rate events
    # ========================
    
    def run_eglp_fixed(self, rate: Optional[float] = None) -> MetricsLogger:
        """Run EGLP with fixed event rate."""
        rate = rate or self.config.event_rate
        
        print("\n" + "="*60)
        print(f"EGLP-Fixed (rate={rate})")
        print("="*60)
        
        self._set_seed(self.config.seed)
        model = self._create_eglp_network()
        controller = FixedRateController(rate=rate, seed=self.config.seed)
        logger = MetricsLogger(f"eglp_fixed_{rate}", self.config.output_dir)
        
        return self._run_eglp_training(model, controller, logger, f"EGLP-Fixed({rate})")
    
    # ========================
    # EGLP-Triggered: Controller-based events
    # ========================
    
    def run_eglp_triggered(self) -> MetricsLogger:
        """Run EGLP with threshold-based event triggering."""
        print("\n" + "="*60)
        print("EGLP-Triggered (ThresholdController)")
        print("="*60)
        
        self._set_seed(self.config.seed)
        model = self._create_eglp_network()
        controller = ThresholdController(
            budget=self.config.event_budget,
            threshold_factor=self.config.threshold_factor,
        )
        logger = MetricsLogger("eglp_triggered", self.config.output_dir)
        
        return self._run_eglp_training(model, controller, logger, "EGLP-Triggered")
    
    def _run_eglp_training(
        self,
        model: EGLPNetwork,
        controller: EventController,
        logger: MetricsLogger,
        name: str,
    ) -> MetricsLogger:
        """Core EGLP hybrid training loop.
        
        Hidden layers: local Hebbian updates modulated by error signal
        Output layer: supervised backprop (only 1 layer, minimal communication)
        """
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer for output layer ONLY
        optimizer = optim.Adam(
            model.get_output_params(), 
            lr=self.config.backprop_lr,
        )
        
        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            epoch_events = 0
            
            for data, target in self.train_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                # Forward pass: hidden layers (no grad) + output layer (grad)
                optimizer.zero_grad()
                output = model(data, store_representations=(epoch == self.config.epochs - 1))
                
                # Compute loss with gradients for output layer
                loss = criterion(output, target)
                
                # Backprop ONLY the output layer
                loss.backward()
                optimizer.step()
                
                # Controller decides event signal (continuous float)
                error_signal = controller.should_trigger(loss.item())
                epoch_events += (1 if error_signal > 0 else 0)
                
                # Local update for hidden layers with error modulation
                model.local_update_all(error_signal)
                model.clear_activations()
                
                # Metrics
                epoch_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
                total += target.size(0)
            
            train_loss = epoch_loss / total
            train_acc = correct / total
            val_loss, val_metrics = self._evaluate(model, self.val_loader, is_eglp=True)
            
            # Compute CKA with backprop representations
            cka_scores = {}
            if self.backprop_representations and epoch == self.config.epochs - 1:
                eglp_reps = model.get_representations()
                for layer_idx in eglp_reps:
                    if layer_idx in self.backprop_representations:
                        cka = compute_cka(
                            eglp_reps[layer_idx],
                            self.backprop_representations[layer_idx]
                        )
                        cka_scores[layer_idx] = cka
            
            # FLOPs (forward only for hidden layers + forward+backward for output)
            flops = estimate_flops(model, (self.config.input_dim,), include_backward=False)
            flops *= len(self.train_loader)
            
            logger.log_epoch(EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_metrics["accuracy"],
                val_precision=val_metrics["precision"],
                val_recall=val_metrics["recall"],
                val_f1=val_metrics["f1"],
                events_triggered=epoch_events,
                flops_estimate=flops,
                cka_scores=cka_scores,
            ))
            
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Events: {epoch_events}")
        
        logger.save()
        return logger
    
    # ========================
    # EXPERIMENT TASKS
    # ========================
    
    def run_ablation_rate(
        self, 
        rates: Optional[List[float]] = None,
    ) -> Dict[float, MetricsLogger]:
        """Sweep event rate ε ∈ [0, 1] and record accuracy vs. communication cost.
        
        Args:
            rates: List of rates to test. Default: [0, 0.1, 0.2, ..., 1.0]
            
        Returns:
            Dict mapping rate to MetricsLogger
        """
        print("\n" + "="*60)
        print("ABLATION: Event Rate Sweep")
        print("="*60)
        
        if rates is None:
            rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        results = {}
        for rate in rates:
            print(f"\nTesting rate = {rate}")
            self._set_seed(self.config.seed)  # Reset seed for fair comparison
            
            model = self._create_eglp_network()
            controller = FixedRateController(rate=rate, seed=self.config.seed)
            logger = MetricsLogger(f"ablation_rate_{rate}", self.config.output_dir)
            
            self._run_eglp_training(model, controller, logger, f"Rate={rate}")
            results[rate] = logger
        
        # Save summary
        self._save_ablation_summary(results, "ablation_rate")
        
        return results
    
    def run_linear_probe(self, backbone_logger: Optional[MetricsLogger] = None) -> MetricsLogger:
        """Freeze EGLP-trained backbone and train a linear head.
        
        Tests representation quality by training only the final classification layer.
        
        Returns:
            MetricsLogger with linear probe results
        """
        print("\n" + "="*60)
        print("LINEAR PROBE: Testing Representation Quality")
        print("="*60)
        
        # First train EGLP backbone
        self._set_seed(self.config.seed)
        backbone = self._create_eglp_network()
        controller = ThresholdController(
            budget=self.config.event_budget,
            threshold_factor=self.config.threshold_factor,
        )
        
        # Optimizer for output layer during backbone training
        backbone_optimizer = optim.Adam(
            backbone.get_output_params(), lr=self.config.backprop_lr
        )
        criterion = nn.CrossEntropyLoss()
        
        print("Training EGLP backbone...")
        for epoch in range(self.config.epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                backbone_optimizer.zero_grad()
                output = backbone(data)
                loss = criterion(output, target)
                loss.backward()
                backbone_optimizer.step()
                
                error_signal = controller.should_trigger(loss.item())
                backbone.local_update_all(error_signal)
                backbone.clear_activations()
        
        # Extract features from frozen backbone (hidden layers only)
        print("\nExtracting features from frozen backbone...")
        
        def extract_features(loader):
            features = []
            labels = []
            
            with torch.no_grad():
                for data, target in loader:
                    data = data.to(self.config.device)
                    
                    # Forward through hidden layers only
                    x = data
                    for layer in backbone.layers:
                        x = layer(x)
                        x = nn.ReLU()(x)
                    
                    features.append(x.cpu())
                    labels.append(target)
            
            return torch.cat(features), torch.cat(labels)
        
        train_features, train_labels = extract_features(self.train_loader)
        val_features, val_labels = extract_features(self.val_loader)
        
        # Train linear probe
        print("Training linear probe...")
        feature_dim = train_features.size(1)
        linear_probe = nn.Linear(feature_dim, self.config.output_dim).to(self.config.device)
        optimizer = optim.Adam(linear_probe.parameters(), lr=0.01)
        
        logger = MetricsLogger("linear_probe", self.config.output_dir)
        
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        probe_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        for epoch in range(10):  # Fewer epochs for linear probe
            linear_probe.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for features, targets in probe_loader:
                features = features.to(self.config.device)
                targets = targets.to(self.config.device)
                
                optimizer.zero_grad()
                output = linear_probe(features)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * features.size(0)
                _, predicted = output.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
            
            train_loss = epoch_loss / total
            train_acc = correct / total
            
            # Validation
            linear_probe.eval()
            val_features_dev = val_features.to(self.config.device)
            val_labels_dev = val_labels.to(self.config.device)
            
            with torch.no_grad():
                val_output = linear_probe(val_features_dev)
                val_loss = criterion(val_output, val_labels_dev).item()
                _, predicted = val_output.max(1)
                val_acc = predicted.eq(val_labels_dev).float().mean().item()
            
            logger.log_epoch(EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
            ))
            
            print(f"Probe Epoch {epoch+1}/10 | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        logger.save()
        return logger
    
    def run_layer_sensitivity(self) -> Dict[str, MetricsLogger]:
        """Enable/disable events selectively for early vs. late layers.
        
        Tests which layers benefit most from global event signals.
        
        Returns:
            Dict mapping configuration name to MetricsLogger
        """
        print("\n" + "="*60)
        print("LAYER SENSITIVITY: Selective Event Gating")
        print("="*60)
        
        num_layers = len(self.config.hidden_dims)  # Only hidden layers
        
        configurations = {
            "all_layers": [True] * num_layers,
            "early_only": [True, True, False] if num_layers == 3 else [True] + [False] * (num_layers - 1),
            "late_only": [False] * (num_layers - 1) + [True],
            "no_events": [False] * num_layers,
        }
        
        results = {}
        for config_name, mask in configurations.items():
            print(f"\nConfiguration: {config_name} (mask={mask})")
            self._set_seed(self.config.seed)
            
            model = self._create_eglp_network()
            model.set_layer_event_mask(mask)
            
            controller = ThresholdController(
                budget=self.config.event_budget,
                threshold_factor=self.config.threshold_factor,
            )
            logger = MetricsLogger(f"sensitivity_{config_name}", self.config.output_dir)
            
            self._run_eglp_training(model, controller, logger, config_name)
            results[config_name] = logger
        
        return results
    
    def run_robustness_test(self) -> Dict[str, MetricsLogger]:
        """Evaluate on Noisy/Rotated MNIST.
        
        Tests model robustness to distribution shift.
        
        Returns:
            Dict mapping test type to MetricsLogger
        """
        print("\n" + "="*60)
        print("ROBUSTNESS: Noisy/Rotated MNIST")
        print("="*60)
        
        # Train model on clean data
        print("Training on clean MNIST...")
        self._set_seed(self.config.seed)
        model = self._create_eglp_network()
        controller = ThresholdController(
            budget=self.config.event_budget,
            threshold_factor=self.config.threshold_factor,
        )
        
        # Create optimizer for output layer
        optimizer = optim.Adam(
            model.get_output_params(), lr=self.config.backprop_lr
        )
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                error_signal = controller.should_trigger(loss.item())
                model.local_update_all(error_signal)
                model.clear_activations()
        
        results = {}
        
        # Test on clean data
        clean_loss, clean_acc = self._evaluate(model, self.test_loader, is_eglp=True)
        print(f"Clean test accuracy: {clean_acc:.4f}")
        
        clean_logger = MetricsLogger("robustness_clean", self.config.output_dir)
        clean_logger.log_epoch(EpochMetrics(
            epoch=0, train_loss=0, train_accuracy=0,
            val_loss=clean_loss, val_accuracy=clean_acc
        ))
        results["clean"] = clean_logger
        
        # Test on noisy data
        print("Testing on noisy data...")
        noisy_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x + 0.3 * torch.randn_like(x)),  # Add noise
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        
        noisy_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=noisy_transform
        )
        noisy_loader = DataLoader(noisy_dataset, batch_size=self.config.batch_size)
        
        noisy_loss, noisy_acc = self._evaluate(model, noisy_loader, is_eglp=True)
        print(f"Noisy test accuracy: {noisy_acc:.4f}")
        
        noisy_logger = MetricsLogger("robustness_noisy", self.config.output_dir)
        noisy_logger.log_epoch(EpochMetrics(
            epoch=0, train_loss=0, train_accuracy=0,
            val_loss=noisy_loss, val_accuracy=noisy_acc
        ))
        results["noisy"] = noisy_logger
        
        # Test on rotated data
        print("Testing on rotated data...")
        rotated_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        
        rotated_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=rotated_transform
        )
        rotated_loader = DataLoader(rotated_dataset, batch_size=self.config.batch_size)
        
        rotated_loss, rotated_acc = self._evaluate(model, rotated_loader, is_eglp=True)
        print(f"Rotated test accuracy: {rotated_acc:.4f}")
        
        rotated_logger = MetricsLogger("robustness_rotated", self.config.output_dir)
        rotated_logger.log_epoch(EpochMetrics(
            epoch=0, train_loss=0, train_accuracy=0,
            val_loss=rotated_loss, val_accuracy=rotated_acc
        ))
        results["rotated"] = rotated_logger
        
        return results
    
    def _save_ablation_summary(
        self, 
        results: Dict[float, MetricsLogger],
        name: str,
    ) -> None:
        """Save ablation study summary for plotting."""
        import json
        from pathlib import Path
        
        summary = {
            "rates": [],
            "final_accuracies": [],
            "total_events": [],
        }
        
        for rate in sorted(results.keys()):
            logger = results[rate]
            summary_data = logger.get_summary()
            
            summary["rates"].append(rate)
            summary["final_accuracies"].append(summary_data.get("final_val_accuracy", 0))
            summary["total_events"].append(summary_data.get("total_events", 0))
        
        output_path = Path(self.config.output_dir) / f"{name}_summary.json"
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nAblation summary saved to {output_path}")
    
    def run_all_experiments(self) -> Dict[str, MetricsLogger]:
        """Run the complete experimental pipeline."""
        print("\n" + "="*60)
        print("RUNNING COMPLETE EXPERIMENTAL PIPELINE")
        print("="*60)
        
        all_results = {}
        
        # Baselines
        all_results["backprop"] = self.run_backprop_baseline()
        all_results["pure_local"] = self.run_pure_local()
        
        # EGLP variants
        all_results["eglp_fixed"] = self.run_eglp_fixed()
        all_results["eglp_triggered"] = self.run_eglp_triggered()
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        for name, logger in all_results.items():
            summary = logger.get_summary()
            print(f"\n{name}:")
            print(f"  Final Val Accuracy: {summary.get('final_val_accuracy', 0):.4f}")
            print(f"  Best Val Accuracy: {summary.get('best_val_accuracy', 0):.4f}")
            print(f"  Total Events: {summary.get('total_events', 0)}")
        
        return all_results
