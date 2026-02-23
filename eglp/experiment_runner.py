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
from .network import EGLPNetwork, create_mlp, create_cnn, FrozenBatchNorm1d, create_predictive_mlp, AdaptiveEGLPNetwork
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
    local_lr: float = 1e-5
    backprop_lr: float = 0.001
    
    # EGLP
    event_budget: int = 1000
    event_rate: float = 0.05
    threshold_factor: float = 2.0
    max_signal: float = 1.0
    
    # EGLP enhancements
    use_anti_hebbian: bool = False
    use_batchnorm: bool = False
    use_consolidation: bool = False
    consolidation_strength: float = 0.01
    use_homeostasis: bool = False
    homeostasis_interval: int = 50  # Apply every N batches
    cosine_lr: bool = False
    cosine_lr_max: float = 5e-4
    cosine_lr_min: float = 1e-5
    soft_wta_k: float = 0.0  # 0 = disabled, e.g. 0.3 = keep top 30%
    event_decay: float = 1.0  # Depth decay γ, 1.0 = no decay
    
    # EGLP enhancements
    gamma_lr: float = 0.01
    inhib_lr: float = 0.01
    feedback_lr: float = 0.01
    inhib_steps: int = 2
    
    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./results"
    
    def __post_init__(self):
        # Auto-configure based on dataset if not explicitly overridden
        if self.dataset.lower() == "cifar10":
            if self.input_dim == 784:  # Check if still default
                self.input_dim = 3072
            
            # CIFAR-10 Best Practice: Deep Pyramidal MLP
            # [Input(3072) -> 3072 -> 2048 -> 1024 -> 512 -> Output(10)]
            # This depth allows for hierarchical feature formation that shallow networks miss.
            if self.hidden_dims is None:
                self.hidden_dims = [3072, 2048, 1024, 512]
            
            # Ensure SoftWTA is active for CIFAR if not manually set
            if self.soft_wta_k == 0.0:
                 self.soft_wta_k = 0.4
                 
        elif self.dataset.lower() == "mnist":
            # MNIST Best Known Configuration: [256, 128] (Default is fine, but explicit)
            if self.hidden_dims is None:
                self.hidden_dims = [256, 128]
        
        # Fallback for other potential datasets
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
        if self.config.dataset.lower() == "mnist":
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
            # update input dim if default
            if self.config.input_dim == 784 and self.config.dataset.lower() != "mnist":
                 pass # Should handled by config init, but just in case
        elif self.config.dataset.lower() == "cifar10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda x: x.view(-1)),  # Flatten 32*32*3
            ])
            train_dataset = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
        
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
    
    def _create_eglp_basic_network(self) -> EGLPNetwork:
        """Create basic EGLP network with local layers (no enhancements)."""
        network = create_mlp(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            lr=self.config.local_lr,
        )
        return network.to(self.config.device)
    
    def _create_eglp_network(self) -> EGLPNetwork:
        """Create enhanced EGLP network with all improvements."""
        network = create_mlp(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            lr=self.config.local_lr,
            use_anti_hebbian=self.config.use_anti_hebbian,
            consolidation_strength=(
                self.config.consolidation_strength if self.config.use_consolidation else 0.0
            ),
            use_batchnorm=self.config.use_batchnorm,
            soft_wta_k=self.config.soft_wta_k if self.config.soft_wta_k > 0 else None,
            event_decay=self.config.event_decay,
        )
        return network.to(self.config.device)

    def _create_eglp_v2_network(self) -> AdaptiveEGLPNetwork:
        """Create EGLP-V2 network with Predictive Coding and Adaptive Feedback."""
        network = create_predictive_mlp(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            lr=self.config.local_lr,
            gamma_lr=self.config.gamma_lr,
            inhib_lr=self.config.inhib_lr,
            feedback_lr=self.config.feedback_lr,
            inhib_steps=self.config.inhib_steps,
            event_decay=self.config.event_decay,
            use_anti_hebbian=self.config.use_anti_hebbian,
            consolidation_strength=(
                self.config.consolidation_strength if self.config.use_consolidation else 0.0
            ),
            use_batchnorm=self.config.use_batchnorm,
            soft_wta_k=self.config.soft_wta_k if self.config.soft_wta_k > 0 else None,
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
    
    def _evaluate_and_store_test(
        self,
        model: nn.Module,
        logger: MetricsLogger,
        name: str,
        is_eglp: bool = False,
    ) -> None:
        """Run dedicated test-set evaluation and store results."""
        test_loss, test_metrics = self._evaluate(model, self.test_loader, is_eglp=is_eglp)
        logger.set_test_metrics(test_metrics)
        
        print(f"\n--- Test Results for {name} ---")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1:        {test_metrics['f1']:.4f}")
        print(f"-------------------------------\n")
    
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
            
            val_acc = val_metrics["accuracy"]
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Store representations for CKA comparison
        self._store_backprop_representations(model)
        
        # Dedicated test-set evaluation
        self._evaluate_and_store_test(model, logger, "Backprop Baseline", is_eglp=False)
        
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
        model = self._create_eglp_basic_network()
        controller = AlwaysOnController(max_signal=self.config.max_signal)
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
        model = self._create_eglp_basic_network()
        controller = FixedRateController(
            rate=rate, 
            seed=self.config.seed,
            max_signal=self.config.max_signal
        )
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
        model = self._create_eglp_basic_network()
        controller = ThresholdController(
            budget=self.config.event_budget,
            threshold_factor=self.config.threshold_factor,
            max_signal=self.config.max_signal,
        )
        logger = MetricsLogger("eglp_triggered", self.config.output_dir)
        
        return self._run_eglp_training(model, controller, logger, "EGLP-Triggered")
    
    # ========================
    # EGLP Baseline: Enhanced with all original heuristics (No PC/AFA)
    # ========================
    
    def run_eglp(self) -> MetricsLogger:
        """Run EGLP Baseline with heuristic enhancements.
        
        Combines:
        - Anti-Hebbian diversity (alternating layers)
        - Synaptic consolidation (EWC-lite)
        - Homeostatic scaling (dead neuron recovery)
        - Cosine LR schedule
        - Frozen BatchNorm
        - Soft-WTA activation (optional)
        - Depth-dependent event decay
        - Higher event rate and max_signal
        """
        print("\n" + "="*60)
        print("EGLP Baseline (Heuristic Enhancements)")
        print(f"  Architecture: {self.config.hidden_dims}")
        print(f"  Anti-Hebbian: {self.config.use_anti_hebbian}")
        print(f"  BatchNorm: {self.config.use_batchnorm}")
        print(f"  Consolidation: {self.config.use_consolidation}")
        print(f"  Homeostasis: {self.config.use_homeostasis}")
        print(f"  Cosine LR: {self.config.cosine_lr}")
        print(f"  Soft-WTA k: {self.config.soft_wta_k}")
        print(f"  Event decay γ: {self.config.event_decay}")
        print(f"  Event rate: {self.config.event_rate}")
        print("="*60)
        
        self._set_seed(self.config.seed)
        model = self._create_eglp_network()
        
        # Warmup BatchNorm if enabled
        if self.config.use_batchnorm:
            print("Running BatchNorm warmup pass...")
            model.warmup_batchnorms(self.train_loader, self.config.device)
        
        controller = ThresholdController(
            budget=self.config.event_budget,
            threshold_factor=self.config.threshold_factor,
            max_signal=self.config.max_signal,
        )
        logger = MetricsLogger("eglp_baseline", self.config.output_dir)
        
        return self._run_eglp_training(
            model, controller, logger, "EGLP Baseline",
            use_cosine_lr=self.config.cosine_lr,
            use_homeostasis=self.config.use_homeostasis,
            use_consolidation=self.config.use_consolidation,
        )

    # ========================
    # EGLP: Predictive Coding & Adaptive Feedback
    # ========================
    
    def run_eglp_v2(self) -> MetricsLogger:
        """Run EGLP with Predictive Coding, Adaptive Feedback, and Learned Inhibition."""
        print("\n" + "="*60)
        print("EGLP (Predictive + Adaptive Feedback + Inhibition)")
        print(f"  Architecture: {self.config.hidden_dims}")
        print(f"  Gamma LR: {self.config.gamma_lr}")
        print(f"  Inhib LR: {self.config.inhib_lr}")
        print(f"  Feedback LR: {self.config.feedback_lr}")
        print(f"  Anti-Hebbian: {self.config.use_anti_hebbian}")
        print(f"  BatchNorm: {self.config.use_batchnorm}")
        print(f"  Consolidation: {self.config.use_consolidation}")
        print(f"  Soft-WTA k: {self.config.soft_wta_k}")
        print("="*60)
        
        self._set_seed(self.config.seed)
        model = self._create_eglp_v2_network()
        
        # Warmup BatchNorm if enabled
        if self.config.use_batchnorm:
            print("Running BatchNorm warmup pass...")
            model.warmup_batchnorms(self.train_loader, self.config.device)
        
        controller = ThresholdController(
            budget=self.config.event_budget,
            threshold_factor=self.config.threshold_factor,
            max_signal=self.config.max_signal,
        )
        logger = MetricsLogger("eglp", self.config.output_dir)
        
        return self._run_eglp_training(
            model, controller, logger, "EGLP",
            use_cosine_lr=self.config.cosine_lr,
            use_homeostasis=self.config.use_homeostasis,
            use_consolidation=self.config.use_consolidation,
        )
    
    def _run_eglp_training(
        self,
        model: EGLPNetwork,
        controller: EventController,
        logger: MetricsLogger,
        name: str,
        use_cosine_lr: bool = False,
        use_homeostasis: bool = False,
        use_consolidation: bool = False,
        train_output: bool = True,
        train_hidden: bool = True,
    ) -> MetricsLogger:
        """Core EGLP hybrid training loop.
        
        Hidden layers: local Hebbian updates modulated by error signal
        Output layer: supervised backprop (only 1 layer, minimal communication)
        
        EGLP additions:
        - Cosine LR schedule for local learning rate
        - Periodic homeostatic scaling
        - Synaptic consolidation snapshots at epoch boundaries
        """
        import math
        
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer for output layer ONLY
        if train_output:
            optimizer = optim.Adam(
                model.get_output_params(), 
                lr=self.config.backprop_lr,
            )
        else:
            optimizer = None
        
        # Store base LR for cosine schedule
        base_lr = self.config.local_lr
        
        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            epoch_events = 0
            batch_count = 0
            
            # --- Cosine LR schedule: update local_lr per epoch ---
            if use_cosine_lr:
                lr_max = self.config.cosine_lr_max
                lr_min = self.config.cosine_lr_min
                T = self.config.epochs
                current_lr = lr_min + 0.5 * (lr_max - lr_min) * (
                    1 + math.cos(epoch * math.pi / T)
                )
                # Update LR in all hidden layers
                for layer in model.layers:
                    if hasattr(layer, 'lr'):
                        layer.lr = current_lr
            
            for data, target in self.train_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                batch_count += 1
                
                # Forward pass: hidden layers (no grad) + output layer (grad)
                if optimizer:
                    optimizer.zero_grad()
                output = model(data, store_representations=(epoch == self.config.epochs - 1))
                
                # Compute loss with gradients for output layer
                loss = criterion(output, target)
                
                # Backprop ONLY the output layer
                if train_output:
                    loss.backward()
                    optimizer.step()
                
                # Controller decides event signal (continuous float)
                if train_hidden:
                    error_signal = controller.should_trigger(loss.item())
                    epoch_events += (1 if error_signal > 0 else 0)
                    
                    # Local update for hidden layers with error modulation
                    model.local_update_all(error_signal, logits=output, target=target)
                    model.clear_activations()
                else:
                    error_signal = 0.0
                
                # --- Homeostatic scaling every N batches ---
                if use_homeostasis and batch_count % self.config.homeostasis_interval == 0:
                    model.homeostatic_scale_all()
                
                # Metrics
                epoch_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
                total += target.size(0)
            
            # --- Synaptic consolidation: snapshot weights at epoch boundary ---
            if use_consolidation:
                model.snapshot_all_weights()
            
            train_loss = epoch_loss / total
            train_acc = correct / total
            val_loss, val_metrics = self._evaluate(model, self.val_loader, is_eglp=True)
            val_acc = val_metrics["accuracy"]
            
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
            
            # Log current LR if cosine schedule is active
            lr_info = ""
            if use_cosine_lr:
                lr_info = f" | LR: {current_lr:.6f}"
            
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
                  f"Events: {epoch_events}{lr_info}")
        
        # Restore base LR after training (in case of cosine schedule)
        if use_cosine_lr:
            for layer in model.layers:
                if hasattr(layer, 'lr'):
                    layer.lr = base_lr
        
        # Dedicated test-set evaluation
        self._evaluate_and_store_test(model, logger, name, is_eglp=True)
        
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
            
            model = self._create_eglp_basic_network()
            controller = FixedRateController(
                rate=rate, 
                seed=self.config.seed,
                max_signal=self.config.max_signal
            )
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
        backbone = self._create_eglp_basic_network()
        controller = ThresholdController(
            budget=self.config.event_budget,
            threshold_factor=self.config.threshold_factor,
            max_signal=self.config.max_signal,
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
        
        # Extract features and train probe
        probe_logger = self._run_linear_probe_on_model(backbone)
        return probe_logger
        
    def _run_linear_probe_on_model(self, model: nn.Module) -> MetricsLogger:
        """Trains a linear probe on the frozen representations of a model."""
        print("\nExtracting features from frozen backbone...")
        
        def extract_features(loader):
            features = []
            labels = []
            
            with torch.no_grad():
                for data, target in loader:
                    data = data.to(self.config.device)
                    
                    x = data
                    for layer in model.layers:
                        x = layer(x)
                        x = nn.ReLU()(x)
                    
                    features.append(x.cpu())
                    labels.append(target)
            
            return torch.cat(features), torch.cat(labels)
        
        train_features, train_labels = extract_features(self.train_loader)
        val_features, val_labels = extract_features(self.val_loader)
        test_features, test_labels = extract_features(self.test_loader)
        
        print("Training linear probe...")
        feature_dim = train_features.size(1)
        linear_probe = nn.Linear(feature_dim, self.config.output_dim).to(self.config.device)
        optimizer = optim.Adam(linear_probe.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        logger = MetricsLogger("linear_probe", self.config.output_dir)
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        probe_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        for epoch in range(10):  # Probe trains fast
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
                epoch=epoch, train_loss=train_loss, train_accuracy=train_acc,
                val_loss=val_loss, val_accuracy=val_acc,
            ))
            
            print(f"Probe Epoch {epoch+1}/10 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
        # Run final test set
        linear_probe.eval()
        test_features_dev = test_features.to(self.config.device)
        test_labels_dev = test_labels.to(self.config.device)
        with torch.no_grad():
            test_output = linear_probe(test_features_dev)
            _, predicted = test_output.max(1)
            test_acc = predicted.eq(test_labels_dev).float().mean().item()
            
        logger.set_test_metrics({"accuracy": test_acc, "precision": 0.0, "recall": 0.0, "f1": 0.0})
        print(f"--- Linear Probe Final Test Acc: {test_acc:.4f} ---")
        
        logger.save()
        return logger

    # ========================
    # DIAGNOSTICS: Hypothesis Validation
    # ========================

    def run_zero_event(self) -> MetricsLogger:
        """Run EGLP with a hardcoded zero-event controller to prove importance of error signal."""
        print("\n" + "="*60)
        print("DIAGNOSTIC: Hard Zero Event (NeverController)")
        print("="*60)
        self._set_seed(self.config.seed)
        model = self._create_eglp_network()
        controller = NeverController()
        logger = MetricsLogger("diagnostic_zero_event", self.config.output_dir)
        return self._run_eglp_training(
            model, controller, logger, "Zero Event",
            use_cosine_lr=self.config.cosine_lr,
            use_homeostasis=self.config.use_homeostasis,
            use_consolidation=self.config.use_consolidation,
            train_output=True,
            train_hidden=True, # Will result in 0 updates due to NeverController
        )

    def run_unsupervised_pretraining(self) -> MetricsLogger:
        """Fully unsupervised Hebbian pretraining of hidden features, followed by linear probe."""
        print("\n" + "="*60)
        print("DIAGNOSTIC: Unsupervised Pretraining + Linear Probe")
        print("="*60)
        self._set_seed(self.config.seed)
        model = self._create_eglp_network()
        controller = AlwaysOnController()
        logger = MetricsLogger("diagnostic_unsupervised", self.config.output_dir)
        
        print("Phase 1/2: Pure Hebbian Pretraining (no labels)...")
        self._run_eglp_training(
            model, controller, logger, "Unsupervised Pretrain",
            use_cosine_lr=self.config.cosine_lr,
            use_homeostasis=self.config.use_homeostasis,
            use_consolidation=self.config.use_consolidation,
            train_output=False, # Crucial: no backprop on classifier
            train_hidden=True,
        )
        
        print("Phase 2/2: Training Linear Probe on frozen features...")
        probe_logger = self._run_linear_probe_on_model(model)
        
        # Merge final linear probe accuracy into the main logger test_metrics
        logger.set_test_metrics({
            "accuracy": probe_logger.test_metrics["accuracy"], 
            "precision": 0.0, "recall": 0.0, "f1": 0.0
        })
        logger.save()
        return logger

    def run_freeze_representation(self, num_retrains: int = 5) -> Dict[str, float]:
        """Freezes representation and repeatedly retrains the head to test stability."""
        print("\n" + "="*60)
        print(f"DIAGNOSTIC: Freeze Representation ({num_retrains} retrains)")
        print("="*60)
        self._set_seed(self.config.seed)
        model = self._create_eglp_network()
        controller = ThresholdController(
            budget=self.config.event_budget,
            threshold_factor=self.config.threshold_factor,
            max_signal=self.config.max_signal,
        )
        base_logger = MetricsLogger("diagnostic_freeze_base", self.config.output_dir)
        
        print("Phase 1: Training full EGLP model...")
        self._run_eglp_training(
            model, controller, base_logger, "Freeze Base Model",
            use_cosine_lr=self.config.cosine_lr,
            use_homeostasis=self.config.use_homeostasis,
            use_consolidation=self.config.use_consolidation,
            train_output=True,
            train_hidden=True,
        )
        
        print(f"Phase 2: Freezing features and retraining output {num_retrains} times...")
        retrain_accs = []
        for i in range(num_retrains):
            print(f"\n  Retrain {i+1}/{num_retrains}...")
            # Reinitialize output layer
            model.output_layer.reset_parameters()
            
            retrain_logger = MetricsLogger(f"diagnostic_freeze_retrain_{i}", self.config.output_dir)
            self._run_eglp_training(
                model, controller, retrain_logger, f"Retrain {i}",
                train_output=True,
                train_hidden=False, # Freeze hidden
            )
            
            test_loss, test_metrics = self._evaluate(model, self.test_loader)
            retrain_accs.append(test_metrics["accuracy"])
            
        mean_acc = sum(retrain_accs) / len(retrain_accs)
        std_acc = (sum((x - mean_acc) ** 2 for x in retrain_accs) / len(retrain_accs)) ** 0.5
        
        print(f"\n--- Freeze Representation Results ---")
        print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        for i, acc in enumerate(retrain_accs):
            print(f"    Run {i+1}: {acc:.4f}")
            
        return {"mean": mean_acc, "std": std_acc, "runs": retrain_accs}
    
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
            
            model = self._create_eglp_basic_network()
            model.set_layer_event_mask(mask)
            
            controller = ThresholdController(
                budget=self.config.event_budget,
                threshold_factor=self.config.threshold_factor,
                max_signal=self.config.max_signal,
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
        model = self._create_eglp_basic_network()
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
                model.local_update_all(error_signal, logits=output, target=target)
                model.clear_activations()
        
        results = {}
        
        # Test on clean data
        clean_loss, clean_metrics = self._evaluate(model, self.test_loader, is_eglp=True)
        print(f"Clean test accuracy: {clean_metrics['accuracy']:.4f}")
        
        clean_logger = MetricsLogger("robustness_clean", self.config.output_dir)
        clean_logger.log_epoch(EpochMetrics(
            epoch=0, train_loss=0, train_accuracy=0,
            val_loss=clean_loss, val_accuracy=clean_metrics["accuracy"],
            val_precision=clean_metrics["precision"],
            val_recall=clean_metrics["recall"],
            val_f1=clean_metrics["f1"],
        ))
        clean_logger.set_test_metrics(clean_metrics)
        results["clean"] = clean_logger
        
        # Setup base transforms
        if self.config.dataset == "cifar10":
            base_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            DsClass = torchvision.datasets.CIFAR10
        else:
            base_norm = transforms.Normalize((0.1307,), (0.3081,))
            DsClass = torchvision.datasets.MNIST

        # Test on noisy data
        print("Testing on noisy data...")
        noisy_transform = transforms.Compose([
            transforms.ToTensor(),
            base_norm,
            transforms.Lambda(lambda x: x + 0.3 * torch.randn_like(x)),  # Add noise
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        
        noisy_dataset = DsClass(
            root="./data", train=False, download=True, transform=noisy_transform
        )
        noisy_loader = DataLoader(noisy_dataset, batch_size=self.config.batch_size)
        
        noisy_loss, noisy_metrics = self._evaluate(model, noisy_loader, is_eglp=True)
        print(f"Noisy test accuracy: {noisy_metrics['accuracy']:.4f}")
        
        noisy_logger = MetricsLogger("robustness_noisy", self.config.output_dir)
        noisy_logger.log_epoch(EpochMetrics(
            epoch=0, train_loss=0, train_accuracy=0,
            val_loss=noisy_loss, val_accuracy=noisy_metrics["accuracy"],
            val_precision=noisy_metrics["precision"],
            val_recall=noisy_metrics["recall"],
            val_f1=noisy_metrics["f1"],
        ))
        noisy_logger.set_test_metrics(noisy_metrics)
        results["noisy"] = noisy_logger
        
        # Test on rotated data
        print("Testing on rotated data...")
        rotated_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.ToTensor(),
            base_norm,
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        
        rotated_dataset = DsClass(
            root="./data", train=False, download=True, transform=rotated_transform
        )
        rotated_loader = DataLoader(rotated_dataset, batch_size=self.config.batch_size)
        
        rotated_loss, rotated_metrics = self._evaluate(model, rotated_loader, is_eglp=True)
        print(f"Rotated test accuracy: {rotated_metrics['accuracy']:.4f}")
        
        rotated_logger = MetricsLogger("robustness_rotated", self.config.output_dir)
        rotated_logger.log_epoch(EpochMetrics(
            epoch=0, train_loss=0, train_accuracy=0,
            val_loss=rotated_loss, val_accuracy=rotated_metrics["accuracy"],
            val_precision=rotated_metrics["precision"],
            val_recall=rotated_metrics["recall"],
            val_f1=rotated_metrics["f1"],
        ))
        rotated_logger.set_test_metrics(rotated_metrics)
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
        all_results["eglp_baseline"] = self.run_eglp()
        all_results["eglp"] = self.run_eglp_v2()
        
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
            if 'test_metrics' in summary:
                tm = summary['test_metrics']
                print(f"  Test Accuracy:  {tm['accuracy']:.4f}")
                print(f"  Test Precision: {tm['precision']:.4f}")
                print(f"  Test Recall:    {tm['recall']:.4f}")
                print(f"  Test F1:        {tm['f1']:.4f}")
        
        return all_results
