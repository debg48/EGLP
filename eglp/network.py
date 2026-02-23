"""EGLP Network wrapper for assembling local layers.

Additional features:
- FrozenBatchNorm1d: BatchNorm-like normalization with no gradients
- Soft-WTA activation: top-k sparsity for feature diversity
- Depth-dependent event scaling: deeper layers get attenuated signals
"""

import torch
import torch.nn as nn
import math
from typing import List, Dict, Optional, Union

from .local_layer import LocalLinear, LocalConv2d
from .event_controller import EventController


class FrozenBatchNorm1d(nn.Module):
    """BatchNorm-like layer with ALL parameters frozen (requires_grad=False).
    
    Unlike standard nn.BatchNorm1d, this will NOT leak gradients through
    the EGLP hidden path. It uses running statistics computed during a
    warmup pass, then stays frozen during training.
    
    This avoids the internal covariate shift problem for Hebbian layers
    while maintaining the no-gradient-leak invariant.
    """
    
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # Affine parameters — frozen
        self.weight = nn.Parameter(torch.ones(num_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        # Once frozen, always use running stats
        self._frozen = False
    
    def freeze(self):
        """Freeze the running statistics. After this, forward always uses stored stats."""
        self._frozen = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if not self._frozen:
                # Warmup mode: update running stats
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)
                
                self.num_batches_tracked += 1
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var + self.momentum * batch_var
                )
            
            # Normalize using running stats (always)
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            return x_norm * self.weight + self.bias


class SoftWTA(nn.Module):
    """Soft Winner-Take-All activation.
    
    After ReLU, keeps only the top-k largest activations per sample
    and zeros the rest. This encourages each sample to activate
    a sparse, diverse subset of neurons.
    
    Replaces pairwise cosine similarity lateral inhibition (O(n²))
    with an efficient O(n log n) sparsity mechanism.
    """
    
    def __init__(self, k_fraction: float = 0.3):
        """
        Args:
            k_fraction: Fraction of neurons to keep active (0-1).
                       E.g. 0.3 means keep top 30% of activations.
        """
        super().__init__()
        self.k_fraction = k_fraction
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Apply ReLU first
            x = torch.relu(x)
            
            # Compute k
            k = max(1, int(x.size(-1) * self.k_fraction))
            
            # Find the k-th largest value per sample
            topk_values, _ = x.topk(k, dim=-1)
            threshold = topk_values[:, -1].unsqueeze(-1)  # (batch, 1)
            
            # Zero out below threshold
            mask = x >= threshold
            return x * mask.float()


class EGLPNetwork(nn.Module):
    """Network wrapper that manages local layers and event broadcasting.
    
    This class:
    - Assembles LocalLinear/LocalConv2d layers for hidden layers (unsupervised)
    - Uses a standard nn.Linear for the output layer (supervised via backprop)
    - Handles forward pass under torch.no_grad() for hidden layers
    - Broadcasts event signals to all (or selected) hidden layers
    - Stores representations for CKA computation
    
    Additional features:
    - Optional FrozenBatchNorm1d after each hidden layer
    - Optional SoftWTA activation for feature diversity
    - Depth-dependent event scaling (deeper layers get weaker signals)
    """
    
    def __init__(
        self,
        layers: nn.ModuleList,
        output_layer: nn.Linear,
        activation: str = "relu",
        layer_event_mask: Optional[List[bool]] = None,
        batchnorms: Optional[nn.ModuleList] = None,
        soft_wta_k: Optional[float] = None,
        event_decay: float = 1.0,
        use_dfa: bool = True,
    ):
        """Initialize EGLP network.
        
        Args:
            layers: ModuleList of LocalLinear/LocalConv2d layers (hidden layers).
            output_layer: Standard nn.Linear for supervised output (has gradients).
            activation: Activation function ("relu", "tanh", "sigmoid").
            layer_event_mask: Optional mask for selective event gating per layer.
                             If None, all hidden layers receive events.
            batchnorms: Optional ModuleList of FrozenBatchNorm1d layers (one per hidden layer).
            soft_wta_k: If set, use SoftWTA activation with this k_fraction instead
                       of standard activation for hidden layers.
            event_decay: Depth-dependent event decay factor γ.
                        Layer l receives event signal E·γ^l. Default 1.0 (no decay).
            use_dfa: If True, use Direct Feedback Alignment (random feedback matrices)
                    instead of gradient norm modulation.
        """
        super().__init__()
        self.layers = layers
        self.output_layer = output_layer
        self.activation_name = activation
        self.event_decay = event_decay
        self.use_dfa = use_dfa
        
        # Set up activation function
        if soft_wta_k is not None and soft_wta_k > 0:
            self.activation = SoftWTA(k_fraction=soft_wta_k)
            self._use_soft_wta = True
        else:
            self._use_soft_wta = False
            if activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "tanh":
                self.activation = nn.Tanh()
            elif activation == "sigmoid":
                self.activation = nn.Sigmoid()
            else:
                raise ValueError(f"Unknown activation: {activation}")
        
        # Optional BatchNorm layers (all frozen, no gradients)
        self.batchnorms = batchnorms
        
        # Event mask (which hidden layers receive events)
        if layer_event_mask is None:
            self.layer_event_mask = [True] * len(layers)
        else:
            if len(layer_event_mask) != len(layers):
                raise ValueError("layer_event_mask must match number of hidden layers")
            self.layer_event_mask = layer_event_mask
        
        # Storage for layer representations (for CKA)
        self.representations: Dict[int, torch.Tensor] = {}
        
        # Initialize Feedback Matrices for DFA
        self.feedback_matrices = nn.ParameterList()
        if self.use_dfa:
            out_dim = output_layer.out_features
            for layer in layers:
                if isinstance(layer, LocalLinear):
                    # Matrix B: (hidden_dim, output_dim)
                    # We project error (batch, output_dim) -> (batch, hidden_dim)
                    # via error @ B.T
                    
                    # Random fixed matrix, non-trainable
                    B = torch.randn(layer.out_features, out_dim) / math.sqrt(out_dim)
                    self.feedback_matrices.append(nn.Parameter(B, requires_grad=False))
                else:
                    # Todo: Implement conv feedback if needed
                    self.feedback_matrices.append(None)
        
        # Verify no gradients on hidden layers
        self._verify_no_grad()
    
    def _verify_no_grad(self) -> None:
        """Verify hidden layer parameters have requires_grad=False.
        
        Output layer is allowed to have gradients (supervised).
        Also checks any batchnorm layers.
        """
        for name, param in self.layers.named_parameters():
            if param.requires_grad:
                raise RuntimeError(
                    f"Gradient leak detected! Hidden layer parameter '{name}' has requires_grad=True"
                )
        if self.batchnorms is not None:
            for name, param in self.batchnorms.named_parameters():
                if param.requires_grad:
                    raise RuntimeError(
                        f"Gradient leak detected! BatchNorm parameter '{name}' has requires_grad=True"
                    )
    
    def forward(self, x: torch.Tensor, store_representations: bool = False) -> torch.Tensor:
        """Forward pass through all layers.
        
        Hidden layers run under torch.no_grad(). The output layer
        runs with gradients enabled for supervised backprop.
        
        Args:
            x: Input tensor.
            store_representations: If True, store each layer's output for CKA.
            
        Returns:
            Output tensor (logits for classification).
        """
        # Hidden layers: no gradients
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                x = layer(x)
                
                # Apply BatchNorm if present (before activation)
                if self.batchnorms is not None and i < len(self.batchnorms):
                    x = self.batchnorms[i](x)
                
                x = self.activation(x)
                
                # Store representation if requested
                if store_representations:
                    self.representations[i] = x.detach().clone()
        
        # Detach to prevent gradients flowing back to hidden layers
        x = x.detach()
        
        # Output layer: gradients enabled for supervised learning
        logits = self.output_layer(x)
        
        if store_representations:
            self.representations[len(self.layers)] = logits.detach().clone()
        
        return logits
    
    def get_output_params(self):
        """Get output layer parameters for the optimizer."""
        return self.output_layer.parameters()
    
    def local_update_all(self, event: float, logits: Optional[torch.Tensor] = None, target: Optional[torch.Tensor] = None) -> None:
        """Broadcast event signal to all hidden layers with neuromodulated updates.
        
        Args:
            event: Continuous error signal from controller (gates magnitude/timing).
            logits: Output logits (batch, num_classes) - needed for DFA.
            target: Target labels (batch,) - needed for DFA.
        """
        if event == 0.0:
            return

        n_layers = len(self.layers)
        
        # Compute error vector for DFA: e = softmax(y) - target_one_hot
        error_vector = None
        if self.use_dfa and logits is not None and target is not None:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                t_one_hot = torch.nn.functional.one_hot(target, num_classes=self.output_layer.out_features).float()
                # Signed error vector
                # (batch, out_dim)
                error_vector = (probs - t_one_hot)

        # Legacy: Gradient-based modulation from the output layer
        # (Only used if use_dfa is False)
        output_grad = None
        if not self.use_dfa and self.output_layer.weight.grad is not None:
            output_grad = self.output_layer.weight.grad
        
        for i in reversed(range(n_layers)):
            # Apply layer-specific event mask
            if not self.layer_event_mask[i]:
                continue
            
            # Depth-dependent scaling: E · γ^l
            scaled_event = event * (self.event_decay ** i)
            
            # Compute modulation
            modulation = None
            
            if self.use_dfa:
                # DFA Modulation: m = e @ B^T
                # e: (batch, out_dim)
                # B: (hidden_dim, out_dim)
                # m: (batch, hidden_dim)
                B = self.feedback_matrices[i]
                if B is not None and error_vector is not None:
                     modulation = torch.mm(error_vector, B.T)
                     # Normalize modulation to keep scale reasonable relative to Hebbian term
                     # We act as if 'event' sets the global learning rate scale
                     # Modulation gives direction and relative intensity
                     # Optional: normalize per batch?
                     # modulation = modulation / (modulation.abs().mean() + 1e-6)
            
            else:
                # Original Gradient Norm Modulation (Positive only)
                if i == n_layers - 1 and output_grad is not None:
                    # Last hidden layer: use output gradient as credit signal
                    # ||∂L/∂w[:, j]|| = how much neuron j matters for reducing loss
                    modulation = output_grad.norm(dim=0)  # (last_hidden_dim,)
                    mod_max = modulation.max()
                    if mod_max > 0:
                        modulation = modulation / mod_max
            
            layer = self.layers[i]
            layer.local_update(scaled_event, modulation=modulation)
    
    def local_update_selective(self, events: List[float]) -> None:
        """Apply different event signals to different hidden layers.
        
        Args:
            events: List of event signals, one per hidden layer.
        """
        if len(events) != len(self.layers):
            raise ValueError("events list must match number of hidden layers")
        
        for layer, event in zip(self.layers, events):
            layer.local_update(event)
    
    def snapshot_all_weights(self) -> None:
        """Take weight snapshots for consolidation in all hidden layers.
        
        Should be called at end of each epoch.
        """
        for layer in self.layers:
            layer.snapshot_weights()
    
    def homeostatic_scale_all(self) -> None:
        """Apply homeostatic scaling to all hidden layers.
        
        Should be called periodically (e.g., every 50 batches).
        """
        for layer in self.layers:
            layer.homeostatic_scale()
    
    def freeze_batchnorms(self) -> None:
        """Freeze all BatchNorm running statistics.
        
        Call after warmup pass to lock in the normalization constants.
        """
        if self.batchnorms is not None:
            for bn in self.batchnorms:
                bn.freeze()
    
    def warmup_batchnorms(self, data_loader, device: str = "cpu") -> None:
        """Run a warmup pass through data to compute BatchNorm running stats.
        
        Args:
            data_loader: DataLoader to compute stats from.
            device: Device to run warmup on.
        """
        if self.batchnorms is None:
            return
        
        self.eval()
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(device)
                x = data
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    if i < len(self.batchnorms):
                        x = self.batchnorms[i](x)
                    x = nn.ReLU()(x)  # Use basic ReLU for warmup
                    layer.clear_activations()
        
        # Freeze after warmup
        self.freeze_batchnorms()
    
    def clear_activations(self) -> None:
        """Clear stored activations in all layers."""
        for layer in self.layers:
            layer.clear_activations()
        self.representations.clear()
    
    def get_representations(self) -> Dict[int, torch.Tensor]:
        """Get stored layer representations for CKA computation."""
        return self.representations
    
    def set_layer_event_mask(self, mask: List[bool]) -> None:
        """Update which hidden layers receive event signals.
        
        Useful for layer sensitivity experiments.
        """
        if len(mask) != len(self.layers):
            raise ValueError("mask must match number of hidden layers")
        self.layer_event_mask = mask
    
    def num_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def create_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    lr: float = 0.01,
    activation: str = "relu",
    weight_clip: float = 1.0,
    max_activation: float = 5.0,
    use_anti_hebbian: bool = False,
    consolidation_strength: float = 0.0,
    use_batchnorm: bool = False,
    soft_wta_k: Optional[float] = None,
    event_decay: float = 1.0,
) -> EGLPNetwork:
    """Factory function to create an MLP with local hidden layers and supervised output.
    
    Args:
        input_dim: Input feature dimension.
        hidden_dims: List of hidden layer dimensions.
        output_dim: Output dimension (number of classes).
        lr: Learning rate for local updates.
        activation: Activation function name.
        weight_clip: Max absolute weight value for clipping.
        max_activation: Max absolute activation value for clamping.
        use_anti_hebbian: If True, alternate Hebbian/anti-Hebbian across layers.
                         Even layers → standard Oja, odd layers → anti-Oja.
        consolidation_strength: Strength of synaptic consolidation (0 = disabled).
        use_batchnorm: If True, add FrozenBatchNorm1d after each hidden layer.
        soft_wta_k: If set, use SoftWTA activation with this fraction.
        event_decay: Depth-dependent event decay factor γ.
        
    Returns:
        EGLPNetwork instance.
    """
    # Hidden layers: local Hebbian learning (unsupervised)
    layers = nn.ModuleList()
    batchnorms = nn.ModuleList() if use_batchnorm else None
    prev_dim = input_dim
    
    for i, hidden_dim in enumerate(hidden_dims):
        # Alternate anti-Hebbian: even layers → Oja, odd layers → anti-Oja
        anti_heb = use_anti_hebbian and (i % 2 == 1)
        
        layers.append(LocalLinear(
            prev_dim, hidden_dim, lr=lr,
            weight_clip=weight_clip,
            max_activation=max_activation,
            anti_hebbian=anti_heb,
            consolidation_strength=consolidation_strength,
        ))
        
        if use_batchnorm:
            batchnorms.append(FrozenBatchNorm1d(hidden_dim))
        
        prev_dim = hidden_dim
    
    # Output layer: standard nn.Linear (supervised via backprop)
    output_layer = nn.Linear(prev_dim, output_dim)
    
    return EGLPNetwork(
        layers, output_layer,
        activation=activation,
        batchnorms=batchnorms,
        soft_wta_k=soft_wta_k,
        event_decay=event_decay,
        use_dfa=True, # Enable DFA by default for MLP
    )


def create_cnn(
    in_channels: int = 1,
    num_classes: int = 10,
    lr: float = 0.01,
) -> EGLPNetwork:
    """Factory function to create a simple CNN for MNIST.
    
    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        lr: Learning rate for local updates.
        
    Returns:
        EGLPNetwork instance.
    """
    layers = nn.ModuleList([
        LocalConv2d(in_channels, 32, kernel_size=3, padding=1, lr=lr),
        LocalConv2d(32, 64, kernel_size=3, padding=1, lr=lr),
    ])
    
    # Output layer would need flattening logic
    # This is a simplified version; extend as needed
    # Calculate flattened size for linear layer
    with torch.no_grad():
        dummy_input = torch.zeros(1, in_channels, 28, 28)
        x = dummy_input
        for layer in layers:
            x = layer(x)
            x = nn.ReLU()(x)
        n_inputs = x.view(1, -1).size(1)
            
    output_layer = nn.Linear(n_inputs, num_classes)
    
    return EGLPNetwork(layers, output_layer, activation="relu")


class AdaptiveEGLPNetwork(EGLPNetwork):
    """EGLP Network with Adaptive Feedback Alignment.
    
    Instead of fixed B_l matrices, the feedback matrices are updated
    based on the correlation between hidden activity and output error.
    """
    def __init__(
        self,
        layers: nn.ModuleList,
        output_layer: nn.Linear,
        feedback_lr: float = 0.01,
        **kwargs
    ):
        super().__init__(layers, output_layer, **kwargs)
        self.feedback_lr = feedback_lr
        
        # Replace fixed feedback matrices with trainable (manually) ones
        self.feedback_matrices = nn.ParameterList()
        out_dim = output_layer.out_features
        for layer in layers:
            # We expect LocalPredictiveLayer or LocalLinear
            B = torch.randn(layer.out_features, out_dim) / math.sqrt(out_dim)
            self.feedback_matrices.append(nn.Parameter(B, requires_grad=False))
            
    def local_update_all(self, event: float, logits: Optional[torch.Tensor] = None, target: Optional[torch.Tensor] = None) -> None:
        if event == 0.0:
            return

        n_layers = len(self.layers)
        
        error_vector = None
        if self.use_dfa and logits is not None and target is not None:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                t_one_hot = torch.nn.functional.one_hot(target, num_classes=self.output_layer.out_features).float()
                error_vector = (probs - t_one_hot)

        for i in reversed(range(n_layers)):
            if not self.layer_event_mask[i]:
                continue
            
            scaled_event = event * (self.event_decay ** i)
            modulation = None
            
            if self.use_dfa:
                B = self.feedback_matrices[i]
                if B is not None and error_vector is not None:
                    modulation = torch.mm(error_vector, B.T)
                    
                    # --- AFA: Adaptive Feedback Alignment Update ---
                    layer = self.layers[i]
                    if hasattr(layer, 'y') and layer.y is not None:
                        # Delta B ~ y * e^T
                        y = layer.y
                        # Average over batch
                        delta_B = self.feedback_lr * event * torch.mm(y.T, error_vector) / y.size(0)
                        
                        # Weight decay for B
                        y_sq = (y ** 2).mean(dim=0, keepdim=True).T # (hidden_dim, 1)
                        decay_B = self.feedback_lr * event * y_sq * B.data
                        
                        B.data -= (delta_B - decay_B) # Minus because error is pred - target
                        
                        # Max-norm constraint on B
                        row_norms = B.data.norm(dim=1, keepdim=True)
                        max_norm = 2.0 # simplified
                        mask = row_norms > max_norm
                        if mask.any():
                            B.data[mask.squeeze(1)] *= (max_norm / row_norms[mask]).unsqueeze(1)
            
            layer = self.layers[i]
            layer.local_update(scaled_event, modulation=modulation)

def create_predictive_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    lr: float = 0.01,
    gamma_lr: float = 0.01,
    inhib_lr: float = 0.01,
    feedback_lr: float = 0.01,
    weight_clip: float = 1.0,
    max_activation: float = 5.0,
    inhib_steps: int = 2,
    event_decay: float = 1.0,
    use_anti_hebbian: bool = False,
    consolidation_strength: float = 0.0,
    use_batchnorm: bool = False,
    soft_wta_k: Optional[float] = None,
) -> AdaptiveEGLPNetwork:
    """Factory function for EGLP-V2 Predictive MLP."""
    layers = nn.ModuleList()
    batchnorms = nn.ModuleList() if use_batchnorm else None
    prev_dim = input_dim
    
    from .local_layer import LocalPredictiveLayer
    from .network import FrozenBatchNorm1d
    
    for i, hidden_dim in enumerate(hidden_dims):
        # CRITICAL FIX: Anti-Hebbian learning mathematically conflicts with Adaptive Feedback (AFA).
        # AFA adapts B to align with W, while Anti-Hebbian forces W to anti-align with B. This causes
        # endless parameter spinning. Since PCAF already uses Learned Lateral Inhibition (V) for
        # feature diversity, we strictly disable Anti-Hebbian for predictive layers.
        if use_anti_hebbian and i == 0:
            print("  [Warning] Disabled Anti-Hebbian for PCAF: Feature diversity is handled by Lateral Inhibition (V).")
        anti_heb = False
        
        layers.append(LocalPredictiveLayer(
            prev_dim, hidden_dim, 
            lr=lr, gamma_lr=gamma_lr, inhib_lr=inhib_lr,
            weight_clip=weight_clip, max_activation=max_activation,
            inhib_steps=inhib_steps,
            anti_hebbian=anti_heb,
            consolidation_strength=consolidation_strength
        ))
        
        if use_batchnorm:
            batchnorms.append(FrozenBatchNorm1d(hidden_dim))
            
        prev_dim = hidden_dim
    
    output_layer = nn.Linear(prev_dim, output_dim)
    
    return AdaptiveEGLPNetwork(
        layers, output_layer,
        feedback_lr=feedback_lr,
        event_decay=event_decay,
        use_dfa=True, 
        activation="relu",
        batchnorms=batchnorms,
        soft_wta_k=soft_wta_k
    )
