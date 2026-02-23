"""Local layers with biologically-inspired learning rules.

These layers implement the three-factor learning rule:
    Δw = η · error_signal · [Local_Term]

where error_signal is a continuous scalar from the controller
(normalized surprise), and Local_Term uses Oja's normalized
Hebbian rule to prevent weight explosion.

Additional features:
- Anti-Hebbian mode: extracts minor components for feature diversity
- Synaptic consolidation: protects important weights from overwriting
- Homeostatic scaling: rescales dead/saturated neurons
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class LocalLinear(nn.Module):
    """Linear layer with local Hebbian learning (Oja's rule).
    
    CRITICAL: No gradients flow through this layer.
    - All parameters have requires_grad=False
    - Forward pass runs under torch.no_grad()
    - Updates computed manually via local_update()
    
    Additional features:
    - anti_hebbian: If True, uses anti-Oja rule to extract minor components
    - consolidation_strength: If > 0, protects important weights via EWC-lite
    - homeostatic scaling: Tracks firing rates and rescales dead/saturated neurons
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lr: float = 0.01,
        bias: bool = True,
        weight_clip: float = 1.0,
        max_activation: float = 5.0,
        decay_factor: float = 1.0,
        anti_hebbian: bool = False,
        consolidation_strength: float = 0.0,
        trace_decay: float = 0.9,
        trace_lambda: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr
        self.weight_clip = weight_clip
        self.max_activation = max_activation
        self.decay_factor = decay_factor
        self.anti_hebbian = anti_hebbian
        self.consolidation_strength = consolidation_strength
        
        # Initialize weights (Xavier uniform)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=False  # CRITICAL: No gradients
        )
        nn.init.xavier_uniform_(self.weight)
        
        # Store initial row norm for max-norm constraint in local_update
        self._init_row_norm = self.weight.data.norm(dim=1).mean().item()
        
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features),
                requires_grad=False
            )
        else:
            self.register_parameter('bias', None)
        
        # Storage for activations (used in local_update)
        self.x: Optional[torch.Tensor] = None  # Input
        self.y: Optional[torch.Tensor] = None  # Output
        
        # --- Synaptic Consolidation (EWC-lite) ---
        # Importance scores per weight, accumulated from |y·x| usage
        self.register_buffer(
            'importance', torch.zeros(out_features, in_features)
        )
        # Snapshot of weights at end of last epoch
        self.register_buffer(
            'weight_snapshot', torch.zeros(out_features, in_features)
        )
        
        # --- Multi-Timescale Plasticity (Eligibility Traces) ---
        self.register_buffer(
            'trace', torch.zeros(out_features, in_features)
        )
        self.trace_decay = trace_decay
        self.trace_lambda = trace_lambda
        
        # --- Homeostatic Scaling ---
        # Exponential moving average of neuron firing rates
        self.register_buffer(
            'firing_rate_ema', torch.ones(out_features) * 0.5  # Start at balanced
        )
        self._ema_decay = 0.99  # Smoothing factor for EMA
        self._homeostasis_target = 0.5  # Target mean activation (normalized)
        self._homeostasis_step = 0  # Step counter
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation storage.
        
        Args:
            x: Input tensor of shape (batch, in_features)
            
        Returns:
            Output tensor of shape (batch, out_features)
        """
        with torch.no_grad():
            # Store input for local update
            self.x = x.detach()
            
            # Linear transformation
            y = torch.mm(x, self.weight.T)
            if self.bias is not None:
                y = y + self.bias
            
            # Store output for local update
            self.y = y.detach()
            
            # Update firing rate EMA (fraction of neurons that are active after ReLU)
            # This runs BEFORE activation in the network, so we check raw pre-activation
            active_fraction = (y > 0).float().mean(dim=0)  # Per-neuron fraction
            self.firing_rate_ema = (
                self._ema_decay * self.firing_rate_ema
                + (1 - self._ema_decay) * active_fraction
            )
            self._homeostasis_step += 1
            
            return y
    
    def local_update(self, event: float, modulation: torch.Tensor = None) -> None:
        """Apply three-factor learning rule with error modulation.
        
        Supports signed feedback from DFA.
        Rule: Δw = η · event · (modulation · (y·x^T - y²·w))
        
        Args:
            event: Continuous global gating signal (scalar).
            modulation: Optional modulation signal.
                       Can be (out_features,) for global modulation
                       OR (batch, out_features) for per-sample DFA.
        """
        if event == 0.0 or self.x is None or self.y is None:
            return
        
        # Get stored activations with clamping
        x = self.x  # (batch, in_features)
        y = torch.clamp(self.y, -self.max_activation, self.max_activation)  # (batch, out_features)
        
        # Handle modulation
        # If modulation is provided, we merge it into the Hebbian and Decay terms
        # efficiently to avoid large tensor expansions.
        
        if modulation is not None:
            # Ensure modulation matches batch dimensions if provided as (out,)
            if modulation.dim() == 1:
                modulation = modulation.unsqueeze(0)  # (1, out)
            
            # modulation is (batch, out) or broadcasts to it
            
            # 1. Modulated Hebbian Term: (m·y) · x^T
            # We modulate the activity y directly before correlation
            y_mod = y * modulation
            hebbian = torch.mm(y_mod.T, x) / x.size(0)
            
            # 2. Modulated Decay Term: m · y² · w
            # We need mean(m · y²) over the batch
            # modulation * y^2 -> (batch, out) -> mean -> (1, out)
            decay_scaler = (modulation * (y ** 2)).mean(dim=0, keepdim=True).T # (out, 1)
            decay = self.decay_factor * decay_scaler * self.weight.data
            
        else:
            # Standard positive Hebbian (unmodulated or implicit +1)
            hebbian = torch.mm(y.T, x) / x.size(0)
            y_sq = (y ** 2).mean(dim=0, keepdim=True)
            decay = self.decay_factor * y_sq.T * self.weight.data

        # Apply anti-Hebbian legacy flag (if we still use it alongside DFA)
        if self.anti_hebbian:
            hebbian = -hebbian
            # If modulation was used, this flips the sign of the ALREADY modulated term
        
        # Core update
        delta_w = self.lr * event * (hebbian - decay)
        
        # --- Synaptic Consolidation penalty ---
        if self.consolidation_strength > 0:
            # Penalize deviation from snapshot
            consolidation_penalty = (
                self.consolidation_strength
                * self.importance
                * (self.weight.data - self.weight_snapshot)
            )
            delta_w -= self.lr * consolidation_penalty
            
            # Update importance
            importance_update = torch.abs(hebbian)
            self.importance = 0.999 * self.importance + 0.001 * importance_update
        
        # Apply update (multi-timescale)
        self.trace = self.trace_decay * self.trace + delta_w
        self.weight.data += delta_w + self.trace_lambda * self.trace
        
        # Max-norm constraint
        row_norms = self.weight.data.norm(dim=1, keepdim=True)
        max_norm = self._init_row_norm * 2.0
        mask = row_norms > max_norm
        if mask.any():
            self.weight.data[mask.squeeze(1)] *= (max_norm / row_norms[mask]).unsqueeze(1)
    
    def snapshot_weights(self) -> None:
        """Take a snapshot of current weights for consolidation.
        
        Should be called at end of each epoch.
        """
        self.weight_snapshot.copy_(self.weight.data)
    
    def homeostatic_scale(self) -> None:
        """Rescale weights of dead/saturated neurons toward target firing rate.
        
        Neurons that fire too little get amplified incoming weights.
        Neurons that fire too much get dampened incoming weights.
        
        Based on biological synaptic scaling (Turrigiano, 2008).
        """
        with torch.no_grad():
            # Compute scaling factor per neuron
            # target / (actual + eps) — dead neurons (low rate) get scaled up
            scale = self._homeostasis_target / (self.firing_rate_ema + 1e-6)
            
            # Clamp to prevent extreme rescaling
            scale = scale.clamp(0.5, 2.0)
            
            # Apply to incoming weights of each neuron (row of weight matrix)
            self.weight.data *= scale.unsqueeze(1)
            
            # Max-norm constraint: prevent rows from growing beyond initial norm
            row_norms = self.weight.data.norm(dim=1, keepdim=True)
            max_norm = self._init_row_norm
            mask = row_norms > max_norm
            if mask.any():
                self.weight.data[mask.squeeze(1)] *= (max_norm / row_norms[mask]).unsqueeze(1)
    
    def clear_activations(self) -> None:
        """Clear stored activations to free memory."""
        self.x = None
        self.y = None
    
    def extra_repr(self) -> str:
        parts = [f"in_features={self.in_features}", f"out_features={self.out_features}", f"lr={self.lr}"]
        if self.anti_hebbian:
            parts.append("anti_hebbian=True")
        if self.consolidation_strength > 0:
            parts.append(f"consolidation={self.consolidation_strength}")
        return ", ".join(parts)


class LocalConv2d(nn.Module):
    """Conv2d layer with local Hebbian learning (Oja's rule).
    
    CRITICAL: No gradients flow through this layer.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        lr: float = 0.01,
        bias: bool = True,
        weight_clip: float = 1.0,
        max_activation: float = 5.0,
        decay_factor: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr = lr
        self.weight_clip = weight_clip
        self.max_activation = max_activation
        self.decay_factor = decay_factor
        
        # Initialize weights
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size),
            requires_grad=False
        )
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')
        
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels),
                requires_grad=False
            )
        else:
            self.register_parameter('bias', None)
        
        # Storage for activations
        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.x_unfold: Optional[torch.Tensor] = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation storage.
        
        Args:
            x: Input tensor of shape (batch, in_channels, H, W)
            
        Returns:
            Output tensor of shape (batch, out_channels, H', W')
        """
        with torch.no_grad():
            self.x = x.detach()
            
            # Use unfold to extract patches for efficient local learning
            # Shape: (batch, in_channels * k * k, L) where L = H' * W'
            self.x_unfold = torch.nn.functional.unfold(
                x, 
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding
            ).detach()
            
            # Standard conv2d forward
            y = torch.nn.functional.conv2d(
                x, self.weight, self.bias,
                stride=self.stride,
                padding=self.padding
            )
            
            self.y = y.detach()
            return y
    
    def local_update(self, event: float) -> None:
        """Apply three-factor learning rule for conv layers.
        
        Uses unfolded input patches for efficient Hebbian update
        with continuous error signal modulation.
        """
        if event == 0.0 or self.x_unfold is None or self.y is None:
            return
        
        batch_size = self.y.size(0)
        
        # Reshape y to (batch, out_channels, L) with clamping
        y_flat = torch.clamp(
            self.y.view(batch_size, self.out_channels, -1),
            -self.max_activation, self.max_activation
        )
        
        # x_unfold is (batch, in_channels * k * k, L)
        x_flat = self.x_unfold
        
        # Hebbian term: average over batch and spatial locations
        # (out_channels, in_channels * k * k)
        hebbian = torch.bmm(y_flat, x_flat.transpose(1, 2)).mean(dim=0)
        hebbian = hebbian / y_flat.size(2)  # Normalize by spatial size
        
        # Oja's normalization
        y_sq = (y_flat ** 2).mean(dim=(0, 2), keepdim=True)  # (1, out_channels, 1)
        w_flat = self.weight.view(self.out_channels, -1)  # (out_channels, in_channels * k * k)
        decay = self.decay_factor * y_sq.squeeze(0).squeeze(-1).unsqueeze(1) * w_flat
        
        # Update with continuous error signal
        delta_w = self.lr * event * (hebbian - decay)
        self.weight.data += delta_w.view_as(self.weight)
        
        # Weight clipping for numerical stability
        self.weight.data.clamp_(-self.weight_clip, self.weight_clip)
    
    def snapshot_weights(self) -> None:
        """No-op for conv layers (consolidation not implemented for conv)."""
        pass
    
    def homeostatic_scale(self) -> None:
        """No-op for conv layers (homeostasis not implemented for conv)."""
        pass
    
    def clear_activations(self) -> None:
        """Clear stored activations to free memory."""
        self.x = None
        self.y = None
        self.x_unfold = None
    
    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, lr={self.lr}"
        )

class LocalPredictiveLayer(nn.Module):
    """Predictive Coding Layer with Adaptive Feedback and Learned Inhibition.
    
    Features:
    - W: Forward feature extraction
    - G: Generative prediction of lower layer
    - V: Learned lateral inhibition
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lr: float = 0.01,
        gamma_lr: float = 0.01,
        inhib_lr: float = 0.01,
        weight_clip: float = 1.0,
        max_activation: float = 5.0,
        inhib_steps: int = 2,
        anti_hebbian: bool = False,
        consolidation_strength: float = 0.0,
        trace_decay: float = 0.9,
        trace_lambda: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.lr_W = lr
        self.lr_G = gamma_lr
        self.lr_V = inhib_lr
        
        self.weight_clip = weight_clip
        self.max_activation = max_activation
        self.inhib_steps = inhib_steps
        self.anti_hebbian = anti_hebbian
        self.consolidation_strength = consolidation_strength
        
        # W: Forward weights
        self.W = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.xavier_uniform_(self.W)
        self._init_row_norm_W = self.W.data.norm(dim=1).mean().item()
        
        # G: Generative weights (predicting input)
        self.G = nn.Parameter(torch.empty(in_features, out_features), requires_grad=False)
        nn.init.xavier_uniform_(self.G)
        self._init_row_norm_G = self.G.data.norm(dim=1).mean().item()
        
        # V: Lateral inhibition (out -> out)
        self.V = nn.Parameter(torch.zeros(out_features, out_features), requires_grad=False)
        # V should always have 0 on the diagonal (no self-inhibition)
        self.V.data.fill_diagonal_(0.0)
        
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        
        # Storage
        self.x = None
        self.y = None
        self.x_hat = None
        
        # --- Synaptic Consolidation (EWC-lite) ---
        self.register_buffer('importance_W', torch.zeros(out_features, in_features))
        self.register_buffer('importance_G', torch.zeros(in_features, out_features))
        self.register_buffer('importance_V', torch.zeros(out_features, out_features))
        
        self.register_buffer('snapshot_W', torch.zeros(out_features, in_features))
        self.register_buffer('snapshot_G', torch.zeros(in_features, out_features))
        self.register_buffer('snapshot_V', torch.zeros(out_features, out_features))
        
        # --- Multi-Timescale Plasticity (Eligibility Traces) ---
        self.register_buffer('trace_W', torch.zeros(out_features, in_features))
        self.trace_decay = trace_decay
        self.trace_lambda = trace_lambda
        
        # --- Homeostatic Scaling ---
        self.register_buffer('firing_rate_ema', torch.ones(out_features) * 0.5)
        self._ema_decay = 0.99
        self._homeostasis_target = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.x = x.detach()
            
            # Initial forward drive
            drive = torch.mm(x, self.W.T) + self.bias
            
            # Settle with lateral inhibition
            y = torch.relu(drive)
            for _ in range(self.inhib_steps):
                # V is strictly positive inhibitory weights, we subtract it
                inhibition = torch.mm(y, self.V.T)
                y = torch.relu(drive - inhibition)
            
            self.y = y.detach()
            
            # Predict lower layer
            self.x_hat = torch.mm(self.y, self.G.T).detach()
            
            # Update firing rate EMA
            active_fraction = (drive > 0).float().mean(dim=0)
            self.firing_rate_ema = (
                self._ema_decay * self.firing_rate_ema
                + (1 - self._ema_decay) * active_fraction
            )
            
            return self.y
            
    def local_update(self, event: float, modulation: torch.Tensor = None) -> None:
        if event == 0.0 or self.x is None or self.y is None:
            return
            
        x = self.x
        y = torch.clamp(self.y, 0, self.max_activation)
        x_hat = self.x_hat
        
        # 1. Update Generative weights G
        # Delta G ~ (x - x_hat) * y^T
        pred_error = x - x_hat
        # Hebbian term for generative model
        delta_G = self.lr_G * event * torch.mm(pred_error.T, y) / x.size(0)
        # Weight decay for G (normalization)
        y_sq = (y ** 2).mean(dim=0, keepdim=True)
        decay_G = self.lr_G * event * y_sq * self.G.data
        self.G.data += delta_G - decay_G
        
        # 2. Update Forward weights W
        # Standard DFA modulation
        if modulation is not None:
            if modulation.dim() == 1:
                modulation = modulation.unsqueeze(0)
            
            # We add a predictive consistency term to modulation:
            # The network shouldn't extract features it can't predict
            # We don't have gradients, so this is an approximation: 
            # How well did G predict this sample?
            # Let's keep it simple: just standard DFA modulation for now to 
            # align W with the global global objective
            
            y_mod = y * modulation
            hebbian_W = torch.mm(y_mod.T, x) / x.size(0)
            
            decay_scaler = (modulation * (y ** 2)).mean(dim=0, keepdim=True).T
            decay_W = decay_scaler * self.W.data
            
        else:
            hebbian_W = torch.mm(y.T, x) / x.size(0)
            y_sq_W = (y ** 2).mean(dim=0, keepdim=True)
            decay_W = y_sq_W.T * self.W.data
            
        # Add predictive term to W update (features should match top-down expectations if provided)
        # Using simple Oja for now on top of DFA
        if self.anti_hebbian:
            hebbian_W = -hebbian_W
            
        delta_W = self.lr_W * event * (hebbian_W - decay_W)
        
        # 3. Update Inhibitory weights V
        # STRUCTURED INHIBITION: Cosine similarity of receptive fields
        # Instead of just noisy co-activation, neurons structurally inhibit those
        # that look for similar features.
        
        # Calculate cosine similarity between rows of W
        W_norm = torch.nn.functional.normalize(self.W.data, p=2, dim=1)
        cos_sim = torch.mm(W_norm, W_norm.T)
        
        # We only want positive similarity (if they are opposite, they don't compete for the same feature)
        # We use standard corr to scale it with activity so dead neurons don't inhibit
        corr = torch.mm(y.T, y) / x.size(0)
        structural_corr = torch.relu(cos_sim) * corr
        
        # But inhibit V from exploding (decay)
        decay_V = self.V.data * 0.01  
        
        delta_V = self.lr_V * event * (structural_corr - decay_V)
        
        # --- Synaptic Consolidation ---
        if self.consolidation_strength > 0:
            pen_W = self.consolidation_strength * self.importance_W * (self.W.data - self.snapshot_W)
            pen_G = self.consolidation_strength * self.importance_G * (self.G.data - self.snapshot_G)
            pen_V = self.consolidation_strength * self.importance_V * (self.V.data - self.snapshot_V)
            
            delta_W -= self.lr_W * pen_W
            delta_G -= self.lr_G * pen_G
            delta_V -= self.lr_V * pen_V
            
            self.importance_W = 0.999 * self.importance_W + 0.001 * torch.abs(hebbian_W)
            self.importance_G = 0.999 * self.importance_G + 0.001 * torch.abs(torch.mm(pred_error.T, y) / x.size(0))
            self.importance_V = 0.999 * self.importance_V + 0.001 * torch.abs(corr)
            
        # Multi-timescale plasticity for W
        self.trace_W = self.trace_decay * self.trace_W + delta_W
        self.W.data += delta_W + self.trace_lambda * self.trace_W
        
        self.G.data += delta_G - decay_G
        self.V.data += delta_V
        
        # enforce constraints on V (positive, zero diag)
        self.V.data.clamp_(min=0.0)
        self.V.data.fill_diagonal_(0.0)
        
        # Max-norm constraint on W and G
        self._apply_max_norm(self.W, self._init_row_norm_W)
        self._apply_max_norm(self.G, self._init_row_norm_G)

    def _apply_max_norm(self, tensor, initial_norm):
        row_norms = tensor.data.norm(dim=1, keepdim=True)
        max_norm = initial_norm * 2.0
        mask = row_norms > max_norm
        if mask.any():
            tensor.data[mask.squeeze(1)] *= (max_norm / row_norms[mask]).unsqueeze(1)
            
    def clear_activations(self) -> None:
        self.x = None
        self.y = None
        self.x_hat = None

    def homeostatic_scale(self) -> None:
        with torch.no_grad():
            scale = self._homeostasis_target / (self.firing_rate_ema + 1e-6)
            scale = scale.clamp(0.5, 2.0)
            
            self.W.data *= scale.unsqueeze(1)
            # Reapply norm constraints
            self._apply_max_norm(self.W, self._init_row_norm_W)
        
    def snapshot_weights(self) -> None:
        self.snapshot_W.copy_(self.W.data)
        self.snapshot_G.copy_(self.G.data)
        self.snapshot_V.copy_(self.V.data)
        
