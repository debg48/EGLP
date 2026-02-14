"""EGLP Network wrapper for assembling local layers."""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union

from .local_layer import LocalLinear, LocalConv2d
from .event_controller import EventController


class EGLPNetwork(nn.Module):
    """Network wrapper that manages local layers and event broadcasting.
    
    This class:
    - Assembles LocalLinear/LocalConv2d layers for hidden layers (unsupervised)
    - Uses a standard nn.Linear for the output layer (supervised via backprop)
    - Handles forward pass under torch.no_grad() for hidden layers
    - Broadcasts event signals to all (or selected) hidden layers
    - Stores representations for CKA computation
    """
    
    def __init__(
        self,
        layers: nn.ModuleList,
        output_layer: nn.Linear,
        activation: str = "relu",
        layer_event_mask: Optional[List[bool]] = None,
    ):
        """Initialize EGLP network.
        
        Args:
            layers: ModuleList of LocalLinear/LocalConv2d layers (hidden layers).
            output_layer: Standard nn.Linear for supervised output (has gradients).
            activation: Activation function ("relu", "tanh", "sigmoid").
            layer_event_mask: Optional mask for selective event gating per layer.
                             If None, all hidden layers receive events.
        """
        super().__init__()
        self.layers = layers
        self.output_layer = output_layer
        self.activation_name = activation
        
        # Set up activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Event mask (which hidden layers receive events)
        if layer_event_mask is None:
            self.layer_event_mask = [True] * len(layers)
        else:
            if len(layer_event_mask) != len(layers):
                raise ValueError("layer_event_mask must match number of hidden layers")
            self.layer_event_mask = layer_event_mask
        
        # Storage for layer representations (for CKA)
        self.representations: Dict[int, torch.Tensor] = {}
        
        # Verify no gradients on hidden layers
        self._verify_no_grad()
    
    def _verify_no_grad(self) -> None:
        """Verify hidden layer parameters have requires_grad=False.
        
        Output layer is allowed to have gradients (supervised).
        """
        for name, param in self.layers.named_parameters():
            if param.requires_grad:
                raise RuntimeError(
                    f"Gradient leak detected! Hidden layer parameter '{name}' has requires_grad=True"
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
    
    def local_update_all(self, event: float) -> None:
        """Broadcast event signal to all hidden layers for local updates.
        
        The output layer is NOT updated here — it's updated via backprop.
        
        Args:
            event: Continuous error signal from controller.
        """
        for i, layer in enumerate(self.layers):
            # Apply layer-specific event mask
            layer_event = event if self.layer_event_mask[i] else 0.0
            layer.local_update(layer_event)
    
    def local_update_selective(self, events: List[float]) -> None:
        """Apply different event signals to different hidden layers.
        
        Args:
            events: List of event signals, one per hidden layer.
        """
        if len(events) != len(self.layers):
            raise ValueError("events list must match number of hidden layers")
        
        for layer, event in zip(self.layers, events):
            layer.local_update(event)
    
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
        
    Returns:
        EGLPNetwork instance.
    """
    # Hidden layers: local Hebbian learning (unsupervised)
    layers = nn.ModuleList()
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(LocalLinear(
            prev_dim, hidden_dim, lr=lr,
            weight_clip=weight_clip,
            max_activation=max_activation,
        ))
        prev_dim = hidden_dim
    
    # Output layer: standard nn.Linear (supervised via backprop)
    output_layer = nn.Linear(prev_dim, output_dim)
    
    return EGLPNetwork(layers, output_layer, activation=activation)


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
