"""EGLP Network wrapper for assembling local layers."""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union

from .local_layer import LocalLinear, LocalConv2d
from .event_controller import EventController


class EGLPNetwork(nn.Module):
    """Network wrapper that manages local layers and event broadcasting.
    
    This class:
    - Assembles LocalLinear/LocalConv2d layers into a network
    - Handles forward pass under torch.no_grad()
    - Broadcasts event signals to all (or selected) layers
    - Stores representations for CKA computation
    """
    
    def __init__(
        self,
        layers: nn.ModuleList,
        activation: str = "relu",
        layer_event_mask: Optional[List[bool]] = None,
    ):
        """Initialize EGLP network.
        
        Args:
            layers: ModuleList of LocalLinear/LocalConv2d layers.
            activation: Activation function ("relu", "tanh", "sigmoid").
            layer_event_mask: Optional mask for selective event gating per layer.
                             If None, all layers receive events.
        """
        super().__init__()
        self.layers = layers
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
        
        # Event mask (which layers receive events)
        if layer_event_mask is None:
            self.layer_event_mask = [True] * len(layers)
        else:
            if len(layer_event_mask) != len(layers):
                raise ValueError("layer_event_mask must match number of layers")
            self.layer_event_mask = layer_event_mask
        
        # Storage for layer representations (for CKA)
        self.representations: Dict[int, torch.Tensor] = {}
        
        # Verify no gradients
        self._verify_no_grad()
    
    def _verify_no_grad(self) -> None:
        """Verify all parameters have requires_grad=False."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                raise RuntimeError(
                    f"Gradient leak detected! Parameter '{name}' has requires_grad=True"
                )
    
    def forward(self, x: torch.Tensor, store_representations: bool = False) -> torch.Tensor:
        """Forward pass through all layers.
        
        Args:
            x: Input tensor.
            store_representations: If True, store each layer's output for CKA.
            
        Returns:
            Output tensor (logits for classification).
        """
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                x = layer(x)
                
                # Apply activation for all but last layer
                if i < len(self.layers) - 1:
                    x = self.activation(x)
                
                # Store representation if requested
                if store_representations:
                    self.representations[i] = x.detach().clone()
            
            return x
    
    def local_update_all(self, event: int) -> None:
        """Broadcast event signal to all layers for local updates.
        
        Args:
            event: Binary event signal (0 or 1) from controller.
        """
        for i, layer in enumerate(self.layers):
            # Apply layer-specific event mask
            layer_event = event if self.layer_event_mask[i] else 0
            layer.local_update(layer_event)
    
    def local_update_selective(self, events: List[int]) -> None:
        """Apply different event signals to different layers.
        
        Args:
            events: List of event signals, one per layer.
        """
        if len(events) != len(self.layers):
            raise ValueError("events list must match number of layers")
        
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
        """Update which layers receive event signals.
        
        Useful for layer sensitivity experiments.
        """
        if len(mask) != len(self.layers):
            raise ValueError("mask must match number of layers")
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
) -> EGLPNetwork:
    """Factory function to create an MLP with local layers.
    
    Args:
        input_dim: Input feature dimension.
        hidden_dims: List of hidden layer dimensions.
        output_dim: Output dimension (number of classes).
        lr: Learning rate for local updates.
        activation: Activation function name.
        
    Returns:
        EGLPNetwork instance.
    """
    layers = nn.ModuleList()
    
    # Input layer
    prev_dim = input_dim
    
    # Hidden layers
    for hidden_dim in hidden_dims:
        layers.append(LocalLinear(prev_dim, hidden_dim, lr=lr))
        prev_dim = hidden_dim
    
    # Output layer
    layers.append(LocalLinear(prev_dim, output_dim, lr=lr))
    
    return EGLPNetwork(layers, activation=activation)


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
        # Note: For CNN, we'll need to handle pooling and flattening separately
        # This is a simplified version; extend as needed
    ])
    
    return EGLPNetwork(layers, activation="relu")
