"""Local layers with biologically-inspired learning rules.

These layers implement the three-factor learning rule:
    Δw = η · I(Event) · [Local_Term]

where I(Event) is a global scalar broadcast by the controller,
and Local_Term uses Oja's normalized Hebbian rule to prevent weight explosion.
"""

import torch
import torch.nn as nn
from typing import Optional


class LocalLinear(nn.Module):
    """Linear layer with local Hebbian learning (Oja's rule).
    
    CRITICAL: No gradients flow through this layer.
    - All parameters have requires_grad=False
    - Forward pass runs under torch.no_grad()
    - Updates computed manually via local_update()
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lr: float = 0.01,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr
        
        # Initialize weights (Xavier uniform)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=False  # CRITICAL: No gradients
        )
        nn.init.xavier_uniform_(self.weight)
        
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
            
            return y
    
    def local_update(self, event: int) -> None:
        """Apply three-factor learning rule.
        
        Implements Oja's normalized Hebbian rule:
            Δw = η · event · (y · x^T - y² · w)
        
        The second term (y² · w) provides normalization to prevent
        weight explosion without requiring gradient-based methods.
        
        Args:
            event: Binary signal (0 or 1) from the event controller.
                   If 0, no update is performed.
        """
        if event == 0 or self.x is None or self.y is None:
            return
        
        # Get stored activations
        x = self.x  # (batch, in_features)
        y = self.y  # (batch, out_features)
        
        # Batch-averaged Hebbian term: y · x^T
        # Shape: (out_features, in_features)
        hebbian = torch.mm(y.T, x) / x.size(0)
        
        # Oja's normalization term: y² · w
        # Prevents weight explosion by decorrelating outputs
        y_sq = (y ** 2).mean(dim=0, keepdim=True)  # (1, out_features)
        decay = y_sq.T * self.weight.data  # (out_features, in_features)
        
        # Three-factor update: Δw = η · event · (hebbian - decay)
        self.weight.data += self.lr * event * (hebbian - decay)
    
    def clear_activations(self) -> None:
        """Clear stored activations to free memory."""
        self.x = None
        self.y = None
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, lr={self.lr}"


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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr = lr
        
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
    
    def local_update(self, event: int) -> None:
        """Apply three-factor learning rule for conv layers.
        
        Uses unfolded input patches for efficient Hebbian update.
        """
        if event == 0 or self.x_unfold is None or self.y is None:
            return
        
        batch_size = self.y.size(0)
        
        # Reshape y to (batch, out_channels, L)
        y_flat = self.y.view(batch_size, self.out_channels, -1)
        
        # x_unfold is (batch, in_channels * k * k, L)
        x_flat = self.x_unfold
        
        # Hebbian term: average over batch and spatial locations
        # (out_channels, in_channels * k * k)
        hebbian = torch.bmm(y_flat, x_flat.transpose(1, 2)).mean(dim=0)
        hebbian = hebbian / y_flat.size(2)  # Normalize by spatial size
        
        # Oja's normalization
        y_sq = (y_flat ** 2).mean(dim=(0, 2), keepdim=True)  # (1, out_channels, 1)
        w_flat = self.weight.view(self.out_channels, -1)  # (out_channels, in_channels * k * k)
        decay = y_sq.squeeze(0).squeeze(-1).unsqueeze(1) * w_flat
        
        # Update
        delta_w = self.lr * event * (hebbian - decay)
        self.weight.data += delta_w.view_as(self.weight)
    
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
