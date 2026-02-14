"""Unit tests for EGLP framework.

Critical test: Verify no gradient leak in EGLP hidden layers,
while output layer correctly supports backprop.
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eglp.local_layer import LocalLinear, LocalConv2d
from eglp.event_controller import ThresholdController, FixedRateController, AlwaysOnController
from eglp.network import EGLPNetwork, create_mlp


class TestNoGradientLeak:
    """Critical tests to ensure no gradients flow through EGLP hidden layers."""
    
    def test_local_linear_no_requires_grad(self):
        """Verify LocalLinear parameters have requires_grad=False."""
        layer = LocalLinear(784, 256, lr=0.01)
        
        for name, param in layer.named_parameters():
            assert param.requires_grad is False, \
                f"Gradient leak detected! Parameter '{name}' has requires_grad=True"
    
    def test_local_conv2d_no_requires_grad(self):
        """Verify LocalConv2d parameters have requires_grad=False."""
        layer = LocalConv2d(1, 32, kernel_size=3, padding=1, lr=0.01)
        
        for name, param in layer.named_parameters():
            assert param.requires_grad is False, \
                f"Gradient leak detected! Parameter '{name}' has requires_grad=True"
    
    def test_eglp_hidden_layers_no_requires_grad(self):
        """Verify EGLP hidden layers have no gradients."""
        network = create_mlp(
            input_dim=784,
            hidden_dims=[256, 128],
            output_dim=10,
            lr=0.01,
        )
        
        # Hidden layers should have no gradients
        for name, param in network.layers.named_parameters():
            assert param.requires_grad is False, \
                f"Gradient leak detected! Hidden layer parameter '{name}' has requires_grad=True"
    
    def test_output_layer_has_gradients(self):
        """Verify output layer has requires_grad=True for supervised learning."""
        network = create_mlp(
            input_dim=784,
            hidden_dims=[256, 128],
            output_dim=10,
            lr=0.01,
        )
        
        # Output layer should have gradients
        for name, param in network.output_layer.named_parameters():
            assert param.requires_grad is True, \
                f"Output layer parameter '{name}' should have requires_grad=True"
    
    def test_forward_pass_hidden_no_grad(self):
        """Verify hidden layer outputs have no gradient fn."""
        layer = LocalLinear(784, 256)
        x = torch.randn(32, 784)
        
        y = layer(x)
        
        # Output should not have gradient fn
        assert y.grad_fn is None, "Output has gradient function - possible gradient leak!"
    
    def test_forward_pass_output_has_grad(self):
        """Verify output layer produces tensors with gradient fn."""
        network = create_mlp(784, [256, 128], 10)
        x = torch.randn(32, 784)
        
        output = network(x)
        
        # Output should have gradient fn (from output_layer)
        assert output.grad_fn is not None, "Output should have gradient function for backprop!"
    
    def test_local_update_modifies_weights(self):
        """Verify local_update actually modifies weights."""
        layer = LocalLinear(784, 256, lr=0.1)
        
        # Store original weights
        original_weights = layer.weight.data.clone()
        
        # Forward pass
        x = torch.randn(32, 784)
        _ = layer(x)
        
        # Local update with continuous error signal
        layer.local_update(event=1.0)
        
        # Weights should be different
        assert not torch.allclose(original_weights, layer.weight.data), \
            "Weights unchanged after local_update!"
    
    def test_local_update_no_change_when_event_zero(self):
        """Verify local_update does nothing when event=0.0."""
        layer = LocalLinear(784, 256, lr=0.1)
        
        x = torch.randn(32, 784)
        _ = layer(x)
        
        original_weights = layer.weight.data.clone()
        layer.local_update(event=0.0)
        
        assert torch.allclose(original_weights, layer.weight.data), \
            "Weights changed when event=0.0!"
    
    def test_hybrid_training_loop(self):
        """Verify complete hybrid training loop: backprop output + local hidden."""
        network = create_mlp(784, [256, 128], 10, lr=0.01)
        controller = ThresholdController(budget=100)
        optimizer = torch.optim.Adam(network.get_output_params(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Store initial hidden weights
        initial_hidden = [l.weight.data.clone() for l in network.layers]
        initial_output = network.output_layer.weight.data.clone()
        
        # Simulate training
        for _ in range(10):
            x = torch.randn(32, 784)
            target = torch.randint(0, 10, (32,))
            
            optimizer.zero_grad()
            output = network(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            error_signal = controller.should_trigger(loss.item())
            network.local_update_all(error_signal)
            network.clear_activations()
        
        # Verify hidden layers still have no gradients
        for name, param in network.layers.named_parameters():
            assert param.requires_grad is False, \
                f"Gradient leak after training! Parameter '{name}'"
        
        # Output layer should have been updated by backprop
        assert not torch.allclose(initial_output, network.output_layer.weight.data), \
            "Output layer weights unchanged - backprop not working!"


class TestLocalLayer:
    """Tests for LocalLinear and LocalConv2d."""
    
    def test_local_linear_forward_shape(self):
        """Test LocalLinear produces correct output shape."""
        layer = LocalLinear(784, 256)
        x = torch.randn(32, 784)
        y = layer(x)
        
        assert y.shape == (32, 256)
    
    def test_local_linear_stores_activations(self):
        """Test LocalLinear stores activations for update."""
        layer = LocalLinear(784, 256)
        x = torch.randn(32, 784)
        
        _ = layer(x)
        
        assert layer.x is not None
        assert layer.y is not None
        assert layer.x.shape == (32, 784)
        assert layer.y.shape == (32, 256)
    
    def test_local_linear_clear_activations(self):
        """Test clear_activations frees memory."""
        layer = LocalLinear(784, 256)
        x = torch.randn(32, 784)
        
        _ = layer(x)
        layer.clear_activations()
        
        assert layer.x is None
        assert layer.y is None
    
    def test_local_conv2d_forward_shape(self):
        """Test LocalConv2d produces correct output shape."""
        layer = LocalConv2d(1, 32, kernel_size=3, padding=1)
        x = torch.randn(16, 1, 28, 28)
        y = layer(x)
        
        assert y.shape == (16, 32, 28, 28)
    
    def test_oja_rule_weight_stability(self):
        """Test Oja's rule with clipping prevents weight explosion."""
        layer = LocalLinear(100, 50, lr=0.1, weight_clip=1.0)
        
        initial_norm = layer.weight.data.norm()
        
        # Many updates
        for _ in range(100):
            x = torch.randn(32, 100)
            _ = layer(x)
            layer.local_update(event=1.0)
        
        final_norm = layer.weight.data.norm()
        
        # With clipping, weight norm should be bounded
        assert torch.isfinite(layer.weight.data).all(), \
            "Weights contain NaN or Inf!"
        assert final_norm < initial_norm * 10, \
            f"Weights exploded! Initial: {initial_norm:.2f}, Final: {final_norm:.2f}"
    
    def test_weight_clipping(self):
        """Test that weight clipping is enforced."""
        clip_val = 0.5
        layer = LocalLinear(100, 50, lr=0.5, weight_clip=clip_val)
        
        # Many aggressive updates
        for _ in range(50):
            x = torch.randn(32, 100)
            _ = layer(x)
            layer.local_update(event=2.0)  # Strong error signal
        
        assert layer.weight.data.max() <= clip_val, \
            f"Weights exceeded clip value {clip_val}: max={layer.weight.data.max()}"
        assert layer.weight.data.min() >= -clip_val, \
            f"Weights below -clip value {clip_val}: min={layer.weight.data.min()}"
    
    def test_error_signal_scales_update(self):
        """Test that larger error signal produces larger weight change."""
        layer_small = LocalLinear(100, 50, lr=0.01, weight_clip=10.0)
        layer_large = LocalLinear(100, 50, lr=0.01, weight_clip=10.0)
        
        # Same initialization
        layer_large.weight.data.copy_(layer_small.weight.data)
        if layer_large.bias is not None:
            layer_large.bias.data.copy_(layer_small.bias.data)
        
        x = torch.randn(32, 100)
        
        _ = layer_small(x)
        _ = layer_large(x)
        
        orig = layer_small.weight.data.clone()
        
        layer_small.local_update(event=0.1)
        layer_large.local_update(event=1.0)
        
        delta_small = (layer_small.weight.data - orig).norm()
        delta_large = (layer_large.weight.data - orig).norm()
        
        assert delta_large > delta_small, \
            f"Larger error signal should produce larger update: {delta_large} vs {delta_small}"


class TestEventController:
    """Tests for event controllers (now returning continuous floats)."""
    
    def test_threshold_controller_warmup(self):
        """Test ThresholdController doesn't trigger during warmup."""
        controller = ThresholdController(budget=100, warmup_steps=10)
        
        for i in range(10):
            event = controller.should_trigger(1.0)
            assert event == 0.0, f"Triggered during warmup at step {i}"
    
    def test_threshold_controller_budget(self):
        """Test ThresholdController respects budget."""
        controller = ThresholdController(budget=5, warmup_steps=0)
        
        # Fill history with low loss
        for _ in range(20):
            controller.should_trigger(1.0)
        
        # Trigger events with high loss
        events = 0
        for _ in range(100):
            signal = controller.should_trigger(10.0)
            if signal > 0:
                events += 1
        
        assert events <= 5, f"Budget exceeded! Triggered {events} events"
    
    def test_threshold_returns_continuous_signal(self):
        """Test ThresholdController returns float, not just 0/1."""
        controller = ThresholdController(budget=100, warmup_steps=0, threshold_factor=1.0)
        
        # Fill history
        for _ in range(20):
            controller.should_trigger(1.0)
        
        # Trigger with higher loss — should get a signal > 0
        signal = controller.should_trigger(5.0)
        assert signal > 0.0, "Should trigger with high loss"
        assert isinstance(signal, float), "Signal should be a float"
    
    def test_fixed_rate_controller_rate(self):
        """Test FixedRateController approximates expected rate."""
        controller = FixedRateController(rate=0.5, seed=42)
        
        events = sum(1 for _ in range(1000) if controller.should_trigger(1.0) > 0)
        
        # Should be roughly 50% (with some variance)
        assert 400 < events < 600, f"Rate off: expected ~500, got {events}"
    
    def test_always_on_controller(self):
        """Test AlwaysOnController always triggers."""
        controller = AlwaysOnController()
        
        for _ in range(100):
            signal = controller.should_trigger(1.0)
            assert signal > 0.0, "AlwaysOnController should always return positive signal"
    
    def test_controller_reset(self):
        """Test controller reset restores state."""
        controller = ThresholdController(budget=10, warmup_steps=0)
        
        # Use budget
        for _ in range(20):
            controller.should_trigger(10.0)
        
        # Reset
        controller.reset()
        
        assert controller.budget == 10
        assert controller.events_triggered == 0


class TestEGLPNetwork:
    """Tests for EGLPNetwork with supervised output."""
    
    def test_network_forward(self):
        """Test network forward pass."""
        network = create_mlp(784, [256, 128], 10)
        x = torch.randn(32, 784)
        
        y = network(x)
        
        assert y.shape == (32, 10)
    
    def test_network_stores_representations(self):
        """Test network stores layer representations."""
        network = create_mlp(784, [256, 128], 10)
        x = torch.randn(32, 784)
        
        _ = network(x, store_representations=True)
        
        reps = network.get_representations()
        assert len(reps) == 3  # 2 hidden + 1 output
    
    def test_network_local_update_all(self):
        """Test broadcasting updates to all hidden layers."""
        network = create_mlp(784, [256, 128], 10, lr=0.1)
        
        # Store original hidden weights
        original = [layer.weight.data.clone() for layer in network.layers]
        
        x = torch.randn(32, 784)
        _ = network(x)
        network.local_update_all(event=1.0)
        
        # All hidden layers should be updated
        for i, layer in enumerate(network.layers):
            assert not torch.allclose(original[i], layer.weight.data), \
                f"Layer {i} weights unchanged!"
    
    def test_layer_event_mask(self):
        """Test selective event gating."""
        network = create_mlp(784, [256, 128], 10, lr=0.1)
        network.set_layer_event_mask([True, False])  # Only 2 hidden layers
        
        original = [layer.weight.data.clone() for layer in network.layers]
        
        x = torch.randn(32, 784)
        _ = network(x)
        network.local_update_all(event=1.0)
        
        # Layer 0 should update, layer 1 should not
        assert not torch.allclose(original[0], network.layers[0].weight.data)
        assert torch.allclose(original[1], network.layers[1].weight.data)
    
    def test_get_output_params(self):
        """Test that get_output_params returns only output layer params."""
        network = create_mlp(784, [256, 128], 10)
        
        output_params = list(network.get_output_params())
        assert len(output_params) == 2  # weight + bias
        
        for p in output_params:
            assert p.requires_grad is True
    
    def test_backward_only_affects_output(self):
        """Test that loss.backward() only creates gradients for output layer."""
        network = create_mlp(784, [256, 128], 10)
        x = torch.randn(32, 784)
        target = torch.randint(0, 10, (32,))
        
        output = network(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Hidden layers should have no gradients
        for name, param in network.layers.named_parameters():
            assert param.grad is None, \
                f"Hidden layer '{name}' has gradients after backward!"
        
        # Output layer should have gradients
        assert network.output_layer.weight.grad is not None, \
            "Output layer should have gradients after backward!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
