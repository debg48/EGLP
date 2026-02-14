"""Unit tests for EGLP framework.

Critical test: Verify no gradient leak in EGLP training.
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
    """Critical tests to ensure no gradients flow through EGLP layers."""
    
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
    
    def test_eglp_network_no_requires_grad(self):
        """Verify full EGLP network has no gradients."""
        network = create_mlp(
            input_dim=784,
            hidden_dims=[256, 128],
            output_dim=10,
            lr=0.01,
        )
        
        for name, param in network.named_parameters():
            assert param.requires_grad is False, \
                f"Gradient leak detected! Parameter '{name}' has requires_grad=True"
    
    def test_forward_pass_no_grad_context(self):
        """Verify forward pass runs under no_grad context."""
        layer = LocalLinear(784, 256)
        x = torch.randn(32, 784)
        
        y = layer(x)
        
        # Output should not have gradient fn
        assert y.grad_fn is None, "Output has gradient function - possible gradient leak!"
    
    def test_local_update_modifies_weights(self):
        """Verify local_update actually modifies weights."""
        layer = LocalLinear(784, 256, lr=0.1)
        
        # Store original weights
        original_weights = layer.weight.data.clone()
        
        # Forward pass
        x = torch.randn(32, 784)
        _ = layer(x)
        
        # Local update with event=1
        layer.local_update(event=1)
        
        # Weights should be different
        assert not torch.allclose(original_weights, layer.weight.data), \
            "Weights unchanged after local_update!"
    
    def test_local_update_no_change_when_event_zero(self):
        """Verify local_update does nothing when event=0."""
        layer = LocalLinear(784, 256, lr=0.1)
        
        x = torch.randn(32, 784)
        _ = layer(x)
        
        original_weights = layer.weight.data.clone()
        layer.local_update(event=0)
        
        assert torch.allclose(original_weights, layer.weight.data), \
            "Weights changed when event=0!"
    
    def test_training_loop_no_backward(self):
        """Verify complete training loop uses no backward pass."""
        network = create_mlp(784, [256, 128], 10, lr=0.01)
        controller = ThresholdController(budget=100)
        
        # Simulate training
        for _ in range(10):
            x = torch.randn(32, 784)
            target = torch.randint(0, 10, (32,))
            
            # Forward pass (must be under no_grad)
            with torch.no_grad():
                output = network(x)
                loss = nn.CrossEntropyLoss()(output, target)
            
            # Get event from controller
            event = controller.should_trigger(loss.item())
            
            # Local update
            network.local_update_all(event)
            network.clear_activations()
        
        # Verify still no gradients
        for name, param in network.named_parameters():
            assert param.requires_grad is False, \
                f"Gradient leak after training! Parameter '{name}'"


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
    
    def test_oja_rule_weight_normalization(self):
        """Test Oja's rule prevents weight explosion."""
        layer = LocalLinear(100, 50, lr=0.1)
        
        initial_norm = layer.weight.data.norm()
        
        # Many updates
        for _ in range(100):
            x = torch.randn(32, 100)
            _ = layer(x)
            layer.local_update(event=1)
        
        final_norm = layer.weight.data.norm()
        
        # Weight norm shouldn't explode
        assert final_norm < initial_norm * 10, \
            f"Weights exploded! Initial: {initial_norm:.2f}, Final: {final_norm:.2f}"


class TestEventController:
    """Tests for event controllers."""
    
    def test_threshold_controller_warmup(self):
        """Test ThresholdController doesn't trigger during warmup."""
        controller = ThresholdController(budget=100, warmup_steps=10)
        
        for i in range(10):
            event = controller.should_trigger(1.0)
            assert event == 0, f"Triggered during warmup at step {i}"
    
    def test_threshold_controller_budget(self):
        """Test ThresholdController respects budget."""
        controller = ThresholdController(budget=5, warmup_steps=0)
        
        # Fill history
        for _ in range(20):
            controller.should_trigger(1.0)
        
        # Trigger events with high loss
        events = 0
        for _ in range(100):
            events += controller.should_trigger(10.0)  # High loss
        
        assert events <= 5, f"Budget exceeded! Triggered {events} events"
    
    def test_fixed_rate_controller_rate(self):
        """Test FixedRateController approximates expected rate."""
        controller = FixedRateController(rate=0.5, seed=42)
        
        events = sum(controller.should_trigger(1.0) for _ in range(1000))
        
        # Should be roughly 50% (with some variance)
        assert 400 < events < 600, f"Rate off: expected ~500, got {events}"
    
    def test_always_on_controller(self):
        """Test AlwaysOnController always triggers."""
        controller = AlwaysOnController()
        
        for _ in range(100):
            assert controller.should_trigger(1.0) == 1
    
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
    """Tests for EGLPNetwork."""
    
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
        """Test broadcasting updates to all layers."""
        network = create_mlp(784, [256, 128], 10, lr=0.1)
        
        # Store original weights
        original = [layer.weight.data.clone() for layer in network.layers]
        
        x = torch.randn(32, 784)
        _ = network(x)
        network.local_update_all(event=1)
        
        # All layers should be updated
        for i, layer in enumerate(network.layers):
            assert not torch.allclose(original[i], layer.weight.data), \
                f"Layer {i} weights unchanged!"
    
    def test_layer_event_mask(self):
        """Test selective event gating."""
        network = create_mlp(784, [256, 128], 10, lr=0.1)
        network.set_layer_event_mask([True, False, True])
        
        original = [layer.weight.data.clone() for layer in network.layers]
        
        x = torch.randn(32, 784)
        _ = network(x)
        network.local_update_all(event=1)
        
        # Layer 0 and 2 should update, layer 1 should not
        assert not torch.allclose(original[0], network.layers[0].weight.data)
        assert torch.allclose(original[1], network.layers[1].weight.data)
        assert not torch.allclose(original[2], network.layers[2].weight.data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
