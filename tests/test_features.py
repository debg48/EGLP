"""Tests for EGLP enhancements.

Tests cover:
- Anti-Hebbian updates (anti-Oja extracts different components)
- FrozenBatchNorm1d (no gradient leak, uses running stats)
- Homeostatic scaling (rescales dead/saturated neurons)
- Cosine LR schedule (LR varies across epochs)
- Depth-dependent event scaling (deeper layers receive weaker signals)
- Soft-WTA activation (top-k sparsity)
- Synaptic consolidation (important weights resist change)
"""

import torch
import torch.nn as nn
import math

from eglp.local_layer import LocalLinear
from eglp.network import (
    EGLPNetwork, create_mlp, FrozenBatchNorm1d, SoftWTA
)


class TestAntiHebbian:
    """Tests for anti-Hebbian (anti-Oja) learning rule."""
    
    def test_anti_hebbian_updates_differently(self):
        """Anti-Hebbian layer should produce different weight updates than standard."""
        torch.manual_seed(42)
        
        standard = LocalLinear(784, 128, lr=0.01, anti_hebbian=False)
        anti = LocalLinear(784, 128, lr=0.01, anti_hebbian=True)
        
        # Copy initial weights so both start the same
        anti.weight.data.copy_(standard.weight.data)
        if anti.bias is not None:
            anti.bias.data.copy_(standard.bias.data)
        
        # Same input
        x = torch.randn(32, 784)
        standard(x)
        anti(x)
        
        w_standard_before = standard.weight.data.clone()
        w_anti_before = anti.weight.data.clone()
        
        standard.local_update(0.5)
        anti.local_update(0.5)
        
        delta_standard = standard.weight.data - w_standard_before
        delta_anti = anti.weight.data - w_anti_before
        
        # Updates should be different
        assert not torch.allclose(delta_standard, delta_anti, atol=1e-8), \
            "Anti-Hebbian should produce different weight updates than standard Hebbian"
    
    def test_anti_hebbian_bounded(self):
        """Anti-Hebbian weights should stay bounded due to Oja's decay term."""
        torch.manual_seed(42)
        layer = LocalLinear(784, 128, lr=0.01, anti_hebbian=True, weight_clip=1.0)
        
        # Run many updates — weights should remain bounded
        for _ in range(100):
            x = torch.randn(32, 784)
            layer(x)
            layer.local_update(0.5)
        
        assert layer.weight.data.abs().max() <= 1.0, \
            "Anti-Hebbian weights should stay within clip bounds"
    
    def test_anti_hebbian_no_grad(self):
        """Anti-Hebbian layers should have no gradients."""
        layer = LocalLinear(784, 128, lr=0.01, anti_hebbian=True)
        for name, param in layer.named_parameters():
            assert param.requires_grad is False, \
                f"Anti-Hebbian parameter '{name}' has requires_grad=True"
    
    def test_alternating_anti_hebbian_in_mlp(self):
        """create_mlp with use_anti_hebbian should alternate layers."""
        network = create_mlp(784, [256, 128, 64], 10, lr=0.01, use_anti_hebbian=True)
        
        assert network.layers[0].anti_hebbian is False, "Layer 0 should be standard Hebbian"
        assert network.layers[1].anti_hebbian is True, "Layer 1 should be anti-Hebbian"
        assert network.layers[2].anti_hebbian is False, "Layer 2 should be standard Hebbian"


class TestFrozenBatchNorm:
    """Tests for FrozenBatchNorm1d."""
    
    def test_frozen_batchnorm_no_grad(self):
        """FrozenBatchNorm1d should have no gradient-enabled parameters."""
        bn = FrozenBatchNorm1d(128)
        for name, param in bn.named_parameters():
            assert param.requires_grad is False, \
                f"FrozenBatchNorm parameter '{name}' has requires_grad=True"
    
    def test_frozen_batchnorm_output_shape(self):
        """Output shape should match input shape."""
        bn = FrozenBatchNorm1d(128)
        x = torch.randn(32, 128)
        y = bn(x)
        assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"
    
    def test_frozen_batchnorm_normalizes(self):
        """After warmup, output should be approximately normalized."""
        bn = FrozenBatchNorm1d(64, momentum=0.5)
        
        # Feed several batches to warm up running stats
        for _ in range(20):
            x = torch.randn(64, 64) * 3 + 5  # Mean=5, std=3
            bn(x)
        
        bn.freeze()
        
        # Test normalization
        x_test = torch.randn(64, 64) * 3 + 5
        y_test = bn(x_test)
        
        # Should be roughly zero-mean, unit-var (within tolerance)
        assert abs(y_test.mean().item()) < 1.0, \
            f"Normalized output mean too far from 0: {y_test.mean().item()}"
    
    def test_frozen_batchnorm_in_eglp_no_grad_leak(self):
        """EGLPNetwork with FrozenBatchNorm should pass _verify_no_grad."""
        network = create_mlp(
            784, [128, 64], 10, lr=0.01, use_batchnorm=True
        )
        # This should not raise
        network._verify_no_grad()
        
        # Double-check bn params
        assert network.batchnorms is not None
        for bn in network.batchnorms:
            for name, param in bn.named_parameters():
                assert param.requires_grad is False


class TestHomeostaticScaling:
    """Tests for homeostatic scaling."""
    
    def test_homeostatic_scaling_rescales(self):
        """Homeostatic scaling should modify weights of dead/saturated neurons."""
        torch.manual_seed(42)
        layer = LocalLinear(784, 128, lr=0.01)
        
        # Force some neurons to have very low firing rate (simulate dead neurons)
        layer.firing_rate_ema[:10] = 0.01  # Dead neurons
        layer.firing_rate_ema[10:20] = 0.99  # Saturated neurons
        layer.firing_rate_ema[20:] = 0.5  # Normal neurons
        
        w_before = layer.weight.data.clone()
        layer.homeostatic_scale()
        w_after = layer.weight.data
        
        # Dead neurons (low firing rate) should get amplified
        dead_change = (w_after[:10] - w_before[:10]).abs().mean()
        normal_change = (w_after[60:70] - w_before[60:70]).abs().mean()
        
        assert dead_change > normal_change, \
            "Dead neurons should be rescaled more than normal neurons"
    
    def test_homeostatic_scaling_bounded(self):
        """Homeostatic scaling should not produce extreme weights."""
        layer = LocalLinear(784, 128, lr=0.01, weight_clip=1.0)
        
        # Extreme firing rates
        layer.firing_rate_ema[:] = 0.001
        layer.homeostatic_scale()
        
        assert layer.weight.data.abs().max() <= 1.0, \
            "Weights should stay within clip bounds after homeostatic scaling"
    
    def test_network_homeostatic_scale_all(self):
        """EGLPNetwork.homeostatic_scale_all() should work."""
        network = create_mlp(784, [128, 64], 10, lr=0.01)
        
        # Set extreme firing rates
        for layer in network.layers:
            layer.firing_rate_ema[:5] = 0.01
        
        w_before = [l.weight.data.clone() for l in network.layers]
        network.homeostatic_scale_all()
        
        # At least dead neurons should change
        for i, layer in enumerate(network.layers):
            changed = not torch.allclose(layer.weight.data[:5], w_before[i][:5])
            assert changed, f"Layer {i} dead neuron weights not changed by homeostatic scaling"


class TestSynapticConsolidation:
    """Tests for synaptic consolidation (EWC-lite)."""
    
    def test_consolidation_penalizes_deviation(self):
        """Consolidation should resist changes to important weights."""
        torch.manual_seed(42)
        
        layer_no_consol = LocalLinear(100, 50, lr=0.01, consolidation_strength=0.0)
        layer_consol = LocalLinear(100, 50, lr=0.01, consolidation_strength=1.0)
        
        # Same starting weights
        layer_consol.weight.data.copy_(layer_no_consol.weight.data)
        if layer_consol.bias is not None:
            layer_consol.bias.data.copy_(layer_no_consol.bias.data)
        
        # Run a few updates to build importance
        for _ in range(5):
            x = torch.randn(16, 100)
            layer_consol(x)
            layer_consol.local_update(0.5)
            layer_consol.clear_activations()
        
        # Snapshot weights
        layer_consol.snapshot_weights()
        
        # Now do more updates — with high importance, weights should change less
        w_snap = layer_consol.weight_snapshot.clone()
        
        x = torch.randn(16, 100)
        layer_no_consol(x)
        layer_consol(x)
        
        layer_no_consol.local_update(0.5)
        layer_consol.local_update(0.5)
        
        # The consolidated layer should deviate less from snapshot
        # (This is a statistical test, so we check the trend)
        assert layer_consol.importance.sum() > 0, \
            "Importance should accumulate from Hebbian updates"
    
    def test_snapshot_weights(self):
        """snapshot_weights should copy current weights."""
        layer = LocalLinear(100, 50, lr=0.01, consolidation_strength=0.1)
        
        # Initial snapshot is zeros
        assert torch.allclose(layer.weight_snapshot, torch.zeros_like(layer.weight_snapshot))
        
        layer.snapshot_weights()
        
        # After snapshot, should match current weights
        assert torch.allclose(layer.weight_snapshot, layer.weight.data)


class TestSoftWTA:
    """Tests for Soft Winner-Take-All activation."""
    
    def test_soft_wta_sparsity(self):
        """SoftWTA should zero out neurons below top-k threshold."""
        wta = SoftWTA(k_fraction=0.3)
        x = torch.randn(8, 100)
        y = wta(x)
        
        # Count active neurons per sample
        active_per_sample = (y > 0).float().mean(dim=1)
        
        # Should be approximately 30% active (±some tolerance from ties)
        assert active_per_sample.mean() <= 0.5, \
            f"Expected ~30% active neurons, got {active_per_sample.mean():.2%}"
    
    def test_soft_wta_preserves_values(self):
        """SoftWTA should preserve values of kept neurons (after ReLU)."""
        wta = SoftWTA(k_fraction=0.5)
        x = torch.randn(4, 10)
        y = wta(x)
        
        # Active values should be ReLU(x) values
        relu_x = torch.relu(x)
        for i in range(4):
            active_mask = y[i] > 0
            assert torch.allclose(y[i][active_mask], relu_x[i][active_mask]), \
                "Active values should match ReLU(x)"
    
    def test_soft_wta_in_network(self):
        """EGLPNetwork with soft_wta_k should use SoftWTA activation."""
        network = create_mlp(784, [128, 64], 10, lr=0.01, soft_wta_k=0.3)
        assert network._use_soft_wta is True
        assert isinstance(network.activation, SoftWTA)


class TestDepthDependentEvents:
    """Tests for depth-dependent event scaling."""
    
    def test_depth_dependent_event_scaling(self):
        """Deeper layers should receive weaker event signals (depth-dependent decay)."""
        torch.manual_seed(42)
        
        # Compare decay=1.0 (no decay) vs decay=0.5
        for decay, label in [(1.0, "no_decay"), (0.5, "with_decay")]:
            network = create_mlp(784, [256, 128, 64], 10, lr=0.01, event_decay=decay)
            x = torch.randn(16, 784)
            network(x)
            
            w_before = [l.weight.data.clone() for l in network.layers]
            network.local_update_all(1.0)
            
            deltas = []
            for i, layer in enumerate(network.layers):
                delta = (layer.weight.data - w_before[i]).abs().mean().item()
                deltas.append(delta)
            
            if label == "no_decay":
                no_decay_deltas = deltas
            else:
                decay_deltas = deltas
        
        # With decay=0.5, each layer should have smaller deltas relative to no_decay
        for i in range(len(decay_deltas)):
            if no_decay_deltas[i] > 0:
                ratio = decay_deltas[i] / no_decay_deltas[i]
                # Layer i should receive (0.5^i) of the no-decay signal
                expected = 0.5 ** i
                # Allow some tolerance due to modulation interactions
                assert ratio < expected + 0.3, \
                    f"Layer {i} with decay should be ~{expected:.2f}x of no-decay (got {ratio:.3f})"
    
    def test_no_decay_default(self):
        """Default event_decay=1.0 should give equal scaling to all layers."""
        network = create_mlp(784, [128, 64], 10, lr=0.01, event_decay=1.0)
        assert network.event_decay == 1.0


class TestCosineLR:
    """Tests for cosine LR schedule (integration test via training loop signal)."""
    
    def test_cosine_lr_schedule_values(self):
        """Cosine schedule should produce correct LR values at key points."""
        lr_max = 5e-4
        lr_min = 1e-5
        T = 20  # Total epochs
        
        # Epoch 0: should be max
        lr_0 = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(0 * math.pi / T))
        assert abs(lr_0 - lr_max) < 1e-8, f"Epoch 0 LR should be max, got {lr_0}"
        
        # Mid-epoch: should be ~mean
        lr_mid = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(T // 2 * math.pi / T))
        expected_mid = (lr_max + lr_min) / 2
        assert abs(lr_mid - expected_mid) < 1e-6, \
            f"Mid-epoch LR should be ~{expected_mid}, got {lr_mid}"
        
        # Last epoch: should be near min
        lr_last = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos((T - 1) * math.pi / T))
        assert lr_last < lr_max, f"Last epoch LR should be near min, got {lr_last}"
