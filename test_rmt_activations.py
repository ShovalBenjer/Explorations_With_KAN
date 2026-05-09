import torch
import pytest
import sys
sys.path.insert(0, '.')

from src.activations.rmt_activation import RhoSinh, TauOscillator, PhiHarmonics, ResonanceLayer


class TestRhoSinh:
    def test_output_shape(self):
        act = RhoSinh()
        x = torch.randn(10, 5)
        y = act(x)
        assert y.shape == x.shape
    
    def test_learnable_parameters(self):
        act = RhoSinh()
        assert hasattr(act, 'scale') and act.scale.requires_grad
        assert hasattr(act, 'alpha') and act.alpha.requires_grad
    
    def test_positive_input(self):
        act = RhoSinh(scale=1.0, alpha=1.0)
        x = torch.tensor([1.0, 2.0, 3.0])
        y = act(x)
        assert torch.all(y > 0)


class TestTauOscillator:
    def test_output_shape(self):
        act = TauOscillator()
        x = torch.randn(10, 5)
        y = act(x)
        assert y.shape == x.shape
    
    def test_oscillation_behavior(self):
        act = TauOscillator(frequency=2.0)
        x = torch.tensor([0.0, 3.14159/2, 3.14159])
        y = act(x)
        assert y[0] > 0  # cos(0) = 1
        assert y[2] < 0  # cos(pi) with damping


class TestPhiHarmonics:
    def test_output_shape(self):
        act = PhiHarmonics(num_harmonics=5)
        x = torch.randn(10, 5)
        y = act(x)
        assert y.shape == x.shape
    
    def test_multiple_harmonics(self):
        act = PhiHarmonics(num_harmonics=3)
        assert act.num_harmonics == 3


class TestResonanceLayer:
    def test_forward_shape(self):
        layer = ResonanceLayer(10, 20)
        x = torch.randn(5, 10)
        y = layer(x)
        assert y.shape == (5, 20)
    
    def test_mixing_weights_influence(self):
        layer = ResonanceLayer(10, 20, activation_types=['rho_sinh', 'tau_oscillator'])
        x = torch.randn(5, 10)
        
        layer.mixing_weights.data = torch.tensor([0.9, 0.1])
        y1 = layer(x)
        
        layer.mixing_weights.data = torch.tensor([0.1, 0.9])
        y2 = layer(x)
        
        assert not torch.allclose(y1, y2)
    
    def test_custom_activation_types(self):
        layer = ResonanceLayer(10, 20, activation_types=['rho_sinh'])
        x = torch.randn(5, 10)
        y = layer(x)
        assert y.shape == (5, 20)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])