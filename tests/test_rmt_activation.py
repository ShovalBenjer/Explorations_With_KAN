import math
import pytest
import numpy as np
import torch

from src.activations.rmt_activation import (
    RMTActivation,
    RMTActivationConfig,
    rho_sinh,
    tau_oscillator,
    phi_harmonics,
)


class TestRhoSinh:
    def test_identity_at_zero(self):
        x = torch.zeros(10)
        result = rho_sinh(x)
        assert torch.allclose(result, torch.zeros(10), atol=1e-6)

    def test_monotonic_increase(self):
        x = torch.linspace(-2, 2, 100)
        result = rho_sinh(x)
        diffs = result[1:] - result[:-1]
        assert (diffs > 0).all()

    def test_scale_parameter(self):
        x = torch.tensor([1.0])
        result_default = rho_sinh(x, rho_scale=1.0)
        result_scaled = rho_sinh(x, rho_scale=2.0)
        assert not torch.allclose(result_default, result_scaled)

    def test_spectral_radius_effect(self):
        x = torch.tensor([0.5])
        result_low = rho_sinh(x, spectral_radius=0.5)
        result_high = rho_sinh(x, spectral_radius=2.0)
        assert not torch.allclose(result_low, result_high)

    def test_output_shape_matches_input(self):
        x = torch.randn(32, 8)
        result = rho_sinh(x)
        assert result.shape == x.shape

    def test_gradient_flow(self):
        x = torch.randn(10, requires_grad=True)
        result = rho_sinh(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_large_input_clamping(self):
        x = torch.tensor([100.0])
        result = rho_sinh(x)
        assert torch.isfinite(result).all()


class TestTauOscillator:
    def test_identity_at_zero_spectral_radius(self):
        x = torch.randn(10)
        result = tau_oscillator(x, tau_frequency=0.0, tau_phase=0.0, spectral_radius=0.0)
        assert torch.allclose(result, x, atol=1e-6)

    def test_periodic_modulation(self):
        x = torch.linspace(0, 2 * math.pi, 100)
        result = tau_oscillator(x, tau_frequency=1.0, tau_phase=0.0)
        assert result.shape == x.shape
        assert not torch.allclose(result, x)

    def test_phase_shift(self):
        x = torch.linspace(0, math.pi, 50)
        result_no_phase = tau_oscillator(x, tau_frequency=1.0, tau_phase=0.0)
        result_with_phase = tau_oscillator(x, tau_frequency=1.0, tau_phase=math.pi / 2)
        assert not torch.allclose(result_no_phase, result_with_phase)

    def test_spectral_radius_effect(self):
        x = torch.tensor([1.0])
        result_low = tau_oscillator(x, spectral_radius=0.1)
        result_high = tau_oscillator(x, spectral_radius=5.0)
        assert not torch.allclose(result_low, result_high)

    def test_gradient_flow(self):
        x = torch.randn(10, requires_grad=True)
        result = tau_oscillator(x, tau_frequency=2.0)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None

    def test_output_shape(self):
        x = torch.randn(16, 4)
        result = tau_oscillator(x)
        assert result.shape == x.shape


class TestPhiHarmonics:
    def test_zero_at_origin(self):
        x = torch.tensor([0.0])
        result = phi_harmonics(x, phi_n_harmonics=3, phi_base_freq=1.0)
        assert torch.allclose(result, x, atol=1e-5)

    def test_harmonics_count_effect(self):
        x = torch.linspace(-2, 2, 50)
        result_low = phi_harmonics(x, phi_n_harmonics=1, phi_base_freq=1.0)
        result_high = phi_harmonics(x, phi_n_harmonics=5, phi_base_freq=1.0)
        assert not torch.allclose(result_low, result_high)

    def test_base_frequency_effect(self):
        x = torch.linspace(-1, 1, 50)
        result_low = phi_harmonics(x, phi_n_harmonics=3, phi_base_freq=0.5)
        result_high = phi_harmonics(x, phi_n_harmonics=3, phi_base_freq=2.0)
        assert not torch.allclose(result_low, result_high)

    def test_spectral_radius_effect(self):
        x = torch.tensor([1.0])
        result_low = phi_harmonics(x, spectral_radius=0.5)
        result_high = phi_harmonics(x, spectral_radius=2.0)
        assert not torch.allclose(result_low, result_high)

    def test_output_shape(self):
        x = torch.randn(20, 10)
        result = phi_harmonics(x)
        assert result.shape == x.shape

    def test_gradient_flow(self):
        x = torch.randn(10, requires_grad=True)
        result = phi_harmonics(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None


class TestRMTActivation:
    def test_default_config(self):
        config = RMTActivationConfig()
        assert config.rho_scale == 1.0
        assert config.tau_frequency == 1.0
        assert config.tau_phase == 0.0
        assert config.phi_n_harmonics == 3
        assert config.phi_base_freq == 1.0
        assert config.spectral_radius == 1.0
        assert config.matrix_dim == 8

    def test_module_creation(self):
        activation = RMTActivation()
        assert hasattr(activation, "rho_scale")
        assert hasattr(activation, "tau_frequency")
        assert hasattr(activation, "tau_phase")
        assert hasattr(activation, "phi_base_freq")
        assert hasattr(activation, "spectral_radius")
        assert hasattr(activation, "resonance_matrix")

    def test_forward_rho_sinh(self):
        activation = RMTActivation()
        x = torch.randn(10)
        result = activation(x, mode="rho_sinh")
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_forward_tau_oscillator(self):
        activation = RMTActivation()
        x = torch.randn(10)
        result = activation(x, mode="tau_oscillator")
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_forward_phi_harmonics(self):
        activation = RMTActivation()
        x = torch.randn(10)
        result = activation(x, mode="phi_harmonics")
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_forward_resonance(self):
        activation = RMTActivation()
        x = torch.randn(3, 8)
        result = activation(x, mode="resonance")
        assert result.shape == x.shape

    def test_invalid_mode_raises(self):
        activation = RMTActivation()
        x = torch.randn(10)
        with pytest.raises(ValueError, match="Unknown activation mode"):
            activation(x, mode="invalid")

    def test_get_spectral_info(self):
        activation = RMTActivation()
        info = activation.get_spectral_info()
        assert "spectral_radius" in info
        assert "eigenvalue_real_parts" in info
        assert "eigenvalue_imag_parts" in info
        assert "rho_scale" in info
        assert "tau_frequency" in info
        assert isinstance(info["spectral_radius"], float)
        assert isinstance(info["eigenvalue_real_parts"], list)

    def test_compute_resonance_spectrum(self):
        activation = RMTActivation()
        spectrum = activation.compute_resonance_spectrum(n_points=64)
        assert "frequencies" in spectrum
        assert "rho_spectrum" in spectrum
        assert "tau_spectrum" in spectrum
        assert "phi_spectrum" in spectrum
        assert "rho_peak_freq" in spectrum
        assert "tau_peak_freq" in spectrum
        assert "phi_peak_freq" in spectrum
        assert len(spectrum["frequencies"]) == 33

    def test_custom_config(self):
        config = RMTActivationConfig(
            rho_scale=0.5,
            tau_frequency=2.0,
            tau_phase=1.0,
            phi_n_harmonics=5,
            phi_base_freq=3.0,
            spectral_radius=1.5,
            matrix_dim=4,
        )
        activation = RMTActivation(config)
        assert activation.config.rho_scale == 0.5
        assert activation.config.tau_frequency == 2.0
        assert activation.config.matrix_dim == 4

    def test_gradient_through_all_modes(self):
        activation = RMTActivation()
        for mode in ["rho_sinh", "tau_oscillator", "phi_harmonics"]:
            x = torch.randn(10, requires_grad=True)
            result = activation(x, mode=mode)
            loss = result.sum()
            loss.backward()
            assert x.grad is not None, f"No gradient for mode {mode}"