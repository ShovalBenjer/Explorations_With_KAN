import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class RMTActivationConfig:
    rho_scale: float = 1.0
    tau_frequency: float = 1.0
    tau_phase: float = 0.0
    phi_n_harmonics: int = 3
    phi_base_freq: float = 1.0
    spectral_radius: float = 1.0
    matrix_dim: int = 8
    clamp_min: float = -10.0
    clamp_max: float = 10.0


def rho_sinh(
    x: torch.Tensor,
    rho_scale: float = 1.0,
    spectral_radius: float = 1.0,
) -> torch.Tensor:
    rho = rho_scale * spectral_radius
    x = torch.clamp(x, -10.0, 10.0)
    scaled = rho * x
    output = torch.sinh(scaled) / (rho + 1e-8)
    return output


def tau_oscillator(
    x: torch.Tensor,
    tau_frequency: float = 1.0,
    tau_phase: float = 0.0,
    spectral_radius: float = 1.0,
) -> torch.Tensor:
    amplitude = spectral_radius / (spectral_radius + 1.0)
    modulation = torch.cos(tau_frequency * x + tau_phase)
    return x * (1.0 + amplitude * modulation)


def phi_harmonics(
    x: torch.Tensor,
    phi_n_harmonics: int = 3,
    phi_base_freq: float = 1.0,
    spectral_radius: float = 1.0,
) -> torch.Tensor:
    result = torch.zeros_like(x)
    for k in range(1, phi_n_harmonics + 1):
        freq = phi_base_freq * k
        weight = spectral_radius / (k * math.pi)
        result = result + weight * torch.sin(freq * x)
    return x + result


class RMTActivation(nn.Module):
    def __init__(self, config: Optional[RMTActivationConfig] = None):
        super().__init__()
        self.config = config or RMTActivationConfig()

        self.rho_scale = nn.Parameter(torch.tensor(self.config.rho_scale))
        self.tau_frequency = nn.Parameter(torch.tensor(self.config.tau_frequency))
        self.tau_phase = nn.Parameter(torch.tensor(self.config.tau_phase))
        self.phi_n_harmonics = self.config.phi_n_harmonics
        self.phi_base_freq = nn.Parameter(torch.tensor(self.config.phi_base_freq))
        self.spectral_radius = nn.Parameter(torch.tensor(self.config.spectral_radius))

        self._init_resonance_matrix()

    def _init_resonance_matrix(self):
        n = self.config.matrix_dim
        gaussian = torch.randn(n, n) / math.sqrt(n)
        eigenvalues = torch.linalg.eigvals(gaussian)
        spectral_rad = torch.max(torch.abs(eigenvalues)).item()
        if spectral_rad > 0:
            gaussian = gaussian / spectral_rad * self.config.spectral_radius
        self.register_buffer("resonance_matrix", gaussian)

    def forward(self, x: torch.Tensor, mode: str = "rho_sinh") -> torch.Tensor:
        x = torch.clamp(x, self.config.clamp_min, self.config.clamp_max)

        if mode == "rho_sinh":
            return rho_sinh(x, self.rho_scale, self.spectral_radius)
        elif mode == "tau_oscillator":
            return tau_oscillator(x, self.tau_frequency, self.tau_phase, self.spectral_radius)
        elif mode == "phi_harmonics":
            return phi_harmonics(x, self.phi_n_harmonics, self.phi_base_freq, self.spectral_radius)
        elif mode == "resonance":
            return self._resonance_forward(x)
        else:
            raise ValueError(f"Unknown activation mode: {mode}")

    def _resonance_forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(-1, self.config.matrix_dim) if x.shape[-1] == self.config.matrix_dim else x.reshape(-1, 1)

        if x_flat.shape[-1] == self.config.matrix_dim:
            projected = x_flat @ self.resonance_matrix
            norms = torch.norm(projected, dim=-1, keepdim=True).clamp(min=1e-8)
            eigenvalue_scale = torch.norm(self.resonance_matrix, p=2)
            x_flat = x_flat * (1.0 + eigenvalue_scale * projected / norms)

        return x_flat.reshape(x.shape)

    def get_spectral_info(self) -> dict:
        eigenvalues = torch.linalg.eigvals(self.resonance_matrix)
        return {
            "spectral_radius": float(torch.max(torch.abs(eigenvalues))),
            "eigenvalue_real_parts": eigenvalues.real.tolist(),
            "eigenvalue_imag_parts": eigenvalues.imag.tolist(),
            "rho_scale": float(self.rho_scale.detach()),
            "tau_frequency": float(self.tau_frequency.detach()),
            "tau_phase": float(self.tau_phase.detach()),
            "phi_base_freq": float(self.phi_base_freq.detach()),
        }

    def compute_resonance_spectrum(self, n_points: int = 256) -> dict:
        t = torch.linspace(-2 * math.pi, 2 * math.pi, n_points)
        with torch.no_grad():
            rho_out = rho_sinh(t, float(self.rho_scale), float(self.spectral_radius))
            tau_out = tau_oscillator(t, float(self.tau_frequency), float(self.tau_phase), float(self.spectral_radius))
            phi_out = phi_harmonics(t, self.phi_n_harmonics, float(self.phi_base_freq), float(self.spectral_radius))

        rho_fft = torch.fft.rfft(rho_out)
        tau_fft = torch.fft.rfft(tau_out)
        phi_fft = torch.fft.rfft(phi_out)

        freqs = torch.fft.rfftfreq(n_points, d=(4 * math.pi / n_points))

        return {
            "frequencies": freqs.tolist(),
            "rho_spectrum": torch.abs(rho_fft).tolist(),
            "tau_spectrum": torch.abs(tau_fft).tolist(),
            "phi_spectrum": torch.abs(phi_fft).tolist(),
            "rho_peak_freq": float(freqs[torch.argmax(torch.abs(rho_fft))]),
            "tau_peak_freq": float(freqs[torch.argmax(torch.abs(tau_fft))]),
            "phi_peak_freq": float(freqs[torch.argmax(torch.abs(phi_fft))]),
        }
