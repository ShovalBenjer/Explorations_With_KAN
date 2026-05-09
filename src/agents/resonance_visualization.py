import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.activations.rmt_activation import (
    RMTActivation,
    RMTActivationConfig,
    rho_sinh,
    tau_oscillator,
    phi_harmonics,
)

logger = logging.getLogger(__name__)


@dataclass
class DynamicsConfig:
    n_steps: int = 200
    dt: float = 0.01
    x_range: tuple = (-3.0, 3.0)
    n_points: int = 500
    noise_level: float = 0.0
    output_dir: str = "resonance_output"


class ResonanceSimulation:
    def __init__(self, config: Optional[DynamicsConfig] = None):
        self.config = config or DynamicsConfig()
        self._output_dir_created = False

    def _ensure_output_dir(self):
        if not self._output_dir_created:
            os.makedirs(self.config.output_dir, exist_ok=True)
            self._output_dir_created = True

    def simulate_activation_dynamics(
        self,
        activation_fn: callable,
        x_range: Optional[tuple] = None,
        n_points: Optional[int] = None,
    ) -> dict:
        x_range = x_range or self.config.x_range
        n_points = n_points or self.config.n_points

        x = np.linspace(x_range[0], x_range[1], n_points)
        x_tensor = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            y = activation_fn(x_tensor).numpy()

        dx = (x_range[1] - x_range[0]) / max(n_points - 1, 1)
        dy = np.gradient(y, dx)

        d2y = np.gradient(dy, dx)

        kinetic = 0.5 * np.sum(dy ** 2) * dx
        potential = 0.5 * np.sum(y ** 2) * dx
        total = kinetic + potential

        return {
            "x": x.tolist(),
            "y": y.tolist(),
            "dy": dy.tolist(),
            "d2y": d2y.tolist(),
            "energy": {
                "kinetic": float(kinetic),
                "potential": float(potential),
                "total": float(total),
            },
        }

    def simulate_temporal_dynamics(
        self,
        activation_fn: callable,
        initial_state: float = 1.0,
        n_steps: Optional[int] = None,
        dt: Optional[float] = None,
        noise_level: Optional[float] = None,
    ) -> dict:
        n_steps = n_steps or self.config.n_steps
        dt = dt or self.config.dt
        noise_level = noise_level if noise_level is not None else self.config.noise_level

        states = np.zeros(n_steps)
        states[0] = initial_state

        for t in range(1, n_steps):
            x_tensor = torch.tensor([states[t - 1]], dtype=torch.float32)
            with torch.no_grad():
                activation_output = activation_fn(x_tensor).item()

            noise = np.random.randn() * noise_level if noise_level > 0 else 0.0
            states[t] = states[t - 1] + dt * (activation_output - states[t - 1]) + noise * math.sqrt(dt)

        time = np.arange(n_steps) * dt

        return {
            "time": time.tolist(),
            "states": states.tolist(),
            "initial_state": initial_state,
            "dt": dt,
            "n_steps": n_steps,
            "final_state": float(states[-1]),
            "converged": abs(states[-1] - states[-2]) < 1e-6 if n_steps > 1 else False,
        }

    def simulate_coupled_oscillators(
        self,
        activation_fn: callable,
        n_oscillators: int = 3,
        coupling_strength: float = 0.1,
        n_steps: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> dict:
        n_steps = n_steps or self.config.n_steps
        dt = dt or self.config.dt

        phases = np.random.uniform(0, 2 * math.pi, n_oscillators)
        amplitudes = np.random.uniform(0.5, 1.5, n_oscillators)
        frequencies = np.random.uniform(0.5, 2.0, n_oscillators)

        trajectory = np.zeros((n_steps, n_oscillators))

        for t in range(n_steps):
            x_tensor = torch.tensor(phases, dtype=torch.float32)
            with torch.no_grad():
                modulated = activation_fn(x_tensor).numpy()

            for i in range(n_oscillators):
                coupling = coupling_strength * np.sum(np.sin(phases - phases[i]))
                phases[i] += dt * (frequencies[i] + modulated[i] * 0.1 + coupling)
                trajectory[t, i] = amplitudes[i] * np.sin(phases[i])

        order_param = np.abs(np.mean(np.exp(1j * phases)))

        return {
            "trajectory": trajectory.tolist(),
            "n_oscillators": n_oscillators,
            "coupling_strength": coupling_strength,
            "order_parameter": float(order_param),
            "frequencies": frequencies.tolist(),
            "amplitudes": amplitudes.tolist(),
        }

    def compute_spectral_analysis(self, activation_fn: callable, n_points: int = 1024) -> dict:
        x = np.linspace(-2 * math.pi, 2 * math.pi, n_points)
        x_tensor = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            y = activation_fn(x_tensor).numpy()

        fft_result = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(n_points, d=(4 * math.pi / n_points))

        magnitudes = np.abs(fft_result)
        phases = np.angle(fft_result)

        peak_idx = np.argmax(magnitudes[1:]) + 1
        peak_freq = freqs[peak_idx]
        peak_magnitude = float(magnitudes[peak_idx])

        total_power = float(np.sum(magnitudes ** 2))
        peak_power = float(magnitudes[peak_idx] ** 2)
        spectral_centroid = float(np.sum(freqs * magnitudes) / (np.sum(magnitudes) + 1e-10))

        return {
            "frequencies": freqs.tolist(),
            "magnitudes": magnitudes.tolist(),
            "phases": phases.tolist(),
            "peak_frequency": float(peak_freq),
            "peak_magnitude": peak_magnitude,
            "total_power": total_power,
            "peak_power_ratio": peak_power / (total_power + 1e-10),
            "spectral_centroid": spectral_centroid,
            "bandwidth": float(np.std(freqs * magnitudes / (np.sum(magnitudes) + 1e-10))),
        }

    def run_full_simulation(
        self,
        activation_config: Optional[RMTActivationConfig] = None,
        activation_modes: Optional[list] = None,
    ) -> dict:
        activation_config = activation_config or RMTActivationConfig()
        activation = RMTActivation(activation_config)
        modes = activation_modes or ["rho_sinh", "tau_oscillator", "phi_harmonics"]

        results = {
            "config": {
                "rho_scale": activation_config.rho_scale,
                "tau_frequency": activation_config.tau_frequency,
                "tau_phase": activation_config.tau_phase,
                "phi_n_harmonics": activation_config.phi_n_harmonics,
                "phi_base_freq": activation_config.phi_base_freq,
                "spectral_radius": activation_config.spectral_radius,
                "matrix_dim": activation_config.matrix_dim,
            },
            "modes": {},
        }

        mode_fns = {
            "rho_sinh": lambda x: rho_sinh(x, activation_config.rho_scale, activation_config.spectral_radius),
            "tau_oscillator": lambda x: tau_oscillator(x, activation_config.tau_frequency, activation_config.tau_phase, activation_config.spectral_radius),
            "phi_harmonics": lambda x: phi_harmonics(x, activation_config.phi_n_harmonics, activation_config.phi_base_freq, activation_config.spectral_radius),
        }

        for mode in modes:
            if mode not in mode_fns:
                continue

            fn = mode_fns[mode]
            dynamics = self.simulate_activation_dynamics(fn)
            temporal = self.simulate_temporal_dynamics(fn)
            spectral = self.compute_spectral_analysis(fn)
            coupled = self.simulate_coupled_oscillators(fn)

            results["modes"][mode] = {
                "dynamics": dynamics,
                "temporal": temporal,
                "spectral": spectral,
                "coupled_oscillators": coupled,
            }

        return results

    def save_simulation_results(self, results: dict, filename: str = "simulation_results.json") -> str:
        self._ensure_output_dir()
        filepath = os.path.join(self.config.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved simulation results to {filepath}")
        return filepath

    def generate_summary(self, results: dict) -> dict:
        summary = {"modes": {}}

        for mode, data in results.get("modes", {}).items():
            dynamics = data.get("dynamics", {})
            temporal = data.get("temporal", {})
            spectral = data.get("spectral", {})
            coupled = data.get("coupled_oscillators", {})

            summary["modes"][mode] = {
                "energy": dynamics.get("energy", {}),
                "converged": temporal.get("converged", False),
                "final_state": temporal.get("final_state"),
                "peak_frequency": spectral.get("peak_frequency"),
                "spectral_centroid": spectral.get("spectral_centroid"),
                "order_parameter": coupled.get("order_parameter"),
                "peak_power_ratio": spectral.get("peak_power_ratio"),
            }

        return summary
