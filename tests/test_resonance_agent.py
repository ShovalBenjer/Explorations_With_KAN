import math
import pytest
import numpy as np
import torch
import tempfile

from src.agents.resonance_agent import (
    ResonanceAgent,
    ResonanceModel,
    ResonanceModelConfig,
    ResonanceTrainingConfig,
)
from src.agents.resonance_visualization import (
    ResonanceSimulation,
    DynamicsConfig,
)
from src.activations.rmt_activation import RMTActivationConfig


class TestResonanceModel:
    def test_default_creation(self):
        model = ResonanceModel()
        assert model.config.input_dim == 1
        assert model.config.hidden_dim == 16
        assert model.config.output_dim == 1

    def test_forward_pass(self):
        model = ResonanceModel()
        x = torch.randn(8, 1)
        output = model(x)
        assert output.shape == (8, 1)
        assert torch.isfinite(output).all()

    def test_custom_config(self):
        config = ResonanceModelConfig(
            input_dim=2,
            hidden_dim=32,
            output_dim=3,
            n_hidden_layers=3,
            activation_config=RMTActivationConfig(rho_scale=0.5),
        )
        model = ResonanceModel(config)
        x = torch.randn(4, 2)
        output = model(x)
        assert output.shape == (4, 3)

    def test_gradient_flow(self):
        model = ResonanceModel()
        x = torch.randn(4, 1)
        output = model(x)
        loss = output.sum()
        loss.backward()
        grad_counts = {"has_grad": 0, "no_grad": 0}
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_counts["has_grad"] += 1
                else:
                    grad_counts["no_grad"] += 1
        assert grad_counts["has_grad"] > 0


class TestResonanceAgent:
    def test_agent_creation(self):
        config = ResonanceTrainingConfig(use_llm_optimization=False)
        agent = ResonanceAgent(training_config=config)
        assert agent.training_config.use_llm_optimization is False
        assert agent._best_loss == float("inf")

    def test_prepare_data(self):
        agent = ResonanceAgent(training_config=ResonanceTrainingConfig(use_llm_optimization=False))
        X = np.random.randn(100, 1).astype(np.float32)
        y = np.sin(X).astype(np.float32)
        train_loader, test_loader = agent.prepare_data(X, y)
        assert len(train_loader) > 0
        assert len(test_loader) > 0

    def test_prepare_data_1d(self):
        agent = ResonanceAgent(training_config=ResonanceTrainingConfig(use_llm_optimization=False))
        X = np.random.randn(100).astype(np.float32)
        y = np.sin(X).astype(np.float32)
        train_loader, test_loader = agent.prepare_data(X, y)
        for batch_x, batch_y in train_loader:
            assert batch_x.dim() == 2
            assert batch_y.dim() == 2
            break

    def test_train_short(self):
        agent = ResonanceAgent(
            model_config=ResonanceModelConfig(
                input_dim=1,
                hidden_dim=8,
                output_dim=1,
                n_hidden_layers=1,
                activation_config=RMTActivationConfig(rho_scale=0.5, matrix_dim=4),
            ),
            training_config=ResonanceTrainingConfig(
                epochs=5,
                batch_size=32,
                use_llm_optimization=False,
                early_stop_patience=50,
            ),
        )
        X = np.linspace(-2, 2, 100).reshape(-1, 1).astype(np.float32)
        y = np.sin(X).astype(np.float32)
        train_loader, test_loader = agent.prepare_data(X, y)
        results = agent.train(train_loader, test_loader, verbose=False)
        assert "history" in results
        assert "best_loss" in results
        assert len(results["history"]) == 5
        assert results["best_loss"] < float("inf")

    def test_predict(self):
        agent = ResonanceAgent(training_config=ResonanceTrainingConfig(use_llm_optimization=False))
        X = np.random.randn(50, 1).astype(np.float32)
        y = np.sin(X).astype(np.float32)
        train_loader, _ = agent.prepare_data(X, y)
        agent.train(train_loader, verbose=False)

        X_new = np.random.randn(10, 1).astype(np.float32)
        predictions = agent.predict(X_new)
        assert predictions.shape == (10, 1)
        assert np.isfinite(predictions).all()

    def test_compare_activations(self):
        agent = ResonanceAgent(training_config=ResonanceTrainingConfig(use_llm_optimization=False))
        X = np.random.randn(50, 1).astype(np.float32)
        y = np.sin(X).astype(np.float32)
        comparison = agent.compare_activations(X, y)
        assert "rho_sinh" in comparison
        assert "tau_oscillator" in comparison
        assert "phi_harmonics" in comparison
        assert "resonance" in comparison
        for mode, info in comparison.items():
            assert "mse" in info
            assert isinstance(info["mse"], float)

    def test_get_spectral_info(self):
        agent = ResonanceAgent(training_config=ResonanceTrainingConfig(use_llm_optimization=False))
        info = agent.get_spectral_info()
        assert "spectral_radius" in info
        assert "rho_scale" in info

    def test_get_resonance_spectrum(self):
        agent = ResonanceAgent(training_config=ResonanceTrainingConfig(use_llm_optimization=False))
        spectrum = agent.get_resonance_spectrum(n_points=64)
        assert "frequencies" in spectrum
        assert "rho_spectrum" in spectrum

    def test_training_history(self):
        agent = ResonanceAgent(
            training_config=ResonanceTrainingConfig(
                epochs=3,
                use_llm_optimization=False,
                early_stop_patience=50,
            ),
        )
        X = np.random.randn(50, 1).astype(np.float32)
        y = np.sin(X).astype(np.float32)
        train_loader, _ = agent.prepare_data(X, y)
        agent.train(train_loader, verbose=False)
        history = agent.get_training_history()
        assert len(history) == 3
        assert "epoch" in history[0]
        assert "train_loss" in history[0]


class TestResonanceSimulation:
    def test_default_config(self):
        config = DynamicsConfig()
        assert config.n_steps == 200
        assert config.dt == 0.01

    def test_simulate_activation_dynamics(self):
        simulation = ResonanceSimulation(DynamicsConfig(n_points=50))
        fn = lambda x: torch.sinh(x)
        result = simulation.simulate_activation_dynamics(fn)
        assert "x" in result
        assert "y" in result
        assert "dy" in result
        assert "d2y" in result
        assert "energy" in result
        assert "kinetic" in result["energy"]
        assert "potential" in result["energy"]
        assert "total" in result["energy"]

    def test_simulate_temporal_dynamics(self):
        simulation = ResonanceSimulation(DynamicsConfig(n_steps=50, dt=0.01))
        fn = lambda x: torch.tanh(x)
        result = simulation.simulate_temporal_dynamics(fn, initial_state=1.0)
        assert "time" in result
        assert "states" in result
        assert "initial_state" in result
        assert "final_state" in result
        assert "converged" in result
        assert len(result["states"]) == 50

    def test_simulate_temporal_with_noise(self):
        simulation = ResonanceSimulation(DynamicsConfig(n_steps=50, dt=0.01))
        fn = lambda x: torch.tanh(x)
        result = simulation.simulate_temporal_dynamics(fn, noise_level=0.1)
        assert len(result["states"]) == 50

    def test_simulate_coupled_oscillators(self):
        simulation = ResonanceSimulation(DynamicsConfig(n_steps=50, dt=0.01))
        fn = lambda x: torch.sin(x)
        result = simulation.simulate_coupled_oscillators(fn, n_oscillators=3, coupling_strength=0.1)
        assert "trajectory" in result
        assert "n_oscillators" in result
        assert "order_parameter" in result
        assert len(result["trajectory"]) == 50

    def test_compute_spectral_analysis(self):
        simulation = ResonanceSimulation()
        fn = lambda x: torch.sin(x)
        result = simulation.compute_spectral_analysis(fn, n_points=64)
        assert "frequencies" in result
        assert "magnitudes" in result
        assert "peak_frequency" in result
        assert "spectral_centroid" in result
        assert "total_power" in result
        assert "peak_power_ratio" in result

    def test_run_full_simulation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            simulation = ResonanceSimulation(DynamicsConfig(
                n_steps=20,
                n_points=50,
                output_dir=tmpdir,
            ))
            results = simulation.run_full_simulation(
                activation_config=RMTActivationConfig(matrix_dim=4),
                activation_modes=["rho_sinh", "tau_oscillator"],
            )
            assert "config" in results
            assert "modes" in results
            assert "rho_sinh" in results["modes"]
            assert "tau_oscillator" in results["modes"]
            for mode in ["rho_sinh", "tau_oscillator"]:
                assert "dynamics" in results["modes"][mode]
                assert "temporal" in results["modes"][mode]
                assert "spectral" in results["modes"][mode]
                assert "coupled_oscillators" in results["modes"][mode]

    def test_save_simulation_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            simulation = ResonanceSimulation(DynamicsConfig(output_dir=tmpdir))
            results = {"test": "data"}
            filepath = simulation.save_simulation_results(results, "test_results.json")
            assert filepath.endswith("test_results.json")
            import os
            assert os.path.exists(filepath)

    def test_generate_summary(self):
        simulation = ResonanceSimulation(DynamicsConfig(n_steps=20, n_points=50))
        results = simulation.run_full_simulation(
            activation_config=RMTActivationConfig(matrix_dim=4),
            activation_modes=["rho_sinh"],
        )
        summary = simulation.generate_summary(results)
        assert "modes" in summary
        assert "rho_sinh" in summary["modes"]
        mode_summary = summary["modes"]["rho_sinh"]
        assert "energy" in mode_summary
        assert "converged" in mode_summary
        assert "peak_frequency" in mode_summary
