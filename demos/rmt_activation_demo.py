import math
import numpy as np
import torch

from src.activations.rmt_activation import (
    RMTActivation,
    RMTActivationConfig,
    rho_sinh,
    tau_oscillator,
    phi_harmonics,
)
from src.agents.resonance_agent import (
    ResonanceAgent,
    ResonanceModelConfig,
    ResonanceTrainingConfig,
)
from src.agents.resonance_visualization import (
    ResonanceSimulation,
    DynamicsConfig,
)


def demo_activation_comparison():
    print("=" * 60)
    print("RMT Activation Function Comparison")
    print("=" * 60)

    x = torch.linspace(-3, 3, 200)
    config = RMTActivationConfig(
        rho_scale=1.0,
        tau_frequency=2.0,
        tau_phase=0.0,
        phi_n_harmonics=3,
        phi_base_freq=1.0,
        spectral_radius=1.0,
    )

    with torch.no_grad():
        rho_out = rho_sinh(x, config.rho_scale, config.spectral_radius)
        tau_out = tau_oscillator(x, config.tau_frequency, config.tau_phase, config.spectral_radius)
        phi_out = phi_harmonics(x, config.phi_n_harmonics, config.phi_base_freq, config.spectral_radius)

    print(f"\nInput range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"\nrho_sinh:    range=[{rho_out.min():.4f}, {rho_out.max():.4f}], mean={rho_out.mean():.4f}")
    print(f"tau_oscillator: range=[{tau_out.min():.4f}, {tau_out.max():.4f}], mean={tau_out.mean():.4f}")
    print(f"phi_harmonics:  range=[{phi_out.min():.4f}, {phi_out.max():.4f}], mean={phi_out.mean():.4f}")

    activation = RMTActivation(config)
    spectral_info = activation.get_spectral_info()
    print(f"\nSpectral radius: {spectral_info['spectral_radius']:.4f}")
    print(f"Rho scale: {spectral_info['rho_scale']:.4f}")
    print(f"Tau frequency: {spectral_info['tau_frequency']:.4f}")
    print(f"Phi base freq: {spectral_info['phi_base_freq']:.4f}")


def demo_resonance_agent_training():
    print("\n" + "=" * 60)
    print("Resonance Agent Training Demo")
    print("=" * 60)

    np.random.seed(42)
    X = np.linspace(-2 * math.pi, 2 * math.pi, 500).reshape(-1, 1).astype(np.float32)
    y = (np.sin(X) + 0.5 * np.cos(2 * X)).astype(np.float32)

    model_config = ResonanceModelConfig(
        input_dim=1,
        hidden_dim=16,
        output_dim=1,
        n_hidden_layers=2,
        activation_config=RMTActivationConfig(
            rho_scale=0.5,
            tau_frequency=1.5,
            phi_n_harmonics=3,
            spectral_radius=0.8,
        ),
        activation_mode="rho_sinh",
    )

    training_config = ResonanceTrainingConfig(
        learning_rate=1e-3,
        epochs=50,
        batch_size=32,
        use_llm_optimization=False,
        early_stop_patience=15,
    )

    agent = ResonanceAgent(
        model_config=model_config,
        training_config=training_config,
    )

    train_loader, test_loader = agent.prepare_data(X, y, train_ratio=0.8)
    results = agent.train(train_loader, test_loader, verbose=True)

    print(f"\nTraining complete!")
    print(f"Best loss: {results['best_loss']:.6f}")
    print(f"Total epochs: {len(results['history'])}")

    predictions = agent.predict(X[:10])
    print(f"\nSample predictions (first 10):")
    for i in range(10):
        print(f"  x={X[i, 0]:.4f}, y_true={y[i, 0]:.4f}, y_pred={predictions[i, 0]:.4f}")

    comparison = agent.compare_activations(X, y)
    print(f"\nActivation comparison (MSE):")
    for mode, info in comparison.items():
        print(f"  {mode}: {info['mse']:.6f}")


def demo_resonance_simulation():
    print("\n" + "=" * 60)
    print("Resonance Dynamics Simulation Demo")
    print("=" * 60)

    config = DynamicsConfig(n_steps=200, dt=0.01, n_points=200, output_dir="resonance_output")
    simulation = ResonanceSimulation(config)

    activation_config = RMTActivationConfig(
        rho_scale=1.0,
        tau_frequency=2.0,
        phi_n_harmonics=3,
        spectral_radius=1.0,
    )

    results = simulation.run_full_simulation(
        activation_config=activation_config,
        activation_modes=["rho_sinh", "tau_oscillator", "phi_harmonics"],
    )

    summary = simulation.generate_summary(results)

    for mode, data in summary["modes"].items():
        print(f"\n--- {mode} ---")
        energy = data.get("energy", {})
        print(f"  Energy: kinetic={energy.get('kinetic', 0):.4f}, potential={energy.get('potential', 0):.4f}, total={energy.get('total', 0):.4f}")
        print(f"  Temporal: converged={data.get('converged')}, final_state={data.get('final_state', 'N/A')}")
        print(f"  Spectral: peak_freq={data.get('peak_frequency', 0):.4f}, centroid={data.get('spectral_centroid', 0):.4f}")
        print(f"  Coupled oscillators: order_param={data.get('order_parameter', 0):.4f}")

    filepath = simulation.save_simulation_results(results)
    print(f"\nFull results saved to: {filepath}")


if __name__ == "__main__":
    demo_activation_comparison()
    demo_resonance_agent_training()
    demo_resonance_simulation()
