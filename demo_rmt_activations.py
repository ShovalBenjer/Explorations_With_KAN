import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.activations.rmt_activation import RhoSinh, TauOscillator, PhiHarmonics, ResonanceLayer
from src.agents.resonance_agent import ResonanceAgent, ResonanceTrainer
from src.agents.resonance_visualization import (
    visualize_activations,
    visualize_layer_mixing,
    visualize_training_history,
    plot_activation_patterns
)


def main():
    print("RMT Activation Demo")
    print("===================")
    
    rho = RhoSinh()
    tau = TauOscillator()
    phi = PhiHarmonics()
    
    x = torch.linspace(-3, 3, 500)
    
    print("\n1. Visualizing activation functions...")
    try:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(x.numpy(), rho(x).detach().numpy())
        plt.title('rho_sinh')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(x.numpy(), tau(x).detach().numpy())
        plt.title('tau_oscillator')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(x.numpy(), phi(x).detach().numpy())
        plt.title('phi_harmonics')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('output/activations.png')
        plt.close()
        print("   Saved: output/activations.png")
    except Exception as e:
        print(f"   Warning: Could not save activation plot: {e}")
    
    print("\n2. Testing ResonanceLayer...")
    layer = ResonanceLayer(10, 20)
    test_input = torch.randn(5, 10)
    output = layer(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Mixing weights: {torch.softmax(layer.mixing_weights, dim=0).detach().numpy()}")
    
    print("\n3. Training ResonanceAgent...")
    agent = ResonanceAgent(input_dim=5, hidden_dims=[32, 16], output_dim=1)
    
    X = torch.randn(100, 5)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16)
    
    trainer = ResonanceTrainer(agent, learning_rate=0.01)
    
    try:
        history = trainer.train(dataloader, epochs=10)
        print(f"   Final loss: {history[-1]['loss']:.6f}")
        print("   Training completed successfully!")
    except Exception as e:
        print(f"   Training error: {e}")
    
    print("\n4. Saving checkpoint...")
    agent.save_checkpoint('output/checkpoint.pt')
    print("   Saved: output/checkpoint.pt")
    
    print("\nDemo complete!")


if __name__ == '__main__':
    import os
    os.makedirs('output', exist_ok=True)
    main()