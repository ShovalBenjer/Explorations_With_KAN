import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List


def visualize_activations(
    activations,
    x_range: tuple = (-5, 5),
    num_points: int = 1000,
    save_path: Optional[str] = None
):
    x = torch.linspace(x_range[0], x_range[1], num_points)
    y = activations(x).detach().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y)
    plt.title('RMT Activation Function')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def visualize_layer_mixing(
    layer,
    save_path: Optional[str] = None
):
    import torch.nn.functional as F
    weights = F.softmax(layer.mixing_weights, dim=0).detach().numpy()
    
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(weights)), weights)
    plt.title('Resonance Layer Mixing Weights')
    plt.xlabel('Activation Type')
    plt.ylabel('Weight')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def visualize_training_history(
    history: List[dict],
    metric: str = 'loss',
    save_path: Optional[str] = None
):
    values = [h[metric] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(values)
    plt.title(f'Training {metric.capitalize()} History')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_activation_patterns(
    model,
    num_samples: int = 100,
    save_path: Optional[str] = None
):
    model.eval()
    patterns = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            x = torch.randn(1, model.network[0].in_features)
            features = model.network[0](x).numpy()
            patterns.append(features)
    
    patterns = np.vstack(patterns)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(patterns.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Activation Patterns')
    plt.xlabel('Sample')
    plt.ylabel('Neuron')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()