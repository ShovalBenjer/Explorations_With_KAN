import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RhoSinh(nn.Module):
    """rho_sinh: Sinh with resonance scaling.
    
    Applies hyperbolic sine with scaling based on resonance parameters.
    """
    
    def __init__(self, scale: float = 1.0, alpha: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.alpha = nn.Parameter(torch.tensor(alpha))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.sinh(self.alpha * x)


class TauOscillator(nn.Module):
    """tau_oscillator: Parametric oscillator activation.
    
    Models oscillatory behavior with learnable frequency and damping.
    """
    
    def __init__(self, frequency: float = 1.0, damping: float = 0.1):
        super().__init__()
        self.frequency = nn.Parameter(torch.tensor(frequency))
        self.damping = nn.Parameter(torch.tensor(damping))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.frequency * x) * torch.exp(-self.damping * x.abs())


class PhiHarmonics(nn.Module):
    """phi_harmonics: Harmonic series activation.
    
    Combines multiple harmonic frequencies for rich activation patterns.
    """
    
    def __init__(self, num_harmonics: int = 5, base_freq: float = 1.0):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.base_freq = nn.Parameter(torch.tensor(base_freq))
        self.amplitudes = nn.Parameter(torch.ones(num_harmonics) * 0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        for i in range(self.num_harmonics):
            result += self.amplitudes[i] * torch.sin(self.base_freq * (i + 1) * x)
        return result / self.num_harmonics


class ResonanceLayer(nn.Module):
    """ResonanceLayer combining multiple RMT activations.
    
    Mixes outputs from multiple activation types with learnable mixing weights.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_types: Optional[list] = None,
        use_bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        
        if activation_types is None:
            activation_types = ['rho_sinh', 'tau_oscillator', 'phi_harmonics']
        
        self.activations = nn.ModuleList()
        for act_type in activation_types:
            if act_type == 'rho_sinh':
                self.activations.append(RhoSinh())
            elif act_type == 'tau_oscillator':
                self.activations.append(TauOscillator())
            elif act_type == 'phi_harmonics':
                self.activations.append(PhiHarmonics())
        
        num_activations = len(self.activations)
        self.mixing_weights = nn.Parameter(torch.ones(num_activations) / num_activations)
        self.neuron_weights = nn.Parameter(torch.ones(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        weights = F.softmax(self.mixing_weights, dim=0)
        
        result = torch.zeros_like(z)
        for i, act in enumerate(self.activations):
            activated = act(z)
            w = self.neuron_weights.unsqueeze(1) if z.dim() == 2 else self.neuron_weights
            result += activated * w * weights[i]
        
        return result