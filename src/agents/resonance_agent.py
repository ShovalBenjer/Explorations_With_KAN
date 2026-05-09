import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import os

from ..activations.rmt_activation import ResonanceLayer


class ResonanceAgent(nn.Module):
    """Neural network using RMT activations for resonance-based learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation_types: Optional[list] = None
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(ResonanceLayer(prev_dim, hidden_dim, activation_types))
            layers.append(nn.Identity())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.optimizer: Optional[optim.Optimizer] = None
        self.global_step = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=0.001)
            self.optimizer = optimizer
        
        if criterion is None:
            criterion = nn.MSELoss()
        
        self.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = self(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss}
    
    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'global_step': self.global_step
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)


class ResonanceTrainer:
    """Trainer for ResonanceAgent with resonance dynamics monitoring."""
    
    def __init__(
        self,
        agent: ResonanceAgent,
        learning_rate: float = 0.001,
        checkpoint_dir: str = 'checkpoints'
    ):
        self.agent = agent
        self.agent.optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
        self.checkpoint_dir = checkpoint_dir
        self.metrics_history: list = []
    
    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        criterion: Optional[nn.Module] = None
    ) -> list:
        if criterion is None:
            criterion = nn.MSELoss()
        
        history = []
        
        for epoch in range(epochs):
            epoch_metrics = self._train_epoch(dataloader, criterion)
            history.append(epoch_metrics)
            self.metrics_history.append(epoch_metrics)
            
            weights = self._get_mixing_weights()
            print(f"Epoch {epoch}, Step {self.agent.global_step}: Loss={epoch_metrics['loss']:.6f}, Resonance mixing: {[f'{w:.3f}' for w in weights]}")
        
        return history
    
    def _train_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        self.agent.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in dataloader:
            self.agent.optimizer.zero_grad()
            output = self.agent(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            self.agent.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.agent.global_step += 1
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def _get_mixing_weights(self) -> list:
        weights = []
        for module in self.agent.network.modules():
            if isinstance(module, ResonanceLayer):
                weights.extend(F.softmax(module.mixing_weights, dim=0).detach().cpu().tolist())
        return weights
    
    def save_checkpoint(self, filename: str = 'resonance_checkpoint.pt') -> None:
        path = os.path.join(self.checkpoint_dir, filename)
        self.agent.save_checkpoint(path)