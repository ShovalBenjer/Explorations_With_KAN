import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.activations.rmt_activation import (
    RMTActivation,
    RMTActivationConfig,
    rho_sinh,
    tau_oscillator,
    phi_harmonics,
)
from src.agents.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "z-ai/glm-5.1"


@dataclass
class ResonanceTrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    batch_size: int = 32
    activation_mode: str = "rho_sinh"
    spectral_radius: float = 1.0
    matrix_dim: int = 8
    use_llm_optimization: bool = True
    llm_optimize_interval: int = 20
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-5


@dataclass
class ResonanceModelConfig:
    input_dim: int = 1
    hidden_dim: int = 16
    output_dim: int = 1
    n_hidden_layers: int = 2
    activation_config: RMTActivationConfig = field(default_factory=RMTActivationConfig)
    activation_mode: str = "rho_sinh"


class ResonanceModel(nn.Module):
    def __init__(self, config: Optional[ResonanceModelConfig] = None):
        super().__init__()
        self.config = config or ResonanceModelConfig()
        self.activation = RMTActivation(self.config.activation_config)

        layers = []
        in_dim = self.config.input_dim
        for i in range(self.config.n_hidden_layers):
            out_dim = self.config.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(RMTActivation(self.config.activation_config))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, self.config.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResonanceAgent:
    def __init__(
        self,
        model_config: Optional[ResonanceModelConfig] = None,
        training_config: Optional[ResonanceTrainingConfig] = None,
        llm: Optional[LLMInterface] = None,
    ):
        self.model_config = model_config or ResonanceModelConfig()
        self.training_config = training_config or ResonanceTrainingConfig()
        self.llm = llm or LLMInterface()
        self.model = ResonanceModel(self.model_config)
        self._training_history: list[dict] = []
        self._best_loss: float = float("inf")
        self._best_state_dict: Optional[dict] = None

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.8,
        noise_level: float = 0.0,
    ) -> tuple[DataLoader, DataLoader]:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(-1)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(-1)

        if noise_level > 0:
            y_tensor = y_tensor + torch.randn_like(y_tensor) * noise_level

        n_total = X_tensor.shape[0]
        n_train = int(n_total * train_ratio)
        indices = torch.randperm(n_total)

        train_dataset = TensorDataset(
            X_tensor[indices[:n_train]],
            y_tensor[indices[:n_train]],
        )
        test_dataset = TensorDataset(
            X_tensor[indices[n_train:]],
            y_tensor[indices[n_train:]],
        )

        train_loader = DataLoader(train_dataset, batch_size=self.training_config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.training_config.batch_size, shuffle=False)

        return train_loader, test_loader

    def train(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        verbose: bool = True,
    ) -> dict:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        criterion = nn.MSELoss()

        patience_counter = 0
        history = []

        for epoch in range(self.training_config.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            test_loss = None
            if test_loader is not None:
                test_loss = self._evaluate(test_loader)

            record = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "test_loss": test_loss,
            }
            history.append(record)
            self._training_history.append(record)

            if avg_train_loss < self._best_loss:
                self._best_loss = avg_train_loss
                self._best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.training_config.epochs} - train_loss: {avg_train_loss:.6f}"
                if test_loss is not None:
                    msg += f" - test_loss: {test_loss:.6f}"
                logger.info(msg)

            if self.training_config.use_llm_optimization and (epoch + 1) % self.training_config.llm_optimize_interval == 0:
                self._llm_optimize(epoch + 1, avg_train_loss, test_loss)

            if patience_counter >= self.training_config.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)

        return {"history": history, "best_loss": self._best_loss}

    def _evaluate(self, data_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _llm_optimize(self, epoch: int, train_loss: float, test_loss: Optional[float]) -> Optional[dict]:
        try:
            spectral_info = self.model.activation.get_spectral_info()

            prompt = (
                f"Current training state at epoch {epoch}:\n"
                f"  Train loss: {train_loss:.6f}\n"
                f"  Test loss: {test_loss:.6f if test_loss else 'N/A'}\n"
                f"  Spectral info: {spectral_info}\n\n"
                "Analyze the resonance patterns and suggest optimal activation parameter adjustments. "
                "Return JSON with keys: 'rho_scale' (float), 'tau_frequency' (float), 'tau_phase' (float), "
                "'phi_base_freq' (float), 'spectral_radius' (float), 'reasoning' (string)."
            )

            suggestions = self.llm.generate_json(
                prompt,
                system_prompt=(
                    "You are an RMT (Random Matrix Theory) activation optimizer. "
                    "Given training progress and spectral information of activation functions, "
                    "suggest parameter adjustments to improve convergence and resonance properties."
                ),
            )

            if "rho_scale" in suggestions and isinstance(suggestions["rho_scale"], (int, float)):
                with torch.no_grad():
                    self.model.activation.rho_scale.fill_(suggestions["rho_scale"])
            if "tau_frequency" in suggestions and isinstance(suggestions["tau_frequency"], (int, float)):
                with torch.no_grad():
                    self.model.activation.tau_frequency.fill_(suggestions["tau_frequency"])
            if "tau_phase" in suggestions and isinstance(suggestions["tau_phase"], (int, float)):
                with torch.no_grad():
                    self.model.activation.tau_phase.fill_(suggestions["tau_phase"])
            if "phi_base_freq" in suggestions and isinstance(suggestions["phi_base_freq"], (int, float)):
                with torch.no_grad():
                    self.model.activation.phi_base_freq.fill_(suggestions["phi_base_freq"])
            if "spectral_radius" in suggestions and isinstance(suggestions["spectral_radius"], (int, float)):
                with torch.no_grad():
                    self.model.activation.spectral_radius.fill_(suggestions["spectral_radius"])

            logger.info(f"LLM optimization applied at epoch {epoch}: {suggestions.get('reasoning', 'N/A')}")
            return suggestions

        except Exception as e:
            logger.warning(f"LLM optimization failed at epoch {epoch}: {e}")
            return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(-1)

        with torch.no_grad():
            output = self.model(X_tensor)

        return output.numpy()

    def compare_activations(self, X: np.ndarray, y: np.ndarray) -> dict:
        modes = ["rho_sinh", "tau_oscillator", "phi_harmonics", "resonance"]
        results = {}

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(-1)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(-1)

        for mode in modes:
            activation = self.model.activation
            with torch.no_grad():
                activated = activation(X_tensor, mode=mode)

            if activated.shape != y_tensor.shape:
                if activated.shape[-1] != y_tensor.shape[-1]:
                    min_dim = min(activated.shape[-1], y_tensor.shape[-1])
                    activated = activated[..., :min_dim]
                    y_tensor_adj = y_tensor[..., :min_dim]
                else:
                    y_tensor_adj = y_tensor
            else:
                y_tensor_adj = y_tensor

            mse = float(torch.mean((activated - y_tensor_adj) ** 2).item())
            results[mode] = {
                "mse": mse,
                "activation_mode": mode,
            }

        return results

    def get_training_history(self) -> list[dict]:
        return self._training_history.copy()

    def get_spectral_info(self) -> dict:
        return self.model.activation.get_spectral_info()

    def get_resonance_spectrum(self, n_points: int = 256) -> dict:
        return self.model.activation.compute_resonance_spectrum(n_points=n_points)