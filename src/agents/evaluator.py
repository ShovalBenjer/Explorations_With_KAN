import logging
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    r2_score: float
    mse: float
    mae: float
    rmse: float
    max_error: float
    mean_absolute_percentage_error: float
    complexity: int
    passes_quality_threshold: bool


class EquationEvaluator:
    def __init__(
        self,
        r2_threshold: float = 0.9,
        mae_threshold: float = 0.1,
        complexity_penalty: float = 0.01,
    ):
        self.r2_threshold = r2_threshold
        self.mae_threshold = mae_threshold
        self.complexity_penalty = complexity_penalty

    @staticmethod
    def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot

    @staticmethod
    def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def compute_max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.max(np.abs(y_true - y_pred)))

    @staticmethod
    def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = np.abs(y_true) > 1e-10
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100

    @staticmethod
    def compute_complexity(expression: str) -> int:
        ops = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh"]
        count = sum(expression.count(op) for op in ops)
        count += expression.count("^") + expression.count("**")
        count += expression.count("+") + expression.count("-")
        count += expression.count("*") + expression.count("/")
        return max(1, count)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        expression: str = "",
    ) -> ValidationMetrics:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        r2 = self.compute_r2(y_true_flat, y_pred_flat)
        mse = self.compute_mse(y_true_flat, y_pred_flat)
        mae = self.compute_mae(y_true_flat, y_pred_flat)
        rmse = self.compute_rmse(y_true_flat, y_pred_flat)
        max_err = self.compute_max_error(y_true_flat, y_pred_flat)
        mape = self.compute_mape(y_true_flat, y_pred_flat)
        complexity = self.compute_complexity(expression) if expression else 0

        passes = r2 >= self.r2_threshold and mae <= self.mae_threshold

        return ValidationMetrics(
            r2_score=r2,
            mse=mse,
            mae=mae,
            rmse=rmse,
            max_error=max_err,
            mean_absolute_percentage_error=mape,
            complexity=complexity,
            passes_quality_threshold=passes,
        )

    def evaluate_kan_model(
        self,
        model,
        dataset: dict,
        expression: str = "",
    ) -> ValidationMetrics:
        model.eval()
        with torch.no_grad():
            y_pred = model(dataset["test_input"])
        y_true = dataset["test_label"].numpy()
        y_pred_np = y_pred.numpy()
        return self.evaluate(y_true, y_pred_np, expression)

    def compare_equations(
        self,
        metrics_list: list[ValidationMetrics],
    ) -> list[tuple[int, float]]:
        scored = []
        for i, m in enumerate(metrics_list):
            score = m.r2_score - self.complexity_penalty * m.complexity
            scored.append((i, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def should_continue_iterating(
        self,
        current_metrics: ValidationMetrics,
        previous_metrics: Optional[ValidationMetrics] = None,
        min_improvement: float = 0.01,
        max_iterations: int = 10,
        current_iteration: int = 1,
    ) -> bool:
        if current_iteration >= max_iterations:
            return False
        if current_metrics.passes_quality_threshold:
            return False
        if previous_metrics is None:
            return True
        improvement = current_metrics.r2_score - previous_metrics.r2_score
        return improvement >= min_improvement
