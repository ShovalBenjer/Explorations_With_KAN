import logging
import uuid
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from kan import KAN, create_dataset

from src.agents.evaluator import EquationEvaluator, ValidationMetrics
from src.agents.storage import EquationStorage, EquationRecord
from src.agents.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

DEFAULT_SYMBOLIC_LIB = ["x", "x^2", "x^3", "x^4", "exp", "log", "sqrt", "tanh", "sin", "abs"]

DEFAULT_KAN_CONFIG = {
    "grid": 5,
    "k": 3,
    "seed": 0,
}

DEFAULT_TRAIN_CONFIG = {
    "opt": "LBFGS",
    "steps": 50,
    "lamb": 5e-5,
    "lamb_entropy": 2.0,
}


@dataclass
class DiscoveryResult:
    equation_id: str
    expression: str
    sympy_expression: Optional[str]
    metrics: ValidationMetrics
    record: EquationRecord
    llm_interpretation: Optional[str] = None


@dataclass
class DiscoveryConfig:
    kan_width: list = field(default_factory=lambda: [2, 1])
    kan_config: dict = field(default_factory=lambda: DEFAULT_KAN_CONFIG)
    train_config: dict = field(default_factory=lambda: DEFAULT_TRAIN_CONFIG)
    symbolic_lib: list = field(default_factory=lambda: DEFAULT_SYMBOLIC_LIB)
    max_iterations: int = 5
    r2_threshold: float = 0.95
    prune_threshold: float = 1e-2
    min_improvement: float = 0.005
    use_llm_hypotheses: bool = True


class EquationDiscoveryAgent:
    def __init__(
        self,
        config: Optional[DiscoveryConfig] = None,
        storage: Optional[EquationStorage] = None,
        evaluator: Optional[EquationEvaluator] = None,
        llm: Optional[LLMInterface] = None,
        storage_dir: Optional[str] = None,
    ):
        self.config = config or DiscoveryConfig()
        self.evaluator = evaluator or EquationEvaluator(
            r2_threshold=self.config.r2_threshold,
        )
        self.storage = storage or EquationStorage(storage_dir=storage_dir)
        self.llm = llm or LLMInterface()
        self._model: Optional[KAN] = None
        self._dataset: Optional[dict] = None
        self._iteration: int = 0
        self._discovery_history: list[dict] = []

    def load_dataset(
        self,
        f: callable,
        n_var: int = 2,
        ranges: list = None,
        train_num: int = 1000,
        test_num: int = 1000,
        noise_level: float = 0.1,
        device: str = "cpu",
        seed: int = 0,
    ) -> dict:
        if ranges is None:
            ranges = [-1, 1]
        self._dataset = create_dataset(
            f=f,
            n_var=n_var,
            train_num=train_num,
            test_num=test_num,
            device=device,
            seed=seed,
        )
        if noise_level > 0:
            noise = torch.randn_like(self._dataset["test_label"]) * noise_level
            self._dataset["test_label"] = self._dataset["test_label"] + noise
        logger.info(f"Loaded dataset: {n_var} variables, {train_num} train, {test_num} test samples")
        return self._dataset

    def load_dataset_from_arrays(
        self,
        train_input: np.ndarray,
        train_label: np.ndarray,
        test_input: np.ndarray,
        test_label: np.ndarray,
    ) -> dict:
        self._dataset = {
            "train_input": torch.tensor(train_input, dtype=torch.float32),
            "train_label": torch.tensor(train_label, dtype=torch.float32),
            "test_input": torch.tensor(test_input, dtype=torch.float32),
            "test_label": torch.tensor(test_label, dtype=torch.float32),
        }
        logger.info(f"Loaded dataset from arrays: train {train_input.shape}, test {test_input.shape}")
        return self._dataset

    def _get_dataset_info(self) -> dict:
        if self._dataset is None:
            return {}
        return {
            "n_var": self._dataset["train_input"].shape[1],
            "train_samples": self._dataset["train_input"].shape[0],
            "test_samples": self._dataset["test_input"].shape[0],
            "train_label_stats": {
                "mean": float(self._dataset["train_label"].mean()),
                "std": float(self._dataset["train_label"].std()),
                "min": float(self._dataset["train_label"].min()),
                "max": float(self._dataset["train_label"].max()),
            },
        }

    def train_kan(
        self,
        width: Optional[list] = None,
        dataset: Optional[dict] = None,
    ) -> KAN:
        width = width or self.config.kan_width
        dataset = dataset or self._dataset

        if dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset first.")

        kan_config = {**self.config.kan_config, "width": width}
        model = KAN(**kan_config)

        train_config = {**self.config.train_config}
        model.fit(dataset, **train_config)

        self._model = model
        self._iteration += 1
        logger.info(f"Trained KAN model (iteration {self._iteration}): width={width}")
        return model

    def prune_kan(self, model: Optional[KAN] = None) -> KAN:
        model = model or self._model
        if model is None:
            raise ValueError("No model to prune. Train a KAN model first.")
        pruned = model.prune()
        self._model = pruned
        logger.info("Pruned KAN model")
        return pruned

    def auto_symbolic(
        self,
        model: Optional[KAN] = None,
        lib: Optional[list] = None,
        r2_threshold: float = 0.1,
    ) -> dict:
        model = model or self._model
        if model is None:
            raise ValueError("No model available. Train a KAN model first.")

        lib = lib or self.config.symbolic_lib

        results = model.auto_symbolic(lib=lib, r2_threshold=r2_threshold)

        symbolic_info = {
            "lib_used": lib,
            "results": str(results) if results else "None",
        }

        try:
            formula = model.symbolic_formula()[0]
            symbolic_info["sympy_expression"] = str(formula)
        except Exception as e:
            logger.warning(f"Could not extract symbolic formula: {e}")
            symbolic_info["sympy_expression"] = None

        self._discovery_history.append({
            "iteration": self._iteration,
            "phase": "symbolic_regression",
            "info": symbolic_info,
        })

        logger.info(f"Auto-symbolic regression complete: {symbolic_info.get('sympy_expression', 'N/A')}")
        return symbolic_info

    def fix_symbolic(
        self,
        model: Optional[KAN] = None,
        fixes: Optional[list] = None,
    ) -> dict:
        model = model or self._model
        if model is None:
            raise ValueError("No model available.")
        if fixes is None:
            return {}

        for layer, inp, out, func in fixes:
            model.fix_symbolic(layer, inp, out, func)

        symbolic_info = {"manual_fixes": fixes}
        try:
            formula = model.symbolic_formula()[0]
            symbolic_info["sympy_expression"] = str(formula)
        except Exception as e:
            logger.warning(f"Could not extract symbolic formula: {e}")
            symbolic_info["sympy_expression"] = None

        return symbolic_info

    def _evaluate_model(
        self,
        model: Optional[KAN] = None,
        expression: str = "",
    ) -> ValidationMetrics:
        model = model or self._model
        if model is None:
            raise ValueError("No model available.")
        if self._dataset is None:
            raise ValueError("No dataset loaded.")

        return self.evaluator.evaluate_kan_model(
            model, self._dataset, expression=expression
        )

    def generate_hypotheses(self, symbolic_results: dict) -> dict:
        if not self.config.use_llm_hypotheses:
            return {"hypotheses": [], "suggested_lib": self.config.symbolic_lib}

        dataset_info = self._get_dataset_info()
        return self.llm.generate_hypothesis(
            symbolic_results=symbolic_results,
            dataset_info=dataset_info,
            iteration=self._iteration,
        )

    def apply_hypotheses(self, hypotheses: dict) -> dict:
        result = {
            "aux_variables": [],
            "updated_lib": self.config.symbolic_lib.copy(),
            "updated_width": self.config.kan_width.copy(),
        }

        for hyp in hypotheses.get("hypotheses", []):
            result["aux_variables"].append({
                "name": hyp.get("name", f"aux_{len(result['aux_variables'])}"),
                "expression": hyp.get("expression", ""),
                "rationale": hyp.get("rationale", ""),
            })

        suggested_lib = hypotheses.get("suggested_lib", [])
        if suggested_lib:
            for func in suggested_lib:
                if func not in result["updated_lib"]:
                    result["updated_lib"].append(func)

        suggested_arch = hypotheses.get("suggested_architecture", {})
        if suggested_arch and "width" in suggested_arch:
            result["updated_width"] = suggested_arch["width"]

        return result

    def run_discovery(
        self,
        f: Optional[callable] = None,
        dataset: Optional[dict] = None,
        n_var: int = 2,
        train_num: int = 1000,
        test_num: int = 1000,
    ) -> DiscoveryResult:
        self._iteration = 0
        self._discovery_history = []

        if dataset is not None:
            self._dataset = dataset
        elif f is not None:
            self.load_dataset(f=f, n_var=n_var, train_num=train_num, test_num=test_num)

        if self._dataset is None:
            raise ValueError("No dataset provided. Pass f or dataset parameter.")

        best_result: Optional[DiscoveryResult] = None
        previous_metrics: Optional[ValidationMetrics] = None
        current_width = self.config.kan_width.copy()
        current_lib = self.config.symbolic_lib.copy()

        for iteration in range(1, self.config.max_iterations + 1):
            self._iteration = iteration
            logger.info(f"=== Discovery iteration {iteration}/{self.config.max_iterations} ===")

            model = self.train_kan(width=current_width)

            pruned = self.prune_kan(model)

            symbolic_results = self.auto_symbolic(pruned, lib=current_lib)

            expression = symbolic_results.get("sympy_expression", "")

            if expression:
                model.fit(self._dataset, **self.config.train_config)
                symbolic_results = self.auto_symbolic(model, lib=current_lib)
                expression = symbolic_results.get("sympy_expression", "")

            metrics = self._evaluate_model(model, expression=expression)

            eq_id = f"eq_{uuid.uuid4().hex[:8]}_iter{iteration}"
            aux_vars = []
            record = EquationRecord(
                equation_id=eq_id,
                expression=expression or "unknown",
                sympy_expression=expression,
                r2_score=metrics.r2_score,
                mse=metrics.mse,
                mae=metrics.mae,
                complexity=metrics.complexity,
                variable_names=[f"x{i}" for i in range(self._dataset["train_input"].shape[1])],
                kan_architecture=current_width,
                symbolic_lib=current_lib,
                iteration=iteration,
                auxiliary_variables=aux_vars,
                hypothesis=None,
                llm_interpretation=None,
            )
            self.storage.store(record)

            result = DiscoveryResult(
                equation_id=eq_id,
                expression=expression or "unknown",
                sympy_expression=expression,
                metrics=metrics,
                record=record,
            )

            if best_result is None or metrics.r2_score > best_result.metrics.r2_score:
                best_result = result

            if metrics.passes_quality_threshold:
                logger.info(f"Quality threshold met at iteration {iteration} (R²={metrics.r2_score:.4f})")
                break

            if not self.evaluator.should_continue_iterating(
                metrics, previous_metrics,
                min_improvement=self.config.min_improvement,
                max_iterations=self.config.max_iterations,
                current_iteration=iteration,
            ):
                logger.info(f"Stopping: insufficient improvement at iteration {iteration}")
                break

            if self.config.use_llm_hypotheses:
                hypotheses = self.generate_hypotheses(symbolic_results)
                applied = self.apply_hypotheses(hypotheses)
                current_lib = applied["updated_lib"]

                if applied["updated_width"] != current_width:
                    current_width = applied["updated_width"]

                aux_vars = applied["aux_variables"]
                record.auxiliary_variables = aux_vars
                record.hypothesis = str(hypotheses.get("hypotheses", []))

            previous_metrics = metrics

        if best_result is not None and self.config.use_llm_hypotheses:
            all_equations = [
                {"expression": r.expression, "r2_score": r.r2_score, "iteration": r.iteration}
                for r in self.storage.list_equations()
            ]
            metrics_dict = {
                "r2_score": best_result.metrics.r2_score,
                "mse": best_result.metrics.mse,
                "mae": best_result.metrics.mae,
            }
            interpretation = self.llm.interpret_results(
                discovered_equations=all_equations,
                dataset_info=self._get_dataset_info(),
                metrics=metrics_dict,
            )
            best_result.llm_interpretation = interpretation
            best_result.record.llm_interpretation = interpretation
            self.storage.store(best_result.record)

        logger.info(f"Discovery complete. Best R²={best_result.metrics.r2_score:.4f}")
        return best_result

    def run_full_pipeline(
        self,
        f: callable,
        n_var: int = 2,
        ranges: list = None,
        train_num: int = 1000,
        test_num: int = 1000,
    ) -> DiscoveryResult:
        return self.run_discovery(f=f, n_var=n_var, ranges=ranges, train_num=train_num, test_num=test_num)