import pytest
import numpy as np
import torch
import tempfile
import os

from src.agents.evaluator import EquationEvaluator, ValidationMetrics
from src.agents.storage import EquationStorage, EquationRecord
from src.agents.llm_interface import LLMInterface


class TestEquationEvaluator:
    def setup_method(self):
        self.evaluator = EquationEvaluator(r2_threshold=0.9, mae_threshold=0.1)

    def test_compute_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert self.evaluator.compute_r2(y, y) == pytest.approx(1.0)

    def test_compute_r2_poor(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        r2 = self.evaluator.compute_r2(y_true, y_pred)
        assert r2 < 0.5

    def test_compute_r2_constant(self):
        y_true = np.array([2.0, 2.0, 2.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        assert self.evaluator.compute_r2(y_true, y_pred) == 1.0

    def test_compute_mse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert self.evaluator.compute_mse(y_true, y_pred) == pytest.approx(0.0)

    def test_compute_mse_nonzero(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 3.0])
        assert self.evaluator.compute_mse(y_true, y_pred) == pytest.approx(1.0)

    def test_compute_mae(self):
        y_true = np.array([1.0, 3.0])
        y_pred = np.array([2.0, 5.0])
        assert self.evaluator.compute_mae(y_true, y_pred) == pytest.approx(1.5)

    def test_compute_rmse(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 3.0])
        assert self.evaluator.compute_rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_compute_complexity(self):
        assert self.evaluator.compute_complexity("sin(x) + exp(x)") == 3
        assert self.evaluator.compute_complexity("x") == 1

    def test_evaluate_returns_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.01, 2.02, 3.03, 4.04, 5.05])
        metrics = self.evaluator.evaluate(y_true, y_pred, expression="x + 1")
        assert isinstance(metrics, ValidationMetrics)
        assert metrics.r2_score > 0.99
        assert metrics.passes_quality_threshold

    def test_evaluate_poor_quality(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        metrics = self.evaluator.evaluate(y_true, y_pred)
        assert not metrics.passes_quality_threshold
        assert metrics.r2_score < 0

    def test_compare_equations(self):
        m1 = ValidationMetrics(
            r2_score=0.95, mse=0.01, mae=0.05, rmse=0.1,
            max_error=0.2, mean_absolute_percentage_error=5.0,
            complexity=3, passes_quality_threshold=True,
        )
        m2 = ValidationMetrics(
            r2_score=0.80, mse=0.05, mae=0.15, rmse=0.22,
            max_error=0.5, mean_absolute_percentage_error=15.0,
            complexity=1, passes_quality_threshold=False,
        )
        ranked = self.evaluator.compare_equations([m1, m2])
        assert ranked[0][0] == 0

    def test_should_continue_iterating(self):
        current = ValidationMetrics(
            r2_score=0.7, mse=0.1, mae=0.2, rmse=0.3,
            max_error=0.5, mean_absolute_percentage_error=10.0,
            complexity=5, passes_quality_threshold=False,
        )
        previous = ValidationMetrics(
            r2_score=0.6, mse=0.15, mae=0.25, rmse=0.38,
            max_error=0.6, mean_absolute_percentage_error=12.0,
            complexity=5, passes_quality_threshold=False,
        )
        assert self.evaluator.should_continue_iterating(current, previous)
        assert not self.evaluator.should_continue_iterating(current, current, min_improvement=0.5)
        assert not self.evaluator.should_continue_iterating(current, max_iterations=1, current_iteration=1)


class TestEquationStorage:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.storage = EquationStorage(storage_dir=self.tmpdir)

    def _make_record(self, eq_id="test_eq", r2=0.95, iteration=1):
        return EquationRecord(
            equation_id=eq_id,
            expression="sin(x) + exp(y)",
            sympy_expression="sin(x) + exp(y)",
            r2_score=r2,
            mse=0.01,
            mae=0.05,
            complexity=3,
            variable_names=["x", "y"],
            kan_architecture=[2, 1],
            symbolic_lib=["sin", "exp"],
            iteration=iteration,
            auxiliary_variables=[],
            hypothesis=None,
            llm_interpretation=None,
        )

    def test_store_and_retrieve(self):
        record = self._make_record()
        self.storage.store(record)
        retrieved = self.storage.retrieve("test_eq")
        assert retrieved is not None
        assert retrieved.expression == "sin(x) + exp(y)"
        assert retrieved.r2_score == 0.95

    def test_list_equations(self):
        self.storage.store(self._make_record("eq1", r2=0.8))
        self.storage.store(self._make_record("eq2", r2=0.95))
        eqs = self.storage.list_equations()
        assert len(eqs) == 2
        assert eqs[0].r2_score >= eqs[1].r2_score

    def test_search_by_r2(self):
        self.storage.store(self._make_record("eq1", r2=0.8))
        self.storage.store(self._make_record("eq2", r2=0.95))
        results = self.storage.search_by_r2(0.9)
        assert len(results) == 1
        assert results[0].equation_id == "eq2"

    def test_search_by_complexity(self):
        self.storage.store(self._make_record("eq1"))
        results = self.storage.search_by_complexity(10)
        assert len(results) == 1

    def test_get_best_equation(self):
        self.storage.store(self._make_record("eq1", r2=0.7))
        self.storage.store(self._make_record("eq2", r2=0.95))
        best = self.storage.get_best_equation()
        assert best.r2_score == 0.95

    def test_delete(self):
        self.storage.store(self._make_record("eq1"))
        assert self.storage.delete("eq1")
        assert self.storage.retrieve("eq1") is None

    def test_persistence(self):
        self.storage.store(self._make_record("persist_eq", r2=0.88))
        storage2 = EquationStorage(storage_dir=self.tmpdir)
        retrieved = storage2.retrieve("persist_eq")
        assert retrieved is not None
        assert retrieved.r2_score == 0.88

    def test_export_all(self):
        self.storage.store(self._make_record("eq1"))
        exported = self.storage.export_all()
        assert len(exported) == 1
        assert isinstance(exported[0], dict)

    def test_record_serialization(self):
        record = self._make_record()
        json_str = record.to_json()
        restored = EquationRecord.from_json(json_str)
        assert restored.equation_id == record.equation_id
        assert restored.r2_score == record.r2_score


class TestLLMInterface:
    def test_init_defaults(self):
        llm = LLMInterface()
        assert llm.model == "z-ai/glm-5.1"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 4096

    def test_init_custom(self):
        llm = LLMInterface(
            model="custom-model",
            temperature=0.5,
            max_tokens=2048,
        )
        assert llm.model == "custom-model"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 2048


class TestDiscoveryConfig:
    def test_default_config(self):
        from src.agents.equation_discovery import DiscoveryConfig
        config = DiscoveryConfig()
        assert config.max_iterations == 5
        assert config.r2_threshold == 0.95
        assert config.use_llm_hypotheses is True
        assert "sin" in config.symbolic_lib
        assert "exp" in config.symbolic_lib

    def test_custom_config(self):
        from src.agents.equation_discovery import DiscoveryConfig
        config = DiscoveryConfig(
            max_iterations=3,
            r2_threshold=0.8,
            use_llm_hypotheses=False,
        )
        assert config.max_iterations == 3
        assert config.r2_threshold == 0.8
        assert config.use_llm_hypotheses is False


class TestEquationDiscoveryAgentInit:
    def test_agent_creation(self):
        from src.agents.equation_discovery import EquationDiscoveryAgent, DiscoveryConfig
        config = DiscoveryConfig(use_llm_hypotheses=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = EquationDiscoveryAgent(config=config, storage_dir=tmpdir)
            assert agent.config.use_llm_hypotheses is False
            assert agent._iteration == 0

    def test_load_dataset_from_arrays(self):
        from src.agents.equation_discovery import EquationDiscoveryAgent, DiscoveryConfig
        config = DiscoveryConfig(use_llm_hypotheses=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = EquationDiscoveryAgent(config=config, storage_dir=tmpdir)
            train_x = np.random.randn(100, 2).astype(np.float32)
            train_y = (np.sin(train_x[:, 0:1]) + np.cos(train_x[:, 1:2])).astype(np.float32)
            test_x = np.random.randn(50, 2).astype(np.float32)
            test_y = (np.sin(test_x[:, 0:1]) + np.cos(test_x[:, 1:2])).astype(np.float32)
            dataset = agent.load_dataset_from_arrays(train_x, train_y, test_x, test_y)
            assert dataset["train_input"].shape == (100, 2)
            assert dataset["train_label"].shape == (100, 1)
